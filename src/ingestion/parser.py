from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.exceptions import ParsingError
from src.logger import get_logger

log = get_logger("Parser")

# ── Compiled patterns ─────────────────────────────────────────────────────────
_RE_RECITAL   = re.compile(r'^\(\s*(\d{1,3})\s*\)\s+\S')          # (1) text…
_RE_CHAPTER   = re.compile(r'^CHAPTER\s+([IVX]+)\s*$', re.I)
_RE_SECTION   = re.compile(r'^Section\s+(\d+)\s*$', re.I)
_RE_ARTICLE   = re.compile(r'^Article\s+(\d+)\s*$', re.I)
_RE_POINT     = re.compile(r'^(\d+)\.\s+\S')                        # "1. Text…"
_RE_SUBPOINT  = re.compile(r'^\(([a-z])\)\s+\S')                    # "(a) text…"

# Lines to skip entirely
_RE_SKIP = re.compile(
    r'^(L\s+\d+/|EN\s+Official|4\.5\.2016|OJ\s+[CL]|\d+\s*$)',
    re.I,
)

_ROMAN = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
    "XI": "11", "XII": "12",
}


# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class _Ctx:
    """Mutable parse context shared across lines."""
    in_recitals: bool = True
    chapter:  Optional[str] = None
    section:  Optional[str] = None
    article:  Optional[str] = None
    point:    Optional[str] = None
    # section is only active while inside the same chapter
    section_chapter: Optional[str] = None


@dataclass
class LegalChunk:
    content:  str
    page:     int
    recital:  Optional[str] = None
    chapter:  Optional[str] = None
    section:  Optional[str] = None
    article:  Optional[str] = None
    point:    Optional[str] = None
    subpoint: Optional[str] = None

    # ── derived ───────────────────────────────────────────────────────────────
    @property
    def reference_path(self) -> str:
        parts: list[str] = []
        if self.recital:
            parts.append(f"Recital {self.recital}")
        else:
            if self.chapter:
                parts.append(f"Chapter {self.chapter}")
            if self.section:
                parts.append(f"Section {self.section}")
            if self.article:
                parts.append(f"Article {self.article}")
            if self.point:
                parts.append(f"Point {self.point}")
            if self.subpoint:
                parts.append(f"Subpoint ({self.subpoint})")
        return " → ".join(parts) or "Document"

    @property
    def level(self) -> str:
        if self.recital:  return "recital"
        if self.subpoint: return "subpoint"
        if self.point:    return "point"
        if self.article:  return "article"
        if self.section:  return "section"
        if self.chapter:  return "chapter"
        return "document"

    def to_document(self) -> Document:
        meta: dict = {
            "page":           self.page,
            "level":          self.level,
            "reference_path": self.reference_path,
        }
        for attr in ("recital", "chapter", "section", "article", "point", "subpoint"):
            val = getattr(self, attr)
            if val is not None:
                meta[attr] = val
        return Document(page_content=self.content.strip(), metadata=meta)


# ── Parser ────────────────────────────────────────────────────────────────────
class GDPRParser:
    """
    Deterministic, regex-only hierarchical parser for the GDPR CELEX PDF.
    Produces one LegalChunk per structural unit.
    """

    def parse(self, pdf_path: str | Path) -> list[LegalChunk]:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception as exc:
            raise ParsingError(f"Cannot load PDF '{pdf_path}': {exc}") from exc

        chunks: list[LegalChunk] = []
        current: Optional[LegalChunk] = None
        ctx = _Ctx()

        def _flush() -> None:
            nonlocal current
            if current and current.content.strip():
                chunks.append(current)
            current = None

        for page_doc in pages:
            page_num: int = page_doc.metadata.get("page", 0)
            for raw_line in page_doc.page_content.splitlines():
                line = raw_line.strip()
                if not line or _RE_SKIP.match(line):
                    continue

                # ── Exit recital phase on first CHAPTER heading ───────────────
                if ctx.in_recitals and _RE_CHAPTER.match(line):
                    ctx.in_recitals = False

                # ── RECITAL ───────────────────────────────────────────────────
                if ctx.in_recitals:
                    m = _RE_RECITAL.match(line)
                    if m:
                        _flush()
                        current = LegalChunk(
                            content=line, page=page_num, recital=m.group(1)
                        )
                    elif current:
                        current.content += " " + line
                    continue

                # ── CHAPTER ───────────────────────────────────────────────────
                m = _RE_CHAPTER.match(line)
                if m:
                    _flush()
                    ctx.chapter  = _ROMAN.get(m.group(1).upper(), m.group(1))
                    ctx.section  = None
                    ctx.article  = None
                    ctx.point    = None
                    ctx.section_chapter = None
                    current = LegalChunk(
                        content=line, page=page_num, chapter=ctx.chapter
                    )
                    continue

                # ── SECTION ───────────────────────────────────────────────────
                m = _RE_SECTION.match(line)
                if m:
                    _flush()
                    ctx.section  = m.group(1)
                    ctx.article  = None
                    ctx.point    = None
                    ctx.section_chapter = ctx.chapter
                    current = LegalChunk(
                        content=line, page=page_num,
                        chapter=ctx.chapter, section=ctx.section,
                    )
                    continue

                # ── ARTICLE ───────────────────────────────────────────────────
                m = _RE_ARTICLE.match(line)
                if m:
                    _flush()
                    ctx.article = m.group(1)
                    ctx.point   = None
                    # section only carried if same chapter
                    sec = ctx.section if ctx.section_chapter == ctx.chapter else None
                    current = LegalChunk(
                        content=line, page=page_num,
                        chapter=ctx.chapter, section=sec, article=ctx.article,
                    )
                    continue

                # ── POINT (only inside an article) ────────────────────────────
                if ctx.article:
                    m = _RE_POINT.match(line)
                    if m:
                        _flush()
                        ctx.point = m.group(1)
                        sec = ctx.section if ctx.section_chapter == ctx.chapter else None
                        current = LegalChunk(
                            content=line, page=page_num,
                            chapter=ctx.chapter, section=sec,
                            article=ctx.article, point=ctx.point,
                        )
                        continue

                # ── SUBPOINT (only inside a point) ────────────────────────────
                if ctx.point:
                    m = _RE_SUBPOINT.match(line)
                    if m:
                        _flush()
                        sec = ctx.section if ctx.section_chapter == ctx.chapter else None
                        current = LegalChunk(
                            content=line, page=page_num,
                            chapter=ctx.chapter, section=sec,
                            article=ctx.article, point=ctx.point,
                            subpoint=m.group(1),
                        )
                        continue

                # ── continuation ──────────────────────────────────────────────
                if current:
                    current.content += "\n" + line

        _flush()
        log.info(f"Parsed {len(chunks)} chunks from '{pdf_path}'")
        return chunks
