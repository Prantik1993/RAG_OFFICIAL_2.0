import re
from typing import List, Optional
from dataclasses import dataclass
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from src.logger import get_logger
from src.exceptions import ParsingError

logger = get_logger("EnhancedParser")


@dataclass
class LegalChunk:
    content: str
    recital: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    point: Optional[str] = None
    subpoint: Optional[str] = None
    page: int = 0

    def to_metadata(self) -> dict:
        meta = {"page": self.page}

        if self.recital:
            meta["recital"] = self.recital
            meta["level"] = "recital"
        elif self.chapter:
            meta["chapter"] = self.chapter
            meta["level"] = "chapter"

            if self.section:
                meta["section"] = self.section
                meta["level"] = "section"

            if self.article:
                meta["article"] = self.article
                meta["level"] = "article"

            if self.point:
                meta["point"] = self.point
                meta["level"] = "point"

            if self.subpoint:
                meta["subpoint"] = self.subpoint
                meta["level"] = "subpoint"

        ref = []
        if self.recital:
            ref.append(f"Recital {self.recital}")
        else:
            if self.chapter:
                ref.append(f"Chapter {self.chapter}")
            if self.section:
                ref.append(f"Section {self.section}")
            if self.article:
                ref.append(f"Article {self.article}")
            if self.point:
                ref.append(f"Point {self.point}")
            if self.subpoint:
                ref.append(f"Subpoint {self.subpoint}")

        meta["reference_path"] = " → ".join(ref) if ref else "Document"
        return meta

    def to_document(self) -> Document:
        return Document(page_content=self.content, metadata=self.to_metadata())


class EnhancedLegalParser:
    """
    GDPR / CELEX compliant hierarchical parser
    """

    RECITAL_PATTERN = re.compile(r'^\(\s*(\d+)\s*\)')
    CHAPTER_PATTERN = re.compile(r'^CHAPTER\s+([IVX]+)', re.IGNORECASE)
    SECTION_PATTERN = re.compile(r'^Section\s+(\d+)', re.IGNORECASE)
    ARTICLE_PATTERN = re.compile(r'^Article\s+(\d+)', re.IGNORECASE)
    POINT_PATTERN = re.compile(r'^(\d+)\.\s+')
    SUBPOINT_PATTERN = re.compile(r'^\(([a-z])\)')

    def __init__(self):
        self.reset_context()

    def _roman_to_arabic(self, roman: str) -> str:
        roman_map = {
            'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
            'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
            'XI': '11', 'XII': '12'
        }
        return roman_map.get(roman.upper(), roman)

    def reset_context(self):
        self.current_chapter = None
        self.current_section = None
        self.current_article = None
        self.current_point = None
        self.in_recital_phase = True
        self.section_active = False

    def parse(self, pdf_path: str) -> List[LegalChunk]:
        try:
            pages = PyPDFLoader(pdf_path).load()
        except Exception as e:
            raise ParsingError(f"Failed to load PDF: {e}")

        chunks: List[LegalChunk] = []
        current_chunk: Optional[LegalChunk] = None

        for page in pages:
            page_num = page.metadata.get("page", 0)
            lines = page.page_content.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # ---- EXIT RECITAL PHASE ----
                if self.in_recital_phase and self.CHAPTER_PATTERN.match(line):
                    self.in_recital_phase = False
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = None

                # ---- RECITAL ----
                if self.in_recital_phase:
                    m = self.RECITAL_PATTERN.match(line)
                    if m:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = LegalChunk(
                            content=line,
                            recital=m.group(1),
                            page=page_num
                        )
                    elif current_chunk:
                        current_chunk.content += " " + line
                    continue

                # ---- CHAPTER ----
                chapter_match = self.CHAPTER_PATTERN.match(line)
                if chapter_match:
                    if current_chunk:
                        chunks.append(current_chunk)

                    self.current_chapter = self._roman_to_arabic(chapter_match.group(1))
                    self.current_section = None
                    self.section_active = False
                    self.current_article = None
                    self.current_point = None

                    current_chunk = LegalChunk(
                        content=line,
                        chapter=self.current_chapter,
                        page=page_num
                    )
                    continue

                # ---- SECTION ----
                section_match = self.SECTION_PATTERN.match(line)
                if section_match:
                    if current_chunk:
                        chunks.append(current_chunk)

                    self.current_section = section_match.group(1)
                    self.section_active = True
                    self.current_article = None
                    self.current_point = None

                    current_chunk = LegalChunk(
                        content=line,
                        chapter=self.current_chapter,
                        section=self.current_section,
                        page=page_num
                    )
                    continue

                # ---- ARTICLE ----
                article_match = self.ARTICLE_PATTERN.match(line)
                if article_match:
                    if current_chunk:
                        chunks.append(current_chunk)

                    self.current_article = article_match.group(1)
                    self.current_point = None

                    current_chunk = LegalChunk(
                        content=line,
                        chapter=self.current_chapter,
                        section=self.current_section if self.section_active else None,
                        article=self.current_article,
                        page=page_num
                    )
                    continue

                # ---- POINT ----
                point_match = self.POINT_PATTERN.match(line)
                if point_match:
                    if current_chunk:
                        chunks.append(current_chunk)

                    self.current_point = point_match.group(1)

                    current_chunk = LegalChunk(
                        content=line,
                        chapter=self.current_chapter,
                        section=self.current_section if self.section_active else None,
                        article=self.current_article,
                        point=self.current_point,
                        page=page_num
                    )
                    continue

                # ---- SUBPOINT ----
                subpoint_match = self.SUBPOINT_PATTERN.match(line)
                if subpoint_match:
                    if current_chunk:
                        chunks.append(current_chunk)

                    current_chunk = LegalChunk(
                        content=line,
                        chapter=self.current_chapter,
                        section=self.current_section if self.section_active else None,
                        article=self.current_article,
                        point=self.current_point,
                        subpoint=subpoint_match.group(1),
                        page=page_num
                    )
                    continue

                # ---- CONTENT ----
                if current_chunk:
                    current_chunk.content += "\n" + line

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Parsed {len(chunks)} hierarchical chunks")
        return chunks