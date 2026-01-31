import re
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader

from src.ingestion.document_structure import (DocumentChunk,LegalReference,DocumentLevel)
from src.logger import get_logger
from src.exceptions import ParsingError

logger = get_logger("CELEXPDFParser")


class CELEXPDFParser:

    CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVX]+)", re.IGNORECASE)
    SECTION_RE = re.compile(r"^Section\s+(\d+)", re.IGNORECASE)
    ARTICLE_RE = re.compile(r"^Article\s+(\d+)", re.IGNORECASE)
    RECITAL_RE = re.compile(r"^\(\s*(\d+)\s*\)")

  
    LEGISLATION_START_MARKER = "HAVE ADOPTED THIS REGULATION"

    def parse(self, pdf_path: str) -> List[DocumentChunk]:
        """
        Parse CELEX PDF into atomic DocumentChunks.
        """
        try:
            pages = PyPDFLoader(pdf_path).load()
        except Exception as e:
            raise ParsingError(f"Failed to load PDF: {e}")

        lines = self._flatten_pages(pages)

        chunks: List[DocumentChunk] = []

        current_chapter: str | None = None
        current_section: str | None = None
        current_chunk: DocumentChunk | None = None
        in_recital_phase: bool = True

        for text, page in lines:

            # ---------- Phase switch ----------
            if in_recital_phase and self.LEGISLATION_START_MARKER in text.upper():
                in_recital_phase = False
                continue

            # ---------- Chapter ----------
            chapter_match = self.CHAPTER_RE.match(text)
            if chapter_match:
                current_chapter = chapter_match.group(1)
                continue

            # ---------- Section ----------
            section_match = self.SECTION_RE.match(text)
            if section_match:
                current_section = section_match.group(1)
                continue

            # ---------- Recitals ----------
            if in_recital_phase:
                recital_match = self.RECITAL_RE.match(text)
                if recital_match:
                    if current_chunk:
                        chunks.append(current_chunk)

                    recital_id = recital_match.group(1)
                    current_chunk = DocumentChunk(
                        content=text,
                        reference=LegalReference(recital=recital_id),
                        page=page,
                        chunk_id=f"recital_{recital_id}",
                        level=DocumentLevel.RECITAL
                    )
                elif current_chunk:
                    current_chunk.content += " " + text

                continue

            # ---------- Articles ----------
            article_match = self.ARTICLE_RE.match(text)
            if article_match:
                if current_chunk:
                    chunks.append(current_chunk)

                article_id = article_match.group(1)
                current_chunk = DocumentChunk(
                    content="",
                    reference=LegalReference(
                        chapter=current_chapter,
                        section=current_section,
                        article=article_id
                    ),
                    page=page,
                    chunk_id=f"article_{article_id}",
                    level=DocumentLevel.ARTICLE
                )
                continue

            # ---------- Accumulate article text ----------
            if current_chunk:
                current_chunk.content += "\n" + text

        # ---------- Flush last chunk ----------
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            "CELEX parsing completed",
            extra={
                "chunks": len(chunks),
                "chapters_seen": sorted(
                    {c.reference.chapter for c in chunks if c.reference.chapter}
                ),
                "sections_seen": sorted(
                    {c.reference.section for c in chunks if c.reference.section}
                ),
            },
        )

        return chunks

    @staticmethod
    def _flatten_pages(pages) -> List[Tuple[str, int]]:
        """
        Convert PDF pages into a flat stream of (line_text, page_number).
        """
        output: List[Tuple[str, int]] = []

        for page in pages:
            page_num = page.metadata.get("page", 0)
            for line in page.page_content.split("\n"):
                clean = line.strip()
                if clean:
                    output.append((clean, page_num))

        return output
