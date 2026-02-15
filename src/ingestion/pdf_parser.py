import re
from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader

from src.exceptions import ParsingError
from src.ingestion.document_structure import DocumentChunk, DocumentLevel, LegalReference
from src.logger import get_logger

logger = get_logger("CELEXPDFParser")


class CELEXPDFParser:
    """Parser tuned for EU regulation hierarchy: recital/chapter/section/article/subsection/point."""

    CHAPTER_RE = re.compile(r"^CHAPTER\s+([IVXLC]+)", re.IGNORECASE)
    SECTION_RE = re.compile(r"^Section\s+(\d+)", re.IGNORECASE)
    ARTICLE_RE = re.compile(r"^Article\s+(\d+)", re.IGNORECASE)
    RECITAL_RE = re.compile(r"^\(\s*(\d+)\s*\)")
    SUBSECTION_RE = re.compile(r"^(\d+)\.\s+(.*)")
    POINT_RE = re.compile(r"^\(([a-z])\)\s+(.*)", re.IGNORECASE)

    LEGISLATION_START_MARKER = "HAVE ADOPTED THIS REGULATION"

    ROMAN_TO_ARABIC = {
        "I": "1",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
        "XI": "11",
        "XII": "12",
    }

    def parse(self, pdf_path: str) -> List[DocumentChunk]:
        try:
            pages = PyPDFLoader(pdf_path).load()
        except Exception as exc:
            raise ParsingError(f"Failed to load PDF: {exc}") from exc

        lines = self._flatten_pages(pages)
        chunks: List[DocumentChunk] = []

        current_chapter: str | None = None
        current_chapter_num: str | None = None
        current_section: str | None = None
        current_article: str | None = None

        in_recital_phase = True
        recital_buffer: list[str] = []
        recital_id: str | None = None
        recital_page = 0

        article_lines: list[tuple[str, int]] = []
        article_page = 0

        for text, page in lines:
            if in_recital_phase and self.LEGISLATION_START_MARKER in text.upper():
                in_recital_phase = False
                if recital_id and recital_buffer:
                    chunks.append(
                        self._build_recital_chunk(recital_id, recital_buffer, recital_page)
                    )
                recital_buffer = []
                recital_id = None
                continue

            chapter_match = self.CHAPTER_RE.match(text)
            if chapter_match:
                current_chapter = chapter_match.group(1).upper()
                current_chapter_num = self.ROMAN_TO_ARABIC.get(current_chapter, current_chapter)
                continue

            section_match = self.SECTION_RE.match(text)
            if section_match:
                current_section = section_match.group(1)
                continue

            if in_recital_phase:
                recital_match = self.RECITAL_RE.match(text)
                if recital_match:
                    if recital_id and recital_buffer:
                        chunks.append(
                            self._build_recital_chunk(recital_id, recital_buffer, recital_page)
                        )
                    recital_id = recital_match.group(1)
                    recital_page = page
                    recital_buffer = [text]
                elif recital_buffer:
                    recital_buffer.append(text)
                continue

            article_match = self.ARTICLE_RE.match(text)
            if article_match:
                if current_article and article_lines:
                    chunks.extend(
                        self._build_article_chunks(
                            article=current_article,
                            lines=article_lines,
                            page=article_page,
                            chapter=current_chapter,
                            chapter_num=current_chapter_num,
                            section=current_section,
                        )
                    )
                current_article = article_match.group(1)
                article_page = page
                article_lines = [text]
                continue

            if current_article:
                article_lines.append(text)

        if recital_id and recital_buffer:
            chunks.append(self._build_recital_chunk(recital_id, recital_buffer, recital_page))

        if current_article and article_lines:
            chunks.extend(
                self._build_article_chunks(
                    article=current_article,
                    lines=article_lines,
                    page=article_page,
                    chapter=current_chapter,
                    chapter_num=current_chapter_num,
                    section=current_section,
                )
            )

        logger.info("CELEX parsing completed", extra={"chunks": len(chunks)})
        return chunks

    def _build_recital_chunk(self, recital_id: str, lines: list[str], page: int) -> DocumentChunk:
        return DocumentChunk(
            content="\n".join(lines),
            reference=LegalReference(recital=recital_id),
            page=page,
            chunk_id=f"recital_{recital_id}",
            level=DocumentLevel.RECITAL,
        )

    def _build_article_chunks(
        self,
        article: str,
        lines: list[tuple[str, int]] | list[str],
        page: int,
        chapter: str | None,
        chapter_num: str | None,
        section: str | None,
    ) -> List[DocumentChunk]:
        text_lines = [line if isinstance(line, str) else line[0] for line in lines]
        full_text = "\n".join(text_lines)
        chunks: list[DocumentChunk] = [
            DocumentChunk(
                content=full_text,
                reference=LegalReference(
                    chapter=chapter,
                    chapter_num=chapter_num,
                    section=section,
                    article=article,
                ),
                page=page,
                chunk_id=f"article_{article}",
                level=DocumentLevel.ARTICLE,
            )
        ]

        current_subsection: str | None = None
        subsection_lines: list[str] = []
        current_point: str | None = None
        point_lines: list[str] = []

        def flush_point():
            nonlocal point_lines, current_point, current_subsection
            if current_point and point_lines:
                chunks.append(
                    DocumentChunk(
                        content=" ".join(point_lines),
                        reference=LegalReference(
                            chapter=chapter,
                            chapter_num=chapter_num,
                            section=section,
                            article=article,
                            subsection=current_subsection,
                            point=current_point,
                        ),
                        page=page,
                        chunk_id=f"article_{article}_s{current_subsection}_p{current_point}",
                        level=DocumentLevel.POINT,
                    )
                )
            point_lines = []
            current_point = None

        def flush_subsection():
            nonlocal subsection_lines, current_subsection
            flush_point()
            if current_subsection and subsection_lines:
                chunks.append(
                    DocumentChunk(
                        content=" ".join(subsection_lines),
                        reference=LegalReference(
                            chapter=chapter,
                            chapter_num=chapter_num,
                            section=section,
                            article=article,
                            subsection=current_subsection,
                        ),
                        page=page,
                        chunk_id=f"article_{article}_s{current_subsection}",
                        level=DocumentLevel.SUBSECTION,
                    )
                )
            subsection_lines = []

        for raw_line in text_lines[1:]:
            subsection_match = self.SUBSECTION_RE.match(raw_line)
            point_match = self.POINT_RE.match(raw_line)

            if subsection_match:
                flush_subsection()
                current_subsection = subsection_match.group(1)
                line_body = subsection_match.group(2).strip()
                subsection_lines = [line_body] if line_body else []
                continue

            if point_match and current_subsection:
                flush_point()
                current_point = point_match.group(1).lower()
                line_body = point_match.group(2).strip()
                point_lines = [line_body] if line_body else []
                subsection_lines.append(raw_line)
                continue

            if current_point:
                point_lines.append(raw_line)

            if current_subsection:
                subsection_lines.append(raw_line)

        flush_subsection()
        return chunks

    @staticmethod
    def _flatten_pages(pages) -> List[Tuple[str, int]]:
        output: List[Tuple[str, int]] = []
        for page in pages:
            page_num = page.metadata.get("page", 0)
            for line in page.page_content.split("\n"):
                clean = line.strip()
                if clean:
                    output.append((clean, page_num))
        return output
