from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class DocumentLevel(str, Enum):
    RECITAL = "recital"
    ARTICLE = "article"
    SUBSECTION = "subsection"
    POINT = "point"


@dataclass(frozen=True)
class LegalReference:
    recital: Optional[str] = None
    chapter: Optional[str] = None
    chapter_num: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    subsection: Optional[str] = None
    point: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "recital": self.recital,
            "chapter": self.chapter,
            "chapter_num": self.chapter_num,
            "section": self.section,
            "article": self.article,
            "subsection": self.subsection,
            "point": self.point,
        }


@dataclass
class DocumentChunk:
    content: str
    reference: LegalReference
    page: int
    chunk_id: str
    level: DocumentLevel

    def to_langchain_document(self):
        from langchain_core.documents import Document

        return Document(
            page_content=self.content.strip(),
            metadata={
                "page": self.page,
                "chunk_id": self.chunk_id,
                "level": self.level.value,
                **self.reference.to_metadata(),
            },
        )
