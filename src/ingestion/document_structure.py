from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class DocumentLevel(str, Enum):
    """
    Logical level of a document chunk.

    Kept intentionally minimal.
    Used by:
    - Hybrid retrieval routing
    - UI display
    - Debugging / analytics
    """
    RECITAL = "recital"
    ARTICLE = "article"


@dataclass(frozen=True)
class LegalReference:
    """
    Flat legal reference metadata.

    DESIGN PRINCIPLE:
    - TAG structure, do NOT reconstruct hierarchy
    - Optimized for metadata filtering in vector databases
    """

    recital: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None

    def to_metadata(self) -> Dict[str, Any]:
        """
        Convert reference to vector-store-friendly metadata.
        """
        return {
            "recital": self.recital,
            "chapter": self.chapter,
            "section": self.section,
            "article": self.article,
        }


@dataclass
class DocumentChunk:
    """
    Atomic ingestion unit.

    One chunk represents:
    - One Recital OR
    - One Article (pre-splitting)

    NOTE:
    - Further splitting happens in the ingestion pipeline
    - This object should stay simple and immutable in meaning
    """

    content: str
    reference: LegalReference
    page: int
    chunk_id: str
    level: DocumentLevel

    def to_langchain_document(self):
        """
        Convert to LangChain Document.

        This is the ONLY place where LangChain is referenced.
        Keeps ingestion logic decoupled from retrieval logic.
        """
        from langchain_core.documents import Document

        return Document(
            page_content=self.content,
            metadata={
                "page": self.page,
                "chunk_id": self.chunk_id,
                "level": self.level.value,
                **self.reference.to_metadata(),
            },
        )
