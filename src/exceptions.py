"""
Domain exceptions — one per layer so callers can catch precisely.
"""


class LegalRAGError(Exception):
    """Base for all domain errors."""


class ParsingError(LegalRAGError):
    """PDF / text extraction failed."""


class IngestionError(LegalRAGError):
    """Ingestion pipeline failed."""


class VectorStoreError(LegalRAGError):
    """FAISS operations failed."""


class RetrievalError(LegalRAGError):
    """Retrieval / query analysis failed."""


class RAGEngineError(LegalRAGError):
    """End-to-end RAG pipeline failed."""
