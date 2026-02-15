"""
Custom exceptions for Legal RAG system
"""


class LegalRAGException(Exception):
    """Base exception"""
    pass


class ParsingError(LegalRAGException):
    """Raised when PDF parsing fails"""
    pass


class DataIngestionError(LegalRAGException):
    """Raised when data ingestion fails"""
    pass


class VectorStoreError(LegalRAGException):
    """Raised when vector store operations fail"""
    pass


class RAGChainError(LegalRAGException):
    """Raised when RAG chain execution fails"""
    pass


class QueryAnalysisError(LegalRAGException):
    """Raised when query analysis fails"""
    pass
