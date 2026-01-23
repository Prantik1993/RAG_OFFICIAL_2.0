class GDPRAppException(Exception):
    """Base exception for the application."""
    pass

class DataIngestionError(GDPRAppException):
    """Raised when PDF loading or splitting fails."""
    pass

class VectorStoreError(GDPRAppException):
    """Raised when FAISS save/load operations fail."""
    pass

class RAGChainError(GDPRAppException):
    """Raised when the LLM generation or retrieval fails."""
    pass

class ParsingError(GDPRAppException):
    """Raised when document structure parsing fails."""
    pass

class QueryRoutingError(GDPRAppException):
    """Raised when query analysis fails."""
    pass

class MetadataError(GDPRAppException):
    """Raised when metadata operations fail."""
    pass