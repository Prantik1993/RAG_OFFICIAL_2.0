from typing import List
from langchain_core.documents import Document
from src.ingestion.pdf_parser import LegalDocumentParser
from src.ingestion.document_structure import DocumentChunk
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config
from src.logger import get_logger
from src.exceptions import DataIngestionError

logger = get_logger("IngestionPipeline")

class IngestionPipeline:
    """
    Orchestrates the complete ingestion process:
    1. Parse PDF with structure extraction
    2. Create chunks with metadata
    3. Convert to LangChain Document format
    """
    
    def __init__(self):
        self.parser = LegalDocumentParser()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )

    
    def run(self, pdf_path: str = None) -> List[Document]:
        """
        Execute the full ingestion pipeline.
        Returns LangChain Document objects ready for embedding.
        """
        try:
            pdf_path = pdf_path or Config.PDF_FILE
            logger.info(f"Starting ingestion pipeline for: {pdf_path}")
            
            # Parse document and extract structure
            chunks = self.parser.parse_document(pdf_path)
            
            # Convert to LangChain Documents
            base_documents = [chunk.to_langchain_document() for chunk in chunks]
            documents = self.text_splitter.split_documents(base_documents)
            
            logger.info(f"Ingestion complete: {len(documents)} documents created")
            self._log_statistics(documents)
            
            return documents
        
        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            raise DataIngestionError(f"Ingestion failed: {e}")
    
    # ... imports remain same ...

    def _log_statistics(self, documents: List[Document]):
        """Log statistics about ingested documents"""
        from collections import Counter
        
        # Count by level
        levels = [doc.metadata.get("level") for doc in documents]
        level_counts = Counter(levels)
        
        logger.info(f"Statistics:")
        logger.info(f"  Total chunks: {len(documents)}")
        logger.info(f"  By level: {dict(level_counts)}")
        
        # Verify Chapters/Sections were captured
        chapters = set(doc.metadata.get("chapter") for doc in documents if doc.metadata.get("chapter"))
        logger.info(f"  Unique Chapters found: {len(chapters)}")