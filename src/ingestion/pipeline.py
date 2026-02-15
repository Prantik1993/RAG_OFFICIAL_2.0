"""
Enhanced Ingestion Pipeline
Uses the new hierarchical parser.
"""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.parser import EnhancedLegalParser
from src.config import Config
from src.logger import get_logger
from src.exceptions import DataIngestionError

logger = get_logger("EnhancedIngestion")


class EnhancedIngestionPipeline:
    """
    Modern ingestion pipeline with hierarchical parsing.
    
    Process:
    1. Parse PDF with full structure extraction
    2. Create chunks with rich metadata
    3. Optional: Split large chunks while preserving metadata
    """
    
    def __init__(self):
        self.parser = EnhancedLegalParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
    
    def run(self, pdf_path: str = None) -> List[Document]:
        """
        Run the complete ingestion pipeline.
        
        Returns:
            List of LangChain Documents ready for embedding
        """
        try:
            pdf_path = pdf_path or Config.PDF_FILE
            logger.info(f"Starting enhanced ingestion for: {pdf_path}")
            
            # Step 1: Parse with hierarchy
            chunks = self.parser.parse(pdf_path)
            logger.info(f"Parsed {len(chunks)} hierarchical chunks")
            
            # Step 2: Convert to LangChain Documents
            documents = [chunk.to_document() for chunk in chunks]
            
            # Step 3: Split large documents while preserving metadata
            final_documents = []
            for doc in documents:
                # If document is too large, split it
                if len(doc.page_content) > Config.CHUNK_SIZE:
                    split_docs = self.splitter.split_documents([doc])
                    # Preserve metadata in splits
                    for split_doc in split_docs:
                        split_doc.metadata = dict(doc.metadata)
                    final_documents.extend(split_docs)
                else:
                    final_documents.append(doc)
            
            logger.info(
                f"Ingestion complete: "
                f"{len(chunks)} base chunks → "
                f"{len(final_documents)} final documents"
            )
            
            # Log statistics
            self._log_statistics(final_documents)
            
            return final_documents
        
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise DataIngestionError(f"Ingestion pipeline failed: {e}")
    
    def _log_statistics(self, documents: List[Document]):
        """Log helpful statistics about parsed content"""
        recitals = set()
        chapters = set()
        sections = set()
        articles = set()
        
        for doc in documents:
            meta = doc.metadata
            if meta.get("recital"):
                recitals.add(meta["recital"])
            if meta.get("chapter"):
                chapters.add(meta["chapter"])
            if meta.get("section"):
                sections.add(meta["section"])
            if meta.get("article"):
                articles.add(meta["article"])
        
        logger.info(
            f"Content statistics: "
            f"{len(recitals)} recitals, "
            f"{len(chapters)} chapters, "
            f"{len(sections)} sections, "
            f"{len(articles)} articles"
        )
