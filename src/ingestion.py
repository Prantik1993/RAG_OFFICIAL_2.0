import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config
from src.logger import get_logger
from src.exceptions import DataIngestionError
from src.exceptions import DataIngestionError

logger = get_logger(__name__)

class IngestionPipeline:
    def load_documents(self):
        try:
            if not os.path.exists(Config.PDF_FILE):
                raise FileNotFoundError(f"PDF file not found at {Config.PDF_FILE}")
            
            logger.info(f"Loading PDF from: {Config.PDF_FILE}")
            loader = PyPDFLoader(Config.PDF_FILE)
            docs = loader.load()
            logger.info(f"Successfully loaded {len(docs)} pages.")
            return docs
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise DataIngestionError(f"Error loading PDF: {e}") from e

    def split_documents(self, docs):
        try:
            logger.info("Splitting documents into chunks...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP,
                separators=["\nArticle ", "\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(docs)
            logger.info(f"Created {len(splits)} chunks.")
            return splits
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise DataIngestionError("Error during text splitting.") from e