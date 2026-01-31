from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.pdf_parser import CELEXPDFParser
from src.config import Config
from src.logger import get_logger
from src.exceptions import DataIngestionError

logger = get_logger("IngestionPipeline")


class IngestionPipeline:
    """
    Industry-standard ingestion pipeline.

    Responsibilities:
    1. Parse CELEX PDF into structured chunks
    2. Convert chunks to LangChain Documents
    3. Split documents for embedding
    """

    def __init__(self):
        self.parser = CELEXPDFParser()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

    def run(self, pdf_path: str | None = None) -> List[Document]:
        try:
            pdf_path = pdf_path or Config.PDF_FILE
            logger.info(f"Starting ingestion pipeline for: {pdf_path}")

            # 1️⃣ Parse PDF
            chunks = self.parser.parse(pdf_path)

            # 2️⃣ Convert to LangChain Documents
            base_documents = [
                chunk.to_langchain_document() for chunk in chunks
            ]

            # 3️⃣ Split for embeddings
            documents = self.splitter.split_documents(base_documents)

            logger.info(
                f"Ingestion completed successfully | "
                f"base_chunks={len(chunks)} | "
                f"final_documents={len(documents)}"
            )

            return documents

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}", exc_info=True)
            raise DataIngestionError(f"Ingestion failed: {e}")
