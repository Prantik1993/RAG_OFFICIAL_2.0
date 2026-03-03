import os
import glob
import logging
from src.logger import setup_logger
setup_logger()
from src.ingestion.pipeline import LegalIngestionPipeline
from src.vector_store.manager import VectorStoreManager
from src.config import Config

logger = logging.getLogger("IngestionScript")

def build_index():
    # 1. Define the path where your 50 PDFs are stored
    # Adjust this path if your raw data is stored elsewhere
    pdf_dir = Config.DATA_DIR
    os.makedirs(pdf_dir, exist_ok=True)
    
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    
    if not pdf_files:
        logger.error(f"No PDFs found in {pdf_dir}. Please place your CELEX documents here before running.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s). Initializing ingestion pipeline...")

    pipeline = LegalIngestionPipeline()
    all_documents = []

    # 2. Parse and chunk every PDF
    for pdf_path in pdf_files:
        logger.info(f"Processing: {os.path.basename(pdf_path)}")
        try:
            docs = pipeline.run(pdf_path)
            all_documents.extend(docs)
            logger.info(f"Successfully extracted {len(docs)} chunks from {os.path.basename(pdf_path)}")
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}", exc_info=True)

    if not all_documents:
        logger.error("No valid documents were extracted. Aborting vector store creation.")
        return

    logger.info(f"Total extracted chunks across all documents: {len(all_documents)}. Building FAISS index...")

    # 3. Create and save the vector store
    manager = VectorStoreManager()
    try:
        manager.create_vectorstore(all_documents)
        logger.info(f"✅ FAISS index generated and saved successfully to {Config.STORAGE_DIR}!")
        logger.info("You can now start the API server.")
    except Exception as e:
        logger.error(f"Failed to build or save vector store: {e}", exc_info=True)

if __name__ == "__main__":
    build_index()