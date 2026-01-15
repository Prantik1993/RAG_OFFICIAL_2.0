import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import Config
from src.logger import get_logger
from src.exceptions import VectorStoreError

logger = get_logger("VectorStore")

class VectorStoreManager:
    """
    Manages the creation, persistence, and loading of the FAISS Vector Database.
    """
    
    def __init__(self):
        try:
            logger.info(f"Initializing Embeddings Model: {Config.EMBEDDING_MODEL}")
            # This handles the 'Embedding' part of your requirement
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        except Exception as e:
            logger.critical(f"Failed to load embedding model: {e}")
            raise VectorStoreError(f"Embedding model initialization failed: {e}")

    def get_vectorstore(self):
        """
        Attempts to load the existing vector store from disk.
        Returns None if not found (signaling that ingestion is needed).
        """
        if os.path.exists(Config.STORAGE_DIR):
            try:
                logger.info(f"Loading existing Vector Store from: {Config.STORAGE_DIR}")
                # allow_dangerous_deserialization is required for local pickle files
                return FAISS.load_local(
                    Config.STORAGE_DIR, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                raise VectorStoreError(f"Could not load index from disk: {e}")
        else:
            logger.warning(f"No existing index found at {Config.STORAGE_DIR}")
            return None

    def create_vectorstore(self, chunks):
        """
        Creates a new vector store from document chunks and saves it to disk.
        """
        try:
            logger.info(f"Creating new Vector Store for {len(chunks)} text chunks...")
            vectorstore = FAISS.from_documents(documents=chunks, embedding=self.embeddings)
            
            logger.info(f"Persisting Vector Store to: {Config.STORAGE_DIR}")
            vectorstore.save_local(Config.STORAGE_DIR)
            logger.info("Vector Store saved successfully.")
            return vectorstore
        except Exception as e:
            logger.error(f"Failed to create/save vector store: {e}")
            raise VectorStoreError(f"Index creation failed: {e}")