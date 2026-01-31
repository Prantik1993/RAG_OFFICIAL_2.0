import os
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import Config
from src.logger import get_logger
from src.exceptions import VectorStoreError

logger = get_logger("VectorStoreManager")


class VectorStoreManager:
    """
    Production-grade Vector Store Manager.

    Responsibilities:
    - Initialize embedding model
    - Create FAISS vector store from documents
    - Persist and load FAISS index
    - Update existing vector store

    Design principles:
    - FAISS is the single source of truth
    - Metadata lives inside Document.metadata
    - No external metadata index
    """

    def __init__(self):
        try:
            logger.info(
                f"Loading embedding model: {Config.EMBEDDING_MODEL}"
            )
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL
            )
        except Exception as e:
            logger.critical(f"Embedding initialization failed: {e}")
            raise VectorStoreError(
                f"Failed to initialize embeddings: {e}"
            )

    # ------------------------------------------------------------------
    # Load existing vector store
    # ------------------------------------------------------------------

    def load_vectorstore(self) -> Optional[FAISS]:
        """
        Load FAISS vector store from disk.
        Returns None if not found.
        """
        index_path = os.path.join(Config.STORAGE_DIR, "index.faiss")

        if not os.path.exists(index_path):
            logger.info("No existing FAISS index found")
            return None

        try:
            logger.info(
                f"Loading FAISS index from {Config.STORAGE_DIR}"
            )
            return FAISS.load_local(
                Config.STORAGE_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise VectorStoreError(
                f"Vector store load failed: {e}"
            )

    # ------------------------------------------------------------------
    # Create new vector store
    # ------------------------------------------------------------------

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a new FAISS vector store from documents and persist it.
        """
        if not documents:
            raise VectorStoreError(
                "Cannot create vector store with empty documents"
            )

        try:
            logger.info(
                f"Creating FAISS index from {len(documents)} documents"
            )

            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings,
            )

            os.makedirs(Config.STORAGE_DIR, exist_ok=True)
            vectorstore.save_local(Config.STORAGE_DIR)

            logger.info("FAISS index created and saved successfully")
            return vectorstore

        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise VectorStoreError(
                f"Vector store creation failed: {e}"
            )

    # ------------------------------------------------------------------
    # Update existing vector store
    # ------------------------------------------------------------------

    def update_vectorstore(
        self, vectorstore: FAISS, new_documents: List[Document]
    ) -> FAISS:
        """
        Add new documents to an existing FAISS vector store.
        """
        if not new_documents:
            logger.warning("No new documents to add")
            return vectorstore

        try:
            logger.info(
                f"Adding {len(new_documents)} documents to FAISS index"
            )

            vectorstore.add_documents(new_documents)
            vectorstore.save_local(Config.STORAGE_DIR)

            logger.info("FAISS index updated successfully")
            return vectorstore

        except Exception as e:
            logger.error(f"Vector store update failed: {e}")
            raise VectorStoreError(
                f"Vector store update failed: {e}"
            )
