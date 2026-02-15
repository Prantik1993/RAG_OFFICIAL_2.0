"""
Vector Store Manager
Handles FAISS vector store creation and management.
"""

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
    Manages FAISS vector store operations.
    """
    
    def __init__(self):
        try:
            logger.info(f"Loading embeddings: {Config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL
            )
            logger.info("Embeddings loaded successfully")
        except Exception as e:
            logger.critical(f"Failed to load embeddings: {e}")
            raise VectorStoreError(f"Embedding initialization failed: {e}")
    
    def load_vectorstore(self) -> Optional[FAISS]:
        """Load existing FAISS index from disk"""
        index_path = os.path.join(Config.STORAGE_DIR, "index.faiss")
        
        if not os.path.exists(index_path):
            logger.info("No existing FAISS index found")
            return None
        
        try:
            logger.info(f"Loading FAISS index from {Config.STORAGE_DIR}")
            return FAISS.load_local(
                Config.STORAGE_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise VectorStoreError(f"Failed to load vector store: {e}")
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create new FAISS index from documents"""
        if not documents:
            raise VectorStoreError("Cannot create vector store from empty documents")
        
        try:
            logger.info(f"Creating FAISS index from {len(documents)} documents")
            
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save to disk
            os.makedirs(Config.STORAGE_DIR, exist_ok=True)
            vectorstore.save_local(Config.STORAGE_DIR)
            
            logger.info("FAISS index created and saved successfully")
            return vectorstore
        
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise VectorStoreError(f"Failed to create vector store: {e}")
    
    def update_vectorstore(self, vectorstore: FAISS, new_documents: List[Document]) -> FAISS:
        """Add new documents to existing FAISS index"""
        if not new_documents:
            logger.warning("No new documents to add")
            return vectorstore
        
        try:
            logger.info(f"Adding {len(new_documents)} documents to FAISS index")
            
            vectorstore.add_documents(new_documents)
            vectorstore.save_local(Config.STORAGE_DIR)
            
            logger.info("FAISS index updated successfully")
            return vectorstore
        
        except Exception as e:
            logger.error(f"Vector store update failed: {e}")
            raise VectorStoreError(f"Failed to update vector store: {e}")