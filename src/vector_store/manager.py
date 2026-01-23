import os
import json
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
    Manages the creation, persistence, and loading of the FAISS Vector Database.
    Also maintains a separate metadata index for efficient filtering.
    """
    
    def __init__(self):
        try:
            logger.info(f"Initializing Embeddings Model: {Config.EMBEDDING_MODEL}")
            self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
            self.metadata_file = os.path.join(Config.METADATA_DIR, "metadata_index.json")
        except Exception as e:
            logger.critical(f"Failed to load embedding model: {e}")
            raise VectorStoreError(f"Embedding model initialization failed: {e}")
    
    def get_vectorstore(self) -> Optional[FAISS]:
        """
        Load existing vector store from disk.
        Returns None if not found.
        """
        index_file = os.path.join(Config.STORAGE_DIR, "index.faiss")
    
        if os.path.exists(index_file):
            try:
                logger.info(f"Loading existing Vector Store from: {Config.STORAGE_DIR}")
                vectorstore = FAISS.load_local(
                    Config.STORAGE_DIR,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector Store loaded successfully")
                return vectorstore
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                return None
        else:
            logger.info(f"No existing index found at {index_file}")
            return None
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents and save to disk.
        Also creates metadata index for efficient filtering.
        """
        try:
            logger.info(f"Creating new Vector Store for {len(documents)} documents...")
            
            # Create FAISS index
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save vector store
            logger.info(f"Persisting Vector Store to: {Config.STORAGE_DIR}")
            os.makedirs(Config.STORAGE_DIR, exist_ok=True)
            vectorstore.save_local(Config.STORAGE_DIR)
            
            # Create and save metadata index
            self._create_metadata_index(documents)
            
            logger.info("Vector Store created and saved successfully")
            return vectorstore
        
        except Exception as e:
            logger.error(f"Failed to create/save vector store: {e}")
            raise VectorStoreError(f"Index creation failed: {e}")
    
    def _create_metadata_index(self, documents: List[Document]):
        """
        Enhanced metadata index with Section and Chapter support.
        Maps article/recital/section/chapter references to document IDs.
        """
        try:
            logger.info("Creating enhanced metadata index...")
            os.makedirs(Config.METADATA_DIR, exist_ok=True)
            
            # Build index structure
            index = {
                "articles": {},      # article_num -> [chunk_ids]
                "recitals": {},      # recital_num -> chunk_id
                "sections": {},      # section_num -> [chunk_ids]
                "chapters": {},      # chapter_num -> [chunk_ids]
                "references": {}     # full_reference -> chunk_id
            }
            
            for i, doc in enumerate(documents):
                chunk_id = doc.metadata.get("chunk_id")
                article = doc.metadata.get("article")
                recital = doc.metadata.get("recital")
                section = doc.metadata.get("section")
                chapter = doc.metadata.get("chapter")
                subsection = doc.metadata.get("subsection")
                point = doc.metadata.get("point")
                
                # Index Recitals
                if recital:
                    index["recitals"][str(recital)] = chunk_id
                    full_ref = f"recital_{recital}"
                    index["references"][full_ref] = chunk_id

                # Index Articles
                if article:
                    if article not in index["articles"]:
                        index["articles"][article] = []
                    index["articles"][article].append(chunk_id)
                    
                    # Full reference
                    full_ref = f"article_{article}"
                    if subsection:
                        full_ref += f"_{subsection}"
                    if point:
                        full_ref += f"_{point}"
                    
                    index["references"][full_ref] = chunk_id
                
                # Index Sections
                if section:
                    if section not in index["sections"]:
                        index["sections"][section] = []
                    index["sections"][section].append(chunk_id)
                    
                    # Add to references
                    index["references"][f"section_{section}"] = chunk_id
                
                # Index Chapters
                if chapter:
                    # Normalize chapter (convert Roman to Arabic if needed)
                    normalized_chapter = self._normalize_chapter(chapter)
                    
                    if normalized_chapter not in index["chapters"]:
                        index["chapters"][normalized_chapter] = []
                    index["chapters"][normalized_chapter].append(chunk_id)
                    
                    # Add to references (store both formats)
                    index["references"][f"chapter_{normalized_chapter}"] = chunk_id
                    index["references"][f"chapter_{chapter}"] = chunk_id  # Original format
            
            # Save to disk
            with open(self.metadata_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            logger.info(
                f"Metadata index created: "
                f"{len(index['articles'])} articles, "
                f"{len(index['recitals'])} recitals, "
                f"{len(index['sections'])} sections, "
                f"{len(index['chapters'])} chapters"
            )
        
        except Exception as e:
            logger.error(f"Failed to create metadata index: {e}")

    def _normalize_chapter(self, chapter_str: str) -> str:
        """Convert Roman numerals to Arabic for consistent indexing"""
        roman_to_arabic = {
            'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
            'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
            'XI': '11'
        }
        return roman_to_arabic.get(str(chapter_str).upper(), str(chapter_str))
    
    def get_metadata_index(self) -> Optional[dict]:
        """Load metadata index from disk"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata index: {e}")
        return None
    
    def update_vectorstore(self, vectorstore: FAISS, new_documents: List[Document]) -> FAISS:
        """
        Add new documents to existing vector store.
        """
        try:
            logger.info(f"Adding {len(new_documents)} new documents to vector store...")
            
            # Add documents to FAISS
            vectorstore.add_documents(new_documents)
            
            # Save updated store
            vectorstore.save_local(Config.STORAGE_DIR)
            
            logger.info("Vector store updated successfully")
            return vectorstore
        
        except Exception as e:
            logger.error(f"Failed to update vector store: {e}")
            raise VectorStoreError(f"Update failed: {e}")