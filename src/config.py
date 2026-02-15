"""
Configuration for Legal RAG System
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
<<<<<<< HEAD
    # --- API KEYS ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is required. Set it in .env file."
        )
    
    # --- PATHS ---
=======
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

>>>>>>> 1838656519ae40225f10abc5643dde520f2e3fee
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "pdfs")
    STORAGE_DIR = os.path.join(BASE_DIR, "storage", "faiss_index")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    PDF_FILE = os.path.join(DATA_DIR, "CELEX_32016R0679_EN_TXT.pdf")
<<<<<<< HEAD
    
    # --- MODELS ---
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- RAG PARAMETERS ---
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    RETRIEVER_K_FINAL = 5
    
    # --- API CONFIGURATION ---
    API_TITLE = "GDPR Legal RAG API v3.0"
    API_VERSION = "3.0"
    API_DESCRIPTION = "LLM-powered RAG with hierarchical document understanding"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # --- INPUT VALIDATION ---
=======

    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300

    RETRIEVER_K_BASE = 20
    RETRIEVER_K_RERANKED = 5
    RETRIEVER_K_FINAL = 3

    MAX_ARTICLE_TITLE_LENGTH = 200

    API_TITLE = "GDPR Legal RAG API"
    API_VERSION = "2.0"
    API_DESCRIPTION = "Advanced RAG system with hierarchy-aware hybrid retrieval for legal documents"
    API_HOST = "0.0.0.0"
    API_PORT = 8000

>>>>>>> 1838656519ae40225f10abc5643dde520f2e3fee
    MAX_QUERY_LENGTH = 2000
