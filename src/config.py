import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- SECRETS ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Validate API key on startup
    if not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is required. "
            "Please set it in your .env file or environment."
        )

    # --- PATHS ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "pdfs")
    STORAGE_DIR = os.path.join(BASE_DIR, "storage", "faiss_index")
    METADATA_DIR = os.path.join(BASE_DIR, "storage", "metadata")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    PDF_FILE = os.path.join(DATA_DIR, "CELEX_32016R0679_EN_TXT.pdf")

    # --- MODELS ---
    LLM_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

    # --- RAG TUNING ---
    CHUNK_SIZE = 1500
    CHUNK_OVERLAP = 300
    
    # Retrieval Strategy
    RETRIEVER_K_BASE = 20
    RETRIEVER_K_RERANKED = 5
    RETRIEVER_K_FINAL = 3
    
    # Query Classification
    EXACT_REFERENCE_CONFIDENCE = 0.8
    
    # PDF Parsing Configuration
    MAX_ARTICLE_TITLE_LENGTH = 200
    
    # API Configuration
    API_TITLE = "GDPR Legal RAG API"
    API_VERSION = "2.0"
    API_DESCRIPTION = "Advanced RAG system with hybrid retrieval for legal documents"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # Input Validation
    MAX_QUERY_LENGTH = 2000