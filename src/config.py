import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # --- SECRETS ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # --- PATHS ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "pdfs")
    STORAGE_DIR = os.path.join(BASE_DIR, "storage", "faiss_index")
    METADATA_DIR = os.path.join(BASE_DIR, "storage", "metadata")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    PDF_FILE = os.path.join(DATA_DIR, "CELEX_32016R0679_EN_TXT.pdf")

    # --- MODELS ---
    LLM_MODEL = "gpt-3.5-turbo"
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
    
    # Article Patterns
    ARTICLE_PATTERN = r"Article\s+(\d+)"
    SUBSECTION_PATTERN = r"^\s*(\d+)\.\s+"
    POINT_PATTERN = r"^\s*\(([a-z])\)\s+"
    CHAPTER_PATTERN = r"CHAPTER\s+([IVX]+)"
    SECTION_PATTERN = r"Section\s+(\d+)"