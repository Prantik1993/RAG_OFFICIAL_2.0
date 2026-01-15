import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # --- SECRETS ---
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # --- PATHS ---
    # Automatically finds the project root directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    STORAGE_DIR = os.path.join(BASE_DIR, "storage", "faiss_index")
    LOG_DIR = os.path.join(BASE_DIR, "logs")
    
    # The specific PDF file to load
    PDF_FILE = os.path.join(DATA_DIR, "CELEX_32016R0679_EN_TXT.pdf")

    # --- MODELS ---
    LLM_MODEL = "gpt-3.5-turbo"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

    # --- RAG TUNING ---
    CHUNK_SIZE = 2000
    CHUNK_OVERLAP = 400
    
    # Search Strategy: Fetch 20 (Broad), Rerank down to 5 (Precise)
    RETRIEVER_K_BASE = 10  
    RETRIEVER_K_FINAL = 3