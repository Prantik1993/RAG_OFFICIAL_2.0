import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "pdfs")
    STORAGE_DIR = os.path.join(BASE_DIR, "storage", "faiss_index")
    METADATA_DIR = os.path.join(BASE_DIR, "storage", "metadata")
    LOG_DIR = os.path.join(BASE_DIR, "logs")

    PDF_FILE = os.path.join(DATA_DIR, "CELEX_32016R0679_EN_TXT.pdf")

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

    MAX_QUERY_LENGTH = 2000
