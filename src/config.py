"""
Configuration  (v4.2) — single source of truth.
All values come from environment variables via .env
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
DATA_DIR      = BASE_DIR / "data" / "pdfs"
STORE_DIR     = BASE_DIR / "storage" / "faiss_index"
LOG_DIR       = BASE_DIR / "logs"
EVAL_DIR      = BASE_DIR / "evaluation"
PROMPTS_DIR   = BASE_DIR / "prompts"

# ── Required secrets ──────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]

# ── Model settings ────────────────────────────────────────────────────────────
LLM_MODEL        = os.getenv("LLM_MODEL",        "gpt-4o-mini")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL   = os.getenv("RERANKER_MODEL",   "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Prompt versioning ─────────────────────────────────────────────────────────
PROMPT_VERSION   = os.getenv("PROMPT_VERSION",   "latest")

# ── Ingestion ─────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_K       = int(os.getenv("RETRIEVAL_K",       "6"))
RETRIEVAL_K_FETCH = int(os.getenv("RETRIEVAL_K_FETCH", "20"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST",    "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
API_TITLE   = "GDPR Legal RAG API"
API_VERSION = "4.2"

MAX_QUERY_LENGTH = 2000

# ── Rate limiting ─────────────────────────────────────────────────────────────
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "15"))
RATE_LIMIT_RPH = int(os.getenv("RATE_LIMIT_RPH", "200"))

# ── Cache ─────────────────────────────────────────────────────────────────────
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
