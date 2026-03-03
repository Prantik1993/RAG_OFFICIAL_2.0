"""
FastAPI Backend
===============
Clean, minimal — one chat endpoint, health, stats, cache clear.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import src.config as cfg
from src.ingestion.pipeline import IngestionPipeline
from src.logger import get_logger
from src.middleware.rate_limiter import RateLimiter
from src.rag.engine import RAGEngine
from src.vector_store.manager import VectorStoreManager

log = get_logger("API")

# ── Globals ───────────────────────────────────────────────────────────────────
_engine:       RAGEngine | None       = None
_rate_limiter: RateLimiter            = RateLimiter()


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    log.info("Server starting…")
    try:
        vs_mgr   = VectorStoreManager()
        pipeline = IngestionPipeline()
        vs       = vs_mgr.load_or_create(pipeline.run)
        _engine  = RAGEngine(vs)
        log.info("Server ready")
    except Exception as exc:
        log.critical(f"Startup failed: {exc}", exc_info=True)
        raise
    yield
    log.info("Server shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=cfg.API_TITLE,
    version=cfg.API_VERSION,
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:      str = Field(..., min_length=1, max_length=cfg.MAX_QUERY_LENGTH)
    session_id: str = Field(default="default", min_length=1, max_length=100)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class ChatResponse(BaseModel):
    answer:   str
    sources:  list[int]
    metadata: dict


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not ready")

    allowed, msg = _rate_limiter.check(req.session_id)
    if not allowed:
        raise HTTPException(429, msg, headers={"Retry-After": "60"})

    try:
        result = _engine.query(req.query, session_id=req.session_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.error(f"Query error: {exc}", exc_info=True)
        raise HTTPException(500, "Internal error")

    answer   = result.get("answer", "")
    ctx_docs = result.get("context", [])
    pages    = sorted({int(d.metadata.get("page", 0)) + 1 for d in ctx_docs})
    refs     = [d.metadata.get("reference_path", "—") for d in ctx_docs[:5]]

    return ChatResponse(
        answer=answer,
        sources=pages,
        metadata={"total_sources": len(ctx_docs), "references": refs},
    )


@app.get("/health")
async def health():
    return {"status": "ok" if _engine else "degraded", "engine": _engine is not None}


@app.get("/metrics")
async def metrics():
    if _engine is None:
        raise HTTPException(503, "Engine not ready")
    return _engine.tracker.stats()


@app.post("/cache/clear")
async def cache_clear():
    if _engine is None:
        raise HTTPException(503, "Engine not ready")
    _engine._cache.clear()
    return {"message": "cache cleared"}


@app.get("/")
async def root():
    return {"name": cfg.API_TITLE, "version": cfg.API_VERSION, "docs": "/docs"}


if __name__ == "__main__":
    uvicorn.run("src.api:app", host=cfg.API_HOST, port=cfg.API_PORT, reload=False)
