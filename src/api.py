"""
FastAPI Backend  (v4.2)
=======================
New endpoints:
  POST /chat              — with prompt_version in response
  GET  /prompts           — list all prompt versions
  GET  /eval/latest       — last evaluation results
  POST /eval/run          — trigger evaluation run (background)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import src.config as cfg
from src.ingestion.pipeline import IngestionPipeline
from src.logger import get_logger
from src.middleware.rate_limiter import RateLimiter
from src.rag.engine import RAGEngine
from src.vector_store.manager import VectorStoreManager

log = get_logger("API")

_engine:       RAGEngine | None = None
_rate_limiter: RateLimiter      = RateLimiter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    log.info("Server starting…")
    try:
        vs_mgr   = VectorStoreManager()
        pipeline = IngestionPipeline()
        vs, bm25 = vs_mgr.load_or_create(pipeline.run)
        _engine  = RAGEngine(vs, bm25)
        log.info("Server ready")
    except Exception as exc:
        log.critical(f"Startup failed: {exc}", exc_info=True)
        raise
    yield
    log.info("Shutdown")


app = FastAPI(
    title=cfg.API_TITLE,
    version=cfg.API_VERSION,
    lifespan=lifespan,
)


# ── Models ────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query:          str = Field(..., min_length=1, max_length=cfg.MAX_QUERY_LENGTH)
    session_id:     str = Field(default="default", min_length=1, max_length=100)
    prompt_version: str | None = Field(default=None)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v


class ChatResponse(BaseModel):
    answer:         str
    sources:        list[int]
    prompt_version: str
    metadata:       dict


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if _engine is None:
        raise HTTPException(503, "Engine not ready")

    allowed, msg = _rate_limiter.check(req.session_id)
    if not allowed:
        raise HTTPException(429, msg, headers={"Retry-After": "60"})

    try:
        result = _engine.query(
            req.query,
            session_id=req.session_id,
            prompt_version=req.prompt_version,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        log.error(f"Query error: {exc}", exc_info=True)
        raise HTTPException(500, "Internal error")

    answer   = result.get("answer", "")
    ctx_docs = result.get("context", [])
    pages    = sorted({int(d.metadata.get("page", 0)) + 1 for d in ctx_docs})
    refs     = [d.metadata.get("reference_path", "—") for d in ctx_docs[:5]]
    sources  = list({d.metadata.get("source_file", "") for d in ctx_docs if d.metadata.get("source_file")})
    reranks  = [d.metadata.get("rerank_score") for d in ctx_docs if "rerank_score" in d.metadata]

    return ChatResponse(
        answer=answer,
        sources=pages,
        prompt_version=result.get("prompt_version", "unknown"),
        metadata={
            "total_sources":  len(ctx_docs),
            "references":     refs,
            "source_files":   sources,
            "rerank_scores":  reranks[:5],
        },
    )


@app.get("/prompts")
async def list_prompts():
    if _engine is None:
        raise HTTPException(503, "Engine not ready")
    return {
        "active": _engine.prompt_registry.get().version,
        "versions": _engine.prompt_registry.list_versions(),
    }


@app.get("/eval/latest")
async def eval_latest():
    import json
    path = cfg.EVAL_DIR / "latest_results.json"
    if not path.exists():
        raise HTTPException(404, "No evaluation results yet. POST /eval/run first.")
    with open(path) as f:
        return json.load(f)


@app.post("/eval/run")
async def eval_run(background_tasks: BackgroundTasks, version: str | None = None, quick: bool = False):
    if _engine is None:
        raise HTTPException(503, "Engine not ready")

    def _run():
        from src.evaluation.ragas_eval import RAGASEvaluator
        RAGASEvaluator(_engine).run(prompt_version=version, limit=5 if quick else None)

    background_tasks.add_task(_run)
    return {"message": "Evaluation started in background. GET /eval/latest when done."}


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
    return {
        "name":           cfg.API_TITLE,
        "version":        cfg.API_VERSION,
        "docs":           "/docs",
        "endpoints":      ["/chat", "/prompts", "/eval/latest", "/eval/run", "/health", "/metrics"],
    }


if __name__ == "__main__":
    uvicorn.run("src.api:app", host=cfg.API_HOST, port=cfg.API_PORT, reload=False)
