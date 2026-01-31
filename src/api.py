import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.manager import VectorStoreManager
from src.rag.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline
from src.config import Config
from src.logger import get_logger

logger = get_logger("API")

# Global instances
engine: RAGEngine | None = None
vectorstore_manager: VectorStoreManager | None = None


# ------------------------------------------------------------------
# Application lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, vectorstore_manager

    try:
        logger.info("Starting API Server initialization")

        # Vector store manager
        vectorstore_manager = VectorStoreManager()

        # Load existing FAISS index
        vectorstore = vectorstore_manager.load_vectorstore()

        # Run ingestion if needed
        if vectorstore is None:
            logger.warning("Vector store not found. Running ingestion pipeline...")
            ingestion = IngestionPipeline()
            documents = ingestion.run()
            vectorstore = vectorstore_manager.create_vectorstore(documents)
            logger.info("Ingestion completed successfully")
        else:
            logger.info("Existing vector store loaded")

        # Initialize RAG Engine
        engine = RAGEngine(vectorstore)
        logger.info("RAG Engine ready")

        yield

        logger.info("API Server shutting down")

    except Exception as e:
        logger.critical(f"Server startup failed: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed: {e}")


app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    session_id: str = Field(default="default_session", min_length=1, max_length=100)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[int]
    query_type: str
    metadata: dict


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="RAG Engine not initialized",
        )

    try:
        logger.info(
            f"Query: '{request.query[:100]}...' | Session: {request.session_id}"
        )

        # Execute RAG
        response = engine.query(
            query=request.query,
            session_id=request.session_id,
        )

        # Defensive extraction
        answer = response.get("answer") or response.get("output") or ""

        context_docs = (
            response.get("context")
            or response.get("documents")
            or []
        )

        # Pages (PDF pages are 0-indexed internally)
        pages = sorted(
            {
                int(doc.metadata.get("page", 0)) + 1
                for doc in context_docs
            }
        )

        # Infer type from retrieved docs
        query_type = (
            context_docs[0].metadata.get("level", "general")
            if context_docs
            else "general"
        )

        metadata = {
            "articles_referenced": sorted(
                {
                    doc.metadata.get("article")
                    for doc in context_docs
                    if doc.metadata.get("article")
                }
            ),
            "recitals_referenced": sorted(
                {
                    doc.metadata.get("recital")
                    for doc in context_docs
                    if doc.metadata.get("recital")
                }
            ),
            "references": [
                f"Article {doc.metadata['article']}"
                if doc.metadata.get("article")
                else f"Recital {doc.metadata['recital']}"
                for doc in context_docs[:3]
            ],
            "total_sources": len(context_docs),
        }

        logger.info(
            f"Response generated | sources={len(context_docs)} | type={query_type}"
        )

        return ChatResponse(
            answer=answer,
            sources=pages,
            query_type=query_type,
            metadata=metadata,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if engine else "degraded",
        "engine_ready": engine is not None,
        "vectorstore_ready": vectorstore_manager is not None,
        "version": Config.API_VERSION,
    }


@app.get("/stats")
async def get_stats():
    """
    Minimal stats endpoint (FAISS-only architecture).
    """
    if not vectorstore_manager:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized",
        )

    return {
        "indexed": True,
        "message": "FAISS index loaded",
    }


@app.get("/")
async def root():
    return {
        "name": Config.API_TITLE,
        "version": Config.API_VERSION,
        "description": Config.API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
    )
