"""
FastAPI Backend for Legal RAG System
"""

import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.manager import VectorStoreManager
from src.rag.engine import EnhancedRAGEngine
from src.ingestion.pipeline import EnhancedIngestionPipeline
from src.config import Config
from src.logger import get_logger

logger = get_logger("API")

# Global instances
engine: EnhancedRAGEngine | None = None
vectorstore_manager: VectorStoreManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global engine, vectorstore_manager
    
    try:
        logger.info("Starting API Server...")
        
        # Initialize vector store manager
        vectorstore_manager = VectorStoreManager()
        
        # Load or create vector store
        vectorstore = vectorstore_manager.load_vectorstore()
        
        if vectorstore is None:
            logger.warning("Vector store not found. Running ingestion...")
            pipeline = EnhancedIngestionPipeline()
            documents = pipeline.run()
            vectorstore = vectorstore_manager.create_vectorstore(documents)
            logger.info("Ingestion completed successfully")
        else:
            logger.info("Loaded existing vector store")
        
        # Initialize RAG engine
        engine = EnhancedRAGEngine(vectorstore)
        logger.info("API Server ready")
        
        yield
        
        logger.info("API Server shutting down")
    
    except Exception as e:
        logger.critical(f"Server startup failed: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed: {e}")


app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    lifespan=lifespan
)


# --- Request/Response Models ---

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
    metadata: dict


# --- Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    if not engine:
        raise HTTPException(
            status_code=503,
            detail="RAG Engine not initialized"
        )
    
    try:
        logger.info(f"Query: '{request.query[:100]}...' | Session: {request.session_id}")
        
        # Execute RAG query
        response = engine.query(
            query=request.query,
            session_id=request.session_id
        )
        
        # Extract answer and context
        answer = response.get("answer", "")
        context_docs = response.get("context", [])
        
        # Extract page numbers (0-indexed internally)
        pages = sorted({
            int(doc.metadata.get("page", 0)) + 1
            for doc in context_docs
        })
        
        # Build metadata
        metadata = {
            "total_sources": len(context_docs),
            "reference_paths": [
                doc.metadata.get("reference_path", "Unknown")
                for doc in context_docs[:5]
            ]
        }
        
        logger.info(f"Response generated | sources={len(context_docs)}")
        
        return ChatResponse(
            answer=answer,
            sources=pages,
            metadata=metadata
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if engine else "degraded",
        "engine_ready": engine is not None,
        "vectorstore_ready": vectorstore_manager is not None,
        "version": Config.API_VERSION
    }


@app.get("/stats")
async def get_stats():
    """Statistics endpoint"""
    if not vectorstore_manager:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    return {
        "indexed": True,
        "message": "FAISS index loaded",
        "version": Config.API_VERSION
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": Config.API_TITLE,
        "version": Config.API_VERSION,
        "description": Config.API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info"
    )
