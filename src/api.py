import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store.manager import VectorStoreManager
from src.rag.engine import RAGEngine
from src.ingestion.pipeline import IngestionPipeline
from src.logger import get_logger

logger = get_logger("API")

# Global variables
engine = None
vectorstore_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, vectorstore_manager
    try:
        logger.info("Starting API Server initialization...")
        
        # Initialize vector store manager
        vectorstore_manager = VectorStoreManager()
        
        # Try to load existing vectorstore
        try:
            vectorstore = vectorstore_manager.get_vectorstore()
        except Exception as e:
            logger.warning(f"Could not load existing vectorstore: {e}")
            vectorstore = None
        
        # Run ingestion if needed
        if vectorstore is None:
            logger.warning("Vector Store not found. Running ingestion pipeline...")
            ingestion = IngestionPipeline()
            documents = ingestion.run()
            vectorstore = vectorstore_manager.create_vectorstore(documents)
            logger.info("Ingestion completed successfully")
        else:
            logger.info("Loaded existing Vector Store")
        
        # Initialize RAG Engine
        engine = RAGEngine(vectorstore)
        logger.info("RAG Engine ready")
        
        yield
        
        logger.info("API Server shutting down...")
    
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        raise RuntimeError(f"Could not initialize application: {e}")
    
    
app = FastAPI(
    title="GDPR Legal RAG API",
    version="2.0",
    description="Advanced RAG system with hybrid retrieval for legal documents",
    lifespan=lifespan
)

# Request/Response Models
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    answer: str
    sources: list[int]
    query_type: str
    metadata: dict = {}

# Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint with intelligent retrieval routing.
    """
    if not request.query or not request.query.strip():
        logger.warning(f"Empty query received. Session: {request.session_id}")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if not engine:
        raise HTTPException(status_code=500, detail="RAG Engine is not initialized.")
    
    try:
        logger.info(f"Query: '{request.query}' | Session: {request.session_id}")
        
        # Execute chain
        chain = engine.get_chain()
        response = chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Extract metadata from context
        context_docs = response.get("context", [])
        
        # Get unique pages
        pages = sorted(set(
            int(doc.metadata.get("page", 0)) + 1 
            for doc in context_docs
        ))
        
        # Get query type from first doc metadata if available
        query_type = "general"
        if context_docs:
            query_type = context_docs[0].metadata.get("level", "general")
        
        # Build metadata
        metadata = {
            "articles_referenced": list(set(
                doc.metadata.get("article") 
                for doc in context_docs 
                if doc.metadata.get("article")
            )),
            "references": [
                doc.metadata.get("full_reference")
                for doc in context_docs[:3]  # Top 3
                if doc.metadata.get("full_reference")
            ]
        }
        
        return ChatResponse(
            answer=response["answer"],
            sources=pages,
            query_type=query_type,
            metadata=metadata
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_ready": engine is not None,
        "vectorstore_ready": vectorstore_manager is not None
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not vectorstore_manager:
        raise HTTPException(status_code=503, detail="System not ready")
    
    metadata_index = vectorstore_manager.get_metadata_index()
    
    if metadata_index:
        return {
            "total_articles": len(metadata_index.get("articles", {})),
            "total_references": len(metadata_index.get("references", {})),
            "indexed": True
        }
    else:
        return {"indexed": False}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)