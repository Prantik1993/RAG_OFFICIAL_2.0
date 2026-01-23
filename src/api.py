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

# Global variables
engine = None
vectorstore_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
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
        logger.info("=" * 60)
        logger.info(f"API Server started successfully on http://{Config.API_HOST}:{Config.API_PORT}")
        logger.info("=" * 60)
        
        yield
        
        # Shutdown
        logger.info("API Server shutting down...")
        logger.info("Cleaning up resources...")
        
        # Add any cleanup logic here if needed
        # e.g., close database connections, flush logs, etc.
        
        logger.info("Shutdown complete")
    
    except Exception as e:
        logger.critical(f"Server startup failed: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize application: {e}")
    
    
app = FastAPI(
    title=Config.API_TITLE,
    version=Config.API_VERSION,
    description=Config.API_DESCRIPTION,
    lifespan=lifespan
)

# Request/Response Models
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=Config.MAX_QUERY_LENGTH)
    session_id: str = Field(default="default_session", min_length=1, max_length=100)
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean query input"""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty or whitespace only")
        return v

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
    
    Args:
        request: ChatRequest containing query and session_id
    
    Returns:
        ChatResponse with answer, sources, query_type, and metadata
    
    Raises:
        HTTPException: 400 for invalid input, 500 for server errors
    """
    if not engine:
        logger.error("RAG Engine not initialized")
        raise HTTPException(
            status_code=503, 
            detail="RAG Engine is not initialized. Please try again later."
        )
    
    try:
        logger.info(f"Query: '{request.query[:100]}...' | Session: {request.session_id}")
        
        # Execute chain
        chain = engine.get_chain()
        response = chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # Extract metadata from context
        context_docs = response.get("context", [])
        
        # FIXED: Get unique pages without adding extra offset
        # PyPDF stores pages as 0-indexed internally, but we want to display 1-indexed
        # Check if your PDF parser already adds 1, if so, remove the +1 here
        pages = sorted(set(
            int(doc.metadata.get("page", 0)) + 1  # Keep +1 if PDF is 0-indexed
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
            "recitals_referenced": list(set(
                doc.metadata.get("recital")
                for doc in context_docs
                if doc.metadata.get("recital")
            )),
            "references": [
                doc.metadata.get("full_reference")
                for doc in context_docs[:3]  # Top 3
                if doc.metadata.get("full_reference")
            ],
            "total_sources": len(context_docs)
        }
        
        logger.info(f"Response generated successfully. Sources: {len(context_docs)}, Type: {query_type}")
        
        return ChatResponse(
            answer=response["answer"],
            sources=pages,
            query_type=query_type,
            metadata=metadata
        )
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your request."
        )

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint with dependency validation.
    
    Returns:
        dict: Health status including component readiness and OpenAI status
    """
    openai_status = "unknown"
    
    try:
        # Check if OpenAI API key is configured
        if Config.OPENAI_API_KEY:
            openai_status = "configured"
        else:
            openai_status = "missing_key"
    except Exception as e:
        logger.error(f"Error checking OpenAI status: {e}")
        openai_status = "error"
    
    # Determine overall status
    overall_status = "healthy"
    if not engine or not vectorstore_manager:
        overall_status = "degraded"
    if openai_status in ["missing_key", "error"]:
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "components": {
            "engine_ready": engine is not None,
            "vectorstore_ready": vectorstore_manager is not None,
            "openai_status": openai_status
        },
        "version": Config.API_VERSION
    }

@app.get("/stats")
async def get_stats():
    """
    Get system statistics including indexed documents.
    
    Returns:
        dict: Statistics about indexed articles, recitals, and references
    
    Raises:
        HTTPException: 503 if system is not ready
    """
    if not vectorstore_manager:
        raise HTTPException(
            status_code=503, 
            detail="Vector store not initialized. System is starting up."
        )
    
    try:
        metadata_index = vectorstore_manager.get_metadata_index()
        
        if metadata_index:
            stats = {
                "indexed": True,
                "total_articles": len(metadata_index.get("articles", {})),
                "total_recitals": len(metadata_index.get("recitals", {})),
                "total_references": len(metadata_index.get("references", {})),
                "articles_list": sorted(
                    metadata_index.get("articles", {}).keys(), 
                    key=lambda x: int(x) if x.isdigit() else 0
                )[:20]  # First 20 articles
            }
            logger.info("Stats retrieved successfully")
            return stats
        else:
            logger.warning("Metadata index not available")
            return {
                "indexed": False,
                "message": "Metadata index not available"
            }
    
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve statistics"
        )

@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": Config.API_TITLE,
        "version": Config.API_VERSION,
        "description": Config.API_DESCRIPTION,
        "docs_url": "/docs",
        "health_check_url": "/health",
        "stats_url": "/stats"
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=Config.API_HOST, 
        port=Config.API_PORT,
        log_level="info"
    )