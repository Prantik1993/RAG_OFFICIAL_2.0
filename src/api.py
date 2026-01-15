import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Path Setup ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine
from src.ingestion import IngestionPipeline
from src.logger import get_logger

logger = get_logger("API")

# Global Variable
engine = None

# --- Lifespan Context Manager (Fixes Deprecation Warning) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    try:
        # Removed emojis to prevent Windows UnicodeEncodeError
        logger.info("Starting API Server initialization...") 
        
        # 1. Prepare Vector Store
        vs_manager = VectorStoreManager()
        vectorstore = vs_manager.get_vectorstore()
        
        # 2. Fallback: Run Ingestion if DB is missing
        if not vectorstore:
            logger.warning("Vector Store not found. Triggering automatic ingestion...")
            ingestion = IngestionPipeline()
            docs = ingestion.load_documents()
            chunks = ingestion.split_documents(docs)
            vectorstore = vs_manager.create_vectorstore(chunks)
            
        # 3. Load RAG Engine
        engine = RAGEngine(vectorstore)
        logger.info("RAG Engine initialized and ready.")
        
        yield
        
        logger.info("API Server shutting down...")
        
    except Exception as e:
        logger.critical(f"Server startup failed: {e}")
        raise RuntimeError(f"Could not initialize application: {e}")

# Initialize App with Lifespan
app = FastAPI(title="GDPR RAG API", version="1.0", lifespan=lifespan)

# --- Data Models ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    answer: str
    sources: list[int]

# --- Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # --- FIX 1: Explicit Guardrail for Empty Input ---
    # This ensures the test receives the expected 400 Bad Request
    if not request.query or not request.query.strip():
        logger.warning(f"Empty query received. Session: {request.session_id}")
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    # -------------------------------------------------

    if not engine:
        raise HTTPException(status_code=500, detail="RAG Engine is not initialized.")
    
    try:
        logger.info(f"Query received: '{request.query}' | Session: {request.session_id}")
        
        # Invoke Chain
        chain = engine.get_chain()
        response = chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        source_pages = sorted(set(int(doc.metadata.get("page", 0)) + 1 for doc in response["context"]))
        
        return ChatResponse(answer=response["answer"], sources=source_pages)
    
    except Exception as e:
        logger.error(f"Internal Server Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)