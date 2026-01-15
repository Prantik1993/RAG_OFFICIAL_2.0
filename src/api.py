import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Path Setup ---
# Ensures we can import 'src' modules regardless of where we run the script from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vector_store import VectorStoreManager
from src.rag_engine import RAGEngine
from src.ingestion import IngestionPipeline
from src.logger import get_logger
from src.exceptions import GDPRAppException

# Initialize Logger
logger = get_logger("API")

# Initialize App
app = FastAPI(title="GDPR RAG API", version="1.0")

# Global Variable to hold the loaded RAG engine
engine = None

# --- Data Models (Pydantic) ---
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_session"

class ChatResponse(BaseModel):
    answer: str
    sources: list[int]

# --- Lifecycle Events ---
@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts. 
    It ensures the Vector Database is ready and loads the RAG Engine.
    """
    global engine
    try:
        logger.info("🚀 Starting API Server initialization...")
        
        # 1. Prepare Vector Store
        vs_manager = VectorStoreManager()
        vectorstore = vs_manager.get_vectorstore()
        
        # 2. Fallback: Run Ingestion if DB is missing
        if not vectorstore:
            logger.warning("⚠️ Vector Store not found. Triggering automatic ingestion...")
            ingestion = IngestionPipeline()
            docs = ingestion.load_documents()
            chunks = ingestion.split_documents(docs)
            vectorstore = vs_manager.create_vectorstore(chunks)
            
        # 3. Load RAG Engine (with Reranker & LLM)
        engine = RAGEngine(vectorstore)
        logger.info("✅ RAG Engine initialized and ready.")
        
    except Exception as e:
        logger.critical(f"❌ Server startup failed: {e}")
        # We raise a RuntimeError to stop the server from starting in a broken state
        raise RuntimeError(f"Could not initialize application: {e}")

# --- Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main Chat Endpoint.
    Receives a query, runs it through the RAG Chain, and returns the answer + sources.
    """
    if not engine:
        raise HTTPException(status_code=500, detail="RAG Engine is not initialized.")
    
    try:
        logger.info(f"📩 Query received: '{request.query}' | Session: {request.session_id}")
        
        # 1. Get the chain from the engine
        chain = engine.get_chain()
        
        # 2. Invoke the chain
        response = chain.invoke(
            {"input": request.query},
            config={"configurable": {"session_id": request.session_id}}
        )
        
        # 3. Extract and Sort Page Numbers (Humans read 1-based, PDF is 0-based)
        # We use a set comprehension to remove duplicates, then sort
        source_pages = sorted(set(int(doc.metadata.get("page", 0)) + 1 for doc in response["context"]))
        
        return ChatResponse(answer=response["answer"], sources=source_pages)
    
    except ValueError as ve:
        # This catches Guardrail errors (e.g. input too long)
        logger.warning(f"🚫 Input Validation Failed: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
        
    except Exception as e:
        # Catch unexpected server errors
        logger.error(f"❌ Internal Server Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred while processing the request.")

# --- Entry Point ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)