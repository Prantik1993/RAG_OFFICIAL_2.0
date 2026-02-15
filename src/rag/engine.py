"""
Enhanced RAG Engine
Uses LLM-powered query analysis and smart retrieval.
"""

from typing import Any
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.retrieval.smart_retriever import SmartRetriever
from src.config import Config
from src.logger import get_logger
from src.exceptions import RAGChainError

logger = get_logger("EnhancedRAGEngine")


class EnhancedRAGEngine:
    """
    Production RAG engine with intelligent query routing.
    """
    
    def __init__(self, vectorstore):
        try:
<<<<<<< HEAD
            logger.info("Initializing Enhanced RAG Engine")
            
=======
            logger.info("Initializing RAG Engine")

            if not Config.OPENAI_API_KEY:
                raise RAGChainError("OPENAI_API_KEY is required to run the RAG engine")

>>>>>>> 1838656519ae40225f10abc5643dde520f2e3fee
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0,
                api_key=Config.OPENAI_API_KEY,
            )
            
            self.smart_retriever = SmartRetriever(vectorstore)
            self._session_store: dict[str, ChatMessageHistory] = {}
            
            logger.info("Enhanced RAG Engine ready")
        
        except Exception as e:
            logger.critical(f"RAG Engine initialization failed: {e}")
            raise RAGChainError(f"Failed to initialize RAG engine: {e}")
    
    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create session history"""
        if session_id not in self._session_store:
            self._session_store[session_id] = ChatMessageHistory()
        return self._session_store[session_id]
    
    def _build_chain(self):
        """Build the RAG chain with smart retrieval"""
        
        # System prompt for answer generation
        system_prompt = """You are an expert legal assistant specialized in GDPR regulation.

CRITICAL RULES:
1. For exact reference queries (e.g., "What is Article 15.1.a?"):
   - Quote the EXACT text from the context
   - Use blockquote format: > quoted text
   - Add a brief explanation if helpful

2. For range queries (e.g., "What articles are in Chapter 2 Section 1?"):
   - List all articles found in the context
   - Provide article numbers and brief descriptions

3. For conceptual queries:
   - Synthesize information from multiple sources
   - Cite specific articles when relevant
   - Provide clear, accurate explanations

4. For comparisons:
   - Highlight similarities and differences
   - Reference specific articles
   - Be objective and precise

5. ALWAYS cite your sources using the reference paths provided in the metadata.

Context:
{context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # Create document chain
        qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
        )
        
        # Wrap smart retriever for LangChain compatibility
        class SmartRetrieverWrapper(BaseRetriever):
            """LangChain-compatible wrapper for SmartRetriever"""
            
            smart_retriever: Any
            
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun | None = None,
            ) -> list[Document]:
                docs, analysis = self.smart_retriever.retrieve(query)
                
                logger.info(
                    f"Retrieved {len(docs)} documents | "
                    f"Intent: {analysis.intent.value} | "
                    f"Confidence: {analysis.confidence:.2f}"
                )
                
                return docs
        
        retriever = SmartRetrieverWrapper(smart_retriever=self.smart_retriever)
        
        # Create retrieval chain
        rag_chain = create_retrieval_chain(retriever, qa_chain)
        
        # Add message history
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    
    def query(self, query: str, session_id: str = "default") -> dict:
        """
        Execute a RAG query.
        
        Args:
            query: User's question
            session_id: Session identifier for conversation history
        
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            # Validate input
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")
            
            if len(query) > Config.MAX_QUERY_LENGTH:
                raise ValueError(f"Query too long (max {Config.MAX_QUERY_LENGTH} characters)")
            
            logger.info(f"Processing query: '{query[:100]}...'")
            

            analysis = self.smart_retriever.analyzer.analyze(query)
            if analysis.confidence < 0.4:
                logger.warning("Low confidence query – generation skipped at engine level")
                return {
                    "answer": (
                        "I can only help with questions related to the GDPR "
                        "and EU data protection law."
                    ),
                    "context": [],
                }

            # Build and execute chain
            chain = self._build_chain()
            
            response = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )
            
            logger.info("Query processed successfully")
            return response
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}", exc_info=True)
            raise RAGChainError(f"Failed to process query: {e}")
