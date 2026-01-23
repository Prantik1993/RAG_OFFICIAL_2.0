from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from src.retrieval.hybrid_retriever import HybridRetriever
from src.config import Config
from src.logger import get_logger
from src.exceptions import RAGChainError

logger = get_logger("RAGEngine")

class RAGEngine:
    """
    Enhanced RAG Engine with intelligent retrieval routing.
    """
    
    def __init__(self, vectorstore):
        try:
            logger.info("Initializing RAG Engine...")
            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0,
                api_key=Config.OPENAI_API_KEY
            )
            
            # Use hybrid retriever instead of basic retriever
            self.hybrid_retriever = HybridRetriever(vectorstore)
            
            self.store = {}  # Memory storage
            
            logger.info("RAG Engine initialized successfully")
        except Exception as e:
            logger.critical(f"Failed to initialize RAG Engine: {e}")
            raise RAGChainError(f"RAG init failed: {e}")
    
    def _validate_input(self, user_input: str):
        """Validate user input"""
        if not user_input or not user_input.strip():
            raise ValueError("Input cannot be empty.")
        if len(user_input) > 2000:
            raise ValueError("Input is too long (max 2000 chars).")
    
    def _get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]
    
    def get_chain(self):
        """
        Build the RAG chain with enhanced prompts and hybrid retrieval.
        """
        try:
            # Document template for formatting retrieved chunks
            document_prompt = ChatPromptTemplate.from_template(
                "Legal Reference: {full_reference}\n"
                "Content: {page_content}\n"
                "Source: Page {page}\n"
            )
            
            # System prompt with enhanced instructions to handle Recitals and Articles
            qa_system_prompt = (
                "You are an expert legal assistant specializing in GDPR regulations.\n\n"
                
                "ROLE AND RESPONSIBILITIES:\n"
                "1. Provide accurate, precise answers based ONLY on the retrieved legal context\n"
                "2. Use exact article or recital references when available\n"
                "3. If information is not in the context, explicitly state this\n"
                "4. Maintain professional, clear language suitable for legal documentation\n\n"
                
                "CITATION RULES (CRITICAL):\n"
                "- ALWAYS cite the specific legal reference for every statement\n"
                "- Format for Articles: [Article X] or [Article X.Y] or [Article X.Y.z]\n"
                "- Format for Recitals: [Recital X]\n"
                "- Also include page numbers: [Article 15, Page 42] or [Recital 10, Page 2]\n"
                "- Multiple references: [Article 6.1.a, Page 8; Recital 40, Page 6]\n"
                "- For definitions or principles, cite the defining article/recital\n\n"
                
                "ANSWER STRUCTURE:\n"
                "1. Direct answer to the question\n"
                "2. Relevant details from the retrieved context\n"
                "3. Related provisions (Recitals/Articles) if applicable\n"
                "4. Practical implications if clear from the text\n\n"
                
                "RETRIEVED LEGAL CONTEXT:\n"
                "{context}\n"
            )
            
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Create question-answer chain
            question_answer_chain = create_stuff_documents_chain(
                self.llm,
                qa_prompt,
                document_prompt=document_prompt,
                document_variable_name="context"
            )
            
            # Create history-aware retriever
            contextualize_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "Given a chat history and the latest user question which might reference context "
                 "from the chat history, formulate a standalone question that can be understood "
                 "without the chat history. Do NOT answer the question, just reformulate it if needed "
                 "and otherwise return it as is. Preserve any article references (e.g., Article 15.1.a) "
                 "or recital references (e.g., Recital 42) exactly."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            
            # Custom retriever wrapper that inherits from BaseRetriever
            class CustomRetrieverWrapper(BaseRetriever):
                hybrid_retriever: Any

                def _get_relevant_documents(
                    self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
                ) -> List[Document]:
                    """
                    Standard method required by BaseRetriever.
                    """
                    # We unpack the tuple (docs, analysis) and return only docs to LangChain
                    docs, analysis = self.hybrid_retriever.retrieve(query)
                    logger.info(f"Retrieved {len(docs)} documents for query type: {analysis.query_type.value}")
                    return docs

            # Initialize with keyword argument
            retriever_wrapper = CustomRetrieverWrapper(hybrid_retriever=self.hybrid_retriever)
            
            history_retriever = create_history_aware_retriever(
                self.llm,
                retriever_wrapper,
                contextualize_prompt
            )
            
            # Create full RAG chain
            rag_chain = create_retrieval_chain(history_retriever, question_answer_chain)
            
            # Wrap with message history
            return RunnableWithMessageHistory(
                rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        
        except Exception as e:
            logger.error(f"Error building RAG chain: {e}")
            raise RAGChainError(f"Chain creation failed: {e}")
    
    def query(self, query: str, session_id: str = "default") -> dict:
        """
        Simplified query method for direct use.
        """
        try:
            self._validate_input(query)
            
            chain = self.get_chain()
            response = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}}
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise RAGChainError(f"Query failed: {e}")