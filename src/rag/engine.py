from typing import List, Any
from langchain_openai import ChatOpenAI
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
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
    Production-grade RAG Engine.

    Responsibilities:
    - Orchestrate retrieval (via HybridRetriever)
    - Build history-aware RAG chain
    - Enforce prompt discipline for legal accuracy
    """

    def __init__(self, vectorstore):
        try:
            logger.info("Initializing RAG Engine")

            self.llm = ChatOpenAI(
                model=Config.LLM_MODEL,
                temperature=0,
                api_key=Config.OPENAI_API_KEY,
            )

            self.hybrid_retriever = HybridRetriever(vectorstore)

            # In-memory session store (replace with Redis in prod if needed)
            self._session_store: dict[str, ChatMessageHistory] = {}

            logger.info("RAG Engine initialized successfully")

        except Exception as e:
            logger.critical(f"RAG Engine initialization failed: {e}")
            raise RAGChainError(f"RAG init failed: {e}")

    # ------------------------------------------------------------------
    # Session handling
    # ------------------------------------------------------------------

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._session_store:
            self._session_store[session_id] = ChatMessageHistory()
        return self._session_store[session_id]

    def _validate_input(self, query: str):
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if len(query) > Config.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long (max {Config.MAX_QUERY_LENGTH} characters)."
            )

    # ------------------------------------------------------------------
    # Chain construction
    # ------------------------------------------------------------------

    def _build_chain(self):
        """
        Build the full RAG chain:
        - History-aware query rewriting
        - Hybrid retrieval
        - Controlled answer synthesis
        """

        # -------- Document formatting --------
        document_prompt = ChatPromptTemplate.from_template(
        "Content:\n{page_content}\n"
        "Source: Page {page}\n"
)


        # -------- System instructions --------
        system_prompt = (
            "You are an expert legal assistant specialized in GDPR.\n\n"

            "EXACT LOOKUP RULES:\n"
            "- If the user asks for a specific Article or Recital:\n"
            "  • Output the EXACT legal text from context\n"
            "  • Do NOT paraphrase or summarize\n"
            "  • Use Markdown blockquote (> text)\n"
            "  • Optionally add ONE short clarification sentence after\n\n"

            "GENERAL QUESTIONS:\n"
            "- You may summarize and explain in neutral legal language\n\n"

            "LEGAL CONTEXT:\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=qa_prompt,
            document_prompt=document_prompt,
            document_variable_name="context",
        )

        # -------- History-aware retriever --------
        contextualize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Given the chat history and the latest question, "
                    "rewrite the question so it is standalone. "
                    "Preserve exact legal references like "
                    "'Article 15.1.a' or 'Recital 42'.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        class HybridRetrieverWrapper(BaseRetriever):
            """
            LangChain-compatible wrapper around HybridRetriever.
            """

            hybrid_retriever: Any

            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun | None = None,
            ) -> List[Document]:
                docs, analysis = self.hybrid_retriever.retrieve(query)

                query_type = (
                    analysis.query_type.value
                    if hasattr(analysis.query_type, "value")
                    else str(analysis.query_type)
                )

                logger.info(
                    f"Retrieved {len(docs)} documents "
                    f"(query_type={query_type}, confidence={analysis.confidence})"
                )
                return docs

        retriever = HybridRetrieverWrapper(
            hybrid_retriever=self.hybrid_retriever
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm,
            retriever,
            contextualize_prompt,
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            qa_chain,
        )

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, query: str, session_id: str = "default") -> dict:
        """
        Execute a RAG query.
        """
        try:
            self._validate_input(query)

            chain = self._build_chain()

            response = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            raise RAGChainError(f"Query execution failed: {e}")
