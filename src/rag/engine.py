"""
RAG Engine  (v4.1)
==================
Receives a query, retrieves via hybrid+rerank, generates one answer.

Pipeline per query:
  SafetyGuardrails → QueryCache → SmartRetriever (FAISS+BM25+CrossEncoder)
  → LangChain RetrievalChain (ChatOpenAI) → output safety → cache → return

One LLM call per query. No routing LLM.
"""

from __future__ import annotations

import time
from typing import Any

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import src.config as cfg
from src.caching.query_cache import QueryCache
from src.exceptions import RAGEngineError
from src.guardrails.safety import SafetyGuardrails
from src.logger import get_logger
from src.monitoring.tracker import LLMTracker
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import SmartRetriever

log = get_logger("RAGEngine")

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are a precise legal assistant specialised in the GDPR (EU 2016/679).

Rules you MUST follow:
1. Base every answer ONLY on the context documents provided.
2. If the answer is not in the context, say exactly: "I could not find that in the GDPR text."
3. For exact article / recital lookups: quote the relevant text, then explain it concisely.
4. For conceptual questions: synthesise across sources, cite article numbers inline (e.g. "Article 6(1)(f)").
5. For follow-up questions: use chat_history to resolve the referent, then answer from context.
6. Never invent article numbers, obligations, or definitions.
7. Never answer questions unrelated to GDPR / EU data protection law.
8. Keep answers concise unless the user explicitly asks for detail.

Context:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ── LangChain retriever adapter ───────────────────────────────────────────────
class _RetrieverAdapter(BaseRetriever):
    """Thin bridge: SmartRetriever → LangChain BaseRetriever interface."""
    smart: Any  # SmartRetriever (Any avoids pydantic forward-ref issues)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs, analysis = self.smart.retrieve(query)
        log.info(
            f"LangChain retrieval | intent={analysis.intent.value} "
            f"docs={len(docs)}"
        )
        return docs


# ── Engine ────────────────────────────────────────────────────────────────────
class RAGEngine:

    def __init__(self, vectorstore: FAISS, bm25: BM25Index) -> None:
        log.info("Initialising RAG Engine (hybrid + rerank)…")
        try:
            self._llm      = ChatOpenAI(
                model=cfg.LLM_MODEL,
                temperature=0,
                api_key=cfg.OPENAI_API_KEY,
            )
            self._retriever = SmartRetriever(vectorstore, bm25)
            self._safety    = SafetyGuardrails()
            self._cache     = QueryCache()
            self._tracker   = LLMTracker()
            self._sessions: dict[str, ChatMessageHistory] = {}
            log.info("RAG Engine ready")
        except Exception as exc:
            raise RAGEngineError(f"Engine init failed: {exc}") from exc

    # ── public ────────────────────────────────────────────────────────────────
    def query(self, query: str, session_id: str = "default") -> dict:
        # 1. Input safety
        safe, reason = self._safety.check(query)
        if not safe:
            log.warning(f"Blocked: {reason}")
            return {"answer": "I can only answer questions about GDPR.", "context": []}

        # 2. Cache hit
        cached = self._cache.get(query)
        if cached:
            log.info("Cache HIT")
            return cached

        # 3. Off-topic fast reject
        analysis = self._retriever._analyzer.analyze(query)
        if analysis.confidence < 0.2:
            return {"answer": "I can only answer questions about GDPR.", "context": []}

        # 4. Build chain + invoke
        try:
            chain = self._build_chain()
            t0    = time.perf_counter()
            result = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )
            latency = (time.perf_counter() - t0) * 1000
            self._tracker.record(latency_ms=latency)
        except Exception as exc:
            log.error(f"Chain error: {exc}", exc_info=True)
            self._tracker.record(latency_ms=0, success=False, error=str(exc))
            raise RAGEngineError(f"Query failed: {exc}") from exc

        # 5. Output safety
        answer = result.get("answer", "")
        if not self._safety.check_output(answer):
            log.error("Unsafe output — suppressed")
            return {"answer": "Unable to generate a safe response.", "context": []}

        # 6. Cache + return
        self._cache.set(query, result)
        return result

    @property
    def tracker(self) -> LLMTracker:
        return self._tracker

    # ── private ───────────────────────────────────────────────────────────────
    def _build_chain(self):
        adapter   = _RetrieverAdapter(smart=self._retriever)
        qa_chain  = create_stuff_documents_chain(self._llm, _PROMPT)
        rag_chain = create_retrieval_chain(adapter, qa_chain)
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory()
        return self._sessions[session_id]
