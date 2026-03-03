from __future__ import annotations
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
from src.retrieval.retriever import SmartRetriever

log = get_logger("RAGEngine")

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """\
You are a precise legal assistant specialised in the GDPR (EU 2016/679).

Rules you MUST follow:
1. Base every answer ONLY on the context documents provided.
2. If the answer is not in the context, say: "I could not find that in the GDPR text."
3. For exact article / recital lookups: quote the relevant text verbatim, then explain it.
4. For conceptual questions: synthesise across sources, cite article numbers inline.
5. For follow-up questions ("describe it", "what does that mean"): use chat_history to
   identify the referent, then answer from context.
6. Never invent article numbers, chapter names, or legal obligations.
7. Keep answers concise unless the user asks for detail.

Context:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


# ── LangChain retriever wrapper ───────────────────────────────────────────────
class _RetrieverWrapper(BaseRetriever):
    """Thin adapter so SmartRetriever works with LangChain chains."""
    smart: Any   # SmartRetriever  (Any avoids pydantic forward-ref issues)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs, analysis = self.smart.retrieve(query)
        log.info(f"Chain retrieval: {len(docs)} docs | {analysis.intent.value}")
        return docs


# ── Engine ────────────────────────────────────────────────────────────────────
class RAGEngine:

    def __init__(self, vectorstore: FAISS) -> None:
        log.info("Initialising RAG Engine…")
        try:
            self._llm = ChatOpenAI(
                model=cfg.LLM_MODEL,
                temperature=0,
                api_key=cfg.OPENAI_API_KEY,
            )
            self._retriever  = SmartRetriever(vectorstore)
            self._safety     = SafetyGuardrails()
            self._cache      = QueryCache()
            self._tracker    = LLMTracker()
            self._sessions: dict[str, ChatMessageHistory] = {}
            log.info("RAG Engine ready")
        except Exception as exc:
            raise RAGEngineError(f"Engine init failed: {exc}") from exc

    # ── public ────────────────────────────────────────────────────────────────
    def query(self, query: str, session_id: str = "default") -> dict:
        # 1. Safety check
        safe, reason = self._safety.check(query)
        if not safe:
            log.warning(f"Query rejected: {reason}")
            return {"answer": "I can only answer questions about GDPR regulations.", "context": []}

        # 2. Cache
        cached = self._cache.get(query)
        if cached:
            return cached

        # 3. Confidence check (off-topic)
        analysis = self._retriever._analyzer.analyze(query)
        if analysis.confidence < 0.2:
            return {"answer": "I can only answer questions about GDPR regulations.", "context": []}

        # 4. Build + run chain
        try:
            chain  = self._build_chain()
            import time
            t0 = time.time()
            result = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )
            latency = (time.time() - t0) * 1000
            self._tracker.record(latency_ms=latency)
        except Exception as exc:
            log.error(f"Chain failed: {exc}", exc_info=True)
            raise RAGEngineError(f"Query failed: {exc}") from exc

        # 5. Output safety
        answer = result.get("answer", "")
        if not self._safety.check_output(answer):
            log.error("Unsafe output detected — suppressing")
            return {"answer": "Unable to generate a safe response. Please rephrase.", "context": []}

        # 6. Cache + return
        self._cache.set(query, result)
        return result

    @property
    def tracker(self) -> LLMTracker:
        return self._tracker

    # ── private ───────────────────────────────────────────────────────────────
    def _build_chain(self):
        wrapper   = _RetrieverWrapper(smart=self._retriever)
        qa_chain  = create_stuff_documents_chain(self._llm, _PROMPT)
        rag_chain = create_retrieval_chain(wrapper, qa_chain)
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
