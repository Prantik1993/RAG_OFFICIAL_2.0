"""
RAG Engine  (v4.2)
==================
- Uses PromptRegistry: active prompt loaded from prompts/vN.yaml
- Prompt version returned in every response for traceability
- source_files list injected into prompt context
- One LLM call per query, no routing LLM
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
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import src.config as cfg
from src.caching.query_cache import QueryCache
from src.exceptions import RAGEngineError
from src.guardrails.safety import SafetyGuardrails
from src.logger import get_logger
from src.monitoring.tracker import LLMTracker
from src.prompts.registry import PromptRegistry
from src.retrieval.bm25_index import BM25Index
from src.retrieval.retriever import SmartRetriever

log = get_logger("RAGEngine")


# ── LangChain retriever adapter ───────────────────────────────────────────────
class _RetrieverAdapter(BaseRetriever):
    smart: Any

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        docs, analysis = self.smart.retrieve(query)
        log.info(f"Retrieved {len(docs)} docs | {analysis.intent.value}")
        return docs


# ── Engine ────────────────────────────────────────────────────────────────────
class RAGEngine:

    def __init__(self, vectorstore: FAISS, bm25: BM25Index) -> None:
        log.info("Initialising RAG Engine…")
        try:
            self._registry  = PromptRegistry()
            self._retriever = SmartRetriever(vectorstore, bm25)
            self._safety    = SafetyGuardrails()
            self._cache     = QueryCache()
            self._tracker   = LLMTracker()
            self._sessions: dict[str, ChatMessageHistory] = {}
            log.info(
                f"RAG Engine ready | "
                f"prompt={self._registry.get().version} | "
                f"model={self._registry.get().model}"
            )
        except Exception as exc:
            raise RAGEngineError(f"Engine init failed: {exc}") from exc

    # ── public ────────────────────────────────────────────────────────────────
    def query(
        self,
        query: str,
        session_id: str = "default",
        prompt_version: str | None = None,
    ) -> dict:
        # 1. Input safety
        safe, reason = self._safety.check(query)
        if not safe:
            log.warning(f"Blocked: {reason}")
            return {"answer": "I can only answer questions about GDPR.", "context": [], "prompt_version": "N/A"}

        # 2. Cache
        cached = self._cache.get(query)
        if cached:
            log.info("Cache HIT")
            return cached

        # 3. Off-topic
        analysis = self._retriever._analyzer.analyze(query)
        if analysis.confidence < 0.2:
            return {"answer": "I can only answer questions about GDPR.", "context": [], "prompt_version": "N/A"}

        # 4. Load prompt
        prompt_cfg = self._registry.get(prompt_version)
        llm = ChatOpenAI(
            model=prompt_cfg.model,
            temperature=prompt_cfg.temperature,
            api_key=cfg.OPENAI_API_KEY,
        )

        # 5. Build chain + invoke
        try:
            chain = self._build_chain(llm, prompt_cfg)
            t0 = time.perf_counter()
            result = chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )
            latency = (time.perf_counter() - t0) * 1000
            self._tracker.record(latency_ms=latency, prompt_version=prompt_cfg.version)
        except Exception as exc:
            log.error(f"Chain error: {exc}", exc_info=True)
            self._tracker.record(latency_ms=0, success=False, error=str(exc))
            raise RAGEngineError(f"Query failed: {exc}") from exc

        # 6. Output safety
        answer = result.get("answer", "")
        if not self._safety.check_output(answer):
            log.error("Unsafe output suppressed")
            return {"answer": "Unable to generate a safe response.", "context": [], "prompt_version": prompt_cfg.version}

        # 7. Attach prompt version to result, cache, return
        result["prompt_version"] = prompt_cfg.version
        self._cache.set(query, result)
        return result

    @property
    def tracker(self) -> LLMTracker:
        return self._tracker

    @property
    def prompt_registry(self) -> PromptRegistry:
        return self._registry

    # ── private ───────────────────────────────────────────────────────────────
    def _build_chain(self, llm, prompt_cfg):
        # Inject available source files into prompt
        source_files = self._get_source_files()
        prompt = self._build_prompt(prompt_cfg, source_files)

        adapter   = _RetrieverAdapter(smart=self._retriever)
        qa_chain  = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(adapter, qa_chain)
        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def _build_prompt(self, prompt_cfg, source_files: str):
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        system = prompt_cfg.system_template.replace("{source_files}", source_files)
        return ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def _get_source_files(self) -> str:
        """List unique source_file values from indexed documents."""
        try:
            docs   = self._retriever._all_docs
            files  = sorted({d.metadata.get("source_file", "unknown") for d in docs[:500]})
            return ", ".join(files) if files else "GDPR EU 2016/679"
        except Exception:
            return "GDPR EU 2016/679"

    def _get_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMessageHistory()
        return self._sessions[session_id]
