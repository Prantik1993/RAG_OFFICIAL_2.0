"""
Smart Retriever
===============
Routes each query to the right retrieval strategy based on QueryAnalysis:

  EXACT   -> metadata filter over all docs, then semantic top-up
  RANGE   -> chapter/section filter, article-level docs only
  SEMANTIC -> pure vector similarity search

No k=5000 hack.  FAISS docstore is iterated once at startup and cached.
"""

from __future__ import annotations

from functools import cached_property
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import src.config as cfg
from src.logger import get_logger
from src.retrieval.query_analyzer import Intent, QueryAnalysis, QueryAnalyzer

log = get_logger("Retriever")


class SmartRetriever:

    def __init__(self, vectorstore: FAISS) -> None:
        self._vs       = vectorstore
        self._analyzer = QueryAnalyzer()

    # ── cached flat list of all indexed docs ──────────────────────────────────
    @cached_property
    def _all_docs(self) -> list[Document]:
        """
        Load every document from the FAISS docstore once and cache it.
        Avoids the repeated similarity_search("", k=5000) anti-pattern.
        """
        store = self._vs.docstore
        docs = [store.search(doc_id) for doc_id in self._vs.index_to_docstore_id.values()]
        log.info(f"Cached {len(docs)} documents from docstore")
        return [d for d in docs if isinstance(d, Document)]

    # ── public ────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, k: Optional[int] = None) -> tuple[list[Document], QueryAnalysis]:
        k = k or cfg.RETRIEVAL_K
        analysis = self._analyzer.analyze(query)

        if analysis.confidence < 0.2:
            log.info("Off-topic query — returning empty")
            return [], analysis

        if analysis.intent == Intent.EXACT:
            docs = self._exact(analysis, k)
        elif analysis.intent == Intent.RANGE:
            docs = self._range(analysis, k)
        else:
            docs = self._semantic(query, k)

        # Fallback: if filters returned nothing, do semantic
        if not docs:
            log.warning("Primary strategy returned 0 docs — semantic fallback")
            docs = self._semantic(query, k)

        log.info(f"Retrieved {len(docs)} docs | intent={analysis.intent.value}")
        return docs, analysis

    # ── strategies ────────────────────────────────────────────────────────────
    def _exact(self, analysis: QueryAnalysis, k: int) -> list[Document]:
        """Filter cached docs by metadata, most-specific first."""
        filt = analysis.filter_dict()
        matched = [d for d in self._all_docs if self._matches(d, filt)]

        # Sort: subpoint > point > article > section > chapter
        matched.sort(key=lambda d: _specificity(d), reverse=True)

        # If very specific (subpoint/point), also include parent article for context
        if analysis.subpoint or analysis.point:
            parent_filt = {k2: v for k2, v in filt.items()
                           if k2 not in ("subpoint", "point")}
            for d in self._all_docs:
                if d not in matched and self._matches(d, parent_filt):
                    matched.append(d)

        result = matched[:k]

        # Always top-up with semantic if we got fewer than k
        if len(result) < k:
            sem = self._semantic(analysis.query, k - len(result))
            seen = {id(d) for d in result}
            result += [d for d in sem if id(d) not in seen]

        return result[:k]

    def _range(self, analysis: QueryAnalysis, k: int) -> list[Document]:
        """Return article-level docs for a chapter/section."""
        matched = []
        for d in self._all_docs:
            if analysis.chapter and str(d.metadata.get("chapter")) != str(analysis.chapter):
                continue
            if analysis.section and str(d.metadata.get("section")) != str(analysis.section):
                continue
            if d.metadata.get("level") == "article":
                matched.append(d)

        matched.sort(key=lambda d: int(d.metadata.get("article", 0) or 0))
        return matched[:k]

    def _semantic(self, query: str, k: int) -> list[Document]:
        return self._vs.similarity_search(query, k=k)

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _matches(doc: Document, filt: dict) -> bool:
        return all(str(doc.metadata.get(k)) == str(v) for k, v in filt.items())


def _specificity(doc: Document) -> int:
    score = 0
    if doc.metadata.get("subpoint"): score += 1000
    if doc.metadata.get("point"):    score += 100
    if doc.metadata.get("article"):  score += 10
    if doc.metadata.get("section"):  score += 5
    if doc.metadata.get("chapter"):  score += 1
    return score
