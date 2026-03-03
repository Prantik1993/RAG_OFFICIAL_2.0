"""
Smart Hybrid Retriever  (v4.1)
==============================
Full pipeline per query:

  1. QueryAnalyzer  — regex classifies intent + extracts refs (zero LLM cost)
  2. Retrieval      — runs TWO parallel strategies:
       a. FAISS semantic search  (dense vectors, conceptual matches)
       b. BM25 keyword search    (sparse, exact legal term matches)
       For EXACT intent, metadata-filtered candidates augment both.
  3. RRF Fusion     — merges dense+sparse ranked lists via Reciprocal Rank Fusion
  4. CrossEncoder   — reranks the fused top-N to final top-K

Flow:
    query
      │
      ├─[FAISS]──top-N dense──┐
      │                        ├─ RRF fusion ─ top-N*2 ─ CrossEncoder ─ top-K
      └─[BM25]──top-N sparse──┘

Config knobs (env vars):
    RETRIEVAL_K        = 6   final docs returned to LLM
    RETRIEVAL_K_FETCH  = 20  candidates fetched per source before reranking
"""

from __future__ import annotations

from functools import cached_property
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

import src.config as cfg
from src.logger import get_logger
from src.retrieval.bm25_index import BM25Index
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.query_analyzer import Intent, QueryAnalysis, QueryAnalyzer
from src.retrieval.reranker import CrossEncoderReranker

log = get_logger("Retriever")


class SmartRetriever:

    def __init__(self, vectorstore: FAISS, bm25: BM25Index) -> None:
        self._vs       = vectorstore
        self._bm25     = bm25
        self._analyzer = QueryAnalyzer()
        self._reranker = CrossEncoderReranker()
        log.info("SmartRetriever initialised (FAISS + BM25 + CrossEncoder)")

    # ── cached flat doc list (for metadata filtering) ─────────────────────────
    @cached_property
    def _all_docs(self) -> list[Document]:
        store = self._vs.docstore
        docs  = [store.search(did) for did in self._vs.index_to_docstore_id.values()]
        docs  = [d for d in docs if isinstance(d, Document)]
        log.info(f"Docstore cached: {len(docs)} documents")
        return docs

    # ── public ────────────────────────────────────────────────────────────────
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> tuple[list[Document], QueryAnalysis]:
        k      = k or cfg.RETRIEVAL_K
        fetch  = cfg.RETRIEVAL_K_FETCH
        analysis = self._analyzer.analyze(query)

        # Off-topic guard
        if analysis.confidence < 0.2:
            log.info("Off-topic — returning empty")
            return [], analysis

        # ── Step 1: Gather candidates ─────────────────────────────────────────
        if analysis.intent == Intent.EXACT:
            candidates = self._exact_candidates(analysis, fetch)
        elif analysis.intent == Intent.RANGE:
            candidates = self._range_candidates(analysis, fetch)
            # range queries: skip reranker, ordering by article number is better
            log.info(f"RANGE: {len(candidates)} docs, skipping reranker")
            return candidates[:k], analysis
        else:
            candidates = self._hybrid_candidates(query, fetch)

        # ── Step 2: Fallback ──────────────────────────────────────────────────
        if not candidates:
            log.warning("All strategies returned 0 — semantic fallback")
            candidates = self._vs.similarity_search(query, k=fetch)

        # ── Step 3: CrossEncoder rerank ───────────────────────────────────────
        final = self._reranker.rerank(query, candidates, k=k)

        log.info(
            f"retrieve done | intent={analysis.intent.value} "
            f"candidates={len(candidates)} → final={len(final)}"
        )
        return final, analysis

    # ── candidate gathering strategies ───────────────────────────────────────
    def _hybrid_candidates(self, query: str, fetch: int) -> list[Document]:
        """
        Run FAISS and BM25 in parallel, fuse with RRF.
        Returns up to fetch*2 unique candidates for the reranker.
        """
        dense  = self._vs.similarity_search(query, k=fetch)
        sparse = self._bm25.search(query, k=fetch)
        fused  = reciprocal_rank_fusion(dense, sparse)
        log.debug(
            f"Hybrid | dense={len(dense)} sparse={len(sparse)} "
            f"fused={len(fused)}"
        )
        return fused

    def _exact_candidates(self, analysis: QueryAnalysis, fetch: int) -> list[Document]:
        """
        For exact refs (Article 15.1.a):
          - Start with metadata-filtered docs (guaranteed correct article)
          - Augment with hybrid search for same article context
          - Fuse via RRF
        """
        filt    = analysis.filter_dict()
        matched = [d for d in self._all_docs if self._matches(d, filt)]
        matched.sort(key=_specificity, reverse=True)

        # Include parent article chunks for full context
        if analysis.subpoint or analysis.point:
            parent = {k: v for k, v in filt.items()
                      if k not in ("subpoint", "point")}
            for d in self._all_docs:
                if d not in matched and self._matches(d, parent):
                    matched.append(d)

        # Hybrid search for same query (catches paraphrased / surrounding context)
        hybrid = self._hybrid_candidates(analysis.query, fetch)

        # Fuse: metadata matches rank-boosted, hybrid fills gaps
        fused = reciprocal_rank_fusion(matched, hybrid)
        return fused

    def _range_candidates(self, analysis: QueryAnalysis, fetch: int) -> list[Document]:
        """Return article-level docs for a chapter/section, sorted by article number."""
        out: list[Document] = []
        for d in self._all_docs:
            if analysis.chapter and str(d.metadata.get("chapter")) != str(analysis.chapter):
                continue
            if analysis.section and str(d.metadata.get("section")) != str(analysis.section):
                continue
            if d.metadata.get("level") == "article":
                out.append(d)
        out.sort(key=lambda d: int(d.metadata.get("article", 0) or 0))
        return out[:fetch]

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
