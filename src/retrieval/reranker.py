"""
CrossEncoder Reranker
=====================
Uses a sentence-transformers CrossEncoder to rerank candidate documents
retrieved by the hybrid retriever.

Why reranking?
  - Hybrid search fetches RETRIEVAL_K_FETCH=20 candidates (broad recall)
  - CrossEncoder scores every (query, doc) pair jointly — much more
    accurate than the bi-encoder cosine similarity used by FAISS
  - Final top-K returned to the LLM are the MOST relevant, not just
    the most similar

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22 MB, runs on CPU in ~50 ms for 20 docs
  - Trained on MS MARCO passage ranking — works well for legal Q&A
  - Can be swapped via RERANKER_MODEL env var

Lazy loading: model is downloaded on first use, cached in memory.
"""

from __future__ import annotations

from functools import cached_property

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import src.config as cfg
from src.logger import get_logger

log = get_logger("Reranker")


class CrossEncoderReranker:
    """
    Reranks a candidate list to the top-k most relevant documents
    for a given query.

    Usage:
        reranker = CrossEncoderReranker()
        top_docs = reranker.rerank(query, candidates, k=6)
    """

    def __init__(self, model_name: str = cfg.RERANKER_MODEL) -> None:
        self._model_name = model_name
        # Don't load model at __init__ — load lazily on first rerank call
        # so startup is fast and model is only in memory when needed.

    @cached_property
    def _model(self) -> CrossEncoder:
        log.info(f"Loading CrossEncoder: {self._model_name}")
        model = CrossEncoder(self._model_name, max_length=512)
        log.info("CrossEncoder ready")
        return model

    # ── public ────────────────────────────────────────────────────────────────
    def rerank(
        self,
        query: str,
        candidates: list[Document],
        k: int,
    ) -> list[Document]:
        """
        Score every (query, doc) pair and return the top-k.

        Args:
            query:      Original user query string
            candidates: Documents from hybrid retrieval (typically 20)
            k:          Number of documents to return after reranking

        Returns:
            Top-k documents sorted by CrossEncoder relevance score (desc)
        """
        if not candidates:
            return []

        # CrossEncoder needs list of [query, text] pairs
        pairs = [[query, doc.page_content] for doc in candidates]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            log.error(f"CrossEncoder predict failed: {exc} — falling back to order")
            return candidates[:k]

        # Attach score to each doc's metadata (useful for debugging / tracing)
        scored: list[tuple[float, Document]] = []
        for score, doc in zip(scores, candidates):
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "rerank_score": round(float(score), 4)},
            )
            scored.append((float(score), doc_copy))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        top_k = [doc for _, doc in scored[:k]]

        log.info(
            f"Reranked {len(candidates)} → {len(top_k)} | "
            f"top score={scored[0][0]:.3f} | bottom={scored[k-1][0]:.3f}"
            if len(scored) >= k else
            f"Reranked {len(candidates)} → {len(top_k)}"
        )
        return top_k

    # ── diagnostics ───────────────────────────────────────────────────────────
    def score_single(self, query: str, text: str) -> float:
        """Score one (query, text) pair. Useful for unit tests."""
        return float(self._model.predict([[query, text]])[0])
