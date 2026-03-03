"""
BM25 Keyword Index
==================
Wraps rank-bm25 to provide keyword / lexical search over the same
document corpus that FAISS handles semantically.

Why BM25 alongside FAISS?
  - FAISS (dense) excels at conceptual / paraphrased queries
  - BM25 (sparse) excels at exact legal terms: "legitimate interest",
    "data portability", "supervisory authority", specific article refs
  - Together they cover both failure modes

Build once at startup from the same docs used for FAISS.
Thread-safe for reads (index is immutable after build).
"""

from __future__ import annotations

import re
import string
from typing import Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.logger import get_logger

log = get_logger("BM25Index")

# Simple English stopwords — keeps legal terms intact
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "or", "and", "not", "this", "that", "it", "its",
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split, remove stopwords."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """
    Keyword index built from a list of LangChain Documents.
    Exposes .search(query, k) -> list[Document] matching FAISS interface.
    """

    def __init__(self) -> None:
        self._docs:  list[Document] = []
        self._index: Optional[BM25Okapi] = None

    # ── build ─────────────────────────────────────────────────────────────────
    def build(self, docs: list[Document]) -> None:
        if not docs:
            raise ValueError("Cannot build BM25 index from empty document list")

        self._docs = docs
        tokenized  = [_tokenize(d.page_content) for d in docs]
        self._index = BM25Okapi(tokenized)
        log.info(f"BM25 index built: {len(docs)} documents")

    # ── search ────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int) -> list[Document]:
        if self._index is None:
            raise RuntimeError("BM25Index not built — call .build() first")

        tokens = _tokenize(query)
        if not tokens:
            log.warning("BM25: empty token list after preprocessing — returning []")
            return []

        scores = self._index.get_scores(tokens)

        # Pair (score, doc), sort descending, take top-k
        ranked = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )
        results = [doc for score, doc in ranked[:k] if score > 0.0]
        log.debug(f"BM25 search '{query[:50]}' → {len(results)} results")
        return results

    # ── helpers ───────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._docs)

    @property
    def is_ready(self) -> bool:
        return self._index is not None
