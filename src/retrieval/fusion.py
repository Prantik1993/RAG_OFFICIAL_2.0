"""
Reciprocal Rank Fusion (RRF)
============================
Merges two ranked lists (FAISS dense + BM25 sparse) into one combined
ranking without needing to normalise scores from different score spaces.

Algorithm (Cormack et al. 2009):
    RRF_score(d) = Σ  1 / (k + rank_i(d))
                    i
    where k=60 (standard constant, dampens top-rank advantage)

Why RRF over score normalisation?
  - FAISS returns cosine similarities (0-1), BM25 returns raw counts
  - Converting them to the same scale requires calibration per corpus
  - RRF only uses ordinal rank position — no calibration needed
  - Empirically outperforms score-based fusion on most retrieval benchmarks
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.logger import get_logger

log = get_logger("RRF")

_RRF_K = 60   # standard constant


def reciprocal_rank_fusion(
    *ranked_lists: list[Document],
    k: int = _RRF_K,
) -> list[Document]:
    """
    Fuse any number of ranked document lists using RRF.

    Args:
        *ranked_lists: One or more lists of Documents, each sorted by
                       relevance (best first).
        k:             RRF constant (default 60).

    Returns:
        Single merged list sorted by fused RRF score (best first).
        Deduplication is by page_content hash.
    """
    scores:   dict[str, float]   = {}
    doc_map:  dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = _doc_key(doc)
            scores[key]  = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc  # keep most recent copy (carries freshest metadata)

    fused = sorted(doc_map.keys(), key=lambda key: scores[key], reverse=True)

    # Attach rrf_score to metadata for traceability
    result: list[Document] = []
    for key in fused:
        doc = doc_map[key]
        result.append(Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, "rrf_score": round(scores[key], 6)},
        ))

    log.debug(
        f"RRF fused {sum(len(r) for r in ranked_lists)} → {len(result)} unique docs"
    )
    return result


def _doc_key(doc: Document) -> str:
    """Stable identity key: prefer reference_path, fall back to content hash."""
    ref = doc.metadata.get("reference_path")
    if ref:
        return ref
    return str(hash(doc.page_content[:200]))
