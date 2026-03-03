"""
Vector Store Manager  (v4.1)
============================
Builds/loads FAISS and also builds the BM25 index from the same docs.
Returns both so the retriever can run hybrid search.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import src.config as cfg
from src.exceptions import VectorStoreError
from src.logger import get_logger
from src.retrieval.bm25_index import BM25Index

log = get_logger("VectorStore")


class VectorStoreManager:

    def __init__(self) -> None:
        log.info(f"Loading embeddings: {cfg.EMBEDDING_MODEL}")
        try:
            self._emb = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
        except Exception as exc:
            raise VectorStoreError(f"Embedding init failed: {exc}") from exc
        log.info("Embeddings ready")

    # ── public ────────────────────────────────────────────────────────────────
    def load_faiss(self) -> Optional[FAISS]:
        index_file = cfg.STORE_DIR / "index.faiss"
        if not index_file.exists():
            log.info("No existing FAISS index found")
            return None
        try:
            vs = FAISS.load_local(
                str(cfg.STORE_DIR),
                self._emb,
                allow_dangerous_deserialization=True,
            )
            log.info(f"FAISS index loaded from {cfg.STORE_DIR}")
            return vs
        except Exception as exc:
            raise VectorStoreError(f"FAISS load failed: {exc}") from exc

    def create_faiss(self, docs: list[Document]) -> FAISS:
        if not docs:
            raise VectorStoreError("No documents to index")
        try:
            log.info(f"Building FAISS from {len(docs)} documents…")
            vs = FAISS.from_documents(docs, self._emb)
            cfg.STORE_DIR.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(cfg.STORE_DIR))
            log.info("FAISS index saved")
            return vs
        except Exception as exc:
            raise VectorStoreError(f"FAISS create failed: {exc}") from exc

    @staticmethod
    def build_bm25(docs: list[Document]) -> BM25Index:
        """Build BM25 keyword index from documents (always in-memory)."""
        bm25 = BM25Index()
        bm25.build(docs)
        return bm25

    def load_or_create(
        self,
        docs_fn: Callable[[], list[Document]],
    ) -> tuple[FAISS, BM25Index]:
        """
        Load FAISS from disk (fast path) or build from scratch.
        BM25 is always rebuilt from docs — it's fast (~1s for 5k docs)
        and doesn't need persistence.

        Returns:
            (faiss_vectorstore, bm25_index)
        """
        vs = self.load_faiss()
        if vs is not None:
            # FAISS loaded from disk — still need docs for BM25
            log.info("Extracting docs from FAISS docstore for BM25…")
            store = vs.docstore
            docs  = [store.search(did) for did in vs.index_to_docstore_id.values()]
            docs  = [d for d in docs if hasattr(d, "page_content")]
        else:
            log.info("Building fresh index…")
            docs = docs_fn()
            vs   = self.create_faiss(docs)

        bm25 = self.build_bm25(docs)
        log.info(f"Stores ready — FAISS: {len(vs.index_to_docstore_id)} | BM25: {len(bm25)}")
        return vs, bm25
