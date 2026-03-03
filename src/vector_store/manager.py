from __future__ import annotations
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import src.config as cfg
from src.exceptions import VectorStoreError
from src.logger import get_logger

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
    def load(self) -> Optional[FAISS]:
        index_file = cfg.STORE_DIR / "index.faiss"
        if not index_file.exists():
            log.info("No existing FAISS index")
            return None
        try:
            vs = FAISS.load_local(
                str(cfg.STORE_DIR),
                self._emb,
                allow_dangerous_deserialization=True,
            )
            log.info(f"Loaded FAISS index from {cfg.STORE_DIR}")
            return vs
        except Exception as exc:
            raise VectorStoreError(f"FAISS load failed: {exc}") from exc

    def create(self, docs: list[Document]) -> FAISS:
        if not docs:
            raise VectorStoreError("No documents provided")
        try:
            log.info(f"Building FAISS index from {len(docs)} documents…")
            vs = FAISS.from_documents(docs, self._emb)
            cfg.STORE_DIR.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(cfg.STORE_DIR))
            log.info("FAISS index saved")
            return vs
        except Exception as exc:
            raise VectorStoreError(f"FAISS create failed: {exc}") from exc

    def load_or_create(self, docs_fn) -> FAISS:
        """Load existing index or call docs_fn() to build one."""
        vs = self.load()
        if vs is not None:
            return vs
        log.info("Building new index…")
        docs = docs_fn()
        return self.create(docs)
