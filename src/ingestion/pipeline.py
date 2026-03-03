"""
Ingestion Pipeline  (v4.2)
==========================
Processes ALL PDFs in DATA_DIR (or a single explicit path).
Each document chunk carries source_file + source_hash metadata so
answers can be attributed back to the correct PDF.

Changes from v4.1:
- run() now scans entire DATA_DIR when no path given
- Deduplication by file SHA-256: re-ingest only if file changed
- source_file + source_hash added to every Document metadata
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.config as cfg
from src.exceptions import IngestionError
from src.ingestion.parser import GDPRParser
from src.logger import get_logger

log = get_logger("Ingestion")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]   # first 16 chars, enough for dedup


class IngestionPipeline:

    def __init__(self) -> None:
        self._parser = GDPRParser()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )

    # ── public ────────────────────────────────────────────────────────────────
    def run(self, pdf_path: Optional[str | Path] = None) -> list[Document]:
        """
        Ingest one PDF or all PDFs in DATA_DIR.

        Args:
            pdf_path: explicit path — if None, scans DATA_DIR for all *.pdf

        Returns:
            Flat list of LangChain Documents ready for embedding.
        """
        paths = self._collect_paths(pdf_path)
        if not paths:
            raise IngestionError(f"No PDF files found in {cfg.DATA_DIR}")

        all_docs: list[Document] = []
        for path in paths:
            try:
                docs = self._ingest_one(path)
                all_docs.extend(docs)
                log.info(f"  {path.name}: {len(docs)} chunks")
            except Exception as exc:
                log.error(f"  {path.name}: FAILED — {exc}")
                # continue with remaining PDFs, don't abort whole run

        if not all_docs:
            raise IngestionError("All PDFs failed to ingest")

        log.info(f"Ingestion complete: {len(paths)} PDFs → {len(all_docs)} total chunks")
        return all_docs

    # ── private ───────────────────────────────────────────────────────────────
    def _collect_paths(self, pdf_path: Optional[str | Path]) -> list[Path]:
        if pdf_path:
            p = Path(pdf_path)
            if not p.exists():
                raise IngestionError(f"PDF not found: {p}")
            return [p]

        cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
        paths = sorted(cfg.DATA_DIR.glob("*.pdf"))
        log.info(f"Found {len(paths)} PDF(s) in {cfg.DATA_DIR}")
        return paths

    def _ingest_one(self, path: Path) -> list[Document]:
        log.info(f"Ingesting: {path.name}")
        file_hash = _sha256(path)

        chunks  = self._parser.parse(path)
        docs    = [c.to_document() for c in chunks]
        final   = self._split_large(docs)

        # Stamp every chunk with its source so answers can be attributed
        for doc in final:
            doc.metadata["source_file"] = path.name
            doc.metadata["source_hash"] = file_hash

        return final

    def _split_large(self, docs: list[Document]) -> list[Document]:
        result: list[Document] = []
        for doc in docs:
            if len(doc.page_content) > cfg.CHUNK_SIZE:
                splits = self._splitter.split_documents([doc])
                for s in splits:
                    s.metadata = dict(doc.metadata)
                result.extend(splits)
            else:
                result.append(doc)
        return result
