"""
Ingestion Pipeline
==================
1. Parse PDF -> hierarchical LegalChunks
2. Split oversized chunks (preserving metadata)
3. Return LangChain Documents ready for embedding
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import src.config as cfg
from src.exceptions import IngestionError
from src.ingestion.parser import GDPRParser
from src.logger import get_logger

log = get_logger("Ingestion")


class IngestionPipeline:

    def __init__(self) -> None:
        self._parser = GDPRParser()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.CHUNK_SIZE,
            chunk_overlap=cfg.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )

    def run(self, pdf_path: Optional[str | Path] = None) -> list[Document]:
        path = Path(pdf_path) if pdf_path else cfg.DATA_DIR / "CELEX_32016R0679_EN_TXT.pdf"
        if not path.exists():
            raise IngestionError(f"PDF not found: {path}")

        try:
            log.info(f"Ingesting: {path}")
            chunks = self._parser.parse(path)
            docs   = [c.to_document() for c in chunks]
            final  = self._split_large(docs)
            self._log_stats(final)
            return final

        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Pipeline failed: {exc}") from exc

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

    @staticmethod
    def _log_stats(docs: list[Document]) -> None:
        recitals = {d.metadata["recital"] for d in docs if "recital" in d.metadata}
        articles = {d.metadata["article"] for d in docs if "article" in d.metadata}
        log.info(
            f"Ingestion done -- {len(docs)} docs | "
            f"{len(recitals)} recitals | {len(articles)} articles"
        )
