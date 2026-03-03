"""
Unit tests for multi-PDF ingestion pipeline.
"""
import shutil
import tempfile
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.pipeline import IngestionPipeline, _sha256


def test_sha256_returns_16_chars(tmp_path):
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    h = _sha256(f)
    assert len(h) == 16
    assert h.isalnum()


def test_sha256_different_files_differ(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"content A")
    f2.write_bytes(b"content B")
    assert _sha256(f1) != _sha256(f2)


def test_collect_paths_single(tmp_path):
    f = tmp_path / "test.pdf"
    f.write_bytes(b"%PDF fake")
    pipeline = IngestionPipeline()
    paths = pipeline._collect_paths(f)
    assert paths == [f]


def test_collect_paths_missing_raises():
    pipeline = IngestionPipeline()
    with pytest.raises(Exception):
        pipeline._collect_paths(Path("/nonexistent/file.pdf"))


def test_no_pdfs_raises(tmp_path, monkeypatch):
    import src.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    pipeline = IngestionPipeline()
    with pytest.raises(Exception, match="No PDF"):
        pipeline.run()
