"""
Unit tests for BM25Index — no network, no FAISS, no LLM.
"""
import pytest
from langchain_core.documents import Document
from src.retrieval.bm25_index import BM25Index, _tokenize


# ── tokenizer ─────────────────────────────────────────────────────────────────
def test_tokenize_removes_stopwords():
    tokens = _tokenize("the data controller shall process")
    assert "the" not in tokens
    assert "shall" not in tokens
    assert "data" in tokens
    assert "controller" in tokens


def test_tokenize_lowercases():
    assert _tokenize("GDPR Article") == ["gdpr", "article"]


def test_tokenize_removes_punctuation():
    tokens = _tokenize("Art. 6(1)(f)")
    assert "." not in " ".join(tokens)
    assert "art" in tokens


# ── index ─────────────────────────────────────────────────────────────────────
@pytest.fixture
def index():
    docs = [
        Document(page_content="data controller processes personal data lawfully",
                 metadata={"article": "1"}),
        Document(page_content="data subject has right to access personal information",
                 metadata={"article": "15"}),
        Document(page_content="supervisory authority shall cooperate with each other",
                 metadata={"article": "60"}),
        Document(page_content="legitimate interest processing consent required",
                 metadata={"article": "6"}),
    ]
    idx = BM25Index()
    idx.build(docs)
    return idx


def test_build(index):
    assert index.is_ready
    assert len(index) == 4


def test_search_returns_relevant(index):
    results = index.search("data controller processes", k=2)
    assert len(results) == 2
    assert "controller" in results[0].page_content


def test_search_exact_legal_term(index):
    results = index.search("supervisory authority cooperate", k=1)
    assert len(results) == 1
    assert results[0].metadata["article"] == "60"


def test_search_empty_query_returns_empty(index):
    results = index.search("the a is", k=3)   # all stopwords
    assert results == []


def test_build_empty_raises():
    idx = BM25Index()
    with pytest.raises(ValueError):
        idx.build([])


def test_not_built_raises():
    idx = BM25Index()
    with pytest.raises(RuntimeError):
        idx.search("anything", k=3)
