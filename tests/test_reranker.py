"""
Unit tests for CrossEncoderReranker.
Model download ~22 MB — skipped in CI if no network.
"""
import pytest
from langchain_core.documents import Document
from src.retrieval.reranker import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker():
    return CrossEncoderReranker()


def _doc(content: str) -> Document:
    return Document(page_content=content, metadata={"article": "test"})


def test_rerank_returns_k_docs(reranker):
    docs = [_doc(f"document number {i} about GDPR data protection") for i in range(10)]
    result = reranker.rerank("data protection rights", docs, k=3)
    assert len(result) == 3


def test_rerank_attaches_score(reranker):
    docs = [_doc("GDPR data subject rights"), _doc("irrelevant content about cats")]
    result = reranker.rerank("data subject rights under GDPR", docs, k=2)
    for d in result:
        assert "rerank_score" in d.metadata


def test_rerank_most_relevant_first(reranker):
    relevant   = _doc("The data subject shall have the right to access personal data under Article 15 GDPR")
    irrelevant = _doc("The weather forecast shows rain tomorrow afternoon")
    docs = [irrelevant, relevant]   # irrelevant first — reranker should flip order

    result = reranker.rerank("right to access personal data Article 15", docs, k=2)
    assert result[0].page_content == relevant.page_content


def test_rerank_empty_returns_empty(reranker):
    assert reranker.rerank("anything", [], k=5) == []


def test_rerank_fewer_than_k(reranker):
    docs = [_doc("only one doc")]
    result = reranker.rerank("query", docs, k=5)
    assert len(result) == 1


def test_score_single(reranker):
    score = reranker.score_single(
        "right to access personal data",
        "Article 15 grants data subjects the right to obtain access to personal data."
    )
    assert isinstance(score, float)
