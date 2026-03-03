"""
Unit tests for RRF fusion — pure logic, no dependencies.
"""
from langchain_core.documents import Document
from src.retrieval.fusion import reciprocal_rank_fusion, _doc_key


def _doc(content: str, ref: str | None = None) -> Document:
    meta = {"reference_path": ref} if ref else {}
    return Document(page_content=content, metadata=meta)


def test_single_list_preserves_order():
    docs = [_doc("a", "ref:a"), _doc("b", "ref:b"), _doc("c", "ref:c")]
    result = reciprocal_rank_fusion(docs)
    assert [d.page_content for d in result] == ["a", "b", "c"]


def test_two_lists_deduplication():
    d1 = _doc("article 15", "art:15")
    d2 = _doc("article 6",  "art:6")
    d3 = _doc("article 15", "art:15")   # same ref as d1

    result = reciprocal_rank_fusion([d1, d2], [d3, d2])
    keys = [d.metadata.get("reference_path") for d in result]
    assert len(keys) == len(set(keys)), "duplicates not removed"


def test_top_ranked_in_both_lists_wins():
    # doc A is rank 1 in list1, rank 1 in list2 → should come first
    a = _doc("doc A", "A")
    b = _doc("doc B", "B")
    c = _doc("doc C", "C")

    result = reciprocal_rank_fusion([a, b, c], [a, c, b])
    assert result[0].page_content == "doc A"


def test_rrf_score_attached_to_metadata():
    docs = [_doc("x", "x"), _doc("y", "y")]
    result = reciprocal_rank_fusion(docs)
    for d in result:
        assert "rrf_score" in d.metadata
        assert d.metadata["rrf_score"] > 0


def test_empty_lists():
    result = reciprocal_rank_fusion([], [])
    assert result == []


def test_one_empty_one_full():
    docs = [_doc("a", "a"), _doc("b", "b")]
    result = reciprocal_rank_fusion([], docs)
    assert len(result) == 2
