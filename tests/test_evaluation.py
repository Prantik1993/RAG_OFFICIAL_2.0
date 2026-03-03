"""
Unit tests for RAGAS metric functions — no engine, no API, no LLM.
"""
from src.evaluation.ragas_eval import (
    _faithfulness,
    _answer_relevancy,
    _context_precision,
    _context_recall,
)


def test_faithfulness_high_when_answer_uses_context():
    ctx    = ["The data controller must ensure lawful processing of personal data"]
    answer = "The controller must process data lawfully according to GDPR principles"
    score  = _faithfulness(answer, ctx)
    assert score > 0.3


def test_faithfulness_zero_empty():
    assert _faithfulness("", []) == 0.0


def test_answer_relevancy_exact_match():
    score = _answer_relevancy(
        "Article 15 grants the right to access personal data",
        "What is the right of access under Article 15?"
    )
    assert score > 0.3


def test_answer_relevancy_zero_empty():
    assert _answer_relevancy("", "") == 0.0


def test_context_precision_relevant_context():
    contexts = ["Article 17 erasure right to be forgotten data subject"]
    gt       = "Article 17 grants the right to erasure of personal data"
    score    = _context_precision(contexts, gt)
    assert score > 0.3


def test_context_precision_zero_empty():
    assert _context_precision([], "anything") == 0.0


def test_context_recall_full_coverage():
    contexts = ["data subject right erasure controller personal information removal deletion"]
    gt       = "data subject erasure right"
    score    = _context_recall(contexts, gt)
    assert score >= 0.5


def test_context_recall_zero_empty():
    assert _context_recall([], "anything") == 0.0
