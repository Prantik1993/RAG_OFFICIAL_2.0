"""
Unit tests for QueryAnalyzer — no LLM, no network, no FAISS needed.
"""
import pytest
from src.retrieval.query_analyzer import Intent, QueryAnalyzer

ana = QueryAnalyzer()


@pytest.mark.parametrize("query,expected_article,expected_point,expected_subpoint", [
    ("What is Article 15?",         "15", None,  None),
    ("Article 15.1",                "15", "1",   None),
    ("Article 15.1.a",              "15", "1",   "a"),
    ("Art. 6(1)(f)",                "6",  "1",   "f"),
    ("article 83",                  "83", None,  None),
    ("Show me Article 6.1.f",       "6",  "1",   "f"),
])
def test_article_extraction(query, expected_article, expected_point, expected_subpoint):
    r = ana.analyze(query)
    assert r.article  == expected_article,  f"{query}: article"
    assert r.point    == expected_point,    f"{query}: point"
    assert r.subpoint == expected_subpoint, f"{query}: subpoint"
    assert r.intent   == Intent.EXACT


@pytest.mark.parametrize("query,expected_recital", [
    ("Recital 42",     "42"),
    ("recital(7)",     "7"),
    ("See recital 108","108"),
])
def test_recital_extraction(query, expected_recital):
    r = ana.analyze(query)
    assert r.recital == expected_recital
    assert r.intent  == Intent.EXACT


@pytest.mark.parametrize("query,expected_chapter", [
    ("Chapter III",  "3"),
    ("Chapter 5",    "5"),
    ("CHAPTER IV",   "4"),
])
def test_chapter_extraction(query, expected_chapter):
    r = ana.analyze(query)
    assert r.chapter == expected_chapter
    assert r.intent  == Intent.RANGE


@pytest.mark.parametrize("query", [
    "What are consent requirements?",
    "Explain the rights of data subjects",
    "How does GDPR handle data breaches?",
])
def test_semantic_intent(query):
    r = ana.analyze(query)
    assert r.intent == Intent.SEMANTIC
    assert r.confidence >= 0.5


def test_offtopic_low_confidence():
    r = ana.analyze("Hi there!")
    assert r.confidence < 0.5
