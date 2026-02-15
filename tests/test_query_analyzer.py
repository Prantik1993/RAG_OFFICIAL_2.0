from src.retrieval.query_analyzer import QueryAnalyzer, QueryType


def test_chapter_section_with_roman_numerals():
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("What is chapter II section III first article?")
    assert result.query_type == QueryType.CHAPTER_SECTION_LOOKUP
    assert result.chapter == "2"
    assert result.section == "3"


def test_exact_article_point():
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("Can you give me Article 2(2)(a) text?")
    assert result.query_type == QueryType.EXACT_REFERENCE
    assert result.article == "2"
    assert result.subsection == "2"
    assert result.point == "a"


def test_typo_in_keywords():
    analyzer = QueryAnalyzer()
    result = analyzer.analyze("chapter v seciton 5 start from whcih articl")
    assert result.query_type in {QueryType.CHAPTER_LOOKUP, QueryType.CHAPTER_SECTION_LOOKUP, QueryType.SECTION_LOOKUP}
