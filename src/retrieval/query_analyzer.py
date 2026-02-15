import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from src.exceptions import QueryRoutingError
from src.logger import get_logger

logger = get_logger("QueryAnalyzer")


class QueryType(Enum):
    EXACT_REFERENCE = "exact_reference"
    ARTICLE_LOOKUP = "article_lookup"
    RECITAL_LOOKUP = "recital_lookup"
    SECTION_LOOKUP = "section_lookup"
    CHAPTER_LOOKUP = "chapter_lookup"
    CHAPTER_SECTION_LOOKUP = "chapter_section_lookup"
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"
    GENERAL = "general"


@dataclass
class QueryAnalysis:
    query_type: QueryType
    original_query: str
    article: Optional[str] = None
    recital: Optional[str] = None
    section: Optional[str] = None
    chapter: Optional[str] = None
    subsection: Optional[str] = None
    point: Optional[str] = None
    confidence: float = 0.0
    extracted_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "article": self.article,
            "recital": self.recital,
            "section": self.section,
            "chapter": self.chapter,
            "subsection": self.subsection,
            "point": self.point,
            "confidence": self.confidence,
        }


class QueryAnalyzer:
    """Parser-style query analyzer with typo-tolerant legal reference extraction."""

    ROMAN_TO_ARABIC = {
        "I": "1",
        "II": "2",
        "III": "3",
        "IV": "4",
        "V": "5",
        "VI": "6",
        "VII": "7",
        "VIII": "8",
        "IX": "9",
        "X": "10",
        "XI": "11",
        "XII": "12",
    }

    def __init__(self):
        self.keywords = {
            "comparison": ["difference", "compare", "versus", "vs", "between"],
            "conceptual": ["what", "explain", "define", "why", "how", "requirements", "rules"],
        }

        self.patterns = {
            "article_point": re.compile(
                r"(?:art(?:icle)?|articl|aritcle)\s*(\d+)\s*(?:[.(]\s*(\d+)\s*[).]?)\s*(?:[.(]\s*([a-z])\s*[).]?)",
                re.IGNORECASE,
            ),
            "article_subsection": re.compile(
                r"(?:art(?:icle)?|articl|aritcle)\s*(\d+)\s*(?:[.(]\s*(\d+)\s*[).]?)",
                re.IGNORECASE,
            ),
            "article_only": re.compile(r"(?:art(?:icle)?|articl|aritcle)\s*(\d+)", re.IGNORECASE),
            "chapter": re.compile(r"(?:chap(?:ter)?|chp|chptr)\s*([ivxlcdm]+|\d+)", re.IGNORECASE),
            "section": re.compile(r"(?:sec(?:tion)?|sction|secton|seciton)\s*([ivxlcdm]+|\d+)", re.IGNORECASE),
            "recital": re.compile(r"(?:recital|whereas)\s*\(?\s*(\d+)\s*\)?", re.IGNORECASE),
            "point_shorthand": re.compile(r"\b(\d+)\s*\.\s*([a-z])\b", re.IGNORECASE),
        }

    def analyze(self, query: str) -> QueryAnalysis:
        try:
            q = " ".join(query.strip().split())
            q_lower = q.lower()

            if recital := self._match(self.patterns["recital"], q):
                return QueryAnalysis(QueryType.RECITAL_LOOKUP, q, recital=recital[0], confidence=0.95)

            chapter = self._normalized_ref(self._match(self.patterns["chapter"], q))
            section = self._normalized_ref(self._match(self.patterns["section"], q))

            if chapter and section:
                return QueryAnalysis(
                    QueryType.CHAPTER_SECTION_LOOKUP,
                    q,
                    chapter=chapter,
                    section=section,
                    confidence=0.95,
                )

            if point_match := self._match(self.patterns["article_point"], q):
                return QueryAnalysis(
                    QueryType.EXACT_REFERENCE,
                    q,
                    article=point_match[0],
                    subsection=point_match[1],
                    point=point_match[2].lower(),
                    confidence=0.95,
                )

            if sub_match := self._match(self.patterns["article_subsection"], q):
                return QueryAnalysis(
                    QueryType.EXACT_REFERENCE,
                    q,
                    article=sub_match[0],
                    subsection=sub_match[1],
                    confidence=0.92,
                )

            # Handles follow-up queries like "give 2.a portion" after history-aware rewrite.
            if shorthand := self._match(self.patterns["point_shorthand"], q):
                return QueryAnalysis(
                    QueryType.EXACT_REFERENCE,
                    q,
                    subsection=shorthand[0],
                    point=shorthand[1].lower(),
                    confidence=0.75,
                )

            if article := self._match(self.patterns["article_only"], q):
                query_type = QueryType.ARTICLE_LOOKUP if any(
                    token in q_lower for token in ["show", "start", "from", "text", "give", "which"]
                ) else QueryType.CONCEPTUAL
                return QueryAnalysis(query_type, q, article=article[0], confidence=0.85)

            if chapter:
                return QueryAnalysis(QueryType.CHAPTER_LOOKUP, q, chapter=chapter, confidence=0.9)

            if section:
                return QueryAnalysis(QueryType.SECTION_LOOKUP, q, section=section, confidence=0.9)

            if any(keyword in q_lower for keyword in self.keywords["comparison"]):
                return QueryAnalysis(QueryType.COMPARISON, q, confidence=0.75)

            if any(keyword in q_lower for keyword in self.keywords["conceptual"]):
                return QueryAnalysis(QueryType.CONCEPTUAL, q, confidence=0.65)

            return QueryAnalysis(QueryType.GENERAL, q, confidence=0.5)

        except Exception as exc:
            logger.error(f"Query analysis failed: {exc}")
            raise QueryRoutingError(f"Failed to analyze query: {exc}") from exc

    def _match(self, pattern: re.Pattern, text: str) -> Optional[tuple[str, ...]]:
        match = pattern.search(text)
        return match.groups() if match else None

    def _normalized_ref(self, match_result: Optional[tuple[str, ...]]) -> Optional[str]:
        if not match_result:
            return None
        raw = match_result[0].upper()
        return self.ROMAN_TO_ARABIC.get(raw, raw)
