"""
Query Analyzer  (regex-only, zero LLM cost)
===========================================
Parses GDPR structural references from free-text queries using compiled
regular expressions.  Falls back to SEMANTIC intent when no reference found.

Handles all common patterns:
  Article 15          -> article=15
  Article 15.1        -> article=15, point=1
  Article 15.1.a      -> article=15, point=1, subpoint=a
  Art. 15(1)(a)       -> article=15, point=1, subpoint=a
  Recital 42          -> recital=42
  Chapter III         -> chapter=3
  Chapter 3 Section 2 -> chapter=3, section=2
  Article 6(1)(f)     -> article=6, point=1, subpoint=f
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.logger import get_logger

log = get_logger("QueryAnalyzer")

# ── Intent ────────────────────────────────────────────────────────────────────
class Intent(str, Enum):
    EXACT    = "exact"      # specific article / recital / point reference
    RANGE    = "range"      # all articles in chapter/section
    SEMANTIC = "semantic"   # conceptual / open question


# ── Compiled patterns ─────────────────────────────────────────────────────────
_ROMAN = {
    "I":"1","II":"2","III":"3","IV":"4","V":"5",
    "VI":"6","VII":"7","VIII":"8","IX":"9","X":"10",
    "XI":"11","XII":"12",
}

# Article 15  |  Art. 15  |  article15
_RE_ARTICLE = re.compile(
    r'\bart(?:icle)?\.?\s*(\d{1,3})'
    r'(?:[.\s(](\d{1,2}))?'          # optional .1 or (1)
    r'(?:[.\s(]([a-z]))?',           # optional .a or (a)
    re.I,
)
# Recital 42  |  recital(42)
_RE_RECITAL = re.compile(r'\brecital\s*\(?(\d{1,3})\)?', re.I)

# Chapter III  |  Chapter 3
_RE_CHAPTER = re.compile(r'\bchapter\s+([IVX]+|\d+)', re.I)

# Section 2
_RE_SECTION = re.compile(r'\bsection\s+(\d+)', re.I)

# "in Chapter X Section Y" range trigger
_RE_RANGE = re.compile(r'\b(chapter|section)\b', re.I)

# Low-confidence signals (greetings / off-topic)
_RE_OFFTOPIC = re.compile(
    r'^(hi|hello|hey|thanks|thank you|bye|weather|joke|who are you)\b',
    re.I,
)


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class QueryAnalysis:
    intent:   Intent
    query:    str

    recital:  Optional[str] = None
    chapter:  Optional[str] = None
    section:  Optional[str] = None
    article:  Optional[str] = None
    point:    Optional[str] = None
    subpoint: Optional[str] = None

    confidence: float = 1.0

    # ── helpers ───────────────────────────────────────────────────────────────
    @property
    def has_ref(self) -> bool:
        return any([self.recital, self.article, self.point, self.subpoint])

    def filter_dict(self) -> dict:
        """Metadata filter for FAISS search."""
        f: dict = {}
        for k in ("recital","chapter","section","article","point","subpoint"):
            v = getattr(self, k)
            if v is not None:
                f[k] = v
        return f


# ── Analyzer ──────────────────────────────────────────────────────────────────
class QueryAnalyzer:
    """
    Stateless, deterministic query classifier.
    Single public method: analyze(query) -> QueryAnalysis
    """

    def analyze(self, query: str) -> QueryAnalysis:
        q = query.strip()

        # Off-topic check
        if _RE_OFFTOPIC.match(q):
            return QueryAnalysis(intent=Intent.SEMANTIC, query=q, confidence=0.1)

        # Try to extract structural reference
        recital = article = point = subpoint = chapter = section = None

        m_rec = _RE_RECITAL.search(q)
        if m_rec:
            recital = m_rec.group(1)

        m_art = _RE_ARTICLE.search(q)
        if m_art:
            article  = m_art.group(1)
            point    = m_art.group(2)   # may be None
            subpoint = m_art.group(3)   # may be None

        m_ch = _RE_CHAPTER.search(q)
        if m_ch:
            raw = m_ch.group(1).upper()
            chapter = _ROMAN.get(raw, raw)

        m_sec = _RE_SECTION.search(q)
        if m_sec:
            section = m_sec.group(1)

        # Determine intent
        if recital or article:
            intent = Intent.EXACT
            confidence = 0.95
        elif chapter or section:
            intent = Intent.RANGE
            confidence = 0.90
        else:
            intent = Intent.SEMANTIC
            confidence = 0.80

        result = QueryAnalysis(
            intent=intent, query=q,
            recital=recital, chapter=chapter, section=section,
            article=article, point=point, subpoint=subpoint,
            confidence=confidence,
        )
        log.info(
            f"Analyzed | intent={intent.value} conf={confidence:.2f} "
            f"ref={result.filter_dict()}"
        )
        return result
