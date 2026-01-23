import re
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass
from src.logger import get_logger
from src.exceptions import QueryRoutingError

logger = get_logger("QueryAnalyzer")

class QueryType(Enum):
    EXACT_REFERENCE = "exact_reference"
    ARTICLE_LOOKUP = "article_lookup"
    RECITAL_LOOKUP = "recital_lookup"
    SECTION_LOOKUP = "section_lookup"
    CHAPTER_LOOKUP = "chapter_lookup"
    CHAPTER_SECTION_LOOKUP = "chapter_section_lookup"  # NEW
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
    extracted_concepts: list = None
    
    def __post_init__(self):
        if self.extracted_concepts is None:
            self.extracted_concepts = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "article": self.article,
            "recital": self.recital,
            "section": self.section,
            "chapter": self.chapter,
            "subsection": self.subsection,
            "point": self.point,
            "confidence": self.confidence
        }

class QueryAnalyzer:
    """
    Enhanced analyzer with Chapter+Section combined detection
    """
    
    def __init__(self):
        self.patterns = {
            # NEW: Combined Chapter + Section pattern
            # Matches: "Chapter 4 Section 3", "Chapter IV Section 2", "Chap 4 Sec 3"
            "chapter_section": re.compile(
                r"(?:chapter|chap)[\s\-]*((?:[IVX]+|\d+))[\s,]+(?:section|sec)[\s\-]*(\d+)",
                re.IGNORECASE
            ),
            
            # Section patterns
            "section": re.compile(
                r"(?:section|sec)[\s\-]*(\d+)", 
                re.IGNORECASE
            ),
            
            # Chapter patterns
            "chapter": re.compile(
                r"(?:chapter|chap)[\s\-]*((?:[IVX]+|\d+))", 
                re.IGNORECASE
            ),
            
            # Recital patterns
            "recital": re.compile(
                r"(?:recital|regulation)(?:[\s\-]*(?:point|clause|part|no|number|num|#|\.))*[\s\-]*\(*(\d+)\)*", 
                re.IGNORECASE
            ),
            
            # Article patterns
            "full_reference_dot": re.compile(r"article\s+(\d+)\.(\d+)\.([a-z])", re.IGNORECASE),
            "full_reference_paren": re.compile(r"article\s+(\d+)\s*\((\d+)\)\s*\(([a-z])\)", re.IGNORECASE),
            "article_subsection_dot": re.compile(r"article\s+(\d+)\.(\d+)", re.IGNORECASE),
            "article_subsection_paren": re.compile(r"article\s+(\d+)\s*\((\d+)\)", re.IGNORECASE),
            "article_only": re.compile(r"article\s+(\d+)(?![.\d(])", re.IGNORECASE),
        }
        
        self.conceptual_keywords = [
            "what is", "what are", "explain", "describe", "how does", "why", "when", 
            "requirements", "rules", "provisions", "obligations", "rights", "principles", 
            "definition", "tell me", "can i", "do i need"
        ]
        
        self.comparison_keywords = [
            "difference", "compare", "versus", "vs", "distinction", "similar", 
            "different", "both", "either", "between"
        ]
    
    def analyze(self, query: str) -> QueryAnalysis:
        try:
            query_lower = query.lower().strip()
            
            # 1. NEW: Check Combined Chapter + Section (HIGHEST PRIORITY)
            combined_match = self.patterns["chapter_section"].search(query)
            if combined_match:
                chapter_id = self._normalize_chapter_id(combined_match.group(1))
                section_id = combined_match.group(2)
                
                logger.info(f"Detected combined query: Chapter {chapter_id}, Section {section_id}")
                
                return QueryAnalysis(
                    query_type=QueryType.CHAPTER_SECTION_LOOKUP,
                    original_query=query,
                    chapter=chapter_id,
                    section=section_id,
                    confidence=0.95
                )
            
            # 2. Check Section (standalone)
            section_match = self.patterns["section"].search(query)
            if section_match and self._is_section_query(query_lower):
                return QueryAnalysis(
                    query_type=QueryType.SECTION_LOOKUP,
                    original_query=query,
                    section=section_match.group(1),
                    confidence=0.95
                )
            
            # 3. Check Chapter (standalone)
            chapter_match = self.patterns["chapter"].search(query)
            if chapter_match and self._is_chapter_query(query_lower):
                chapter_id = self._normalize_chapter_id(chapter_match.group(1))
                return QueryAnalysis(
                    query_type=QueryType.CHAPTER_LOOKUP,
                    original_query=query,
                    chapter=chapter_id,
                    confidence=0.95
                )
            
            # 4. Check Recitals
            recital_match = self.patterns["recital"].search(query)
            if recital_match:
                return QueryAnalysis(
                    query_type=QueryType.RECITAL_LOOKUP,
                    original_query=query,
                    recital=recital_match.group(1),
                    confidence=0.95
                )

            # 5. Check Exact Article References
            exact_result = self._check_exact_reference(query)
            if exact_result:
                return exact_result
            
            # 6. Check Comparison
            if any(keyword in query_lower for keyword in self.comparison_keywords):
                articles = self._extract_article_numbers(query)
                return QueryAnalysis(
                    query_type=QueryType.COMPARISON,
                    original_query=query,
                    article=articles[0] if articles else None,
                    confidence=0.8,
                    extracted_concepts=articles
                )
            
            # 7. Check Conceptual
            if any(keyword in query_lower for keyword in self.conceptual_keywords):
                articles = self._extract_article_numbers(query)
                return QueryAnalysis(
                    query_type=QueryType.CONCEPTUAL,
                    original_query=query,
                    article=articles[0] if articles else None,
                    confidence=0.7,
                    extracted_concepts=articles
                )
            
            # 8. Default Fallback
            return QueryAnalysis(
                query_type=QueryType.GENERAL,
                original_query=query,
                confidence=0.5
            )
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise QueryRoutingError(f"Failed to analyze query: {e}")
    
    def _is_section_query(self, query_lower: str) -> bool:
        """Check if this is truly asking for a section"""
        section_indicators = [
            "section start", "start from", "which article", 
            "show section", "what is section", "display section",
            "section contain", "in section", "section has"
        ]
        return any(indicator in query_lower for indicator in section_indicators)
    
    def _is_chapter_query(self, query_lower: str) -> bool:
        """Check if this is truly asking for a chapter"""
        chapter_indicators = [
            "chapter start", "start from", "which article",
            "show chapter", "what is chapter", "display chapter",
            "chapter contain", "in chapter", "chapter has"
        ]
        return any(indicator in query_lower for indicator in chapter_indicators)
    
    def _normalize_chapter_id(self, chapter_str: str) -> str:
        """Convert Roman numerals to Arabic for consistent lookup"""
        roman_to_arabic = {
            'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
            'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
            'XI': '11'
        }
        return roman_to_arabic.get(chapter_str.upper(), chapter_str)
    
    def _check_exact_reference(self, query: str) -> Optional[QueryAnalysis]:
        """Check all article regex patterns"""
        
        # Full Reference (Art 1.2.a)
        match = self.patterns["full_reference_dot"].search(query)
        if match:
             return QueryAnalysis(
                 query_type=QueryType.EXACT_REFERENCE,
                 original_query=query,
                 article=match.group(1),
                 subsection=match.group(2),
                 point=match.group(3),
                 confidence=0.95
             )
        
        # Full Reference (Art 1(2)(a))
        match = self.patterns["full_reference_paren"].search(query)
        if match:
             return QueryAnalysis(
                 query_type=QueryType.EXACT_REFERENCE,
                 original_query=query,
                 article=match.group(1),
                 subsection=match.group(2),
                 point=match.group(3),
                 confidence=0.95
             )

        # Subsection (Art 1.2)
        match = self.patterns["article_subsection_dot"].search(query)
        if match:
            return QueryAnalysis(
                query_type=QueryType.EXACT_REFERENCE,
                original_query=query,
                article=match.group(1),
                subsection=match.group(2),
                confidence=0.9
            )

        # Subsection (Art 1(2))
        match = self.patterns["article_subsection_paren"].search(query)
        if match:
            return QueryAnalysis(
                query_type=QueryType.EXACT_REFERENCE,
                original_query=query,
                article=match.group(1),
                subsection=match.group(2),
                confidence=0.9
            )

        # Article Only (Art 1)
        match = self.patterns["article_only"].search(query)
        if match:
            query_lower = query.lower()
            lookup_phrases = ["show", "display", "get", "find", "retrieve", "read"]
            q_type = QueryType.ARTICLE_LOOKUP if any(p in query_lower for p in lookup_phrases) else QueryType.CONCEPTUAL
            
            return QueryAnalysis(
                query_type=q_type, 
                original_query=query, 
                article=match.group(1), 
                confidence=0.85 if q_type == QueryType.ARTICLE_LOOKUP else 0.7
            )

        return None

    def _extract_article_numbers(self, query: str) -> list:
        """Extract all article numbers"""
        pattern = re.compile(r"article\s+(\d+)", re.IGNORECASE)
        matches = pattern.findall(query)
        return list(set(matches))