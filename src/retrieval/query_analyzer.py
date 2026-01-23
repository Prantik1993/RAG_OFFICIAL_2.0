import re
from enum import Enum
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from src.logger import get_logger
from src.exceptions import QueryRoutingError

logger = get_logger("QueryAnalyzer")

class QueryType(Enum):
    EXACT_REFERENCE = "exact_reference"
    ARTICLE_LOOKUP = "article_lookup"
    RECITAL_LOOKUP = "recital_lookup"
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"
    GENERAL = "general"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    original_query: str
    article: Optional[str] = None
    recital: Optional[str] = None
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
            "subsection": self.subsection,
            "point": self.point,
            "confidence": self.confidence
        }

class QueryAnalyzer:
    """
    Analyzes user queries with ROBUST pattern matching.
    Uses Keyword Arguments to prevent field mismatch errors.
    """
    
    def __init__(self):
        self.patterns = {
            # 1. RECITAL (Matches "Recital 1", "Reg Point 1")
            "recital": re.compile(
                r"(?:recital|regulation)(?:[\s\-]*(?:point|clause|part|no|number|num|#|\.))*[\s\-]*\(*(\d+)\)*", 
                re.IGNORECASE
            ),
            
            # 2. FULL REFERENCE (Article 15.1.a)
            "full_reference_dot": re.compile(r"article\s+(\d+)\.(\d+)\.([a-z])", re.IGNORECASE),
            "full_reference_paren": re.compile(r"article\s+(\d+)\s*\((\d+)\)\s*\(([a-z])\)", re.IGNORECASE),
            
            # 3. SUBSECTION REFERENCE (Article 15.1)
            "article_subsection_dot": re.compile(r"article\s+(\d+)\.(\d+)", re.IGNORECASE),
            "article_subsection_paren": re.compile(r"article\s+(\d+)\s*\((\d+)\)", re.IGNORECASE),
            
            # 4. ARTICLE ONLY (Article 15)
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
            
            # 1. Check Recitals
            recital_match = self.patterns["recital"].search(query)
            if recital_match:
                return QueryAnalysis(
                    query_type=QueryType.RECITAL_LOOKUP,
                    original_query=query,
                    recital=recital_match.group(1), # Explicit kwarg
                    confidence=0.95
                )

            # 2. Check Exact Article References
            exact_result = self._check_exact_reference(query)
            if exact_result:
                return exact_result
            
            # 3. Check Comparison
            if any(keyword in query_lower for keyword in self.comparison_keywords):
                articles = self._extract_article_numbers(query)
                return QueryAnalysis(
                    query_type=QueryType.COMPARISON,
                    original_query=query,
                    article=articles[0] if articles else None,
                    confidence=0.8,
                    extracted_concepts=articles
                )
            
            # 4. Check Conceptual
            if any(keyword in query_lower for keyword in self.conceptual_keywords):
                articles = self._extract_article_numbers(query)
                return QueryAnalysis(
                    query_type=QueryType.CONCEPTUAL,
                    original_query=query,
                    article=articles[0] if articles else None,
                    confidence=0.7,
                    extracted_concepts=articles
                )
            
            # 5. Default Fallback
            return QueryAnalysis(
                query_type=QueryType.GENERAL,
                original_query=query,
                confidence=0.5
            )
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise QueryRoutingError(f"Failed to analyze query: {e}")
    
    def _check_exact_reference(self, query: str) -> Optional[QueryAnalysis]:
        """Helper to check all article regex patterns using KEYWORD ARGUMENTS"""
        
        # A. Full Reference (Art 1.2.a)
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
        
        # B. Full Reference (Art 1(2)(a))
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

        # C. Subsection (Art 1.2)
        match = self.patterns["article_subsection_dot"].search(query)
        if match:
            return QueryAnalysis(
                query_type=QueryType.EXACT_REFERENCE,
                original_query=query,
                article=match.group(1),
                subsection=match.group(2), # <--- This was going into 'recital' before!
                confidence=0.9
            )

        # D. Subsection (Art 1(2))
        match = self.patterns["article_subsection_paren"].search(query)
        if match:
            return QueryAnalysis(
                query_type=QueryType.EXACT_REFERENCE,
                original_query=query,
                article=match.group(1),
                subsection=match.group(2),
                confidence=0.9
            )

        # E. Article Only (Art 1)
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
        """Extract all article numbers from query for comparison/conceptual analysis"""
        pattern = re.compile(r"article\s+(\d+)", re.IGNORECASE)
        matches = pattern.findall(query)
        return list(set(matches))
    
    def should_use_metadata_filter(self, analysis: QueryAnalysis) -> bool:
        if analysis.query_type == QueryType.RECITAL_LOOKUP:
            return True
        return analysis.query_type in [QueryType.EXACT_REFERENCE, QueryType.ARTICLE_LOOKUP] and analysis.article is not None