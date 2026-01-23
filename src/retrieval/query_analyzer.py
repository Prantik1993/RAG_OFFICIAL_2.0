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
    RECITAL_LOOKUP = "recital_lookup"  # <--- NEW
    CONCEPTUAL = "conceptual"
    COMPARISON = "comparison"
    GENERAL = "general"

@dataclass
class QueryAnalysis:
    query_type: QueryType
    original_query: str
    article: Optional[str] = None
    recital: Optional[str] = None      # <--- NEW
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
            "recital": self.recital,   # <--- NEW
            "subsection": self.subsection,
            "point": self.point,
            "confidence": self.confidence
        }

class QueryAnalyzer:
    """
    Analyzes user queries with FLEXIBLE pattern matching.
    Handles Articles and Recitals.
    """
    
    def __init__(self):
        self.patterns = {
            "recital": re.compile(r"recital\s+(?:\()?(\d+)(?:\))?", re.IGNORECASE), # <--- NEW
            
            # Articles
            "full_reference_dot": re.compile(r"article\s+(\d+)\.(\d+)\.([a-z])", re.IGNORECASE),
            "full_reference_paren": re.compile(r"article\s+(\d+)\s*\((\d+)\)\s*\(([a-z])\)", re.IGNORECASE),
            "article_subsection_dot": re.compile(r"article\s+(\d+)\.(\d+)", re.IGNORECASE),
            "article_subsection_paren": re.compile(r"article\s+(\d+)\s*\((\d+)\)", re.IGNORECASE),
            "article_subsection_part": re.compile(r"article\s+(\d+)\s+(?:part|point|section|subsection|para|paragraph)\s+(\d+)", re.IGNORECASE),
            "article_subsection_underscore": re.compile(r"article\s+(\d+)[_-](\d+)", re.IGNORECASE),
            "article_only": re.compile(r"article\s+(\d+)(?![.\d(])", re.IGNORECASE),
        }
        
        self.conceptual_keywords = [
            "what is", "what are", "explain", "describe", "how does",
            "why", "when", "requirements", "rules", "provisions",
            "obligations", "rights", "principles", "definition", "tell me"
        ]
        
        self.comparison_keywords = [
            "difference", "compare", "versus", "vs", "distinction",
            "similar", "different", "both", "either"
        ]
    
    def analyze(self, query: str) -> QueryAnalysis:
        try:
            query_lower = query.lower().strip()
            
            # 1. Check Recitals (Priority)
            recital_match = self.patterns["recital"].search(query)
            if recital_match:
                return QueryAnalysis(
                    query_type=QueryType.RECITAL_LOOKUP,
                    original_query=query,
                    recital=recital_match.group(1),
                    confidence=0.95
                )

            # 2. Check Exact References
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
            
            # 5. Default
            return QueryAnalysis(
                query_type=QueryType.GENERAL,
                original_query=query,
                confidence=0.5
            )
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise QueryRoutingError(f"Failed to analyze query: {e}")
    
    # ... (The rest of the methods _check_exact_reference etc. remain the same)
    def _check_exact_reference(self, query: str) -> Optional[QueryAnalysis]:
        # (Copy your existing code for _check_exact_reference here)
        # ... (Same as your file)
        match = self.patterns["full_reference_dot"].search(query)
        if match:
             return QueryAnalysis(query_type=QueryType.EXACT_REFERENCE, original_query=query, article=match.group(1), subsection=match.group(2), point=match.group(3), confidence=0.95)
        
        match = self.patterns["full_reference_paren"].search(query)
        if match:
             return QueryAnalysis(query_type=QueryType.EXACT_REFERENCE, original_query=query, article=match.group(1), subsection=match.group(2), point=match.group(3), confidence=0.95)

        match = self.patterns["article_subsection_dot"].search(query)
        if match:
            return QueryAnalysis(query_type=QueryType.EXACT_REFERENCE, original_query=query, article=match.group(1), subsection=match.group(2), confidence=0.9)

        match = self.patterns["article_subsection_paren"].search(query)
        if match:
            return QueryAnalysis(query_type=QueryType.EXACT_REFERENCE, original_query=query, article=match.group(1), subsection=match.group(2), confidence=0.9)

        match = self.patterns["article_subsection_part"].search(query)
        if match:
            return QueryAnalysis(query_type=QueryType.EXACT_REFERENCE, original_query=query, article=match.group(1), subsection=match.group(2), confidence=0.85)

        match = self.patterns["article_subsection_underscore"].search(query)
        if match:
            return QueryAnalysis(query_type=QueryType.EXACT_REFERENCE, original_query=query, article=match.group(1), subsection=match.group(2), confidence=0.85)

        match = self.patterns["article_only"].search(query)
        if match:
            query_lower = query.lower()
            lookup_phrases = ["show", "display", "get", "find", "retrieve", "read"]
            query_type = QueryType.ARTICLE_LOOKUP if any(p in query_lower for p in lookup_phrases) else QueryType.CONCEPTUAL
            return QueryAnalysis(query_type=query_type, original_query=query, article=match.group(1), confidence=0.85 if query_type == QueryType.ARTICLE_LOOKUP else 0.7)

        return None

    def _extract_article_numbers(self, query: str) -> list:
        pattern = re.compile(r"article\s+(\d+)", re.IGNORECASE)
        matches = pattern.findall(query)
        return list(set(matches))