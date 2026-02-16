from typing import Optional
from dataclasses import dataclass
from enum import Enum
import json
import time

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.config import Config
from src.logger import get_logger
from src.monitoring.llm_tracker import tracker

logger = get_logger("LLMQueryAnalyzer")


class QueryIntent(str, Enum):
    """Types of queries the system can handle"""
    EXACT_LOOKUP = "exact_lookup"  # "What is Article 15.1.a"
    RANGE_QUERY = "range_query"    # "Show articles in Chapter 2 Section 1"
    CONCEPTUAL = "conceptual"       # "What are consent requirements"
    COMPARISON = "comparison"       # "Compare Article 6 and 7"
    GENERAL = "general"             # General questions about the document


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    intent: QueryIntent
    original_query: str
    
    # Extracted references
    recital: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    article: Optional[str] = None
    point: Optional[str] = None
    subpoint: Optional[str] = None
    
    # For range queries
    start_article: Optional[str] = None
    end_article: Optional[str] = None
    
    # Confidence
    confidence: float = 0.0
    
    # Explanation from LLM
    reasoning: str = ""
    
    def has_exact_reference(self) -> bool:
        """Check if query has specific reference"""
        return any([
            self.recital,
            self.article,
            self.point,
            self.subpoint
        ])
    
    def to_filter_dict(self) -> dict:
        """Convert to metadata filter for vector store"""
        filter_dict = {}
        
        if self.recital:
            filter_dict["recital"] = self.recital
        if self.chapter:
            filter_dict["chapter"] = self.chapter
        if self.section:
            filter_dict["section"] = self.section
        if self.article:
            filter_dict["article"] = self.article
        if self.point:
            filter_dict["point"] = self.point
        if self.subpoint:
            filter_dict["subpoint"] = self.subpoint
        
        return filter_dict


class LLMQueryAnalyzer:
    """
    Uses GPT to understand user queries with high accuracy.
    No brittle regex patterns - handles typos, natural language, etc.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=Config.OPENAI_API_KEY
        )
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query analyzer for a GDPR legal assistant.

    IMPORTANT: Analyze queries in context. Follow-up questions are VALID.

    If the query:
    - Explicitly references GDPR structure (recital, article, chapter, section, point, GDPR), OR
    - Contains GDPR-related terms (personal data, data subject, controller, processor, consent, processing, criminal convictions, etc.), OR
    - Quotes GDPR text directly, OR
    - Is a follow-up question referring to previous GDPR content (using words like "it", "that", "above", "this article", etc.), OR
    - Asks to summarize/describe/explain something mentioned earlier
    → Treat as VALID legal query
    → Set confidence HIGH (0.7–1.0)

    If the query is casual chat, jokes, greetings, or clearly unrelated to data protection:
    → Set confidence LOW (0.0–0.2)

    The document has this structure:
    - Recitals: (1), (2), ..., (108)
    - Chapters: CHAPTER I, II, III, IV, ... (Roman numerals)
    - Sections: Section 1, 2, 3, ... (within chapters)
    - Articles: Article 1, 2, 3, ... (within sections or chapters)
    - Points: 1., 2., 3., ... (within articles)
    - Subpoints: (a), (b), (c), ... (within points)

    Examples of VALID queries (HIGH confidence 0.7-1.0):
    - "What is Article 15.1.a?" 
    - "What does 'personal data' mean?"
    - "'personal data' means any information..." (quoted GDPR text)
    - "which article is the above query" (follow-up)
    - "can you describe within 20 words above article" (follow-up)
    - "describe it in short" (follow-up)
    - "what does that mean" (follow-up)
    - "tell me more about it" (follow-up)

    Examples of INVALID queries (LOW confidence 0.0-0.2):
    - "What's the weather today?"
    - "Tell me a joke"
    - "Hello, how are you?"
    - "Who won the Super Bowl?"

    CRITICAL: The word "article" in a follow-up context ("above article", "that article") indicates a GDPR question. Set HIGH confidence.

    IMPORTANT RULES:
    1. Convert Roman numerals (I, II, III, IV, V) to Arabic (1, 2, 3, 4, 5)
    2. Handle typos and variations
    3. For "Article X.Y.Z", extract: article=X, point=Y, subpoint=Z
    4. Follow-up questions that reference "above", "that", "it", "this" are VALID if discussing GDPR content

    Respond ONLY with valid JSON in this format:
    {{
        "intent": "exact_lookup|range_query|conceptual|comparison|general",
        "recital": "number or null",
        "chapter": "number or null",
        "section": "number or null",
        "article": "number or null",
        "point": "number or null",
        "subpoint": "letter or null",
        "start_article": "number or null (for range queries)",
        "confidence": 0.0-1.0,
        "reasoning": "brief explanation of your analysis"
    }}"""),
        ("human", "{query}")
    ])

    
    def analyze(self, query: str) -> QueryAnalysis:
        """Analyze user query using LLM with monitoring"""
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing query: '{query}'")
            
            # Call LLM
            chain = self.analysis_prompt | self.llm
            response = chain.invoke({"query": query})
            
            # Track the call
            usage = response.response_metadata.get('token_usage', {})
            tracker.track(
                model=self.llm.model_name,
                operation="query_analysis",
                start_time=start_time,
                success=True,
                tokens=usage.get('total_tokens', 0)
            )
            
            # Parse JSON response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            
            result = json.loads(content)
            
            # Create QueryAnalysis object
            analysis = QueryAnalysis(
                intent=QueryIntent(result["intent"]),
                original_query=query,
                recital=result.get("recital"),
                chapter=self._normalize_chapter(result.get("chapter")),
                section=result.get("section"),
                article=result.get("article"),
                point=result.get("point"),
                subpoint=result.get("subpoint"),
                start_article=result.get("start_article"),
                confidence=result.get("confidence", 0.8),
                reasoning=result.get("reasoning", "")
            )
            
            logger.info(
                f"Query analysis complete: intent={analysis.intent.value}, "
                f"confidence={analysis.confidence:.2f}"
            )
            
            return analysis
        
        except Exception as e:
            logger.error(f"Query analysis failed: {e}", exc_info=True)
            
            # Track failed call
            tracker.track(
                model=self.llm.model_name,
                operation="query_analysis",
                start_time=start_time,
                success=False,
                error=str(e)
            )
            
            # Fallback to general query
            return QueryAnalysis(
                intent=QueryIntent.GENERAL,
                original_query=query,
                confidence=0.3,
                reasoning=f"Analysis failed: {str(e)}"
            )
    
    def _normalize_chapter(self, chapter: Optional[str]) -> Optional[str]:
        """Convert Roman numerals to Arabic numbers"""
        if not chapter:
            return None
        
        roman_map = {
            'I': '1', 'II': '2', 'III': '3', 'IV': '4', 'V': '5',
            'VI': '6', 'VII': '7', 'VIII': '8', 'IX': '9', 'X': '10',
            'XI': '11', 'XII': '12'
        }
        
        chapter_upper = str(chapter).upper().strip()
        return roman_map.get(chapter_upper, chapter)