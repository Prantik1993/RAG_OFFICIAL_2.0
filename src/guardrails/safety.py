"""
Content safety guardrails with smart follow-up question detection
"""
import re
from typing import Tuple
from src.logger import get_logger

logger = get_logger("SafetyGuardrails")


class SafetyGuardrails:
    """Minimal yet production-grade safety checks"""
    
    # Only block clear attack patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all|previous|prior)\s+instructions",
        r"disregard.*instructions",
        r"new\s+role\s*:",
        r"system\s+prompt",
        r"<\|im_",
        r"</s>",
    ]
    
    # Follow-up question indicators (these are SAFE)
    FOLLOWUP_INDICATORS = [
        r"\b(it|this|that|these|those)\b",
        r"\b(describe|explain|summarize|tell me more)\b",
        r"\b(in \d+ words|briefly|in short)\b",
        r"\b(what does|what is|how does)\b.*\b(mean|work)\b",
        r"\bmore details?\b",
        r"\bcan you\b",
    ]
    
    def validate_input(self, query: str) -> Tuple[bool, str]:
        """
        Validate user input - minimal checks only
        
        Returns: (is_safe, reason)
        """
        if not query or not query.strip():
            return False, "Empty query"
        
        if len(query) > 2000:
            return False, "Query too long"
        
        query_lower = query.strip().lower()
        
        # Check if it's a follow-up question (ALLOW these)
        for pattern in self.FOLLOWUP_INDICATORS:
            if re.search(pattern, query_lower):
                logger.debug("Follow-up question detected - allowed")
                return True, "Safe"
        
        # Only block obvious injection attempts
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, query_lower):
                logger.warning(f"Injection pattern detected: {pattern}")
                return False, "Invalid request"
        
        # Check for excessive special characters (>50%)
        special_count = sum(not c.isalnum() and not c.isspace() for c in query)
        if len(query) > 0 and special_count / len(query) > 0.5:
            return False, "Invalid characters"
        
        return True, "Safe"
    
    def validate_output(self, response: str) -> bool:
        """Check if LLM leaked system info"""
        if len(response.strip()) < 10:
            return False
        
        response_lower = response.lower()
        
        # Only block if system prompt is leaked
        leak_patterns = ["you are an expert legal assistant", "langchain"]
        for pattern in leak_patterns:
            if pattern in response_lower:
                logger.warning("System leak detected in output")
                return False
        
        return True