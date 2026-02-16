"""
Track LLM usage and costs
"""
import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from src.logger import get_logger

logger = get_logger("LLMTracker")


@dataclass
class LLMCall:
    """Single LLM call record"""
    timestamp: str
    model: str
    operation: str  # "query_analysis" or "answer_generation"
    tokens: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


class LLMTracker:
    """Minimal LLM usage tracker"""
    
    COST_PER_1K_TOKENS = 0.0003  # gpt-4o-mini average
    
    def __init__(self, log_file: str = "logs/llm_metrics.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.calls = []
    
    def track(
        self, 
        model: str, 
        operation: str, 
        start_time: float, 
        success: bool,
        tokens: int = 0,
        error: Optional[str] = None
    ):
        """Log LLM call"""
        call = LLMCall(
            timestamp=datetime.utcnow().isoformat(),
            model=model,
            operation=operation,
            tokens=tokens,
            latency_ms=(time.time() - start_time) * 1000,
            success=success,
            error=error
        )
        
        self.calls.append(call)
        
        # Log to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(call)) + '\n')
        
        # Log summary
        if success:
            cost = (tokens / 1000) * self.COST_PER_1K_TOKENS
            logger.info(f"{operation}: {call.latency_ms:.0f}ms, {tokens} tokens, ${cost:.6f}")
        else:
            logger.error(f"{operation} failed: {error}")
    
    def get_stats(self) -> dict:
        """Get basic stats"""
        if not self.calls:
            return {"total_calls": 0}
        
        successful = [c for c in self.calls if c.success]
        total_tokens = sum(c.tokens for c in successful)
        total_cost = (total_tokens / 1000) * self.COST_PER_1K_TOKENS
        
        return {
            "total_calls": len(self.calls),
            "successful": len(successful),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "avg_latency_ms": round(sum(c.latency_ms for c in successful) / len(successful)) if successful else 0
        }


# Global tracker
tracker = LLMTracker()