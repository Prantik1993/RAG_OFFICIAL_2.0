"""
Simple in-memory query cache
"""
import hashlib
from typing import Optional
from src.logger import get_logger

logger = get_logger("QueryCache")


class QueryCache:
    """Simple LRU cache for query responses"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = {}
    
    def _normalize(self, query: str) -> str:
        """Normalize query for cache key"""
        return ' '.join(query.lower().strip().split())
    
    def _key(self, query: str) -> str:
        """Generate cache key"""
        normalized = self._normalize(query)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[dict]:
        """Get cached response"""
        key = self._key(query)
        if key in self._cache:
            logger.info("Cache HIT")
            return self._cache[key]
        logger.debug("Cache MISS")
        return None
    
    def set(self, query: str, response: dict):
        """Cache response"""
        key = self._key(query)
        
        # Simple LRU: remove oldest if full
        if len(self._cache) >= self.max_size:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = response
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()
        logger.info("Cache cleared")


# Global cache
query_cache = QueryCache(max_size=1000)