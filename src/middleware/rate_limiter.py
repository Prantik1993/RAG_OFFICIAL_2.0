"""
Simple token bucket rate limiter
"""
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Tuple
from src.logger import get_logger

logger = get_logger("RateLimiter")


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_minute: int = 10, requests_per_hour: int = 100):
        self.rpm = requests_per_minute
        self.rph = requests_per_hour
        self.buckets = defaultdict(lambda: {
            'minute': {'count': 0, 'reset': datetime.now()},
            'hour': {'count': 0, 'reset': datetime.now()}
        })
    
    def check_limit(self, identifier: str) -> Tuple[bool, str]:
        """
        Check rate limit for identifier (session_id or IP)
        
        Returns: (allowed, message)
        """
        now = datetime.now()
        bucket = self.buckets[identifier]
        
        # Reset minute counter if expired
        if now > bucket['minute']['reset']:
            bucket['minute'] = {'count': 0, 'reset': now + timedelta(minutes=1)}
        
        # Check minute limit
        if bucket['minute']['count'] >= self.rpm:
            return False, f"Rate limit: {self.rpm}/minute. Try again in 60s"
        
        # Reset hour counter if expired
        if now > bucket['hour']['reset']:
            bucket['hour'] = {'count': 0, 'reset': now + timedelta(hours=1)}
        
        # Check hour limit
        if bucket['hour']['count'] >= self.rph:
            return False, f"Rate limit: {self.rph}/hour. Try again later"
        
        # Increment both counters
        bucket['minute']['count'] += 1
        bucket['hour']['count'] += 1
        
        return True, "OK"