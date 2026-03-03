"""
Rate Limiter  (token-bucket, per session_id)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta

import src.config as cfg
from src.logger import get_logger

log = get_logger("RateLimiter")


class RateLimiter:

    def __init__(
        self,
        rpm: int = cfg.RATE_LIMIT_RPM,
        rph: int = cfg.RATE_LIMIT_RPH,
    ) -> None:
        self._rpm = rpm
        self._rph = rph
        self._buckets: dict = defaultdict(lambda: {
            "m": {"n": 0, "reset": datetime.utcnow()},
            "h": {"n": 0, "reset": datetime.utcnow()},
        })

    def check(self, session_id: str) -> tuple[bool, str]:
        now    = datetime.utcnow()
        bucket = self._buckets[session_id]

        # reset counters if windows expired
        if now >= bucket["m"]["reset"]:
            bucket["m"] = {"n": 0, "reset": now + timedelta(minutes=1)}
        if now >= bucket["h"]["reset"]:
            bucket["h"] = {"n": 0, "reset": now + timedelta(hours=1)}

        if bucket["m"]["n"] >= self._rpm:
            return False, f"Rate limit: {self._rpm} req/min. Retry in 60 s."
        if bucket["h"]["n"] >= self._rph:
            return False, f"Rate limit: {self._rph} req/hour. Retry later."

        bucket["m"]["n"] += 1
        bucket["h"]["n"] += 1
        return True, "ok"
