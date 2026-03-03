"""
Query Cache
===========
Simple in-memory LRU cache keyed on normalised query text.
Thread-safe for the single-process FastAPI use-case.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Optional

import src.config as cfg
from src.logger import get_logger

log = get_logger("QueryCache")


class QueryCache:

    def __init__(self, max_size: int = cfg.CACHE_MAX_SIZE) -> None:
        self._max  = max_size
        self._data: OrderedDict[str, dict] = OrderedDict()

    # ── public ────────────────────────────────────────────────────────────────
    def get(self, query: str) -> Optional[dict]:
        key = self._key(query)
        if key in self._data:
            self._data.move_to_end(key)   # refresh LRU position
            log.debug("Cache HIT")
            return self._data[key]
        log.debug("Cache MISS")
        return None

    def set(self, query: str, value: dict) -> None:
        key = self._key(query)
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self._max:
            self._data.popitem(last=False)   # evict oldest

    def clear(self) -> None:
        self._data.clear()
        log.info("Cache cleared")

    def __len__(self) -> int:
        return len(self._data)

    # ── private ───────────────────────────────────────────────────────────────
    @staticmethod
    def _key(query: str) -> str:
        normalised = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalised.encode()).hexdigest()
