
from __future__ import annotations
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import src.config as cfg
from src.logger import get_logger

log = get_logger("Tracker")

_LOG_FILE = cfg.LOG_DIR / "llm_calls.jsonl"


@dataclass
class _Call:
    ts:         str
    latency_ms: float
    success:    bool
    error:      Optional[str] = None


class LLMTracker:

    def __init__(self) -> None:
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._calls: list[_Call] = []

    def record(self, latency_ms: float, success: bool = True, error: Optional[str] = None) -> None:
        call = _Call(
            ts=datetime.now(timezone.utc).isoformat(),
            latency_ms=round(latency_ms, 1),
            success=success,
            error=error,
        )
        self._calls.append(call)
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(call)) + "\n")
        if success:
            log.info(f"LLM call: {latency_ms:.0f} ms")
        else:
            log.error(f"LLM call failed: {error}")

    def stats(self) -> dict:
        total = len(self._calls)
        if not total:
            return {"total_calls": 0}
        ok  = [c for c in self._calls if c.success]
        avg = sum(c.latency_ms for c in ok) / len(ok) if ok else 0
        return {
            "total_calls":    total,
            "successful":     len(ok),
            "avg_latency_ms": round(avg),
        }
