"""
LLM Usage Tracker  (v4.2)
=========================
Now tracks prompt_version per call for eval correlation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import src.config as cfg
from src.logger import get_logger

log = get_logger("Tracker")

_LOG_FILE = cfg.LOG_DIR / "llm_calls.jsonl"


@dataclass
class _Call:
    ts:             str
    latency_ms:     float
    success:        bool
    prompt_version: str = "unknown"
    error:          Optional[str] = None


class LLMTracker:

    def __init__(self) -> None:
        cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._calls: list[_Call] = []

    def record(
        self,
        latency_ms: float,
        success: bool = True,
        error: Optional[str] = None,
        prompt_version: str = "unknown",
    ) -> None:
        call = _Call(
            ts=datetime.now(timezone.utc).isoformat(),
            latency_ms=round(latency_ms, 1),
            success=success,
            prompt_version=prompt_version,
            error=error,
        )
        self._calls.append(call)
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(call)) + "\n")
        if success:
            log.info(f"LLM call: {latency_ms:.0f} ms | prompt_v={prompt_version}")
        else:
            log.error(f"LLM call failed: {error}")

    def stats(self) -> dict:
        total = len(self._calls)
        if not total:
            return {"total_calls": 0}
        ok  = [c for c in self._calls if c.success]
        avg = sum(c.latency_ms for c in ok) / len(ok) if ok else 0

        # Per-version breakdown
        by_version: dict = {}
        for c in self._calls:
            v = c.prompt_version
            by_version.setdefault(v, {"calls": 0, "latencies": []})
            by_version[v]["calls"] += 1
            if c.success:
                by_version[v]["latencies"].append(c.latency_ms)

        version_stats = {
            v: {
                "calls": d["calls"],
                "avg_latency_ms": round(sum(d["latencies"]) / len(d["latencies"])) if d["latencies"] else 0,
            }
            for v, d in by_version.items()
        }

        return {
            "total_calls":    total,
            "successful":     len(ok),
            "avg_latency_ms": round(avg),
            "by_prompt_version": version_stats,
        }
