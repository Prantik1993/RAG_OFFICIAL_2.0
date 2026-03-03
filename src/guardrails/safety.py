"""
Safety Guardrails
=================
Minimal, production-grade checks.  No over-blocking.
"""

from __future__ import annotations

import re
from src.logger import get_logger

log = get_logger("Safety")

# Only block clear prompt-injection patterns
_INJECTION = re.compile(
    r'(ignore\s+(all|previous|prior)\s+instructions'
    r'|disregard.*instructions'
    r'|new\s+role\s*:'
    r'|system\s+prompt'
    r'|<\|im_start\|>'
    r'|</s>)',
    re.I,
)

# System prompt leak markers
_LEAK = re.compile(r'(you are an expert legal|langchain)', re.I)


class SafetyGuardrails:

    def check(self, query: str) -> tuple[bool, str]:
        q = query.strip()
        if not q:
            return False, "empty query"
        if len(q) > 2000:
            return False, "query too long"
        if _INJECTION.search(q):
            return False, "injection pattern"
        # >60% special chars is likely garbage / attack
        special = sum(not c.isalnum() and not c.isspace() for c in q)
        if len(q) > 0 and special / len(q) > 0.6:
            return False, "malformed input"
        return True, "ok"

    def check_output(self, response: str) -> bool:
        if len(response.strip()) < 5:
            return False
        if _LEAK.search(response):
            log.warning("System prompt leak detected")
            return False
        return True
