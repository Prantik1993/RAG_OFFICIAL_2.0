"""
Prompt Registry
===============
Loads versioned prompts from prompts/*.yaml at runtime.
Active version controlled by PROMPT_VERSION env var.

Why YAML prompts?
- Git-tracked: every prompt change is a diff, fully reversible
- Version in API response: you know exactly which prompt generated an answer
- A/B testing: run v1 and v2 simultaneously, compare RAGAS scores
- Zero code change to update a prompt

Usage:
    from src.prompts.registry import PromptRegistry
    registry = PromptRegistry()
    prompt   = registry.get()           # active version
    prompt   = registry.get("2")        # explicit version
    prompt   = registry.get("latest")   # highest version number
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import src.config as cfg
from src.logger import get_logger

log = get_logger("PromptRegistry")

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


class PromptConfig:
    """Parsed prompt YAML with convenience accessors."""

    def __init__(self, data: dict) -> None:
        self._d = data

    @property
    def version(self) -> str:
        return str(self._d["version"])

    @property
    def description(self) -> str:
        return self._d.get("description", "")

    @property
    def system_template(self) -> str:
        return self._d["system"]

    @property
    def temperature(self) -> float:
        return float(self._d.get("temperature", 0))

    @property
    def model(self) -> str:
        return self._d.get("model", cfg.LLM_MODEL)

    @property
    def changelog(self) -> list[str]:
        return self._d.get("changelog", [])

    def to_langchain_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", self.system_template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def to_dict(self) -> dict:
        return {
            "version":     self.version,
            "description": self.description,
            "model":       self.model,
            "temperature": self.temperature,
            "changelog":   self.changelog,
        }


class PromptRegistry:
    """
    Loads all prompt YAML files from the prompts/ directory.
    Thread-safe (reads only after init).
    """

    def __init__(self, prompts_dir: Optional[Path] = None) -> None:
        self._dir = prompts_dir or _PROMPTS_DIR
        self._prompts: dict[str, PromptConfig] = {}
        self._load_all()

    # ── public ────────────────────────────────────────────────────────────────
    def get(self, version: Optional[str] = None) -> PromptConfig:
        """
        Return a PromptConfig.
          version=None    → uses PROMPT_VERSION env var, falls back to "latest"
          version="latest"→ highest version number
          version="2"     → exactly v2
        """
        v = version or cfg.PROMPT_VERSION
        if v == "latest":
            return self._latest()
        if v not in self._prompts:
            log.warning(f"Prompt version '{v}' not found — using latest")
            return self._latest()
        return self._prompts[v]

    def list_versions(self) -> list[dict]:
        return [p.to_dict() for p in sorted(
            self._prompts.values(), key=lambda p: p.version
        )]

    # ── private ───────────────────────────────────────────────────────────────
    def _load_all(self) -> None:
        if not self._dir.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self._dir}")

        for yaml_file in sorted(self._dir.glob("v*.yaml")):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                cfg_obj = PromptConfig(data)
                self._prompts[cfg_obj.version] = cfg_obj
                log.info(f"Loaded prompt v{cfg_obj.version}: {cfg_obj.description}")
            except Exception as exc:
                log.error(f"Failed to load {yaml_file}: {exc}")

        if not self._prompts:
            raise RuntimeError(f"No prompt YAML files found in {self._dir}")

    def _latest(self) -> PromptConfig:
        return self._prompts[max(self._prompts.keys(), key=lambda v: int(v))]
