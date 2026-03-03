"""
Unit tests for PromptRegistry — reads from actual prompts/ directory.
"""
import pytest
from pathlib import Path
from src.prompts.registry import PromptRegistry, PromptConfig

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@pytest.fixture(scope="module")
def registry():
    return PromptRegistry(_PROMPTS_DIR)


def test_loads_at_least_one_version(registry):
    assert len(registry.list_versions()) >= 1


def test_get_latest(registry):
    p = registry.get("latest")
    assert isinstance(p, PromptConfig)
    assert p.version is not None


def test_get_explicit_version(registry):
    p = registry.get("1")
    assert p.version == "1"


def test_get_unknown_falls_back_to_latest(registry):
    p = registry.get("999")
    assert p is not None   # fallback, no exception


def test_prompt_has_required_fields(registry):
    p = registry.get("1")
    assert "{context}" in p.system_template
    assert p.model
    assert p.temperature >= 0


def test_to_langchain_prompt(registry):
    from langchain_core.prompts import ChatPromptTemplate
    p      = registry.get("1")
    prompt = p.to_langchain_prompt()
    assert isinstance(prompt, ChatPromptTemplate)


def test_list_versions_sorted(registry):
    versions = [v["version"] for v in registry.list_versions()]
    assert versions == sorted(versions)
