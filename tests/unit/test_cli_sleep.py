"""Tests for the ``agentling sleep`` CLI command's LLM client wiring.

The CLI must build its client exactly like the server does — through the
wire-format-aware ``llm_api_key`` / ``llm_base_url`` config properties and
the ``wire_format`` factory parameter — so an agentling deployed on
``AGENT_WIRE_FORMAT=responses`` runs its manual sleep cycle over the same
backend as its live traffic instead of silently falling back to the
Anthropic Messages path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.core.llm import MockLLMClient


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    for var in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY", "OPENAI_BASE_URL",
        "AGENT_WIRE_FORMAT", "AGENT_CONFIG", "AGENT_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    # Keep the AgentConfig() inside _sleep_command from reading a repo .env.
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def _run_sleep_capturing_factory(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run ``_sleep_command`` with a spy factory; return its kwargs."""
    import agentlings.core.llm as llm_module
    from agentlings.__main__ import _sleep_command

    captured: dict = {}

    def spy_factory(**kwargs):
        captured.update(kwargs)
        return MockLLMClient(tool_names=[])

    # _sleep_command imports create_llm_client function-locally at call
    # time, so patching the source module is effective. Belt-and-braces:
    # the empty-capture assert below fails loudly if that import ever
    # moves to module level.
    monkeypatch.setattr(llm_module, "create_llm_client", spy_factory)
    # An empty data dir means light sleep finds nothing and the cycle
    # returns immediately — the client construction is all we exercise.
    _sleep_command(None)
    assert captured, "spy factory was never invoked — patching is ineffective"
    return captured


def test_sleep_cli_uses_wire_format_selection(_clean_env) -> None:
    _clean_env.setenv("AGENT_WIRE_FORMAT", "responses")
    _clean_env.setenv("AGENT_MODEL", "gpt-5-mini")
    _clean_env.setenv("OPENAI_API_KEY", "sk-openai")
    _clean_env.setenv("OPENAI_BASE_URL", "https://gateway.example")

    captured = _run_sleep_capturing_factory(_clean_env)

    assert captured["wire_format"] == "responses"
    assert captured["api_key"] == "sk-openai"
    assert captured["base_url"] == "https://gateway.example"


def test_sleep_cli_messages_path_unchanged(_clean_env) -> None:
    _clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant")
    _clean_env.setenv("ANTHROPIC_BASE_URL", "http://ollama:11434")

    captured = _run_sleep_capturing_factory(_clean_env)

    assert captured["wire_format"] == "messages"
    assert captured["api_key"] == "sk-ant"
    assert captured["base_url"] == "http://ollama:11434"


def test_sleep_cli_mock_backend_needs_no_credentials(_clean_env) -> None:
    """Mock + responses without OPENAI_API_KEY must run, not raise —
    credential validation only applies to real backends."""
    _clean_env.setenv("AGENT_LLM_BACKEND", "mock")
    _clean_env.setenv("AGENT_WIRE_FORMAT", "responses")

    captured = _run_sleep_capturing_factory(_clean_env)

    assert captured["backend"] == "mock"
    assert captured["api_key"] == ""
    assert captured["base_url"] is None
