"""Tests for system prompt construction.

The system prompt is the agent's identity and operating constraints. These tests
verify that the prompt is assembled correctly from config, memory, and data
directory awareness — and critically, that execution rules are always present
as an architectural requirement regardless of custom system prompts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agentlings.config import AgentConfig
from agentlings.core.memory_models import MemoryEntry, MemoryStore
from agentlings.core.prompt import build_system_prompt

EXECUTION_RULES = [
    "PLAN FIRST",
    "STAY FOCUSED",
    "DELIVER INCREMENTALLY",
    "PREFER COMPLETE OVER AMBITIOUS",
    "CLARIFY EARLY",
]


def _all_text(prompt: list[dict]) -> str:
    """Concatenate all prompt block texts for full-prompt assertions."""
    return "\n".join(block["text"] for block in prompt)


# ---------------------------------------------------------------------------
# Execution rules — architectural requirement, always present
# ---------------------------------------------------------------------------

class TestExecutionRulesAlwaysPresent:
    """Execution rules are non-negotiable framework constraints.

    They must appear in every agent's prompt regardless of whether the operator
    defines a custom system prompt. This is an architectural requirement — the
    rules prevent agents from running unbounded completion loops that time out
    and waste resources. An operator can shape the agent's personality and
    domain expertise, but cannot opt out of operational discipline.
    """

    def test_present_with_yaml_prompt(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config)
        text = _all_text(prompt)
        for rule in EXECUTION_RULES:
            assert rule in text, f"Execution rule '{rule}' missing from YAML-configured agent"

    def test_present_with_default_prompt(self, tmp_path: Path) -> None:
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path,
        )
        prompt = build_system_prompt(config)
        text = _all_text(prompt)
        for rule in EXECUTION_RULES:
            assert rule in text, f"Execution rule '{rule}' missing from default agent"

    def test_rules_in_separate_block_from_identity(self, test_config: AgentConfig) -> None:
        """Execution rules must be a distinct block so they survive prompt caching independently.

        If the operator changes their system prompt, only that block's cache
        invalidates — the rules block stays cached.
        """
        prompt = build_system_prompt(test_config)
        identity_block = prompt[0]["text"]
        rules_block = prompt[1]["text"]
        assert "You are a test agent." in identity_block
        assert "PLAN FIRST" not in identity_block
        assert "PLAN FIRST" in rules_block

    def test_rules_block_has_cache_control(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config)
        assert prompt[1]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# YAML-defined system prompt
# ---------------------------------------------------------------------------

class TestYamlSystemPrompt:
    """When the operator defines a system prompt in agent.yaml, it becomes the identity block."""

    def test_yaml_prompt_used_verbatim(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config)
        assert prompt[0]["text"].strip() == "You are a test agent."

    def test_yaml_prompt_not_mixed_with_default(self, test_config: AgentConfig) -> None:
        """The identity block must contain only the operator's text, not the default template."""
        prompt = build_system_prompt(test_config)
        assert "Be concise. Use code blocks" not in prompt[0]["text"]


# ---------------------------------------------------------------------------
# Default system prompt (no YAML override)
# ---------------------------------------------------------------------------

class TestDefaultSystemPrompt:
    """When no YAML prompt is defined, the framework generates an identity block."""

    def _default_config(self, tmp_path: Path) -> AgentConfig:
        return AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path,
        )

    def test_includes_agent_identity(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(self._default_config(tmp_path))
        assert "agentling" in prompt[0]["text"]
        assert "A lightweight AI agent" in prompt[0]["text"]

    def test_cache_control_is_ephemeral(self, tmp_path: Path) -> None:
        prompt = build_system_prompt(self._default_config(tmp_path))
        assert prompt[0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Memory injection
# ---------------------------------------------------------------------------

class TestMemoryInjection:
    """Long-term memory is injected as a separate prompt block after the rules."""

    def test_memory_entries_appear_in_prompt(self, test_config: AgentConfig) -> None:
        memory = MemoryStore(entries=[
            MemoryEntry(
                key="test-fact",
                value="important value",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        prompt = build_system_prompt(test_config, memory=memory)
        text = _all_text(prompt)
        assert "test-fact" in text
        assert "important value" in text

    def test_empty_memory_produces_no_block(self, test_config: AgentConfig) -> None:
        """An empty memory store must not add a blank block to the prompt.

        A blank block wastes prompt cache slots and could confuse the LLM
        with an empty "Memory" section.
        """
        memory = MemoryStore(entries=[])
        prompt = build_system_prompt(test_config, memory=memory)
        assert len(prompt) == 2  # identity + rules, no memory block

    def test_custom_injection_template(self, test_config: AgentConfig) -> None:
        """Operators can override the memory injection template for domain-specific framing."""
        memory = MemoryStore(entries=[
            MemoryEntry(
                key="k1", value="v1",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        custom = "CUSTOM MEMORY:\n{entries}"
        prompt = build_system_prompt(test_config, memory=memory, injection_prompt=custom)
        text = _all_text(prompt)
        assert "CUSTOM MEMORY:" in text


# ---------------------------------------------------------------------------
# Data directory awareness
# ---------------------------------------------------------------------------

class TestDataDirAwareness:
    """The agent is told about its data directory so it can read journals and logs."""

    def test_data_dir_path_in_prompt(self, test_config: AgentConfig, tmp_data_dir: Path) -> None:
        prompt = build_system_prompt(test_config, data_dir=tmp_data_dir)
        text = _all_text(prompt)
        assert str(tmp_data_dir) in text
        assert "journals" in text

    def test_data_dir_block_has_cache_control(self, test_config: AgentConfig, tmp_data_dir: Path) -> None:
        prompt = build_system_prompt(test_config, data_dir=tmp_data_dir)
        for block in prompt:
            assert block.get("cache_control") == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Combined blocks — ordering matters
# ---------------------------------------------------------------------------

class TestPromptBlockOrdering:
    """Blocks must appear in a consistent order: identity, rules, memory, data dir.

    The LLM reads blocks sequentially. Identity sets the persona, rules set
    constraints, memory provides context, data dir enables capabilities.
    """

    def test_full_ordering(self, test_config: AgentConfig, tmp_data_dir: Path) -> None:
        memory = MemoryStore(entries=[
            MemoryEntry(
                key="k1", value="v1",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        prompt = build_system_prompt(test_config, memory=memory, data_dir=tmp_data_dir)
        assert len(prompt) == 4
        assert "You are a test agent." in prompt[0]["text"]  # identity
        assert "PLAN FIRST" in prompt[1]["text"]             # rules
        assert "k1" in prompt[2]["text"]                     # memory
        assert str(tmp_data_dir) in prompt[3]["text"]        # data dir

    def test_all_blocks_have_cache_control(self, test_config: AgentConfig, tmp_data_dir: Path) -> None:
        memory = MemoryStore(entries=[
            MemoryEntry(
                key="k1", value="v1",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        prompt = build_system_prompt(test_config, memory=memory, data_dir=tmp_data_dir)
        for i, block in enumerate(prompt):
            assert block.get("cache_control") == {"type": "ephemeral"}, (
                f"Block {i} missing cache_control"
            )
