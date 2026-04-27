"""Tests for system prompt construction.

The system prompt is the agent's identity and operating constraints. These tests
verify that the prompt is assembled correctly from config, memory, and data
directory awareness — and that operator-controlled flags (data_dir_awareness,
memory injection) toggle the optional blocks cleanly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agentlings.config import AgentConfig, AgentDefinition
from agentlings.core.memory_models import MemoryEntry, MemoryStore
from agentlings.core.prompt import build_system_prompt


def _all_text(prompt: list[dict]) -> str:
    """Concatenate all prompt block texts for full-prompt assertions."""
    return "\n".join(block["text"] for block in prompt)


class TestYamlSystemPrompt:
    """When the operator defines a system prompt in agent.yaml, it becomes the identity block."""

    def test_yaml_prompt_used_verbatim(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config)
        assert prompt[0]["text"].strip() == "You are a test agent."

    def test_yaml_prompt_not_mixed_with_default(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config)
        assert "Be concise. Use code blocks" not in prompt[0]["text"]

    def test_identity_block_has_cache_control(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config)
        assert prompt[0]["cache_control"] == {"type": "ephemeral"}


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

    def test_no_framework_planning_block(self, tmp_path: Path) -> None:
        """The framework no longer staples a PLAN-FIRST block onto every prompt.

        Operators own discipline rules via their own ``system_prompt``. Small
        models (Ollama) get derailed by verbose meta-instructions.
        """
        prompt = build_system_prompt(self._default_config(tmp_path))
        text = _all_text(prompt)
        assert "PLAN FIRST" not in text


class TestMemoryInjection:
    """Long-term memory is injected as a separate prompt block after identity."""

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
        memory = MemoryStore(entries=[])
        prompt = build_system_prompt(test_config, memory=memory)
        assert len(prompt) == 1  # identity only

    def test_custom_injection_template(self, test_config: AgentConfig) -> None:
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


class TestDataDirAwareness:
    """The data-directory awareness block is opt-out via ``data_dir_awareness``."""

    def test_data_dir_path_in_prompt_by_default(
        self, test_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
        prompt = build_system_prompt(test_config, data_dir=tmp_data_dir)
        text = _all_text(prompt)
        assert str(tmp_data_dir) in text
        assert "journals" in text

    def test_no_block_when_data_dir_omitted(self, test_config: AgentConfig) -> None:
        prompt = build_system_prompt(test_config, data_dir=None)
        text = _all_text(prompt)
        assert "Data Directory" not in text

    def test_no_block_when_awareness_disabled(self, tmp_path: Path) -> None:
        """``data_dir_awareness=False`` suppresses the block even when data_dir is supplied."""
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path,
        )
        config._definition = AgentDefinition(data_dir_awareness=False)
        prompt = build_system_prompt(config, data_dir=tmp_path)
        text = _all_text(prompt)
        assert "Data Directory" not in text
        assert str(tmp_path) not in text

    def test_data_dir_block_has_cache_control(
        self, test_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
        prompt = build_system_prompt(test_config, data_dir=tmp_data_dir)
        for block in prompt:
            assert block.get("cache_control") == {"type": "ephemeral"}


class TestPromptBlockOrdering:
    """Blocks appear in a consistent order: identity, memory, data dir."""

    def test_full_ordering(self, test_config: AgentConfig, tmp_data_dir: Path) -> None:
        memory = MemoryStore(entries=[
            MemoryEntry(
                key="k1", value="v1",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        prompt = build_system_prompt(test_config, memory=memory, data_dir=tmp_data_dir)
        assert len(prompt) == 3
        assert "You are a test agent." in prompt[0]["text"]  # identity
        assert "k1" in prompt[1]["text"]                     # memory
        assert str(tmp_data_dir) in prompt[2]["text"]        # data dir

    def test_all_blocks_have_cache_control(
        self, test_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
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
