from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agentlings.config import AgentConfig
from agentlings.core.memory_models import MemoryEntry, MemoryStore
from agentlings.core.prompt import build_system_prompt


def test_yaml_system_prompt_used(test_config: AgentConfig) -> None:
    prompt = build_system_prompt(test_config)
    assert len(prompt) == 1
    assert prompt[0]["text"].strip() == "You are a test agent."


def test_default_prompt_has_cache_control(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
    )
    prompt = build_system_prompt(config)
    assert prompt[0]["cache_control"] == {"type": "ephemeral"}


def test_default_prompt_when_no_yaml(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
    )
    prompt = build_system_prompt(config)
    assert "agentling" in prompt[0]["text"]
    assert "A lightweight AI agent" in prompt[0]["text"]


def test_memory_injection(test_config: AgentConfig) -> None:
    memory = MemoryStore(entries=[
        MemoryEntry(
            key="test-fact",
            value="important value",
            recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
    ])
    prompt = build_system_prompt(test_config, memory=memory)
    assert len(prompt) >= 2
    memory_block = prompt[1]["text"]
    assert "test-fact" in memory_block
    assert "important value" in memory_block


def test_empty_memory_omitted(test_config: AgentConfig) -> None:
    memory = MemoryStore(entries=[])
    prompt = build_system_prompt(test_config, memory=memory)
    assert len(prompt) == 1


def test_data_dir_awareness(test_config: AgentConfig, tmp_data_dir: Path) -> None:
    prompt = build_system_prompt(test_config, data_dir=tmp_data_dir)
    assert len(prompt) == 2
    assert str(tmp_data_dir) in prompt[1]["text"]
    assert "journals" in prompt[1]["text"]


def test_memory_and_data_dir(test_config: AgentConfig, tmp_data_dir: Path) -> None:
    memory = MemoryStore(entries=[
        MemoryEntry(
            key="k1", value="v1",
            recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
    ])
    prompt = build_system_prompt(test_config, memory=memory, data_dir=tmp_data_dir)
    assert len(prompt) == 3
    assert "k1" in prompt[1]["text"]
    assert str(tmp_data_dir) in prompt[2]["text"]


def test_custom_injection_prompt(test_config: AgentConfig) -> None:
    memory = MemoryStore(entries=[
        MemoryEntry(
            key="k1", value="v1",
            recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
        ),
    ])
    custom = "CUSTOM MEMORY:\n{entries}"
    prompt = build_system_prompt(test_config, memory=memory, injection_prompt=custom)
    assert "CUSTOM MEMORY:" in prompt[1]["text"]
