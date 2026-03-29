from __future__ import annotations

from pathlib import Path

from agentlings.config import AgentConfig
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
