from __future__ import annotations

from pathlib import Path

from agentlings.config import AgentConfig
from agentlings.prompt import build_system_prompt


def test_default_prompt_includes_name(test_config: AgentConfig) -> None:
    prompt = build_system_prompt(test_config)
    assert len(prompt) == 1
    assert "test-agent" in prompt[0]["text"]
    assert "A test agent" in prompt[0]["text"]


def test_default_prompt_has_cache_control(test_config: AgentConfig) -> None:
    prompt = build_system_prompt(test_config)
    assert prompt[0]["cache_control"] == {"type": "ephemeral"}


def test_file_override(tmp_path: Path) -> None:
    prompt_file = tmp_path / "custom_prompt.txt"
    prompt_file.write_text("You are a custom agent.")
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        agent_system_prompt_file=str(prompt_file),
    )
    prompt = build_system_prompt(config)
    assert prompt[0]["text"] == "You are a custom agent."
    assert prompt[0]["cache_control"] == {"type": "ephemeral"}
