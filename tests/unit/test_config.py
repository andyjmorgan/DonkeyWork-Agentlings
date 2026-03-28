from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.config import AgentConfig


def test_defaults(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=data_dir,
    )
    assert config.agent_model == "claude-sonnet-4-6"
    assert config.agent_max_tokens == 4096
    assert config.agent_host == "0.0.0.0"
    assert config.agent_port == 8420
    assert config.agent_name == "agentling"
    assert config.agent_log_level == "INFO"
    assert config.agent_llm_backend == "anthropic"
    assert config.agent_system_prompt_file is None


def test_data_dir_created(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    assert not data_dir.exists()
    AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=data_dir,
    )
    assert data_dir.exists()


def test_custom_values(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
        agent_model="claude-opus-4-6",
        agent_port=9000,
        agent_name="my-agent",
        agent_llm_backend="mock",
    )
    assert config.agent_model == "claude-opus-4-6"
    assert config.agent_port == 9000
    assert config.agent_name == "my-agent"
    assert config.agent_llm_backend == "mock"


def test_llm_backend_validation(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path,
            agent_llm_backend="invalid",  # type: ignore[arg-type]
        )


def test_enabled_tools_empty(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
    )
    assert config.enabled_tools == []


def test_enabled_tools_parsed(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
        agent_tools="bash, filesystem",
    )
    assert config.enabled_tools == ["bash", "filesystem"]


