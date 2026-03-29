from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.config import AgentConfig, AgentDefinition, SkillConfig


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
    assert config.agent_log_level == "INFO"
    assert config.agent_llm_backend == "anthropic"


def test_default_agent_identity(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
    )
    assert config.agent_name == "agentling"
    assert config.agent_description == "A lightweight AI agent"
    assert config.enabled_tools == []
    assert config.skills == []
    assert config.system_prompt is None


def test_data_dir_created(tmp_path: Path) -> None:
    data_dir = tmp_path / "nested" / "data"
    assert not data_dir.exists()
    AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=data_dir,
    )
    assert data_dir.exists()


def test_llm_backend_validation(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path,
            agent_llm_backend="invalid",  # type: ignore[arg-type]
        )


class TestYAMLConfig:
    def test_loads_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: my-agent\n"
            "description: My custom agent\n"
            "tools:\n"
            "  - bash\n"
            "  - filesystem\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.agent_name == "my-agent"
        assert config.agent_description == "My custom agent"
        assert config.enabled_tools == ["bash", "filesystem"]

    def test_skills_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "skills:\n"
            "  - id: ops\n"
            "    name: Operations\n"
            "    description: Cluster operations\n"
            "    tags: [k8s, devops]\n"
            "  - id: files\n"
            "    name: File Management\n"
            "    description: Manage config files\n"
            "    tags: [files]\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert len(config.skills) == 2
        assert config.skills[0].id == "ops"
        assert config.skills[0].tags == ["k8s", "devops"]
        assert config.skills[1].id == "files"

    def test_system_prompt_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "system_prompt: |\n"
            "  You are a helpful agent.\n"
            "  Be concise.\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert "You are a helpful agent." in config.system_prompt
        assert "Be concise." in config.system_prompt

    def test_no_yaml_uses_defaults(self, tmp_path: Path) -> None:
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path,
        )
        assert config.agent_name == "agentling"
        assert config.enabled_tools == []
        assert config.skills == []

    def test_partial_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text("name: minimal-agent\n")
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.agent_name == "minimal-agent"
        assert config.agent_description == "A lightweight AI agent"
        assert config.enabled_tools == []

    def test_empty_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text("")
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.agent_name == "agentling"


class TestAgentDefinition:
    def test_defaults(self) -> None:
        defn = AgentDefinition()
        assert defn.name == "agentling"
        assert defn.tools == []
        assert defn.skills == []
        assert defn.system_prompt is None

    def test_skill_config(self) -> None:
        skill = SkillConfig(
            id="test", name="Test", description="A test skill", tags=["a", "b"]
        )
        assert skill.id == "test"
        assert skill.tags == ["a", "b"]
