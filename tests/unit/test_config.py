from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.config import (
    AgentConfig,
    AgentDefinition,
    MemoryConfig,
    SkillConfig,
    SleepConfig,
    TelemetryConfig,
)


def test_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    data_dir = tmp_path / "data"
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=data_dir,
        _env_file=None,
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

    def test_new_config_sections_default_none(self) -> None:
        defn = AgentDefinition()
        assert defn.memory is None
        assert defn.sleep is None
        assert defn.telemetry is None

    def test_bash_timeout_default(self) -> None:
        defn = AgentDefinition()
        assert defn.bash_timeout == 30

    def test_bash_timeout_custom(self) -> None:
        defn = AgentDefinition(bash_timeout=120)
        assert defn.bash_timeout == 120

    def test_bash_timeout_zero_rejected(self) -> None:
        with pytest.raises(Exception):
            AgentDefinition(bash_timeout=0)

    def test_bash_timeout_negative_rejected(self) -> None:
        with pytest.raises(Exception):
            AgentDefinition(bash_timeout=-1)


class TestMemoryConfig:
    def test_defaults(self) -> None:
        config = MemoryConfig()
        assert config.token_budget == 2000
        assert config.injection_prompt is None

    def test_custom_values(self) -> None:
        config = MemoryConfig(token_budget=500, injection_prompt="custom: {entries}")
        assert config.token_budget == 500


class TestSleepConfig:
    def test_defaults(self) -> None:
        config = SleepConfig()
        assert config.schedule == "0 2 * * *"
        assert config.journal_retention_days == 30
        assert config.conversation_retention_days == 14
        assert config.memory_max_entries == 50
        assert config.model is None
        assert config.summary_prompt is None
        assert config.consolidation_prompt is None


class TestTelemetryConfig:
    def test_defaults(self) -> None:
        config = TelemetryConfig()
        assert config.enabled is False
        assert config.protocol == "http"
        assert config.insecure is True


class TestYAMLWithNewSections:
    def test_memory_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "memory:\n"
            "  token_budget: 500\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.memory_config is not None
        assert config.memory_config.token_budget == 500

    def test_sleep_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "sleep:\n"
            "  schedule: '0 3 * * *'\n"
            "  memory_max_entries: 100\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.sleep_config is not None
        assert config.sleep_config.schedule == "0 3 * * *"
        assert config.sleep_config.memory_max_entries == 100

    def test_bash_timeout_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "bash_timeout: 90\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.definition.bash_timeout == 90

    def test_telemetry_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "telemetry:\n"
            "  enabled: true\n"
            "  endpoint: http://otel:4318\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.telemetry_config is not None
        assert config.telemetry_config.enabled is True
        assert config.telemetry_config.endpoint == "http://otel:4318"
