from __future__ import annotations

from pathlib import Path

from agentlings.config import AgentConfig
from agentlings.protocol.agent_card import generate_agent_card


def test_generated_card_fields(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert card.name == "test-agent"
    assert card.description == "A test agent"
    # 1.0 cards expose transport URLs via supported_interfaces; there is no
    # top-level ``url`` field anymore.
    urls = [iface.url for iface in card.supported_interfaces]
    assert (
        f"http://{test_config.agent_host}:{test_config.agent_port}/a2a" in urls
    )


def test_generated_card_capabilities(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert card.capabilities.streaming is False
    assert card.capabilities.push_notifications is False


def test_generated_card_security(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert card.security_schemes is not None
    assert "apiKey" in card.security_schemes
    # security_requirements is a repeated SecurityRequirement; we advertise
    # a single requirement keying on ``apiKey``.
    assert len(card.security_requirements) == 1
    assert "apiKey" in card.security_requirements[0].schemes


def test_skills_from_yaml(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert len(card.skills) == 1
    assert card.skills[0].id == "testing"
    assert card.skills[0].name == "Testing"
    assert card.skills[0].description == "A test skill"
    assert list(card.skills[0].tags) == ["test"]


def test_default_skill_when_no_yaml(tmp_path: Path) -> None:
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
    )
    card = generate_agent_card(config)
    assert len(card.skills) == 1
    assert card.skills[0].id == "agentling"
    assert card.skills[0].name == "agentling"


def test_multiple_skills(tmp_path: Path) -> None:
    yaml_file = tmp_path / "agent.yaml"
    yaml_file.write_text(
        "name: multi\n"
        "description: Multi-skill agent\n"
        "skills:\n"
        "  - id: ops\n"
        "    name: Operations\n"
        "    description: Cluster ops\n"
        "    tags: [k8s]\n"
        "  - id: files\n"
        "    name: Files\n"
        "    description: File management\n"
        "    tags: [files]\n"
        "  - id: monitoring\n"
        "    name: Monitoring\n"
        "    description: Observability\n"
        "    tags: [metrics, logs]\n"
    )
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        agent_config=str(yaml_file),
    )
    card = generate_agent_card(config)
    assert len(card.skills) == 3
    assert [s.id for s in card.skills] == ["ops", "files", "monitoring"]
