from __future__ import annotations

from agentlings.protocol.agent_card import generate_agent_card
from agentlings.config import AgentConfig


def test_generated_card_fields(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert card.name == "test-agent"
    assert card.description == "A test agent"
    assert card.url == f"http://{test_config.agent_host}:{test_config.agent_port}/a2a"


def test_generated_card_capabilities(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert card.capabilities.streaming is False
    assert card.capabilities.push_notifications is False


def test_generated_card_skill(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert len(card.skills) == 1
    assert card.skills[0].id == "test-agent"
    assert card.skills[0].name == "test-agent"


def test_generated_card_security(test_config: AgentConfig) -> None:
    card = generate_agent_card(test_config)
    assert card.security_schemes is not None
    assert "apiKey" in card.security_schemes
    assert card.security == [{"apiKey": []}]
