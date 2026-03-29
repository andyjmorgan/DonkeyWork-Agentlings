"""Agent Card generation for A2A service discovery."""

from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    APIKeySecurityScheme,
)

from agentlings.config import AgentConfig


def generate_agent_card(config: AgentConfig) -> AgentCard:
    """Build an A2A AgentCard from the agent's configuration.

    Args:
        config: Agent configuration containing name, description, and URL settings.

    Returns:
        A populated AgentCard describing this agent's capabilities and endpoint.
    """
    if config.agent_external_url:
        url = config.agent_external_url.rstrip("/") + "/a2a"
    else:
        url = f"http://{config.agent_host}:{config.agent_port}/a2a"
    return AgentCard(
        name=config.agent_name,
        description=config.agent_description,
        url=url,
        version="0.1.0",
        skills=[
            AgentSkill(
                id=config.agent_name,
                name=config.agent_name,
                description=config.agent_description,
                tags=[],
            )
        ],
        capabilities=AgentCapabilities(streaming=True, pushNotifications=False),
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        securitySchemes={
            "apiKey": APIKeySecurityScheme(
                type="apiKey",
                in_="header",
                name="X-API-Key",
            )
        },
        security=[{"apiKey": []}],
    )
