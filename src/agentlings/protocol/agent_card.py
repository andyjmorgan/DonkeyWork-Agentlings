"""Agent Card generation from configuration."""

from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    APIKeySecurityScheme,
)

from agentlings.config import AgentConfig


def generate_agent_card(config: AgentConfig) -> AgentCard:
    """Generate an A2A Agent Card from the agent's configuration.

    The Agent Card is the single source of truth for agent identity.
    The MCP tool schema is derived from it.

    Args:
        config: The agent configuration.

    Returns:
        A fully populated ``AgentCard`` instance.
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
