"""Agent Card generation from configuration."""

from __future__ import annotations

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    APIKeySecurityScheme,
    OpenIdConnectSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.constants import PROTOCOL_VERSION_1_0, TransportProtocol

from agentlings.config import AgentConfig


def generate_agent_card(config: AgentConfig) -> AgentCard:
    """Generate an A2A Agent Card from the agent's configuration.

    Skills are taken from the YAML definition if provided, otherwise a
    single default skill is generated from the agent's name and description.

    Args:
        config: The agent configuration.

    Returns:
        A fully populated ``AgentCard`` instance.
    """
    if config.agent_external_url:
        url = config.agent_external_url.rstrip("/") + "/a2a"
    else:
        url = f"http://{config.agent_host}:{config.agent_port}/a2a"

    if config.skills:
        skills = [
            AgentSkill(
                id=s.id,
                name=s.name,
                description=s.description,
                tags=list(s.tags),
            )
            for s in config.skills
        ]
    else:
        skills = [
            AgentSkill(
                id=config.agent_name,
                name=config.agent_name,
                description=config.agent_description,
                tags=[],
            )
        ]

    security_schemes: dict[str, SecurityScheme] = {
        "apiKey": SecurityScheme(
            api_key_security_scheme=APIKeySecurityScheme(
                location="header",
                name="X-API-Key",
            )
        )
    }
    # Two requirements in the list mean "apiKey OR oidc", mirroring the
    # either-credential-works behaviour of the auth middleware.
    security_requirements = [
        SecurityRequirement(schemes={"apiKey": StringList(list=[])})
    ]

    oauth = config.oauth_config
    if oauth is not None:
        security_schemes["oidc"] = SecurityScheme(
            open_id_connect_security_scheme=OpenIdConnectSecurityScheme(
                open_id_connect_url=(
                    oauth.issuer.rstrip("/") + "/.well-known/openid-configuration"
                ),
            )
        )
        security_requirements.append(
            SecurityRequirement(schemes={"oidc": StringList(list=[])})
        )

    return AgentCard(
        name=config.agent_name,
        description=config.agent_description,
        version="0.1.0",
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_1_0,
            ),
        ],
        skills=skills,
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        security_schemes=security_schemes,
        security_requirements=security_requirements,
    )
