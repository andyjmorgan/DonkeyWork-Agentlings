"""Agent configuration from environment variables and optional YAML agent definition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """Memory subsystem configuration.

    Attributes:
        token_budget: Maximum tokens for the memory block injected into the system prompt.
        injection_prompt: Override template for the memory injection block.
            Receives ``{entries}`` placeholder. ``None`` uses the built-in default.
    """

    token_budget: int = 2000
    injection_prompt: str | None = None


class SleepConfig(BaseModel):
    """Nightly sleep cycle configuration.

    Attributes:
        schedule: Cron expression for when to run (default 2am daily).
        journal_retention_days: How long to keep journal files.
        conversation_retention_days: How long to keep JSONL conversation files.
        memory_max_entries: Hard cap on memory entries after consolidation.
        model: Model override for sleep LLM calls (``None`` uses agent default).
        summary_prompt: Override for the per-conversation summary prompt.
        consolidation_prompt: Override for the REM consolidation prompt.
    """

    schedule: str = "0 2 * * *"
    journal_retention_days: int = 30
    conversation_retention_days: int = 14
    memory_max_entries: int = 50
    model: str | None = None
    summary_prompt: str | None = None
    consolidation_prompt: str | None = None


class TelemetryConfig(BaseModel):
    """OpenTelemetry configuration.

    Attributes:
        enabled: Whether telemetry is active.
        endpoint: OTLP collector endpoint URL.
        protocol: Collector protocol (``"http"`` or ``"grpc"``).
        service_name: Service name for spans and metrics.
        insecure: Disable TLS for the collector connection.
    """

    enabled: bool = False
    endpoint: str = "http://localhost:4318"
    protocol: str = "http"
    service_name: str = "agentling"
    insecure: bool = True


class SkillConfig(BaseModel):
    """A skill advertised in the Agent Card.

    Attributes:
        id: Unique skill identifier.
        name: Human-readable skill name.
        description: What this skill does.
        tags: Searchable tags for discovery.
    """

    id: str
    name: str
    description: str
    tags: list[str] = Field(default_factory=list)


class AgentDefinition(BaseModel):
    """Agent identity and behaviour loaded from YAML.

    Attributes:
        name: Agent name used in the Agent Card and MCP tool.
        description: Agent description for discovery.
        tools: Tool names or groups to enable (e.g. ``["bash", "filesystem"]``).
        skills: Skills to advertise in the Agent Card.
        system_prompt: The system prompt sent to the LLM.
    """

    name: str = "agentling"
    description: str = "A lightweight AI agent"
    tools: list[str] = Field(default_factory=list)
    skills: list[SkillConfig] = Field(default_factory=list)
    system_prompt: str | None = None
    memory: MemoryConfig | None = None
    sleep: SleepConfig | None = None
    telemetry: TelemetryConfig | None = None


class AgentConfig(BaseSettings):
    """Runtime configuration for an agentling instance.

    Secrets and runtime settings come from environment variables.
    Agent identity (name, description, skills, tools, system prompt) comes
    from a YAML file pointed to by ``AGENT_CONFIG``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    agent_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"
    agent_max_tokens: int = 4096
    agent_host: str = "0.0.0.0"
    agent_port: int = 8420
    agent_data_dir: Path = Path("./data")
    agent_log_level: str = "INFO"
    agent_llm_backend: Literal["anthropic", "mock"] = "anthropic"
    agent_external_url: str | None = None
    agent_config: str | None = None
    agent_otel_endpoint: str | None = None
    agent_otel_protocol: str = "http"
    agent_otel_insecure: bool = True

    _definition: AgentDefinition = AgentDefinition()

    @model_validator(mode="after")
    def _init(self) -> AgentConfig:
        self.agent_data_dir.mkdir(parents=True, exist_ok=True)
        if self.agent_config:
            self._definition = _load_definition(self.agent_config)
        return self

    @property
    def definition(self) -> AgentDefinition:
        """The agent definition loaded from YAML (or defaults)."""
        return self._definition

    @property
    def agent_name(self) -> str:
        """Agent name from the YAML definition."""
        return self._definition.name

    @property
    def agent_description(self) -> str:
        """Agent description from the YAML definition."""
        return self._definition.description

    @property
    def enabled_tools(self) -> list[str]:
        """Tool names/groups to activate from the YAML definition."""
        return self._definition.tools

    @property
    def system_prompt(self) -> str | None:
        """System prompt from the YAML definition."""
        return self._definition.system_prompt

    @property
    def skills(self) -> list[SkillConfig]:
        """Skills to advertise from the YAML definition."""
        return self._definition.skills

    @property
    def memory_config(self) -> MemoryConfig | None:
        """Memory configuration from the YAML definition."""
        return self._definition.memory

    @property
    def sleep_config(self) -> SleepConfig | None:
        """Sleep cycle configuration from the YAML definition."""
        return self._definition.sleep

    @property
    def telemetry_config(self) -> TelemetryConfig | None:
        """Telemetry configuration, with env vars overriding YAML values."""
        base = self._definition.telemetry
        if self.agent_otel_endpoint:
            if base is None:
                base = TelemetryConfig(enabled=True)
            base = base.model_copy(update={
                "enabled": True,
                "endpoint": self.agent_otel_endpoint,
                "protocol": self.agent_otel_protocol,
                "insecure": self.agent_otel_insecure,
            })
        return base


def _load_definition(path: str) -> AgentDefinition:
    """Load an ``AgentDefinition`` from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        A validated ``AgentDefinition``.
    """
    raw = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    logger.info("loaded agent definition from %s", path)
    return AgentDefinition.model_validate(data)
