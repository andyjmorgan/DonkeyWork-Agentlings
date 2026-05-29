"""Agent configuration from environment variables and optional YAML agent definition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal


import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class MemoryConfig(BaseModel):
    """Memory subsystem configuration.

    Attributes:
        token_budget: Maximum tokens for the memory block injected into the system prompt.
        injection_prompt: Override template for the memory injection block.
            Receives ``{entries}`` placeholder. ``None`` uses the built-in default.
        inject_into_prompt: When ``False``, the memory store is still active and
            the ``memory_*`` tools are still callable, but no memory block is
            stapled into the system prompt. The agent must read memory
            explicitly via the tool. Useful for small/local models where the
            prompt budget is precious.
    """

    model_config = ConfigDict(extra="ignore")

    token_budget: int = 2000
    injection_prompt: str | None = None
    inject_into_prompt: bool = True


class SleepConfig(BaseModel):
    """Nightly sleep cycle configuration.

    Attributes:
        enabled: When ``False``, the sleep cycle is not scheduled even if
            the block is present. Set this for backends that lack the
            Anthropic batches API (e.g. Ollama's compatibility layer).
        schedule: Cron expression for when to run (default 2am daily).
        journal_retention_days: How long to keep journal files.
        conversation_retention_days: How long to keep JSONL conversation files.
        memory_max_entries: Hard cap on memory entries after consolidation.
        model: Model override for sleep LLM calls (``None`` uses agent default).
        summary_prompt: Override for the per-conversation summary prompt.
        consolidation_prompt: Override for the REM consolidation prompt.
        consolidation_max_tokens: Output token budget for the REM consolidation
            call. Must be large enough to re-emit the entire memory (up to
            ``memory_max_entries`` entries) as JSON; the per-turn agent default
            (4096) truncates the response and the consolidation is dropped.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    schedule: str = "0 2 * * *"
    journal_retention_days: int = 30
    conversation_retention_days: int = 14
    memory_max_entries: int = 50
    model: str | None = None
    summary_prompt: str | None = None
    consolidation_prompt: str | None = None
    consolidation_max_tokens: int = 16384


ThinkingMode = Literal["off", "budget", "adaptive"]
ThinkingEffort = Literal["low", "medium", "high", "xhigh", "max"]
ThinkingDisplay = Literal["summarized"]

INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"


class ThinkingConfig(BaseModel):
    """Extended-thinking configuration for Anthropic models.

    Three modes cover the model-generation split that landed in early 2026:

    * ``mode: "off"`` (default) — no ``thinking`` block is attached. Use
      this for non-Anthropic backends (Ollama) and for cost-sensitive
      live calls.

    * ``mode: "budget"`` — legacy shape, supported on Claude Sonnet 3.7,
      Sonnet 4/4.5, Opus 4/4.1/4.5, and Haiku 4.5. Sends
      ``thinking={"type": "enabled", "budget_tokens": N}``. Set
      ``interleaved: true`` to add the ``interleaved-thinking-2025-05-14``
      beta header (required on Opus 4-4.5 and Sonnet 4-4.5 to think
      between tool calls; not supported on Haiku 4.5). Anthropic requires
      ``budget_tokens >= 1024`` and ``< max_tokens`` (except when
      ``interleaved: true``, where the budget is a per-turn total across
      all thinking blocks and may exceed ``max_tokens``).

    * ``mode: "adaptive"`` — the post-Sonnet-4.6 shape. Sends
      ``thinking={"type": "adaptive"}`` and lets the model decide budget
      per request. Required on Opus 4.7+ (legacy budget returns 400);
      recommended on Sonnet 4.6 and Opus 4.6. Optional ``effort`` knob
      (``low|medium|high|xhigh|max``) goes into ``output_config.effort``.
      Optional ``display: "summarized"`` opts back into summarized
      thinking content on Opus 4.7+ (default is empty thinking blocks).
      Interleaved thinking is automatic in this mode — no flag needed.

    A ``model_validator`` rejects combinations that the API would reject
    (e.g. ``effort`` set when ``mode != "adaptive"``). The LLM client logs
    a warning at construction if the configured mode looks wrong for the
    active model, but does not refuse — the user may swap models without
    re-validating YAML, and a clear log line is better than a startup
    crash.
    """

    model_config = ConfigDict(extra="ignore")

    mode: ThinkingMode = "off"
    budget_tokens: int = Field(ge=1024, default=8192)
    interleaved: bool = False
    effort: ThinkingEffort | None = None
    display: ThinkingDisplay | None = None

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "ThinkingConfig":
        if self.mode != "adaptive":
            if self.effort is not None:
                raise ValueError(
                    "thinking.effort is only valid when thinking.mode == 'adaptive'"
                )
            if self.display is not None:
                raise ValueError(
                    "thinking.display is only valid when thinking.mode == 'adaptive'"
                )
        if self.mode != "budget" and self.interleaved:
            raise ValueError(
                "thinking.interleaved is only valid when thinking.mode == 'budget' "
                "(interleaved thinking is implicit on adaptive mode)"
            )
        return self


class TelemetryConfig(BaseModel):
    """OpenTelemetry configuration.

    Attributes:
        enabled: Whether telemetry is active.
        endpoint: OTLP collector endpoint URL.
        protocol: Collector protocol (``"http"`` or ``"grpc"``).
        service_name: Service name for spans and metrics.
        insecure: Disable TLS for the collector connection.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    endpoint: str = "http://localhost:4318"
    protocol: str = "http"
    service_name: str = "agentling"
    insecure: bool = True
    headers: dict[str, str] = Field(default_factory=dict)


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
        bash_timeout: Default timeout in seconds for bash tool commands.
        max_tool_result_chars: Maximum character length of a single tool result
            before it is truncated (with a marker telling the model it was cut)
            on its way back to the LLM. Guards against a runaway tool (e.g. a
            recursive grep sweeping up minified assets) producing a multi-MB
            result that blows the request body past the API's hard size limit
            and fails the turn with a 413. Defaults to 32000; set ``0`` to
            disable truncation.
        data_dir_awareness: When ``True`` (default), append a system-prompt
            block telling the agent where its data directory lives and how
            to read journals and conversation logs. Set to ``False`` for
            agents without filesystem tools or when the prompt budget is
            tight (e.g. small local models).
        send_name_header: When ``True`` (default), send the agentling's ``name``
            as the ``x-agentling-name`` header on every LLM request so a
            deployment can attribute upstream traffic to a specific agentling.
            Set to ``False`` to omit it (e.g. when a shared proxy injects its
            own identity header).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = "agentling"
    description: str = "A lightweight AI agent"
    tools: list[str] = Field(default_factory=list)
    skills: list[SkillConfig] = Field(default_factory=list)
    system_prompt: str | None = None
    bash_timeout: int = Field(ge=1, default=50)
    # Keep the default in sync with completion.DEFAULT_MAX_TOOL_RESULT_CHARS
    # (literal here to avoid a config<->core import cycle). 0 disables truncation.
    max_tool_result_chars: int = Field(ge=0, default=32_000)
    data_dir_awareness: bool = True
    send_name_header: bool = True
    memory: MemoryConfig | None = None
    sleep: SleepConfig | None = None
    telemetry: TelemetryConfig | None = None
    thinking: ThinkingConfig | None = None


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
    anthropic_base_url: str | None = None
    agent_api_key: str = ""
    agent_model: str = "claude-sonnet-4-6"
    agent_max_tokens: int = 4096
    agent_host: str = "0.0.0.0"
    agent_port: int = 8420
    agent_data_dir: Path = Path("./data")
    agent_skills_dir: Path | None = None
    agent_log_level: str = "INFO"
    agent_llm_backend: Literal["anthropic", "mock"] = "anthropic"
    agent_external_url: str | None = None
    agent_config: str | None = None
    agent_otel_endpoint: str | None = None
    agent_otel_protocol: str = "http"
    agent_otel_insecure: bool = True
    agent_otel_headers: str = ""
    agent_task_await_seconds: int = 60
    agent_tools_dir: Path | None = None

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
    def skills_dir(self) -> Path | None:
        """Filesystem root for runtime instruction-skills (Open Skills spec).

        Opt-in: when ``AGENT_SKILLS_DIR`` is unset, the agent does not scan
        anywhere. This matches ``AGENT_TOOLS_DIR`` so the two folder-scan
        env vars share one mental model.
        """
        return self.agent_skills_dir

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
            updates: dict[str, Any] = {
                "enabled": True,
                "endpoint": self.agent_otel_endpoint,
                "protocol": self.agent_otel_protocol,
                "insecure": self.agent_otel_insecure,
            }
            if self.agent_otel_headers:
                updates["headers"] = _parse_headers(self.agent_otel_headers)
            base = base.model_copy(update=updates)
        return base


def _parse_headers(raw: str) -> dict[str, str]:
    """Parse a comma-separated ``key=value`` string into a headers dict.

    Example: ``"Authorization=Bearer tok,X-Custom=val"``
    """
    headers: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            k, v = pair.split("=", 1)
            headers[k.strip()] = v.strip()
    return headers


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
