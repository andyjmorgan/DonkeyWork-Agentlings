"""Agent configuration from environment variables and optional YAML agent definition."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal


import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _parse_base_url_host(url: str) -> str | None:
    """Extract the normalised hostname from a base URL, or ``None``.

    ``None`` means the URL has no parseable ``scheme://host`` form (e.g.
    a scheme-less ``api.openai.com``, which urlparse reads as a path).
    The hostname is lowercased and FQDN-trailing-dot-stripped —
    ``api.openai.com.`` is the same host as ``api.openai.com`` to DNS
    and must classify identically.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return None
    return parsed.hostname.lower().rstrip(".")


def _is_openai_host(url: str) -> bool:
    """Whether a URL points at OpenAI itself (any host under openai.com)."""
    host = _parse_base_url_host(url)
    return host is not None and (
        host == "openai.com" or host.endswith(".openai.com")
    )


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
            the block is present.
        batch: When ``True`` (default), deep-sleep summaries are submitted to
            the Anthropic Message Batches API (50% cost, parallel, but may sit
            for up to the poll timeout). When ``False``, each summary is run
            as a sequential live ``complete()`` call instead — immediate,
            full price, and usable against backends without a batches API
            (Ollama's compatibility layer, the OpenAI Responses wire
            format). Backends that advertise no batch support degrade to
            the live path automatically even when this is ``True``. The
            ``model`` override below is honoured on both paths; combine
            with ``batch: false`` to run sleep on a cheaper/faster model
            than the agent uses for live turns.
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
    batch: bool = True
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


class OAuthConfig(BaseModel):
    """OAuth/OIDC bearer-token validation for the HTTP surface.

    The agentling acts purely as an OAuth 2.0 Resource Server: it validates
    the ``Authorization: Bearer`` JWT's signature against the issuer's
    published JWKS and checks ``iss``/``aud``/``exp``. It never issues tokens,
    inspects user identity, or enforces scopes — a validly-signed,
    correctly-audienced, unexpired token from the trusted issuer is accepted.

    Attributes:
        enabled: Whether bearer-token validation is active. When ``False`` the
            HTTP surface is protected by the API key alone.
        issuer: The token issuer URL, matched against the ``iss`` claim and
            advertised to clients (e.g. ``https://auth.example.com/realms/x``).
        audience: The resource identifier this server validates against the
            token's ``aud`` claim. A token whose ``aud`` (string or array)
            contains this value is accepted.
        jwks_uri: The issuer's JWKS endpoint. When ``None`` it is resolved at
            first use from the issuer's OIDC discovery document
            (``{issuer}/.well-known/openid-configuration``).
        algorithms: Permitted JWS signing algorithms.
        required_roles: Roles the token must carry (ALL of them) in addition to
            a valid signature and ``iss``/``aud``/``exp``. Empty (the default)
            preserves audience-only behaviour, so existing agentlings are
            unaffected. Roles are read from Keycloak-style ``realm_access.roles``
            and ``resource_access.<client>.roles``, plus a flat ``roles`` claim.
            This is how a deployment scopes access to specific principals
            (e.g. an operator role) without the framework hard-coding any policy.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    issuer: str = ""
    audience: str = ""
    jwks_uri: str | None = None
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    required_roles: list[str] = Field(default_factory=list)


class IconsConfig(BaseModel):
    """Icon URLs advertised on the MCP surface.

    Each value is the full icon address — an HTTPS URL or a ``data:`` URI —
    advertised as a single MCP ``Icon`` on the relevant surface. We send one
    icon per surface rather than a multi-resolution set because real clients
    (the MCP Inspector included) render every entry in the ``icons`` array
    side by side instead of selecting a best fit. A ``None`` value omits icons
    for that surface.

    Attributes:
        server: Icon URL for the MCP server (``serverInfo.icons``).
        spawn: Icon URL for the main spawn tool.
        task: Icon URL for the ``__get_task`` tool.
    """

    model_config = ConfigDict(extra="ignore")

    server: str | None = None
    spawn: str | None = None
    task: str | None = None


class A2AConfig(BaseModel):
    """A2A protocol surface configuration.

    Attributes:
        streaming: Advertise and enable A2A ``message/stream`` handling. The
            streaming path observes durable Agentlings tasks; client disconnects
            do not cancel task execution.
        tool_progress_summaries: When enabled, Agentlings may ask the model for
            short human-readable tool action summaries and stream them to A2A
            clients. This is intentionally separate from ``streaming`` so the
            backbone can exist before richer progress events.
    """

    model_config = ConfigDict(extra="ignore")

    streaming: bool = False
    tool_progress_summaries: bool = False


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
    oauth: OAuthConfig | None = None
    icons: IconsConfig | None = None
    a2a: A2AConfig | None = None


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
    openai_api_key: str = ""
    openai_base_url: str | None = None
    agent_shared_llm_key: bool = False
    agent_wire_format: Literal["messages", "responses"] = "messages"
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
    agent_oauth_issuer: str | None = None
    agent_oauth_audience: str | None = None
    agent_oauth_jwks_uri: str | None = None
    agent_oauth_required_roles: str | None = None  # comma-separated; overrides YAML
    agent_a2a_streaming: bool | None = None
    agent_a2a_tool_progress_summaries: bool | None = None

    _definition: AgentDefinition = AgentDefinition()

    @property
    def _uses_real_llm(self) -> bool:
        """Whether a real LLM backend is active (mock uses no
        credentials, no model, no endpoint). The single source of the
        mock exemption used by credential/model validation."""
        return self.agent_llm_backend != "mock"

    @model_validator(mode="after")
    def _init(self) -> AgentConfig:
        self.agent_data_dir.mkdir(parents=True, exist_ok=True)
        if self.agent_config:
            self._definition = _load_definition(self.agent_config)
        if self.agent_wire_format == "responses" and self._uses_real_llm:
            # Validation below is scoped to deployments that actually use
            # the responses wire format — an OPENAI_BASE_URL exported for
            # unrelated tooling must not abort a messages/mock deployment
            # that never reads it.
            if (
                self.openai_base_url
                and _parse_base_url_host(self.openai_base_url) is None
            ):
                # A scheme-less base URL parses as a path (no hostname),
                # which would both break the HTTP client and bypass the
                # openai.com fallback guard — reject it outright.
                raise ValueError(
                    f"OPENAI_BASE_URL ({self.openai_base_url!r}) is not a "
                    "valid http(s) URL with a hostname"
                )
            # The built-in AGENT_MODEL default is a Messages-format
            # Claude model that api.openai.com does not serve; selecting
            # the responses wire format without choosing a model
            # explicitly is a misconfiguration foot-gun, so it fails
            # fast.
            if "agent_model" not in self.model_fields_set:
                raise ValueError(
                    "AGENT_WIRE_FORMAT=responses requires AGENT_MODEL to "
                    "be set explicitly — the built-in default "
                    f"({self.agent_model!r}) is a Messages-format model "
                    "that a Responses endpoint will not serve"
                )
            # Same foot-gun for the sleep-cycle model override — but any
            # gateway indication (explicitly claude-* AGENT_MODEL, the
            # shared-key opt-in, or a non-OpenAI base URL) means the
            # endpoint may well serve Claude models over the responses
            # protocol, mirroring the agent_model exemption. Hard-fail
            # only when pointing at api.openai.com itself, which
            # certainly does not serve them.
            sleep = self._definition.sleep
            if sleep and sleep.model and sleep.model.startswith("claude-"):
                # An explicit api.openai.com destination is definitive — it
                # cannot serve Messages-format models, so no gateway hint
                # (shared key, claude agent model) may soften it to a warn.
                openai_destination = bool(
                    self.openai_base_url
                    and _is_openai_host(self.openai_base_url)
                )
                gateway_indicated = not openai_destination and (
                    self.agent_model.startswith("claude-")
                    or self.agent_shared_llm_key
                    or bool(self.openai_base_url)
                )
                if gateway_indicated:
                    logger.warning(
                        "sleep.model (%s) is a Claude model on the "
                        "responses wire format — assuming the configured "
                        "gateway serves it",
                        sleep.model,
                    )
                else:
                    raise ValueError(
                        f"sleep.model ({sleep.model!r}) is a "
                        "Messages-format model, which api.openai.com "
                        "will not serve — set a Responses-served model, "
                        "remove the override, or point OPENAI_BASE_URL "
                        "at a gateway that serves it"
                    )
        return self

    @property
    def definition(self) -> AgentDefinition:
        """The agent definition loaded from YAML (or defaults)."""
        return self._definition

    @property
    def llm_api_key(self) -> str:
        """The API key for the active wire format.

        On the ``responses`` wire format, ``OPENAI_API_KEY`` is used.
        Reusing ``ANTHROPIC_API_KEY`` (a gateway serving both protocols
        behind one inbound key) requires **explicit opt-in** via
        ``AGENT_SHARED_LLM_KEY=true`` — credential routing is never
        inferred from the URL alone: any host, including a typo'd one or
        a third-party provider (Azure OpenAI, ...), would otherwise
        receive the Anthropic secret as a Bearer token. Even with the
        opt-in, hosts under openai.com and unparseable base URLs are
        refused outright.

        The mock backend uses no credentials, so it always resolves to an
        empty key — irrelevant credential validation must never block a
        mock deployment.

        Raises:
            ValueError: When ``AGENT_WIRE_FORMAT=responses`` (non-mock)
                and no usable key is configured.
        """
        if not self._uses_real_llm:
            return ""
        if self.agent_wire_format == "responses":
            if self.openai_api_key:
                return self.openai_api_key
            gateway_host = (
                _parse_base_url_host(self.openai_base_url)
                if self.openai_base_url else None
            )
            if (
                self.agent_shared_llm_key
                and gateway_host is not None
                and not _is_openai_host(self.openai_base_url or "")
            ):
                logger.warning(
                    "OPENAI_API_KEY unset — AGENT_SHARED_LLM_KEY is set, "
                    "using ANTHROPIC_API_KEY for the responses wire "
                    "format against gateway host %s",
                    gateway_host,
                )
                return self.anthropic_api_key
            raise ValueError(
                "AGENT_WIRE_FORMAT=responses requires OPENAI_API_KEY. "
                "To reuse ANTHROPIC_API_KEY against a non-OpenAI gateway, "
                "set AGENT_SHARED_LLM_KEY=true and point OPENAI_BASE_URL "
                "at the gateway (openai.com hosts and unparseable URLs "
                "are always refused)"
            )
        return self.anthropic_api_key

    @property
    def llm_base_url(self) -> str | None:
        """The endpoint override for the active wire format.

        ``OPENAI_BASE_URL`` deliberately does not fall back to
        ``ANTHROPIC_BASE_URL`` — that variable points at a Messages-shaped
        endpoint, which would be wrong for the Responses wire format. The
        mock backend makes no HTTP calls, so it always resolves to
        ``None``.
        """
        if not self._uses_real_llm:
            return None
        if self.agent_wire_format == "responses":
            return self.openai_base_url
        return self.anthropic_base_url

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
    def oauth_config(self) -> OAuthConfig | None:
        """OAuth configuration, with env vars overriding YAML values.

        Returns ``None`` unless OAuth is enabled, either by an ``oauth`` block
        in the YAML definition with ``enabled: true`` or by setting
        ``AGENT_OAUTH_ISSUER`` (which implies enablement).
        """
        base = self._definition.oauth
        if self.agent_oauth_issuer:
            if base is None:
                base = OAuthConfig(enabled=True)
            updates: dict[str, Any] = {
                "enabled": True,
                "issuer": self.agent_oauth_issuer,
            }
            if self.agent_oauth_audience:
                updates["audience"] = self.agent_oauth_audience
            if self.agent_oauth_jwks_uri:
                updates["jwks_uri"] = self.agent_oauth_jwks_uri
            if self.agent_oauth_required_roles:
                updates["required_roles"] = [
                    r.strip()
                    for r in self.agent_oauth_required_roles.split(",")
                    if r.strip()
                ]
            base = base.model_copy(update=updates)
        if base is None or not base.enabled:
            return None
        return base

    @property
    def a2a_config(self) -> A2AConfig:
        """A2A surface configuration, with env vars overriding YAML values."""
        base = self._definition.a2a or A2AConfig()
        updates: dict[str, Any] = {}
        if self.agent_a2a_streaming is not None:
            updates["streaming"] = self.agent_a2a_streaming
        if self.agent_a2a_tool_progress_summaries is not None:
            updates["tool_progress_summaries"] = self.agent_a2a_tool_progress_summaries
        if updates:
            base = base.model_copy(update=updates)
        return base

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
