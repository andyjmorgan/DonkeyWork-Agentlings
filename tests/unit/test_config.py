from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.config import (
    AgentConfig,
    AgentDefinition,
    MemoryConfig,
    OAuthConfig,
    SkillConfig,
    SleepConfig,
    TelemetryConfig,
    ThinkingConfig,
)


def _clear_oauth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "AGENT_OAUTH_ISSUER",
        "AGENT_OAUTH_AUDIENCE",
        "AGENT_OAUTH_JWKS_URI",
        "AGENT_A2A_STREAMING",
        "AGENT_A2A_TOOL_PROGRESS_SUMMARIES",
    ):
        monkeypatch.delenv(var, raising=False)


def test_oauth_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_oauth_env(monkeypatch)
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        _env_file=None,
    )
    assert config.oauth_config is None


def test_oauth_from_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_oauth_env(monkeypatch)
    yaml_file = tmp_path / "agent.yaml"
    yaml_file.write_text(
        "name: secured\n"
        "description: Secured agent\n"
        "oauth:\n"
        "  enabled: true\n"
        "  issuer: https://auth.donkeywork.dev/realms/Agents\n"
        "  audience: donkeywork-agents-api\n"
    )
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        agent_config=str(yaml_file),
        _env_file=None,
    )
    oauth = config.oauth_config
    assert oauth is not None
    assert oauth.issuer == "https://auth.donkeywork.dev/realms/Agents"
    assert oauth.audience == "donkeywork-agents-api"
    assert oauth.jwks_uri is None
    assert oauth.algorithms == ["RS256"]


def test_oauth_yaml_disabled_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _clear_oauth_env(monkeypatch)
    yaml_file = tmp_path / "agent.yaml"
    yaml_file.write_text(
        "name: secured\n"
        "oauth:\n"
        "  enabled: false\n"
        "  issuer: https://auth.donkeywork.dev/realms/Agents\n"
    )
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        agent_config=str(yaml_file),
        _env_file=None,
    )
    assert config.oauth_config is None


def test_oauth_env_enables_and_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the issuer via env enables OAuth and overrides YAML values."""
    monkeypatch.setenv("AGENT_OAUTH_ISSUER", "https://env-issuer/realms/X")
    monkeypatch.setenv("AGENT_OAUTH_AUDIENCE", "env-audience")
    monkeypatch.setenv(
        "AGENT_OAUTH_JWKS_URI", "https://env-issuer/realms/X/protocol/openid-connect/certs"
    )
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        _env_file=None,
    )
    oauth = config.oauth_config
    assert oauth is not None
    assert oauth.enabled is True
    assert oauth.issuer == "https://env-issuer/realms/X"
    assert oauth.audience == "env-audience"
    assert oauth.jwks_uri.endswith("/protocol/openid-connect/certs")


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
    assert config.anthropic_base_url is None


def test_anthropic_base_url_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointing at Ollama is via env, mirroring how operators actually set it."""
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://192.168.69.21:11434")
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path / "data",
        _env_file=None,
    )
    assert config.anthropic_base_url == "http://192.168.69.21:11434"


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


def test_skills_and_tools_dirs_opt_in_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both folder-scan env vars share opt-in semantics: unset means no scan."""
    monkeypatch.delenv("AGENT_SKILLS_DIR", raising=False)
    monkeypatch.delenv("AGENT_TOOLS_DIR", raising=False)
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
        _env_file=None,
    )
    assert config.skills_dir is None
    assert config.agent_tools_dir is None


def test_skills_dir_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_SKILLS_DIR", str(tmp_path / "skills"))
    config = AgentConfig(
        anthropic_api_key="sk-test",
        agent_api_key="key",
        agent_data_dir=tmp_path,
        _env_file=None,
    )
    assert config.skills_dir == tmp_path / "skills"


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


class TestWireFormatConfig:
    """The AGENT_WIRE_FORMAT selector and the OPENAI_* endpoint settings."""

    def _base_kwargs(self, tmp_path: Path) -> dict:
        return {
            "agent_api_key": "key",
            "agent_data_dir": tmp_path / "data",
            # The responses wire format requires an explicit model.
            "agent_model": "gpt-5-mini",
            "_env_file": None,
        }

    def test_default_wire_format_is_messages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AGENT_WIRE_FORMAT", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-test", **self._base_kwargs(tmp_path),
        )
        assert config.agent_wire_format == "messages"

    def test_wire_format_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENT_WIRE_FORMAT", "responses")
        config = AgentConfig(
            anthropic_api_key="sk-test", **self._base_kwargs(tmp_path),
        )
        assert config.agent_wire_format == "responses"

    def test_invalid_wire_format_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            AgentConfig(
                anthropic_api_key="sk-test",
                agent_wire_format="grpc",  # type: ignore[arg-type]
                **self._base_kwargs(tmp_path),
            )

    def test_openai_settings_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.example")
        config = AgentConfig(
            anthropic_api_key="sk-test", **self._base_kwargs(tmp_path),
        )
        assert config.openai_api_key == "sk-openai"
        assert config.openai_base_url == "https://gateway.example"

    def test_llm_api_key_messages_uses_anthropic(self, tmp_path: Path) -> None:
        config = AgentConfig(
            anthropic_api_key="sk-anthropic",
            openai_api_key="sk-openai",
            **self._base_kwargs(tmp_path),
        )
        assert config.llm_api_key == "sk-anthropic"

    def test_llm_api_key_responses_prefers_openai(self, tmp_path: Path) -> None:
        config = AgentConfig(
            anthropic_api_key="sk-anthropic",
            openai_api_key="sk-openai",
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        assert config.llm_api_key == "sk-openai"

    def test_llm_api_key_fallback_requires_explicit_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One inbound key can serve both protocols — but ONLY with the
        explicit AGENT_SHARED_LLM_KEY opt-in plus a gateway URL.
        Credential routing is never inferred from DNS alone."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-shared",
            openai_base_url="https://gateway.example",
            agent_shared_llm_key=True,
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        assert config.llm_api_key == "sk-shared"

    def test_llm_api_key_no_fallback_without_opt_in(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A gateway URL alone (Azure OpenAI, a typo'd host, ...) must
        NOT route the Anthropic key anywhere."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-shared",
            openai_base_url="https://myco.openai.azure.com",
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        with pytest.raises(ValueError, match="AGENT_SHARED_LLM_KEY"):
            _ = config.llm_api_key

    def test_llm_api_key_responses_refuses_anthropic_key_for_openai(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without a base-URL override the fallback would send the
        Anthropic secret as a Bearer token to api.openai.com — the config
        must fail fast instead of leaking it."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-ant-secret",
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = config.llm_api_key

    def test_llm_base_url_per_wire_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-test",
            anthropic_base_url="http://ollama:11434",
            openai_base_url="https://gateway.example",
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        assert config.llm_base_url == "https://gateway.example"

    def test_llm_base_url_responses_does_not_fall_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ANTHROPIC_BASE_URL points at a Messages-shaped endpoint — it must
        never leak into the responses wire format."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-test",
            anthropic_base_url="http://ollama:11434",
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        assert config.llm_base_url is None

    def test_llm_base_url_messages_uses_anthropic(self, tmp_path: Path) -> None:
        config = AgentConfig(
            anthropic_api_key="sk-test",
            anthropic_base_url="http://ollama:11434",
            **self._base_kwargs(tmp_path),
        )
        assert config.llm_base_url == "http://ollama:11434"

    @pytest.mark.parametrize("openai_url", [
        "https://api.openai.com",
        "https://api.openai.com/v1",
        "https://api.openai.com/v1/responses",
        "https://openai.com",
        "https://eu.api.openai.com/v1",
        # FQDN trailing dot — same host to DNS, must classify identically.
        "http://api.openai.com./v1",
        "https://API.OPENAI.COM.",
    ])
    def test_fallback_refused_for_openai_hosts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, openai_url: str,
    ) -> None:
        """Even WITH the shared-key opt-in, any openai.com host form
        still refuses the Anthropic key."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-ant-secret",
            openai_base_url=openai_url,
            agent_shared_llm_key=True,
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            _ = config.llm_api_key

    def test_responses_requires_explicit_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The built-in Claude model default would be sent to a Responses
        endpoint that cannot serve it — selecting the responses wire
        format without an explicit AGENT_MODEL must fail fast."""
        monkeypatch.delenv("AGENT_MODEL", raising=False)
        kwargs = self._base_kwargs(tmp_path)
        kwargs.pop("agent_model")
        with pytest.raises(Exception, match="AGENT_MODEL"):
            AgentConfig(
                openai_api_key="sk-openai",
                agent_wire_format="responses",
                **kwargs,
            )

    def test_responses_model_from_env_counts_as_explicit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AGENT_MODEL", "gpt-5-mini")
        kwargs = self._base_kwargs(tmp_path)
        kwargs.pop("agent_model")
        config = AgentConfig(
            openai_api_key="sk-openai",
            agent_wire_format="responses",
            **kwargs,
        )
        assert config.agent_model == "gpt-5-mini"

    def test_schemeless_base_url_rejected_at_config_time(
        self, tmp_path: Path,
    ) -> None:
        """A scheme-less URL parses as a path (no hostname) — it would
        both break the HTTP client and bypass the openai.com fallback
        guard, so it is rejected outright."""
        with pytest.raises(Exception, match="OPENAI_BASE_URL"):
            AgentConfig(
                openai_api_key="sk-openai",
                openai_base_url="api.openai.com",
                agent_wire_format="responses",
                **self._base_kwargs(tmp_path),
            )

    def test_gateway_fallback_logs_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The opted-in ANTHROPIC_API_KEY fallback must be visible."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = AgentConfig(
            anthropic_api_key="sk-shared",
            openai_base_url="https://gateway.example",
            agent_shared_llm_key=True,
            agent_wire_format="responses",
            **self._base_kwargs(tmp_path),
        )
        with caplog.at_level("WARNING"):
            assert config.llm_api_key == "sk-shared"
        assert any(
            "gateway.example" in r.message and "ANTHROPIC_API_KEY" in r.message
            for r in caplog.records
        )

    def test_schemeless_url_ignored_on_messages_deployment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An OPENAI_BASE_URL exported for unrelated tooling must not
        abort a messages/mock deployment that never reads it."""
        monkeypatch.setenv("OPENAI_BASE_URL", "api.openai.com")
        config = AgentConfig(
            anthropic_api_key="sk-test",
            **self._base_kwargs(tmp_path),
        )
        assert config.agent_wire_format == "messages"
        mock_config = AgentConfig(
            agent_llm_backend="mock",
            agent_wire_format="responses",
            **{**self._base_kwargs(tmp_path), "agent_model": "gpt-5-mini"},
        )
        assert mock_config.llm_base_url is None

    def test_claude_sleep_model_warns_when_gateway_base_url(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-OpenAI base URL indicates a gateway that may serve
        Claude models over responses — warn, don't fail (mirroring the
        agent_model exemption)."""
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: claude-haiku-4-5\n"
        )
        with caplog.at_level("WARNING"):
            config = AgentConfig(
                openai_api_key="sk-openai",
                openai_base_url="https://gateway.example",
                agent_wire_format="responses",
                agent_config=str(agent_yaml),
                **self._base_kwargs(tmp_path),
            )
        assert config.sleep_config.model == "claude-haiku-4-5"
        assert any("sleep.model" in r.message for r in caplog.records)

    def test_claude_sleep_model_still_fails_against_openai_itself(
        self, tmp_path: Path,
    ) -> None:
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: claude-haiku-4-5\n"
        )
        with pytest.raises(Exception, match="sleep.model"):
            AgentConfig(
                openai_api_key="sk-openai",
                openai_base_url="https://api.openai.com/v1",
                agent_wire_format="responses",
                agent_config=str(agent_yaml),
                **self._base_kwargs(tmp_path),
            )

    def test_openai_destination_overrides_gateway_hints(
        self, tmp_path: Path,
    ) -> None:
        """An explicit api.openai.com base URL is definitive: no gateway
        hint (claude-* agent model, shared key) may soften the claude-*
        sleep.model rejection to a warning — OpenAI cannot serve it."""
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: claude-haiku-4-5\n"
        )
        kwargs = self._base_kwargs(tmp_path)
        kwargs["agent_model"] = "claude-sonnet-4-6"
        with pytest.raises(Exception, match="sleep.model"):
            AgentConfig(
                openai_api_key="sk-openai",
                openai_base_url="https://api.openai.com/v1",
                agent_wire_format="responses",
                agent_config=str(agent_yaml),
                **kwargs,
            )

    def test_claude_sleep_model_warns_when_agent_model_also_claude(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An explicitly claude-* AGENT_MODEL means the gateway serves
        Claude over responses — a claude-* sleep.model is then consistent
        and warned about, not rejected (same exemption as AGENT_MODEL)."""
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: claude-haiku-4-5\n"
        )
        kwargs = self._base_kwargs(tmp_path)
        kwargs["agent_model"] = "claude-sonnet-4-6"
        with caplog.at_level("WARNING"):
            config = AgentConfig(
                openai_api_key="sk-openai",
                agent_wire_format="responses",
                agent_config=str(agent_yaml),
                **kwargs,
            )
        assert config.sleep_config.model == "claude-haiku-4-5"
        assert any("sleep.model" in r.message for r in caplog.records)

    def test_claude_sleep_model_rejected_under_responses(
        self, tmp_path: Path,
    ) -> None:
        """A leftover claude-* sleep.model would burn one 400 per
        conversation every night — reject it at config time."""
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: claude-haiku-4-5\n"
        )
        with pytest.raises(Exception, match="sleep.model"):
            AgentConfig(
                openai_api_key="sk-openai",
                agent_wire_format="responses",
                agent_config=str(agent_yaml),
                **self._base_kwargs(tmp_path),
            )

    def test_responses_sleep_model_accepted(self, tmp_path: Path) -> None:
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: gpt-5-nano\n"
        )
        config = AgentConfig(
            openai_api_key="sk-openai",
            agent_wire_format="responses",
            agent_config=str(agent_yaml),
            **self._base_kwargs(tmp_path),
        )
        assert config.sleep_config.model == "gpt-5-nano"

    def test_claude_sleep_model_fine_on_messages_path(self, tmp_path: Path) -> None:
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: t\n"
            "sleep:\n"
            "  model: claude-haiku-4-5\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_config=str(agent_yaml),
            **self._base_kwargs(tmp_path),
        )
        assert config.sleep_config.model == "claude-haiku-4-5"

    def test_mock_backend_credentials_resolve_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """llm_api_key/llm_base_url are backend-aware: mock uses no
        credentials, so the responses key requirement never fires and
        call sites need no special-casing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
        kwargs = self._base_kwargs(tmp_path)
        kwargs.pop("agent_model")
        config = AgentConfig(
            agent_llm_backend="mock",
            agent_wire_format="responses",
            **kwargs,
        )
        assert config.llm_api_key == ""
        assert config.llm_base_url is None

    def test_mock_backend_exempt_from_model_requirement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AGENT_MODEL", raising=False)
        kwargs = self._base_kwargs(tmp_path)
        kwargs.pop("agent_model")
        config = AgentConfig(
            agent_llm_backend="mock",
            agent_wire_format="responses",
            **kwargs,
        )
        assert config.agent_wire_format == "responses"


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
        assert defn.a2a is None

    def test_a2a_config_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: a2a-agent\n"
            "description: A2A agent\n"
            "a2a:\n"
            "  streaming: true\n"
            "  tool_progress_summaries: true\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.a2a_config.streaming is True
        assert config.a2a_config.tool_progress_summaries is True

    def test_a2a_config_env_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: a2a-agent\n"
            "a2a:\n"
            "  streaming: false\n"
            "  tool_progress_summaries: false\n"
        )
        monkeypatch.setenv("AGENT_A2A_STREAMING", "true")
        monkeypatch.setenv("AGENT_A2A_TOOL_PROGRESS_SUMMARIES", "true")
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
            _env_file=None,
        )
        assert config.a2a_config.streaming is True
        assert config.a2a_config.tool_progress_summaries is True

    def test_bash_timeout_default(self) -> None:
        defn = AgentDefinition()
        assert defn.bash_timeout == 50

    def test_bash_timeout_custom(self) -> None:
        defn = AgentDefinition(bash_timeout=120)
        assert defn.bash_timeout == 120

    def test_bash_timeout_zero_rejected(self) -> None:
        with pytest.raises(Exception):
            AgentDefinition(bash_timeout=0)

    def test_bash_timeout_negative_rejected(self) -> None:
        with pytest.raises(Exception):
            AgentDefinition(bash_timeout=-1)


class TestThinkingConfig:
    def test_defaults(self) -> None:
        cfg = ThinkingConfig()
        assert cfg.mode == "off"
        assert cfg.budget_tokens == 8192
        assert cfg.interleaved is False
        assert cfg.effort is None
        assert cfg.display is None

    def test_budget_mode_with_interleaved(self) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=4096, interleaved=True)
        assert cfg.mode == "budget"
        assert cfg.interleaved is True

    def test_adaptive_with_effort_and_display(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort="medium", display="summarized")
        assert cfg.mode == "adaptive"
        assert cfg.effort == "medium"
        assert cfg.display == "summarized"

    def test_budget_tokens_below_min_rejected(self) -> None:
        with pytest.raises(Exception):
            ThinkingConfig(mode="budget", budget_tokens=512)

    def test_effort_only_valid_in_adaptive_mode(self) -> None:
        with pytest.raises(Exception, match="adaptive"):
            ThinkingConfig(mode="budget", effort="medium")

    def test_display_only_valid_in_adaptive_mode(self) -> None:
        with pytest.raises(Exception, match="adaptive"):
            ThinkingConfig(mode="off", display="summarized")

    def test_interleaved_only_valid_in_budget_mode(self) -> None:
        with pytest.raises(Exception, match="budget"):
            ThinkingConfig(mode="adaptive", interleaved=True)

    def test_invalid_effort_value_rejected(self) -> None:
        with pytest.raises(Exception):
            ThinkingConfig(mode="adaptive", effort="ultra")

    def test_agent_definition_thinking_optional(self) -> None:
        defn = AgentDefinition()
        assert defn.thinking is None
        defn2 = AgentDefinition(thinking=ThinkingConfig(mode="adaptive", effort="medium"))
        assert defn2.thinking is not None
        assert defn2.thinking.mode == "adaptive"


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
        assert config.enabled is True
        assert config.schedule == "0 2 * * *"
        assert config.journal_retention_days == 30
        assert config.conversation_retention_days == 14
        assert config.memory_max_entries == 50
        assert config.model is None
        assert config.summary_prompt is None
        assert config.consolidation_prompt is None

    def test_disabled_flag_parses(self) -> None:
        """Backends without the batches API (e.g. Ollama) flip this to false."""
        config = SleepConfig(enabled=False)
        assert config.enabled is False

    def test_batch_defaults_true(self) -> None:
        assert SleepConfig().batch is True

    def test_batch_false_parses_from_yaml(self, tmp_path: Path) -> None:
        """The batches workaround: keep the sleep cycle but run deep-sleep
        summaries as live calls (Ollama, the responses wire format)."""
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "sleep:\n"
            "  batch: false\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
            _env_file=None,
        )
        assert config.sleep_config is not None
        assert config.sleep_config.batch is False
        assert config.sleep_config.enabled is True

    def test_disabled_from_yaml(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "agent.yaml"
        yaml_file.write_text(
            "name: test\n"
            "description: test\n"
            "sleep:\n"
            "  enabled: false\n"
        )
        config = AgentConfig(
            anthropic_api_key="sk-test",
            agent_api_key="key",
            agent_data_dir=tmp_path / "data",
            agent_config=str(yaml_file),
        )
        assert config.sleep_config is not None
        assert config.sleep_config.enabled is False


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
