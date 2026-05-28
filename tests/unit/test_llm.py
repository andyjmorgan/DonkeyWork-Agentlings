from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlings.config import INTERLEAVED_THINKING_BETA, ThinkingConfig
from agentlings.core.llm import (
    ANTHROPIC_BETA_HEADER,
    CONTEXT_ID_HEADER,
    NAME_HEADER,
    TASK_ID_HEADER,
    AnthropicLLMClient,
    MockLLMClient,
    _build_thinking_kwargs,
    create_llm_client,
)


class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        client = MockLLMClient()
        response = await client.complete(
            system=[{"type": "text", "text": "You are helpful."}],
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tools=[],
        )
        assert len(response.content) == 1
        assert response.content[0]["type"] == "text"
        assert "hello" in response.content[0]["text"]
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_use_response(self) -> None:
        client = MockLLMClient(tool_names=["shell"])
        response = await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "run shell command"}]}
            ],
            tools=[{"name": "shell", "description": "exec", "input_schema": {}}],
        )
        assert len(response.content) == 1
        assert response.content[0]["type"] == "tool_use"
        assert response.content[0]["name"] == "shell"
        assert response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_tool_result_gets_text_response(self) -> None:
        client = MockLLMClient(tool_names=["shell"])
        response = await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "run shell"}]},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "shell", "input": {}}]},
                {"role": "tool", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "output"}]},
            ],
            tools=[],
        )
        assert response.content[0]["type"] == "text"
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_no_tool_match_gives_text(self) -> None:
        client = MockLLMClient(tool_names=["kubectl"])
        response = await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "just chat"}]}
            ],
            tools=[],
        )
        assert response.content[0]["type"] == "text"


class TestAnthropicBatchResults:
    """Verify batch_results correctly awaits the SDK's coroutine before iterating.

    The Anthropic SDK's batches.results() returns a coroutine that resolves
    to an async iterator. Missing the await causes 'async for requires an
    object with __aiter__ method, got coroutine' — which silently killed the
    sleep cycle's journal write in production.
    """

    @pytest.mark.asyncio
    async def test_batch_results_awaits_before_iterating(self) -> None:
        """Simulate the real SDK pattern: results() is a coroutine returning an async iterator."""
        succeeded_entry = MagicMock()
        succeeded_entry.custom_id = "ctx-1"
        succeeded_entry.result.type = "succeeded"
        text_block = MagicMock()
        text_block.model_dump.return_value = {"type": "text", "text": "summary"}
        succeeded_entry.result.message.content = [text_block]

        failed_entry = MagicMock()
        failed_entry.custom_id = "ctx-2"
        failed_entry.result.type = "errored"
        failed_entry.result.error = "rate limit"

        async def _async_iter():
            for entry in [succeeded_entry, failed_entry]:
                yield entry

        mock_client = MagicMock()
        mock_client.messages.batches.results = AsyncMock(return_value=_async_iter())

        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client._client = mock_client

        results = await client.batch_results("batch-123")

        mock_client.messages.batches.results.assert_awaited_once_with("batch-123")
        assert len(results) == 2
        assert results[0].custom_id == "ctx-1"
        assert results[0].status == "succeeded"
        assert results[0].content == [{"type": "text", "text": "summary"}]
        assert results[1].custom_id == "ctx-2"
        assert results[1].status == "failed"
        assert results[1].error == "rate limit"


class TestContextIdHeader:
    """The ``x-agentling-context-id`` header lets a backend correlate the
    fan-out of Messages calls one task makes back to a single session.
    """

    def _stub_client(self) -> tuple[AnthropicLLMClient, AsyncMock]:
        response = MagicMock()
        response.content = []
        response.stop_reason = "end_turn"
        response.usage = None
        create = AsyncMock(return_value=response)
        mock_client = MagicMock()
        mock_client.messages.create = create

        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client._client = mock_client
        client._model = "claude-sonnet-4-6"
        client._max_tokens = 128
        return client, create

    @pytest.mark.asyncio
    async def test_context_and_task_id_forwarded_as_headers(self) -> None:
        client, create = self._stub_client()

        await client.complete(
            system=[], messages=[], tools=[],
            context_id="ctx-abc", task_id="task-123",
        )

        create.assert_awaited_once()
        extra_headers = create.await_args.kwargs["extra_headers"]
        assert extra_headers == {
            CONTEXT_ID_HEADER: "ctx-abc",
            TASK_ID_HEADER: "task-123",
        }

    @pytest.mark.asyncio
    async def test_only_provided_ids_become_headers(self) -> None:
        client, create = self._stub_client()

        await client.complete(system=[], messages=[], tools=[], task_id="task-only")

        extra_headers = create.await_args.kwargs["extra_headers"]
        assert extra_headers == {TASK_ID_HEADER: "task-only"}

    @pytest.mark.asyncio
    async def test_no_header_when_ids_absent(self) -> None:
        client, create = self._stub_client()

        await client.complete(system=[], messages=[], tools=[])

        assert "extra_headers" not in create.await_args.kwargs

    @pytest.mark.asyncio
    async def test_mock_records_last_context_and_task_id(self) -> None:
        client = MockLLMClient()
        await client.complete(
            system=[],
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=[],
            context_id="ctx-xyz",
            task_id="task-xyz",
        )
        assert client.last_context_id == "ctx-xyz"
        assert client.last_task_id == "task-xyz"


class TestNameHeader:
    """The static ``x-agentling-name`` header is baked into the client's
    default headers once, so every request attributes traffic to the agentling
    without per-call threading.
    """

    def test_name_set_as_default_header(self) -> None:
        client = create_llm_client(
            backend="anthropic", api_key="sk-test", agent_name="office-k3s",
        )
        assert isinstance(client, AnthropicLLMClient)
        assert client._client.default_headers[NAME_HEADER] == "office-k3s"

    def test_no_name_header_when_unset(self) -> None:
        client = create_llm_client(backend="anthropic", api_key="sk-test")
        assert isinstance(client, AnthropicLLMClient)
        assert NAME_HEADER not in client._client.default_headers


class TestFactory:
    def test_mock_backend(self) -> None:
        client = create_llm_client(backend="mock")
        assert isinstance(client, MockLLMClient)

    def test_anthropic_backend(self) -> None:
        from agentlings.core.llm import AnthropicLLMClient

        client = create_llm_client(backend="anthropic", api_key="sk-test")
        assert isinstance(client, AnthropicLLMClient)

    def test_anthropic_backend_with_base_url(self) -> None:
        """``base_url`` reaches the underlying Anthropic SDK client.

        This is the core wiring for Ollama compatibility — if it regresses,
        every Ollama request silently goes back to api.anthropic.com.
        """
        from agentlings.core.llm import AnthropicLLMClient

        client = create_llm_client(
            backend="anthropic",
            api_key="ollama",
            model="qwen3:4b",
            base_url="http://localhost:11434",
        )
        assert isinstance(client, AnthropicLLMClient)
        assert str(client._client.base_url).rstrip("/") == "http://localhost:11434"

    def test_anthropic_backend_empty_api_key_with_base_url(self) -> None:
        """An empty api_key + base_url should fall back to a placeholder so
        the Anthropic SDK doesn't refuse to construct. Backends that ignore
        the header (Ollama) accept this happily.
        """
        from agentlings.core.llm import AnthropicLLMClient

        client = AnthropicLLMClient(
            api_key="",
            model="qwen3:4b",
            max_tokens=128,
            base_url="http://localhost:11434",
        )
        assert client._client.api_key == "unset"


class TestThinkingHelper:
    """``_build_thinking_kwargs`` translates ThinkingConfig → API kwargs.

    These cover the three modes plus the budget>=max_tokens self-protection
    path, without booting an HTTP client.
    """

    def test_off_returns_none_triple(self) -> None:
        block, output, beta = _build_thinking_kwargs(None, max_tokens=4096)
        assert (block, output, beta) == (None, None, None)
        block, output, beta = _build_thinking_kwargs(
            ThinkingConfig(mode="off"), max_tokens=4096,
        )
        assert (block, output, beta) == (None, None, None)

    def test_budget_mode_emits_enabled_block(self) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=2048)
        block, output, beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block == {"type": "enabled", "budget_tokens": 2048}
        assert output is None
        assert beta is None

    def test_budget_mode_interleaved_adds_beta_header(self) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=2048, interleaved=True)
        block, _output, beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block == {"type": "enabled", "budget_tokens": 2048}
        assert beta == INTERLEAVED_THINKING_BETA

    def test_budget_over_max_tokens_drops_block(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=8192)
        with caplog.at_level("WARNING"):
            block, _output, _beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block is None
        assert any("budget_tokens" in r.message for r in caplog.records)

    def test_interleaved_budget_may_exceed_max_tokens(self) -> None:
        # With interleaved, the budget is a per-turn total across all
        # thinking blocks and may legally exceed max_tokens.
        cfg = ThinkingConfig(mode="budget", budget_tokens=8192, interleaved=True)
        block, _output, beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block == {"type": "enabled", "budget_tokens": 8192}
        assert beta == INTERLEAVED_THINKING_BETA

    def test_adaptive_mode_emits_adaptive_block_with_effort(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort="medium")
        block, output, beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block == {"type": "adaptive"}
        assert output == {"effort": "medium"}
        assert beta is None

    def test_adaptive_mode_emits_display(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", display="summarized", effort="low")
        block, _output, _beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block == {"type": "adaptive", "display": "summarized"}

    def test_adaptive_without_effort_omits_output_addition(self) -> None:
        cfg = ThinkingConfig(mode="adaptive")
        block, output, _beta = _build_thinking_kwargs(cfg, max_tokens=4096)
        assert block == {"type": "adaptive"}
        assert output is None


class TestAnthropicClientThinkingIntegration:
    """The client surfaces thinking via the right kwargs + headers."""

    def _stub_client(self, thinking: ThinkingConfig | None = None) -> tuple[AnthropicLLMClient, AsyncMock]:
        response = MagicMock()
        response.content = []
        response.stop_reason = "end_turn"
        response.usage = None
        create = AsyncMock(return_value=response)
        mock_client = MagicMock()
        mock_client.messages.create = create
        client = AnthropicLLMClient.__new__(AnthropicLLMClient)
        client._client = mock_client
        client._model = "claude-sonnet-4-6"
        client._max_tokens = 4096
        client._thinking = thinking
        return client, create

    @pytest.mark.asyncio
    async def test_adaptive_thinking_appears_in_kwargs(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort="medium")
        client, create = self._stub_client(thinking=cfg)
        await client.complete(system=[], messages=[], tools=[])
        kwargs = create.await_args.kwargs
        assert kwargs["thinking"] == {"type": "adaptive"}
        assert kwargs["output_config"] == {"effort": "medium"}
        assert "extra_headers" not in kwargs or ANTHROPIC_BETA_HEADER not in kwargs.get(
            "extra_headers", {}
        )

    @pytest.mark.asyncio
    async def test_budget_interleaved_emits_block_plus_beta_header(self) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=2048, interleaved=True)
        client, create = self._stub_client(thinking=cfg)
        await client.complete(system=[], messages=[], tools=[])
        kwargs = create.await_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert kwargs["extra_headers"][ANTHROPIC_BETA_HEADER] == INTERLEAVED_THINKING_BETA

    @pytest.mark.asyncio
    async def test_no_thinking_when_unconfigured(self) -> None:
        client, create = self._stub_client(thinking=None)
        await client.complete(system=[], messages=[], tools=[])
        kwargs = create.await_args.kwargs
        assert "thinking" not in kwargs

    @pytest.mark.asyncio
    async def test_existing_output_schema_merges_with_effort(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort="high")
        client, create = self._stub_client(thinking=cfg)
        await client.complete(
            system=[], messages=[], tools=[],
            output_schema={"type": "object"},
        )
        oc = create.await_args.kwargs["output_config"]
        assert oc["format"]["type"] == "json_schema"
        assert oc["effort"] == "high"


class TestThinkingModelMismatchWarning:
    """The startup warning must not false-positive on adaptive-capable models.

    Regression for the 0.9.0 release where `claude-sonnet-4-` startswith-matched
    `claude-sonnet-4-6` (which IS adaptive-capable) and warned spuriously.
    """

    def test_adaptive_on_sonnet_4_6_does_not_warn(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from agentlings.core.llm import _warn_if_mode_mismatches_model

        with caplog.at_level("WARNING"):
            _warn_if_mode_mismatches_model("adaptive", "claude-sonnet-4-6")
        assert not any("does not support adaptive" in r.message for r in caplog.records)

    def test_adaptive_on_sonnet_4_5_does_warn(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from agentlings.core.llm import _warn_if_mode_mismatches_model

        with caplog.at_level("WARNING"):
            _warn_if_mode_mismatches_model("adaptive", "claude-sonnet-4-5")
        assert any("does not support adaptive" in r.message for r in caplog.records)

    def test_budget_on_opus_4_7_does_warn(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from agentlings.core.llm import _warn_if_mode_mismatches_model

        with caplog.at_level("WARNING"):
            _warn_if_mode_mismatches_model("budget", "claude-opus-4-7")
        assert any("rejects" in r.message for r in caplog.records)

    def test_adaptive_on_opus_4_6_does_not_warn(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from agentlings.core.llm import _warn_if_mode_mismatches_model

        with caplog.at_level("WARNING"):
            _warn_if_mode_mismatches_model("adaptive", "claude-opus-4-6")
        assert not any("does not support adaptive" in r.message for r in caplog.records)


class TestThinkingFactory:
    """The factory threads thinking through to whichever client it builds."""

    def test_mock_backend_records_thinking(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort="medium")
        client = create_llm_client(backend="mock", thinking=cfg)
        assert isinstance(client, MockLLMClient)
        assert client.thinking_config == cfg

    def test_mock_backend_drops_off_mode(self) -> None:
        client = create_llm_client(backend="mock", thinking=ThinkingConfig(mode="off"))
        assert client.thinking_config is None  # type: ignore[union-attr]

    def test_anthropic_backend_threads_thinking(self) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=2048)
        client = create_llm_client(
            backend="anthropic", api_key="k", model="claude-sonnet-4-5",
            max_tokens=4096, thinking=cfg,
        )
        assert client._thinking == cfg  # type: ignore[union-attr]


class TestMockLLMDelay:
    """``delay-N`` markers in user messages produce genuine async sleeps."""

    @pytest.mark.asyncio
    async def test_delay_marker_sleeps_requested_seconds(self) -> None:
        client = MockLLMClient()
        start = time.monotonic()
        response = await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "delay-1 hi"}]}
            ],
            tools=[],
        )
        elapsed = time.monotonic() - start
        assert elapsed >= 0.9, f"expected ~1s delay, got {elapsed:.2f}s"
        assert elapsed < 2.0, f"delay should be bounded, got {elapsed:.2f}s"
        assert response.content[0]["type"] == "text"
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_fractional_delay_supported(self) -> None:
        client = MockLLMClient()
        start = time.monotonic()
        await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "delay-0.5 hi"}]}
            ],
            tools=[],
        )
        elapsed = time.monotonic() - start
        assert 0.4 <= elapsed < 1.5, f"expected ~0.5s delay, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_no_delay_without_marker(self) -> None:
        client = MockLLMClient()
        start = time.monotonic()
        await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "hello there"}]}
            ],
            tools=[],
        )
        elapsed = time.monotonic() - start
        assert elapsed < 0.1, f"no marker should mean no sleep, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_delay_skipped_on_tool_result_loop(self) -> None:
        """Tool-result turns must not re-trip the sleep each iteration."""
        client = MockLLMClient(tool_names=["shell"])
        # Simulate a tool-result message that still has the delay marker
        # somewhere in prior history text but the *last* message is a
        # tool_result.
        start = time.monotonic()
        await client.complete(
            system=[],
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "delay-5 run shell"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "shell", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "delay-5 output"},
                ]},
            ],
            tools=[],
        )
        elapsed = time.monotonic() - start
        # The last turn is a tool_result — must NOT sleep regardless of text.
        assert elapsed < 0.1, f"tool-result turn should skip delay, got {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_delay_cancellable(self) -> None:
        """Awaiting a long delay is cooperative — task.cancel() wakes it."""
        import asyncio

        client = MockLLMClient()

        async def _run() -> None:
            await client.complete(
                system=[],
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "delay-60 hi"}]}
                ],
                tools=[],
            )

        task = asyncio.create_task(_run())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
