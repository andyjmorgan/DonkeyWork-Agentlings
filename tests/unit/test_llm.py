from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentlings.core.llm import AnthropicLLMClient, MockLLMClient, create_llm_client


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
