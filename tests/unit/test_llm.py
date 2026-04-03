from __future__ import annotations

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
