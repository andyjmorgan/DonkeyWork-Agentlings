from __future__ import annotations

import pytest

from agentlings.core.llm import MockLLMClient, create_llm_client


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


class TestFactory:
    def test_mock_backend(self) -> None:
        client = create_llm_client(backend="mock")
        assert isinstance(client, MockLLMClient)

    def test_anthropic_backend(self) -> None:
        from agentlings.core.llm import AnthropicLLMClient

        client = create_llm_client(backend="anthropic", api_key="sk-test")
        assert isinstance(client, AnthropicLLMClient)
