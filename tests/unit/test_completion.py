"""Tests for the extracted completion cycle."""

from __future__ import annotations

import pytest

from agentlings.core.completion import run_completion
from agentlings.core.llm import MockLLMClient
from agentlings.tools.registry import ToolRegistry


@pytest.fixture
def tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tools(["bash"])
    return registry


class TestRunCompletion:
    async def test_simple_text_response(self, tools: ToolRegistry) -> None:
        llm = MockLLMClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = await run_completion(llm, [], messages, tools)
        assert result.content[0]["type"] == "text"
        assert len(result.turns) == 1
        assert result.turns[0].tool_results == []

    async def test_tool_use_produces_multiple_turns(self, tools: ToolRegistry) -> None:
        llm = MockLLMClient(tool_names=["bash"])
        messages = [{"role": "user", "content": [{"type": "text", "text": "run bash please"}]}]
        result = await run_completion(llm, [], messages, tools)
        assert len(result.turns) >= 2
        assert result.turns[0].tool_results != []
        assert result.content[0]["type"] == "text"

    async def test_messages_mutated_in_place(self, tools: ToolRegistry) -> None:
        llm = MockLLMClient(tool_names=["bash"])
        messages = [{"role": "user", "content": [{"type": "text", "text": "run bash"}]}]
        original_len = len(messages)
        await run_completion(llm, [], messages, tools)
        assert len(messages) > original_len
