"""Unit tests for the MCP tool handler's defensive validation.

The MCP SDK client validates arguments against ``inputSchema`` before
sending them, so non-numeric or negative ``waitSeconds`` values never
reach the handler over the wire. These tests invoke the handler's
internal call_tool coroutine directly to verify the defense-in-depth
envelope shaping that kicks in if schema validation is ever bypassed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from a2a.types import AgentCard

from agentlings.config import AgentConfig
from agentlings.core.llm import MockLLMClient
from agentlings.core.loop import MessageLoop
from agentlings.core.store import JournalStore
from agentlings.protocol.agent_card import generate_agent_card
from agentlings.protocol.mcp import create_mcp_server
from agentlings.tools.registry import ToolRegistry


@pytest.fixture
def mcp_server(tmp_data_dir: Path, test_config: AgentConfig):
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    tools.register_tools(["bash"])
    llm = MockLLMClient(tool_names=tools.tool_names())
    loop = MessageLoop(config=test_config, store=store, llm=llm, tools=tools)
    agent_card: AgentCard = generate_agent_card(test_config)
    server = create_mcp_server(loop=loop, agent_card=agent_card, config=test_config)
    return server, agent_card


async def _invoke(server, agent_name: str, arguments: dict) -> dict:
    # The lowlevel MCP Server stores its registered handlers in
    # ``request_handlers`` via internal decorators. Reach the call_tool
    # coroutine via its registered entry for a direct, SDK-free invocation.
    from mcp.types import CallToolRequest, CallToolRequestParams

    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=agent_name, arguments=arguments),
    )
    result = await handler(request)
    text = result.root.content[0].text
    return json.loads(text)


class TestWaitSecondsDefensiveValidation:
    @pytest.mark.asyncio
    async def test_non_numeric_wait_seconds(self, mcp_server) -> None:
        server, card = mcp_server
        payload = await _invoke(
            server, card.name,
            {"taskId": "abc", "waitSeconds": "nope"},
        )
        assert payload.get("error") == "invalid_input"

    @pytest.mark.asyncio
    async def test_negative_wait_seconds(self, mcp_server) -> None:
        server, card = mcp_server
        payload = await _invoke(
            server, card.name,
            {"taskId": "abc", "waitSeconds": -1},
        )
        assert payload.get("error") == "invalid_input"

    @pytest.mark.asyncio
    async def test_null_wait_seconds_treated_as_zero(self, mcp_server) -> None:
        """``waitSeconds=None`` is valid and behaves as ``0``."""
        server, card = mcp_server
        payload = await _invoke(
            server, card.name,
            {"taskId": "nonexistent", "waitSeconds": None},
        )
        # waitSeconds passes validation; fails downstream as task_not_found.
        assert payload.get("error") == "task_not_found"

    @pytest.mark.asyncio
    async def test_valid_numeric_string_wait_seconds(self, mcp_server) -> None:
        """JSON numbers arrive as floats; string-numbers also cast cleanly."""
        server, card = mcp_server
        payload = await _invoke(
            server, card.name,
            {"taskId": "nonexistent", "waitSeconds": "2.5"},
        )
        # Valid float-string passes validation; fails as task_not_found.
        assert payload.get("error") == "task_not_found"
