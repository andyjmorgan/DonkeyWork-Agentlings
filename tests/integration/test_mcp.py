from __future__ import annotations

import pytest

from tests.integration.mcp_client import MCPTestClient


class TestMCPToolsList:
    @pytest.mark.asyncio
    async def test_lists_single_tool(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_tool_name_matches_agent(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        assert tools[0]["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_tool_schema_has_message(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        schema = tools[0]["inputSchema"]
        assert "message" in schema["properties"]
        assert "message" in schema["required"]

    @pytest.mark.asyncio
    async def test_tool_schema_has_optional_context_id(
        self, mcp_client: MCPTestClient
    ) -> None:
        tools = await mcp_client.list_tools()
        schema = tools[0]["inputSchema"]
        assert "contextId" in schema["properties"]
        assert "contextId" not in schema.get("required", [])


class TestMCPToolCall:
    @pytest.mark.asyncio
    async def test_call_returns_context_id(self, mcp_client: MCPTestClient) -> None:
        result = await mcp_client.call_tool("hello")
        assert result.context_id is not None
        assert len(result.context_id) > 0

    @pytest.mark.asyncio
    async def test_call_returns_message(self, mcp_client: MCPTestClient) -> None:
        result = await mcp_client.call_tool("hello")
        assert len(result.message) > 0

    @pytest.mark.asyncio
    async def test_continuation_with_context_id(
        self, mcp_client: MCPTestClient
    ) -> None:
        r1 = await mcp_client.call_tool("hello")
        r2 = await mcp_client.call_tool("follow up", context_id=r1.context_id)
        assert r2.context_id == r1.context_id
