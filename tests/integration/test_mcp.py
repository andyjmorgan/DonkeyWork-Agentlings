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
    async def test_tool_schema_has_task_fields(
        self, mcp_client: MCPTestClient
    ) -> None:
        tools = await mcp_client.list_tools()
        schema = tools[0]["inputSchema"]
        props = schema["properties"]
        assert "message" in props
        assert "taskId" in props
        assert "contextId" in props
        assert "waitSeconds" in props
        # v2: nothing is required; server validates mutual exclusion at call time.
        assert schema.get("required", []) == []

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
    async def test_call_returns_completed_status(
        self, mcp_client: MCPTestClient
    ) -> None:
        result = await mcp_client.call_tool("hello")
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_call_returns_task_id(self, mcp_client: MCPTestClient) -> None:
        result = await mcp_client.call_tool("hello")
        assert result.task_id is not None
        assert len(result.task_id) > 0

    @pytest.mark.asyncio
    async def test_continuation_with_context_id(
        self, mcp_client: MCPTestClient
    ) -> None:
        r1 = await mcp_client.call_tool("hello")
        r2 = await mcp_client.call_tool("follow up", context_id=r1.context_id)
        assert r2.context_id == r1.context_id


class TestMCPTaskPolling:
    """Tests that exercise the poll path with a completed task."""

    @pytest.mark.asyncio
    async def test_poll_completed_task_returns_final_response(
        self, mcp_client: MCPTestClient
    ) -> None:
        created = await mcp_client.call_tool("hello")
        polled = await mcp_client.poll_task(
            task_id=created.task_id,  # type: ignore[arg-type]
            context_id=created.context_id,
        )
        assert polled.status == "completed"
        assert polled.message == created.message
        assert polled.task_id == created.task_id

    @pytest.mark.asyncio
    async def test_poll_without_context_id_still_works(
        self, mcp_client: MCPTestClient
    ) -> None:
        created = await mcp_client.call_tool("hello")
        polled = await mcp_client.poll_task(created.task_id)  # type: ignore[arg-type]
        assert polled.status == "completed"

    @pytest.mark.asyncio
    async def test_poll_unknown_task_returns_error(
        self, mcp_client: MCPTestClient
    ) -> None:
        polled = await mcp_client.poll_task("nonexistent-task-id")
        # Error envelope uses raw["error"] and our dataclass maps it to status.
        assert polled.raw.get("error") == "task_not_found"


class TestMCPInputValidation:
    @pytest.mark.asyncio
    async def test_missing_message_and_task_id(
        self, mcp_client: MCPTestClient
    ) -> None:
        result = await mcp_client.call_invalid({})
        assert result.raw.get("error") == "invalid_input"

    @pytest.mark.asyncio
    async def test_both_message_and_task_id(
        self, mcp_client: MCPTestClient
    ) -> None:
        result = await mcp_client.call_invalid(
            {"message": "hi", "taskId": "some-task"}
        )
        assert result.raw.get("error") == "invalid_input"

    @pytest.mark.asyncio
    async def test_whitespace_only_message_rejected(
        self, mcp_client: MCPTestClient
    ) -> None:
        """Engine-level validation must be caught and shaped cleanly.

        Regression: when whitespace-only messages slipped past the
        handler's ``message == ""`` literal check, the ``engine.spawn()``
        raised ``InvalidTaskInputError`` which the handler attempted to
        match in an ``except`` clause — but the name was not imported,
        causing a ``NameError`` to propagate instead of a clean envelope.
        """
        result = await mcp_client.call_invalid({"message": "   \n\t  "})
        assert result.raw.get("error") == "invalid_input"
