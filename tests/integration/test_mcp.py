from __future__ import annotations

import pytest

from tests.integration.mcp_client import MCPTestClient


def _by_name(tools: list[dict], suffix: str = "") -> dict:
    for t in tools:
        if suffix:
            if t["name"].endswith(suffix):
                return t
        else:
            if not t["name"].endswith("__get_task"):
                return t
    raise AssertionError(f"no tool found matching suffix={suffix!r}")


class TestMCPToolsList:
    @pytest.mark.asyncio
    async def test_lists_two_tools(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        assert len(tools) == 2

    @pytest.mark.asyncio
    async def test_spawn_tool_name_matches_agent(
        self, mcp_client: MCPTestClient
    ) -> None:
        tools = await mcp_client.list_tools()
        spawn = _by_name(tools)
        assert spawn["name"] == "test-agent"

    @pytest.mark.asyncio
    async def test_get_task_tool_name(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        get_task = _by_name(tools, "__get_task")
        assert get_task["name"] == "test-agent__get_task"

    @pytest.mark.asyncio
    async def test_spawn_schema_shape(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        schema = _by_name(tools)["inputSchema"]
        props = schema["properties"]
        assert set(props.keys()) == {"message", "contextId"}
        assert schema.get("required", []) == ["message"]
        assert schema.get("additionalProperties") is False

    @pytest.mark.asyncio
    async def test_get_task_schema_shape(self, mcp_client: MCPTestClient) -> None:
        tools = await mcp_client.list_tools()
        schema = _by_name(tools, "__get_task")["inputSchema"]
        props = schema["properties"]
        assert set(props.keys()) == {"taskId", "contextId", "waitSeconds"}
        assert schema.get("required", []) == ["taskId"]
        assert schema.get("additionalProperties") is False


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
    """Tests that exercise the get_task tool against a completed task."""

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
        assert polled.raw.get("error") == "task_not_found"


class TestMCPInputValidation:
    """Schema-level validation comes from the MCP SDK's jsonschema pass.

    With the split into two tools, mutual-exclusion is enforced by shape:
    the spawn tool has no ``taskId`` field, the get_task tool has no
    ``message`` field. Both schemas declare ``additionalProperties: false``
    and required fields, so the SDK rejects malformed payloads before the
    handler runs. Schema rejections come back as ``isError=True`` with a
    plain-text ``"Input validation error"`` message — not the handler's
    JSON envelope.
    """

    @pytest.mark.asyncio
    async def test_spawn_rejects_extra_property(
        self, mcp_client: MCPTestClient
    ) -> None:
        is_error, text = await mcp_client.call_raw(
            {"message": "hi", "taskId": "some-task"}
        )
        assert is_error
        assert "validation" in text.lower()

    @pytest.mark.asyncio
    async def test_spawn_rejects_missing_message(
        self, mcp_client: MCPTestClient
    ) -> None:
        is_error, text = await mcp_client.call_raw({})
        assert is_error
        assert "validation" in text.lower()

    @pytest.mark.asyncio
    async def test_get_task_rejects_missing_task_id(
        self, mcp_client: MCPTestClient
    ) -> None:
        is_error, text = await mcp_client.call_raw({}, on_get_task=True)
        assert is_error
        assert "validation" in text.lower()

    @pytest.mark.asyncio
    async def test_get_task_rejects_extra_property(
        self, mcp_client: MCPTestClient
    ) -> None:
        is_error, text = await mcp_client.call_raw(
            {"taskId": "abc", "message": "no"}, on_get_task=True
        )
        assert is_error
        assert "validation" in text.lower()

    @pytest.mark.asyncio
    async def test_whitespace_only_message_rejected(
        self, mcp_client: MCPTestClient
    ) -> None:
        """Whitespace-only messages pass schema validation but fail at the
        engine; the handler maps ``InvalidTaskInputError`` onto an
        ``invalid_input`` JSON envelope."""
        result = await mcp_client.call_invalid({"message": "   \n\t  "})
        assert result.raw.get("error") == "invalid_input"
