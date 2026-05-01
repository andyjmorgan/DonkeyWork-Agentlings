from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)


@dataclass
class MCPToolCallResult:
    """Parsed response from a tool call.

    Attributes:
        context_id: Parent context ID, always present.
        message: Final response text (empty when status != completed).
        status: ``completed``, ``working``, ``failed``, ``cancelled``, or
            ``error`` (app-level error envelope).
        task_id: The task this call refers to, if any.
        raw: The raw decoded JSON payload for tests that need every field.
    """

    context_id: str | None
    message: str
    status: str
    task_id: str | None
    raw: Any


class MCPTestClient:
    def __init__(self, base_url: str, api_key: str = "") -> None:
        self._url = f"{base_url}/mcp"
        self._headers = {"X-API-Key": api_key} if api_key else {}

    async def list_tools(self) -> list[dict[str, Any]]:
        async with streamablehttp_client(
            self._url, headers=self._headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": t.inputSchema,
                    }
                    for t in result.tools
                ]

    async def _spawn_tool_name(self, session: ClientSession) -> str:
        tools = await session.list_tools()
        for t in tools.tools:
            if not t.name.endswith("__get_task"):
                return t.name
        raise RuntimeError("no spawn tool found")

    async def _get_task_tool_name(self, session: ClientSession) -> str:
        tools = await session.list_tools()
        for t in tools.tools:
            if t.name.endswith("__get_task"):
                return t.name
        raise RuntimeError("no get_task tool found")

    async def _call_named(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolCallResult:
        async with streamablehttp_client(
            self._url, headers=self._headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                text_content = result.content[0].text  # type: ignore[attr-defined]
                parsed = json.loads(text_content)
                return MCPToolCallResult(
                    context_id=parsed.get("contextId"),
                    message=parsed.get("message", ""),
                    status=parsed.get("status", parsed.get("error", "")),
                    task_id=parsed.get("taskId") or parsed.get("activeTaskId"),
                    raw=parsed,
                )

    async def _call_spawn(self, arguments: dict[str, Any]) -> MCPToolCallResult:
        async with streamablehttp_client(
            self._url, headers=self._headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_name = await self._spawn_tool_name(session)
                result = await session.call_tool(tool_name, arguments)
                text_content = result.content[0].text  # type: ignore[attr-defined]
                parsed = json.loads(text_content)
                return MCPToolCallResult(
                    context_id=parsed.get("contextId"),
                    message=parsed.get("message", ""),
                    status=parsed.get("status", parsed.get("error", "")),
                    task_id=parsed.get("taskId") or parsed.get("activeTaskId"),
                    raw=parsed,
                )

    async def _call_get_task(self, arguments: dict[str, Any]) -> MCPToolCallResult:
        async with streamablehttp_client(
            self._url, headers=self._headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_name = await self._get_task_tool_name(session)
                result = await session.call_tool(tool_name, arguments)
                text_content = result.content[0].text  # type: ignore[attr-defined]
                parsed = json.loads(text_content)
                return MCPToolCallResult(
                    context_id=parsed.get("contextId"),
                    message=parsed.get("message", ""),
                    status=parsed.get("status", parsed.get("error", "")),
                    task_id=parsed.get("taskId") or parsed.get("activeTaskId"),
                    raw=parsed,
                )

    async def call_tool(
        self, message: str, context_id: str | None = None
    ) -> MCPToolCallResult:
        """Send a new message and wait for completion or a task handle."""
        args: dict[str, Any] = {"message": message}
        if context_id is not None:
            args["contextId"] = context_id
        return await self._call_spawn(args)

    async def poll_task(
        self,
        task_id: str,
        context_id: str | None = None,
        wait_seconds: float = 0.0,
    ) -> MCPToolCallResult:
        """Poll an existing task for its current state via the get_task tool."""
        args: dict[str, Any] = {"taskId": task_id, "waitSeconds": wait_seconds}
        if context_id is not None:
            args["contextId"] = context_id
        return await self._call_get_task(args)

    async def call_invalid(
        self,
        arguments: dict[str, Any],
        *,
        on_get_task: bool = False,
    ) -> MCPToolCallResult:
        """Issue an arbitrary arguments payload that the handler will shape
        as a JSON envelope (engine-level errors only)."""
        if on_get_task:
            return await self._call_get_task(arguments)
        return await self._call_spawn(arguments)

    async def call_raw(
        self,
        arguments: dict[str, Any],
        *,
        on_get_task: bool = False,
    ) -> tuple[bool, str]:
        """Issue a raw call and return (isError, content_text).

        For asserting SDK-level schema validation responses, which are plain
        text (not the handler's JSON envelope) and would otherwise fail to
        parse via ``_call_named``.
        """
        async with streamablehttp_client(
            self._url, headers=self._headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_name = (
                    await self._get_task_tool_name(session)
                    if on_get_task
                    else await self._spawn_tool_name(session)
                )
                result = await session.call_tool(tool_name, arguments)
                text = result.content[0].text  # type: ignore[attr-defined]
                return bool(result.isError), text
