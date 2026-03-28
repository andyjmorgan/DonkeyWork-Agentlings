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
    context_id: str | None
    message: str
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

    async def call_tool(
        self, message: str, context_id: str | None = None
    ) -> MCPToolCallResult:
        async with streamablehttp_client(
            self._url, headers=self._headers
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                tool_name = tools.tools[0].name

                args: dict[str, Any] = {"message": message}
                if context_id is not None:
                    args["contextId"] = context_id

                result = await session.call_tool(tool_name, args)

                text_content = result.content[0].text  # type: ignore[attr-defined]
                parsed = json.loads(text_content)

                return MCPToolCallResult(
                    context_id=parsed.get("contextId"),
                    message=parsed.get("message", ""),
                    raw=parsed,
                )
