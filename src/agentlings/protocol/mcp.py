"""MCP protocol server exposing the agentling as a single tool."""

from __future__ import annotations

import json
import logging
from typing import Any

from a2a.types import AgentCard
from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool

from agentlings.core.loop import MessageLoop

logger = logging.getLogger(__name__)


def create_mcp_server(
    loop: MessageLoop,
    agent_card: AgentCard,
) -> Server:
    """Create an MCP server with a single tool derived from the Agent Card.

    The tool accepts a natural language message and an optional context ID
    for multi-turn conversations, delegating to the shared message loop.

    Args:
        loop: The message loop to forward requests to.
        agent_card: The agent card whose name and description define the tool.

    Returns:
        A configured MCP ``Server`` instance.
    """
    server = Server(agent_card.name)

    tool = Tool(
        name=agent_card.name,
        description=(
            f"{agent_card.description}. Returns a contextId for multi-turn "
            "conversations — pass it back to continue."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Natural language request to the agent",
                },
                "contextId": {
                    "type": "string",
                    "description": (
                        "Context ID from a previous response. Not required for "
                        "the first message. Include to continue an existing conversation."
                    ),
                },
            },
            "required": ["message"],
        },
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [tool]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name != agent_card.name:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        message = arguments.get("message", "")
        context_id = arguments.get("contextId")

        result = await loop.process_message(
            text=message,
            context_id=context_id,
            stream=False,
            via="mcp",
        )

        response_text = _extract_text(result.content)
        response_data = {
            "contextId": result.context_id,
            "message": response_text,
        }

        return [TextContent(type="text", text=json.dumps(response_data))]

    return server


def _extract_text(content: list[dict[str, Any]]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
