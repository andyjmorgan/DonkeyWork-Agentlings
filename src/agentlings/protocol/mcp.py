"""MCP protocol server exposing the agentling as a single task-aware tool.

The tool has four input fields:

- ``contextId`` (optional) — which conversation to append to or continue.
- ``message`` (optional) — a new request to the agent.
- ``taskId`` (optional) — an existing task to poll.
- ``waitSeconds`` (optional) — when polling, how long to block for completion.

``message`` and ``taskId`` are mutually exclusive. Sending ``message`` starts a
task; sending ``taskId`` polls an existing one. In either case the response is
a ``CallToolResult`` whose ``structuredContent`` tells the caller whether the
task completed (and the final response) or is still working (with the taskId
to poll later).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from a2a.types import AgentCard
from mcp.server.lowlevel import Server
from mcp.types import TextContent, Tool

from agentlings.config import AgentConfig
from agentlings.core.loop import MessageLoop
from agentlings.core.task import (
    ContextBusyError,
    InvalidTaskInputError,
    TaskContextMismatchError,
    TaskNotFoundError,
    TaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)


def create_mcp_server(
    loop: MessageLoop,
    agent_card: AgentCard,
    config: AgentConfig,
) -> Server:
    """Create an MCP server with a single task-aware tool.

    Args:
        loop: The message loop exposing the shared task engine.
        agent_card: The agent card whose name and description define the tool.
        config: Agent configuration (for the await timeout cap).

    Returns:
        A configured MCP ``Server`` instance.
    """
    server = Server(agent_card.name)
    engine = loop.engine
    await_seconds = float(getattr(config, "agent_task_await_seconds", 60))

    tool = Tool(
        name=agent_card.name,
        description=(
            f"{agent_card.description}\n\n"
            "Sends a message to the agent (set `message`) or polls an existing "
            "task (set `taskId`). The response's `structuredContent.status` is "
            "either `completed` (final response in `message`) or `working` "
            "(retry with `taskId` to poll). `message` and `taskId` are mutually "
            "exclusive. `waitSeconds` applies only to polls and is capped at "
            f"{int(await_seconds)} seconds."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "contextId": {
                    "type": "string",
                    "description": (
                        "Context ID from a previous response. Optional. Include "
                        "to continue an existing conversation."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": (
                        "Natural language request to the agent. Mutually "
                        "exclusive with `taskId`."
                    ),
                },
                "taskId": {
                    "type": "string",
                    "description": (
                        "Task ID returned by a prior call that yielded a working "
                        "status. Mutually exclusive with `message`."
                    ),
                },
                "waitSeconds": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "Maximum seconds to block on a poll, capped server-side. "
                        "Ignored when `message` is provided."
                    ),
                },
            },
            "additionalProperties": False,
        },
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [tool]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name != agent_card.name:
            return _error_payload("unknown_tool", f"Unknown tool: {name}")

        message = arguments.get("message")
        task_id = arguments.get("taskId")
        context_id = arguments.get("contextId")
        wait_seconds = float(arguments.get("waitSeconds") or 0)

        # Input validation — one of message XOR taskId.
        if (message is None or message == "") and not task_id:
            return _error_payload(
                "invalid_input",
                "Either `message` or `taskId` must be provided.",
            )
        if message and task_id:
            return _error_payload(
                "invalid_input",
                "`message` and `taskId` are mutually exclusive.",
            )

        try:
            if task_id:
                state = await engine.poll(
                    task_id=task_id,
                    context_id=context_id,
                    wait_seconds=wait_seconds,
                    cap_seconds=await_seconds,
                )
            else:
                assert message is not None
                state = await engine.spawn(
                    message=message,
                    context_id=context_id,
                    via="mcp",
                    await_seconds=await_seconds,
                )
        except ContextBusyError as e:
            return _error_payload(
                "context_busy",
                str(e),
                active_task_id=e.active_task_id,
                context_id=e.context_id,
            )
        except TaskNotFoundError as e:
            return _error_payload("task_not_found", str(e), task_id=e.task_id)
        except TaskContextMismatchError as e:
            return _error_payload(
                "task_context_mismatch",
                str(e),
                task_id=e.task_id,
            )
        except InvalidTaskInputError as e:
            return _error_payload("invalid_input", str(e))

        return _format_state(state, await_seconds)

    return server


def _format_state(state: TaskState, await_seconds: float) -> list[TextContent]:
    """Turn a ``TaskState`` into an MCP ``CallToolResult`` payload."""
    structured: dict[str, Any] = {
        "taskId": state.task_id,
        "contextId": state.context_id,
        "status": state.status.value,
    }
    if state.status == TaskStatus.COMPLETED:
        response_text = _extract_text(state.content)
        structured["message"] = response_text
        envelope = {
            "contextId": state.context_id,
            "taskId": state.task_id,
            "message": response_text,
            "status": state.status.value,
        }
        return [TextContent(
            type="text",
            text=json.dumps(envelope),
        )]

    if state.status == TaskStatus.WORKING:
        structured["pollAfterMs"] = 10_000
        human = (
            f"Task {state.task_id} still running. "
            f"Call again with taskId={state.task_id} to poll "
            f"(waitSeconds up to {int(await_seconds)})."
        )
        envelope = {
            "contextId": state.context_id,
            "taskId": state.task_id,
            "status": state.status.value,
            "message": human,
            "pollAfterMs": 10_000,
        }
        return [TextContent(type="text", text=json.dumps(envelope))]

    # Failed or cancelled — return an error-shaped payload.
    envelope = {
        "contextId": state.context_id,
        "taskId": state.task_id,
        "status": state.status.value,
        "error": state.error or state.status.value,
    }
    return [TextContent(type="text", text=json.dumps(envelope))]


def _error_payload(error: str, message: str, **extra: Any) -> list[TextContent]:
    """Return an MCP tool result envelope for an application-level error."""
    payload: dict[str, Any] = {"error": error, "message": message, **extra}
    return [TextContent(type="text", text=json.dumps(payload))]


def _extract_text(content: list[dict[str, Any]]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
