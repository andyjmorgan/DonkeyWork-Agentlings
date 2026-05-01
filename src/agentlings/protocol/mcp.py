"""MCP protocol server exposing the agentling as task-aware tools.

Two tools are registered, mirroring A2A's ``message/send`` + ``tasks/get``:

- ``<agent_name>`` — start a new task. Inputs: ``message`` (required),
  ``contextId`` (optional).
- ``<agent_name>__get_task`` — poll an existing task. Inputs: ``taskId``
  (required), ``contextId`` (optional), ``waitSeconds`` (optional).

Both schemas declare ``additionalProperties: false`` so the MCP SDK's
input validator rejects unknown fields before the handler runs. Required
fields are enforced the same way. The handler keeps only genuine
business-logic checks (engine dispatch, error mapping). Responses are
the same message-shape envelope in both cases.
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
from agentlings.core.telemetry import otel_span

logger = logging.getLogger(__name__)


def create_mcp_server(
    loop: MessageLoop,
    agent_card: AgentCard,
    config: AgentConfig,
) -> Server:
    """Create an MCP server exposing spawn + get_task tools.

    Args:
        loop: The message loop exposing the shared task engine.
        agent_card: The agent card whose name and description define the tools.
        config: Agent configuration (for the await timeout cap).

    Returns:
        A configured MCP ``Server`` instance.
    """
    server = Server(agent_card.name)
    engine = loop.engine
    await_seconds = float(getattr(config, "agent_task_await_seconds", 60))

    spawn_name = agent_card.name
    get_task_name = f"{agent_card.name}__get_task"

    spawn_tool = Tool(
        name=spawn_name,
        description=(
            f"{agent_card.description}\n\n"
            "Start a new task. The response's `status` is either `completed` "
            f"(final response in `message`) or `working` — in the working "
            f"case, retry via `{get_task_name}` with the returned `taskId`."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "contextId": {
                    "type": "string",
                    "description": (
                        "Context ID from a previous response. Optional. "
                        "Include to continue an existing conversation."
                    ),
                },
                "message": {
                    "type": "string",
                    "description": "Natural language request to the agent.",
                },
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    )

    get_task_tool = Tool(
        name=get_task_name,
        description=(
            f"Poll the state of a task previously returned by `{spawn_name}` "
            "whose status was `working`. The response uses the same envelope "
            "shape: `status` is `completed`, `working`, `failed`, or "
            f"`cancelled`. `waitSeconds` is capped at {int(await_seconds)} "
            "seconds server-side."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "taskId": {
                    "type": "string",
                    "description": (
                        "Task ID returned by a prior call that yielded a "
                        "working status."
                    ),
                },
                "contextId": {
                    "type": "string",
                    "description": (
                        "Optional context ID. When provided, the server "
                        "asserts the task belongs to it."
                    ),
                },
                "waitSeconds": {
                    "type": "number",
                    "minimum": 0,
                    "description": (
                        "Maximum seconds to block on this poll, capped "
                        "server-side."
                    ),
                },
            },
            "required": ["taskId"],
            "additionalProperties": False,
        },
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [spawn_tool, get_task_tool]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        context_id = arguments.get("contextId")

        if name == spawn_name:
            op = "spawn"
            message = arguments.get("message")
            task_id = None
            wait_seconds = 0.0
        elif name == get_task_name:
            op = "poll"
            message = None
            task_id = arguments.get("taskId")
            wait_seconds = arguments.get("waitSeconds", 0.0)
        else:
            with otel_span("agentling.mcp.call_tool", {
                "agent.via": "mcp",
                "mcp.tool_name": name,
                "mcp.operation": "unknown",
            }) as span:
                span.set_attribute("mcp.outcome", "unknown_tool")
            return _error_payload("unknown_tool", f"Unknown tool: {name}")

        with otel_span("agentling.mcp.call_tool", {
            "agent.via": "mcp",
            "mcp.tool_name": name,
            "mcp.operation": op,
            "task.context_id": context_id or "",
            "task.id": task_id or "",
            "mcp.message_chars": len(message or ""),
        }) as span:
            try:
                if op == "poll":
                    assert task_id is not None
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
                span.set_attribute("mcp.outcome", "context_busy")
                return _error_payload(
                    "context_busy",
                    str(e),
                    active_task_id=e.active_task_id,
                    context_id=e.context_id,
                )
            except TaskNotFoundError as e:
                span.set_attribute("mcp.outcome", "task_not_found")
                return _error_payload("task_not_found", str(e), task_id=e.task_id)
            except TaskContextMismatchError as e:
                span.set_attribute("mcp.outcome", "task_context_mismatch")
                return _error_payload(
                    "task_context_mismatch",
                    str(e),
                    task_id=e.task_id,
                )
            except InvalidTaskInputError as e:
                span.set_attribute("mcp.outcome", "invalid_input")
                return _error_payload("invalid_input", str(e))

            span.set_attribute("mcp.outcome", state.status.value)
            span.set_attribute("task.status", state.status.value)
            span.set_attribute("task.id", state.task_id)
            return _format_state(state, await_seconds, get_task_name)

    return server


def _format_state(
    state: TaskState, await_seconds: float, get_task_name: str
) -> list[TextContent]:
    """Turn a ``TaskState`` into an MCP ``CallToolResult`` payload."""
    if state.status == TaskStatus.COMPLETED:
        response_text = _extract_text(state.content)
        envelope = {
            "contextId": state.context_id,
            "taskId": state.task_id,
            "message": response_text,
            "status": state.status.value,
        }
        return [TextContent(type="text", text=json.dumps(envelope))]

    if state.status == TaskStatus.WORKING:
        human = (
            f"Task {state.task_id} still running. Call `{get_task_name}` "
            f"with taskId={state.task_id} to poll "
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
