"""Unit tests for the MCP tool handler's schema-validation layering.

The MCP SDK (1.10+) runs ``jsonschema.validate`` against each tool's
declared ``inputSchema`` inside ``@server.call_tool()`` before the
handler runs. The handler owns only genuine business logic (engine
dispatch, error mapping); schema-shape validation is delegated to the
SDK. The contract tests below assert our schemas are tight enough for
the SDK to reject:

- malformed value types,
- missing required fields, and
- unknown ("extra") fields (``additionalProperties: false``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from a2a.types import AgentCard
from mcp.types import CallToolRequest, CallToolRequestParams

from agentlings.config import AgentConfig
from agentlings.core.llm import MockLLMClient
from agentlings.core.loop import MessageLoop
from agentlings.core.store import JournalStore
from agentlings.protocol.agent_card import generate_agent_card
from agentlings.core.task import TaskState, TaskStatus
from agentlings.protocol.mcp import _format_state, create_mcp_server
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


async def _invoke(server, tool_name: str, arguments: dict) -> str:
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name=tool_name, arguments=arguments),
    )
    result = await handler(request)
    return result.root.content[0].text


@pytest.mark.asyncio
async def test_get_task_schema_rejects_bad_wait_seconds(mcp_server) -> None:
    """The SDK rejects ``waitSeconds`` of the wrong type before the
    handler runs."""
    server, card = mcp_server
    text = await _invoke(
        server,
        f"{card.name}__get_task",
        {"taskId": "abc", "waitSeconds": "not-a-number"},
    )
    assert "Input validation error" in text


@pytest.mark.asyncio
async def test_spawn_schema_rejects_missing_message(mcp_server) -> None:
    """The spawn tool requires ``message`` — empty payload is a schema
    violation, not a handler-level branch."""
    server, card = mcp_server
    text = await _invoke(server, card.name, {})
    assert "Input validation error" in text


@pytest.mark.asyncio
async def test_spawn_schema_rejects_extra_property(mcp_server) -> None:
    """``additionalProperties: false`` is the bouncer — extras don't reach
    the handler. Catches an LLM that fabricates a field like ``priority``
    or tries to smuggle ``taskId`` onto a spawn call."""
    server, card = mcp_server
    text = await _invoke(
        server,
        card.name,
        {"message": "hi", "taskId": "abc"},
    )
    assert "Input validation error" in text


@pytest.mark.asyncio
async def test_get_task_schema_rejects_missing_task_id(mcp_server) -> None:
    """The get_task tool requires ``taskId`` — empty payload is rejected
    by the SDK, not the handler."""
    server, card = mcp_server
    text = await _invoke(server, f"{card.name}__get_task", {})
    assert "Input validation error" in text


@pytest.mark.asyncio
async def test_get_task_schema_rejects_extra_property(mcp_server) -> None:
    """Extras on the poll path are likewise schema-rejected — e.g. an LLM
    re-supplying ``message`` while polling."""
    server, card = mcp_server
    text = await _invoke(
        server,
        f"{card.name}__get_task",
        {"taskId": "abc", "message": "no"},
    )
    assert "Input validation error" in text


def test_working_envelope_points_at_get_task_tool() -> None:
    """The working-state envelope must name the get_task tool the LLM
    should call to poll. Otherwise the LLM either retries the spawn tool
    (creating a new task) or hallucinates a tool name."""
    state = TaskState(
        task_id="task-123",
        context_id="ctx-abc",
        status=TaskStatus.WORKING,
    )
    [content] = _format_state(state, await_seconds=2.0, get_task_name="agent__get_task")
    payload = json.loads(content.text)
    assert payload["status"] == "working"
    assert payload["taskId"] == "task-123"
    assert "agent__get_task" in payload["message"]
    assert "task-123" in payload["message"]
