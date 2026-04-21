"""Unit tests for the MCP tool handler's schema-validation layering.

The MCP SDK (1.10+) runs ``jsonschema.validate`` against the declared
``inputSchema`` inside ``@server.call_tool()`` before the handler runs.
The handler owns only genuine business logic (XOR between ``message``
and ``taskId``, engine dispatch); schema-shape validation is delegated
to the SDK. The contract test below asserts our schema is tight enough
for the SDK to reject a malformed ``waitSeconds`` value.
"""

from __future__ import annotations

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


@pytest.mark.asyncio
async def test_schema_validates_wait_seconds_type(mcp_server) -> None:
    """The MCP SDK's inputSchema validation rejects malformed `waitSeconds`
    before the handler runs. This asserts our declared schema is tight
    enough for the SDK to catch a type violation."""
    server, card = mcp_server
    from mcp.types import CallToolRequest, CallToolRequestParams
    handler = server.request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name=card.name,
            arguments={"taskId": "abc", "waitSeconds": "not-a-number"},
        ),
    )
    result = await handler(request)
    text = result.root.content[0].text
    assert "Input validation error" in text
