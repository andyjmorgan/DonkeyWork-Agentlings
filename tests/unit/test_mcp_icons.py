"""Unit tests for MCP icon advertisement.

We advertise exactly one ``Icon`` per surface (server, spawn tool, get_task
tool). The spec models ``icons`` as a multi-resolution set for clients to pick
a best fit from, but real clients (the MCP Inspector among them) render every
entry side by side — so a single icon is the only thing that displays cleanly.
These tests pin that single-icon contract and the MIME inference.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from a2a.types import AgentCard
from mcp.types import ListToolsRequest, PaginatedRequestParams

from agentlings.config import AgentConfig
from agentlings.core.llm import MockLLMClient
from agentlings.core.loop import MessageLoop
from agentlings.core.store import JournalStore
from agentlings.protocol.agent_card import generate_agent_card
from agentlings.protocol.mcp import _icon_set, _mime_from_url, create_mcp_server
from agentlings.tools.registry import ToolRegistry


def test_icon_set_returns_none_for_unset() -> None:
    assert _icon_set(None) is None
    assert _icon_set("") is None


def test_icon_set_wraps_single_icon() -> None:
    icons = _icon_set("https://cdn.example.com/icons/agentling.png")
    assert icons is not None
    assert len(icons) == 1
    icon = icons[0]
    assert icon.src == "https://cdn.example.com/icons/agentling.png"
    assert icon.mimeType == "image/png"
    assert icon.sizes is None


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://x/a.png", "image/png"),
        ("https://x/a.jpg", "image/jpeg"),
        ("https://x/a.jpeg", "image/jpeg"),
        ("https://x/a.svg", "image/svg+xml"),
        ("https://x/a.webp", "image/webp"),
        ("https://x/a.PNG", "image/png"),
        ("https://x/icon", None),
        ("https://x/a.gif", None),
    ],
)
def test_mime_from_url(url: str, expected: str | None) -> None:
    assert _mime_from_url(url) == expected


def test_icon_set_omits_mime_for_data_uri() -> None:
    uri = "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4="
    icons = _icon_set(uri)
    assert icons is not None
    assert len(icons) == 1
    assert icons[0].src == uri
    assert icons[0].mimeType is None


def _config_with_icons(tmp_path: Path, tmp_data_dir: Path) -> AgentConfig:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        "name: test-agent\n"
        "description: A test agent\n"
        "tools:\n"
        "  - bash\n"
        "icons:\n"
        "  server: https://cdn.example.com/icons/agentling.png\n"
        "  spawn: https://cdn.example.com/icons/play.png\n"
        "  task: https://cdn.example.com/icons/task.svg\n"
    )
    return AgentConfig(
        anthropic_api_key="test-key",
        agent_api_key="test-agent-key",
        agent_data_dir=tmp_data_dir,
        agent_llm_backend="mock",
        agent_config=str(agent_yaml),
    )


@pytest.mark.asyncio
async def test_server_and_tools_each_advertise_one_icon(
    tmp_path: Path, tmp_data_dir: Path
) -> None:
    config = _config_with_icons(tmp_path, tmp_data_dir)
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    tools.register_tools(["bash"])
    llm = MockLLMClient(tool_names=tools.tool_names())
    loop = MessageLoop(config=config, store=store, llm=llm, tools=tools)
    agent_card: AgentCard = generate_agent_card(config)
    server = create_mcp_server(loop=loop, agent_card=agent_card, config=config)

    assert server.icons is not None
    assert [i.src for i in server.icons] == [
        "https://cdn.example.com/icons/agentling.png"
    ]

    handler = server.request_handlers[ListToolsRequest]
    request = ListToolsRequest(method="tools/list", params=PaginatedRequestParams())
    result = await handler(request)
    by_name = {t.name: t for t in result.root.tools}

    spawn = by_name["test-agent"]
    assert [i.src for i in spawn.icons] == ["https://cdn.example.com/icons/play.png"]
    assert spawn.icons[0].mimeType == "image/png"

    get_task = by_name["test-agent__get_task"]
    assert [i.src for i in get_task.icons] == ["https://cdn.example.com/icons/task.svg"]
    assert get_task.icons[0].mimeType == "image/svg+xml"


@pytest.mark.asyncio
async def test_icons_omitted_when_unconfigured(
    tmp_data_dir: Path, test_config: AgentConfig
) -> None:
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    tools.register_tools(["bash"])
    llm = MockLLMClient(tool_names=tools.tool_names())
    loop = MessageLoop(config=test_config, store=store, llm=llm, tools=tools)
    agent_card: AgentCard = generate_agent_card(test_config)
    server = create_mcp_server(loop=loop, agent_card=agent_card, config=test_config)

    assert server.icons is None
    handler = server.request_handlers[ListToolsRequest]
    request = ListToolsRequest(method="tools/list", params=PaginatedRequestParams())
    result = await handler(request)
    assert all(t.icons is None for t in result.root.tools)
