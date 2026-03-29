from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.config import AgentConfig
from agentlings.llm import MockLLMClient
from agentlings.loop import MessageLoop
from agentlings.store import ContextNotFoundError, JournalStore
from agentlings.tools import ToolRegistry


@pytest.fixture
def loop_deps(tmp_data_dir: Path, test_config: AgentConfig):
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    tools.register_tools(["bash", "filesystem"])
    llm = MockLLMClient(tool_names=tools.tool_names())
    loop = MessageLoop(config=test_config, store=store, llm=llm, tools=tools)
    return loop, store


class TestProcessMessage:
    @pytest.mark.asyncio
    async def test_new_context_created(self, loop_deps) -> None:
        loop, store = loop_deps
        result = await loop.process_message("hello")
        assert result.context_id is not None
        assert store.exists(result.context_id)

    @pytest.mark.asyncio
    async def test_response_has_content(self, loop_deps) -> None:
        loop, _ = loop_deps
        result = await loop.process_message("hello")
        assert len(result.content) > 0
        assert result.content[0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_continuation_with_context_id(self, loop_deps) -> None:
        loop, _ = loop_deps
        r1 = await loop.process_message("hello")
        r2 = await loop.process_message("follow up", context_id=r1.context_id)
        assert r2.context_id == r1.context_id

    @pytest.mark.asyncio
    async def test_unknown_context_id_creates_it(self, loop_deps) -> None:
        loop, store = loop_deps
        result = await loop.process_message("hello", context_id="external-id")
        assert result.context_id == "external-id"
        assert store.exists("external-id")

    @pytest.mark.asyncio
    async def test_tool_execution_flow(self, loop_deps) -> None:
        loop, store = loop_deps
        result = await loop.process_message("run bash echo test")
        assert result.context_id is not None
        messages = store.replay(result.context_id)
        assert len(messages) >= 3

    @pytest.mark.asyncio
    async def test_journal_records_conversation(self, loop_deps) -> None:
        loop, store = loop_deps
        result = await loop.process_message("hello")
        messages = store.replay(result.context_id)
        roles = [m["role"] for m in messages]
        assert "user" in roles
        assert "assistant" in roles

    @pytest.mark.asyncio
    async def test_via_parameter_recorded(self, loop_deps) -> None:
        loop, store = loop_deps
        result = await loop.process_message("hello", via="mcp")
        path = store._path(result.context_id)
        import json
        lines = path.read_text().strip().splitlines()
        first = json.loads(lines[0])
        assert first["via"] == "mcp"
