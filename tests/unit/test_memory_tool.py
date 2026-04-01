"""Tests for the memory_edit tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.core.memory_store import MemoryFileStore
from agentlings.tools.memory import MEMORY_TOOL_DEFINITION, _memory_edit, init_memory_tool
from agentlings.tools.registry import TOOL_GROUPS, ToolRegistry


class TestMemoryToolRegistration:
    def test_memory_group_exists(self) -> None:
        assert "memory" in TOOL_GROUPS
        assert "memory_edit" in TOOL_GROUPS["memory"]

    def test_register_memory_group(self) -> None:
        registry = ToolRegistry()
        registry.register_tools(["memory"])
        assert "memory_edit" in registry.tool_names()

    def test_schema_has_operation(self) -> None:
        schema = MEMORY_TOOL_DEFINITION["input_schema"]
        assert "operation" in schema["properties"]
        assert schema["properties"]["operation"]["enum"] == ["set", "remove", "list"]


class TestMemoryEditOperations:
    @pytest.fixture(autouse=True)
    def _init_store(self, tmp_data_dir: Path) -> None:
        store = MemoryFileStore(tmp_data_dir)
        init_memory_tool(store)

    def test_set(self) -> None:
        result = _memory_edit(operation="set", key="test-key", value="test-value")
        assert not result.is_error
        assert "test-key" in result.output

    def test_set_requires_key(self) -> None:
        result = _memory_edit(operation="set", value="v")
        assert result.is_error

    def test_set_requires_value(self) -> None:
        result = _memory_edit(operation="set", key="k")
        assert result.is_error

    def test_list_empty(self) -> None:
        result = _memory_edit(operation="list")
        assert not result.is_error
        assert "empty" in result.output.lower()

    def test_list_with_entries(self) -> None:
        _memory_edit(operation="set", key="k1", value="v1")
        result = _memory_edit(operation="list")
        assert "k1" in result.output

    def test_remove(self) -> None:
        _memory_edit(operation="set", key="k1", value="v1")
        result = _memory_edit(operation="remove", key="k1")
        assert not result.is_error
        list_result = _memory_edit(operation="list")
        assert "empty" in list_result.output.lower()

    def test_remove_requires_key(self) -> None:
        result = _memory_edit(operation="remove")
        assert result.is_error

    def test_unknown_operation(self) -> None:
        result = _memory_edit(operation="bogus")
        assert result.is_error
