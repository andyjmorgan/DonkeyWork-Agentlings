"""Memory tool: lets the agent read and write its own long-term memory."""

from __future__ import annotations

import json
from typing import Any

from agentlings.core.memory_store import MemoryFileStore
from agentlings.tools.registry import ToolResult

_MEMORY_STORE: MemoryFileStore | None = None


def init_memory_tool(store: MemoryFileStore) -> None:
    """Bind the memory tool to a concrete store instance.

    Must be called before the tool is executed. Typically called during
    server startup after the ``MemoryFileStore`` is created.

    Args:
        store: The memory file store to use for all memory operations.
    """
    global _MEMORY_STORE
    _MEMORY_STORE = store


def _memory_edit(operation: str, key: str | None = None, value: str | None = None) -> ToolResult:
    """Execute a memory operation: set, remove, or list.

    Args:
        operation: One of ``"set"``, ``"remove"``, or ``"list"``.
        key: Required for ``set`` and ``remove`` operations.
        value: Required for ``set`` operation.
    """
    if _MEMORY_STORE is None:
        return ToolResult(output="Memory not initialized", is_error=True)

    if operation == "list":
        entries = _MEMORY_STORE.list()
        if not entries:
            return ToolResult(output="Memory is empty.")
        lines = [f"- **{e.key}**: {e.value}" for e in entries]
        return ToolResult(output="\n".join(lines))

    if operation == "set":
        if not key or not value:
            return ToolResult(
                output="Both 'key' and 'value' are required for set operation",
                is_error=True,
            )
        store = _MEMORY_STORE.set(key, value)
        return ToolResult(output=f"Memory updated: {key} ({len(store.entries)} entries total)")

    if operation == "remove":
        if not key:
            return ToolResult(
                output="'key' is required for remove operation",
                is_error=True,
            )
        store = _MEMORY_STORE.remove(key)
        return ToolResult(output=f"Memory entry '{key}' removed ({len(store.entries)} entries total)")

    return ToolResult(
        output=f"Unknown operation: {operation}. Use 'set', 'remove', or 'list'.",
        is_error=True,
    )


MEMORY_TOOL_DEFINITION: dict[str, Any] = {
    "name": "memory_edit",
    "description": (
        "Read and write your long-term memory. Operations:\n"
        "- set: Store a fact by key (upserts if key exists)\n"
        "- remove: Delete a fact by key\n"
        "- list: Show all current memory entries"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["set", "remove", "list"],
                "description": "The memory operation to perform",
            },
            "key": {
                "type": "string",
                "description": "Memory entry key (required for set and remove)",
            },
            "value": {
                "type": "string",
                "description": "Memory entry value (required for set)",
            },
        },
        "required": ["operation"],
    },
    "execute_fn": _memory_edit,
}
