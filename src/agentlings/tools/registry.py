"""Tool registry for managing and executing agent tools."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

TOOL_GROUPS: dict[str, list[str]] = {
    "bash": ["bash"],
    "filesystem": [
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
    ],
    "memory": ["memory_edit"],
}


@dataclass
class ToolResult:
    """Result of executing a tool.

    Attributes:
        output: The tool's output text.
        is_error: Whether the execution resulted in an error.
    """

    output: str
    is_error: bool = False


class ToolRegistry:
    """Registry for tools the agent can invoke during LLM completion loops."""

    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        execute_fn: Callable[..., ToolResult],
    ) -> None:
        """Register a tool with its schema and execution function.

        Args:
            name: Unique tool identifier.
            description: Human-readable description for the LLM.
            input_schema: JSON Schema describing the tool's input parameters.
            execute_fn: Callable that performs the tool's action and returns a ``ToolResult``.
        """
        self._tools[name] = _ToolEntry(
            name=name,
            description=description,
            input_schema=input_schema,
            execute_fn=execute_fn,
        )
        logger.info("registered tool: %s", name)

    def list_schemas(self) -> list[dict[str, Any]]:
        """Return tool schemas in the format expected by the Anthropic API."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def tool_names(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(self._tools.keys())

    async def execute(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """Execute a registered tool by name.

        Args:
            name: The tool to execute.
            input_dict: Keyword arguments to pass to the tool function.

        Returns:
            The tool's result, or an error result if the tool is unknown.
        """
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(output=f"Unknown tool: {name}", is_error=True)

        logger.debug("executing tool: %s", name)
        fn = entry.execute_fn
        if asyncio.iscoroutinefunction(fn):
            return await fn(**input_dict)
        return await asyncio.to_thread(fn, **input_dict)

    def register_tools(self, enabled: list[str]) -> None:
        """Register built-in tools by name or group.

        Args:
            enabled: List of tool names or group names (e.g. ``"bash"``, ``"filesystem"``)
                     to activate from the built-in registry.
        """
        from agentlings.tools.builtins import BUILTIN_REGISTRY
        from agentlings.tools.memory import MEMORY_TOOL_DEFINITION

        all_tools: dict[str, dict[str, Any]] = {
            **BUILTIN_REGISTRY,
            MEMORY_TOOL_DEFINITION["name"]: MEMORY_TOOL_DEFINITION,
        }

        resolved: set[str] = set()
        for name in enabled:
            if name in TOOL_GROUPS:
                resolved.update(TOOL_GROUPS[name])
            else:
                resolved.add(name)

        for tool_name in sorted(resolved):
            if tool_name in all_tools:
                defn = all_tools[tool_name]
                self.register(
                    name=defn["name"],
                    description=defn["description"],
                    input_schema=defn["input_schema"],
                    execute_fn=defn["execute_fn"],
                )
            else:
                logger.warning("unknown tool: %s", tool_name)


@dataclass
class _ToolEntry:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute_fn: Callable[..., ToolResult]
