"""Tool registry for managing and executing agent tools."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from agentlings.tools.decorator import Tool

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

    def register_tool_object(self, tool: "Tool[Any]") -> None:
        """Register a ``@tool``-decorated ``Tool`` instance.

        Bridges the decorator surface to the existing registry shape: the
        tool's input schema and metadata are extracted via
        ``to_anthropic_dict()``; execution is wrapped so a unified
        ``ToolResult`` is returned regardless of the underlying function's
        return type.

        Args:
            tool: The ``Tool`` produced by the ``@tool`` decorator.
        """
        async def _execute(**kwargs: Any) -> ToolResult:
            try:
                output = await tool.call(kwargs)
            except Exception as e:  # noqa: BLE001 — surface to LLM, not crash
                return ToolResult(
                    output=f"Tool {tool.name} failed: {e}",
                    is_error=True,
                )
            return ToolResult(output=str(output), is_error=False)

        self.register(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            execute_fn=_execute,
        )

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
        try:
            fn = entry.execute_fn
            if asyncio.iscoroutinefunction(fn):
                return await fn(**input_dict)
            return await asyncio.to_thread(fn, **input_dict)
        except Exception:
            logger.exception("tool %s raised an unhandled exception", name)
            return ToolResult(
                output=f"Tool {name} failed with an internal error",
                is_error=True,
            )

    def register_tools(self, enabled: list[str], bash_timeout: int = 30) -> None:
        """Register built-in tools by name or group.

        Args:
            enabled: List of tool names or group names (e.g. ``"bash"``, ``"filesystem"``)
                     to activate from the built-in registry.
            bash_timeout: Default timeout in seconds for the bash tool.
        """
        from agentlings.tools.builtins import build_builtin_registry
        from agentlings.tools.memory import MEMORY_TOOL_DEFINITION

        all_tools: dict[str, dict[str, Any]] = {
            **build_builtin_registry(bash_timeout),
            MEMORY_TOOL_DEFINITION["name"]: MEMORY_TOOL_DEFINITION,
        }

        resolved: set[str] = set()
        for name in enabled:
            if name in TOOL_GROUPS:
                resolved.update(TOOL_GROUPS[name])
            else:
                resolved.add(name)

        unknown = sorted(resolved - all_tools.keys())
        if unknown:
            raise ValueError(
                f"Unknown tools in agent config: {unknown}. "
                f"Available: {sorted(all_tools.keys())}"
            )

        for tool_name in sorted(resolved):
            defn = all_tools[tool_name]
            self.register(
                name=defn["name"],
                description=defn["description"],
                input_schema=defn["input_schema"],
                execute_fn=defn["execute_fn"],
            )


@dataclass
class _ToolEntry:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute_fn: Callable[..., ToolResult]
