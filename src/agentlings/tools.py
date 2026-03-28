from __future__ import annotations

import asyncio
import logging
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _ToolEntry] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        execute_fn: Callable[..., ToolResult],
    ) -> None:
        self._tools[name] = _ToolEntry(
            name=name,
            description=description,
            input_schema=input_schema,
            execute_fn=execute_fn,
        )
        logger.info("registered tool: %s", name)

    def list_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(output=f"Unknown tool: {name}", is_error=True)

        logger.debug("executing tool: %s", name)
        fn = entry.execute_fn
        if asyncio.iscoroutinefunction(fn):
            return await fn(**input_dict)
        return await asyncio.to_thread(fn, **input_dict)

    def register_builtins(self) -> None:
        self.register(
            name="shell",
            description="Execute a shell command and return its output.",
            input_schema={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default 30)",
                    },
                },
                "required": ["command"],
            },
            execute_fn=_shell,
        )
        self.register(
            name="read_file",
            description="Read the contents of a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                },
                "required": ["path"],
            },
            execute_fn=_read_file,
        )
        self.register(
            name="write_file",
            description="Write content to a file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
            execute_fn=_write_file,
        )


@dataclass
class _ToolEntry:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute_fn: Callable[..., ToolResult]


def _shell(command: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout
        if result.stderr:
            output += result.stderr
        return ToolResult(
            output=output.strip(),
            is_error=result.returncode != 0,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            output=f"Command timed out after {timeout} seconds",
            is_error=True,
        )


def _read_file(path: str) -> ToolResult:
    try:
        content = Path(path).read_text(encoding="utf-8")
        return ToolResult(output=content)
    except (FileNotFoundError, PermissionError) as e:
        return ToolResult(output=str(e), is_error=True)


def _write_file(path: str, content: str) -> ToolResult:
    try:
        Path(path).write_text(content, encoding="utf-8")
        return ToolResult(output=f"Written {len(content)} bytes to {path}")
    except (PermissionError, OSError) as e:
        return ToolResult(output=str(e), is_error=True)
