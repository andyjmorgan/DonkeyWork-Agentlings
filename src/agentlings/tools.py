from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30

TOOL_GROUPS: dict[str, list[str]] = {
    "bash": ["bash"],
    "filesystem": [
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "search_files",
    ],
}


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

    def register_tools(self, enabled: list[str]) -> None:
        resolved: set[str] = set()
        for name in enabled:
            if name in TOOL_GROUPS:
                resolved.update(TOOL_GROUPS[name])
            else:
                resolved.add(name)

        for tool_name in sorted(resolved):
            if tool_name in _BUILTIN_REGISTRY:
                defn = _BUILTIN_REGISTRY[tool_name]
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


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------


def _bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> ToolResult:
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


def _read_file(path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
    try:
        p = Path(path)
        lines = p.read_text(encoding="utf-8").splitlines()
        selected = lines[offset : offset + limit]
        numbered = [f"{i + offset + 1}\t{line}" for i, line in enumerate(selected)]
        output = "\n".join(numbered)
        if offset + limit < len(lines):
            output += f"\n... ({len(lines) - offset - limit} more lines)"
        return ToolResult(output=output)
    except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
        return ToolResult(output=str(e), is_error=True)


def _write_file(path: str, content: str) -> ToolResult:
    try:
        Path(path).write_text(content, encoding="utf-8")
        return ToolResult(output=f"Written {len(content)} bytes to {path}")
    except (PermissionError, OSError) as e:
        return ToolResult(output=str(e), is_error=True)


def _edit_file(
    path: str, old_text: str, new_text: str, replace_all: bool = False
) -> ToolResult:
    try:
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        count = content.count(old_text)

        if count == 0:
            return ToolResult(output="old_text not found in file", is_error=True)
        if count > 1 and not replace_all:
            return ToolResult(
                output=f"old_text found {count} times — set replace_all to replace all occurrences",
                is_error=True,
            )

        if replace_all:
            new_content = content.replace(old_text, new_text)
        else:
            new_content = content.replace(old_text, new_text, 1)

        p.write_text(new_content, encoding="utf-8")
        replacements = count if replace_all else 1
        return ToolResult(output=f"Replaced {replacements} occurrence(s) in {path}")
    except (FileNotFoundError, PermissionError) as e:
        return ToolResult(output=str(e), is_error=True)


def _list_directory(path: str = ".") -> ToolResult:
    try:
        p = Path(path)
        if not p.is_dir():
            return ToolResult(output=f"Not a directory: {path}", is_error=True)

        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries:
            prefix = "[DIR]" if entry.is_dir() else "[FILE]"
            lines.append(f"{prefix} {entry.name}")

        return ToolResult(output="\n".join(lines) if lines else "(empty directory)")
    except (PermissionError, OSError) as e:
        return ToolResult(output=str(e), is_error=True)


def _search_files(path: str, pattern: str) -> ToolResult:
    try:
        p = Path(path)
        matches = sorted(str(m) for m in p.rglob(pattern))
        if not matches:
            return ToolResult(output="No matches found")
        return ToolResult(output="\n".join(matches))
    except (PermissionError, OSError) as e:
        return ToolResult(output=str(e), is_error=True)


# ---------------------------------------------------------------------------
# Built-in tool registry
# ---------------------------------------------------------------------------


_BUILTIN_REGISTRY: dict[str, dict[str, Any]] = {
    "bash": {
        "name": "bash",
        "description": "Execute a shell command and return its output.",
        "input_schema": {
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
        "execute_fn": _bash,
    },
    "read_file": {
        "name": "read_file",
        "description": "Read the contents of a file with line numbers. Supports offset and limit for large files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (0-based, default 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read (default 2000)",
                },
            },
            "required": ["path"],
        },
        "execute_fn": _read_file,
    },
    "write_file": {
        "name": "write_file",
        "description": "Write content to a file, creating it if it doesn't exist or overwriting if it does.",
        "input_schema": {
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
        "execute_fn": _write_file,
    },
    "edit_file": {
        "name": "edit_file",
        "description": "Replace text in a file. Finds old_text and replaces with new_text. Fails if old_text is not found or matches multiple times (unless replace_all is true).",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "The text to replace it with",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace all occurrences (default false)",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
        "execute_fn": _edit_file,
    },
    "list_directory": {
        "name": "list_directory",
        "description": "List the contents of a directory, showing [DIR] and [FILE] prefixes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to list (default: current directory)",
                },
            },
        },
        "execute_fn": _list_directory,
    },
    "search_files": {
        "name": "search_files",
        "description": "Recursively search for files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to search in",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (e.g. '*.py', '**/*.json')",
                },
            },
            "required": ["path", "pattern"],
        },
        "execute_fn": _search_files,
    },
}
