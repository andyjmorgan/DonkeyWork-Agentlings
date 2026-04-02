"""Built-in tool implementations: bash shell and filesystem operations."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agentlings.tools.registry import ToolResult

DEFAULT_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _make_bash(default_timeout: int = DEFAULT_TIMEOUT) -> callable:
    """Create a bash tool function with a configurable default timeout."""

    def _bash(command: str, timeout: int | None = None) -> ToolResult:
        effective_timeout = timeout if timeout is not None else default_timeout
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            output = output.strip()
            is_error = result.returncode != 0
            if not output:
                output = (
                    f"Command failed with exit code {result.returncode}"
                    if is_error
                    else "(no output)"
                )
            return ToolResult(output=output, is_error=is_error)
        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"Command timed out after {effective_timeout} seconds",
                is_error=True,
            )

    return _bash


def _read_file(path: str, offset: int = 0, limit: int = 2000) -> ToolResult:
    """Read a text file and return its contents with line numbers.

    Args:
        path: Path to the file to read.
        offset: Zero-based line number to start reading from.
        limit: Maximum number of lines to return.

    Returns:
        Numbered lines of the file, with a truncation notice if applicable.
    """
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
    """Write content to a file, creating or overwriting as needed.

    Args:
        path: Path to the file to write.
        content: The text content to write.
    """
    try:
        Path(path).write_text(content, encoding="utf-8")
        return ToolResult(output=f"Written {len(content)} bytes to {path}")
    except (PermissionError, OSError) as e:
        return ToolResult(output=str(e), is_error=True)


def _edit_file(
    path: str, old_text: str, new_text: str, replace_all: bool = False
) -> ToolResult:
    """Find and replace text in a file.

    Fails if ``old_text`` is not found. If it matches multiple times,
    fails unless ``replace_all`` is set.

    Args:
        path: Path to the file to edit.
        old_text: Exact text to find.
        new_text: Replacement text.
        replace_all: If true, replace all occurrences instead of requiring a unique match.
    """
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
    """List directory contents with ``[DIR]`` and ``[FILE]`` prefixes.

    Args:
        path: Path to the directory to list. Defaults to the current directory.
    """
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
    """Recursively search for files matching a glob pattern.

    Args:
        path: Root directory to search from.
        pattern: Glob pattern to match (e.g. ``"*.py"``, ``"**/*.json"``).
    """
    try:
        p = Path(path)
        matches = sorted(str(m) for m in p.rglob(pattern))
        if not matches:
            return ToolResult(output="No matches found")
        return ToolResult(output="\n".join(matches))
    except (PermissionError, OSError) as e:
        return ToolResult(output=str(e), is_error=True)


# ---------------------------------------------------------------------------
# Registry mapping tool names to their definitions
# ---------------------------------------------------------------------------


def build_builtin_registry(bash_timeout: int = DEFAULT_TIMEOUT) -> dict[str, dict[str, Any]]:
    """Build the built-in tool registry with configurable defaults."""
    return {
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
                    "description": f"Timeout in seconds (default {bash_timeout})",
                },
            },
            "required": ["command"],
        },
        "execute_fn": _make_bash(bash_timeout),
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


BUILTIN_REGISTRY = build_builtin_registry()
