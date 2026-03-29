"""Append-only JSONL conversation journal with compaction-aware replay."""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any

from agentlings.core.models import CompactionEntry, MessageEntry

logger = logging.getLogger(__name__)


class ContextNotFoundError(Exception):
    """Raised when a context ID does not correspond to an existing journal file."""


class JournalStore:
    """Manages per-context JSONL journal files for conversation persistence.

    Each context gets its own file at ``{data_dir}/{context_id}.jsonl``.
    Entries are append-only and never modified or deleted.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ctx_id: str) -> Path:
        return self._data_dir / f"{ctx_id}.jsonl"

    def create(self, ctx_id: str) -> None:
        """Create an empty journal file for a new context.

        Args:
            ctx_id: The context identifier.
        """
        path = self._path(ctx_id)
        path.touch(exist_ok=True)
        logger.info("created context %s", ctx_id)

    def exists(self, ctx_id: str) -> bool:
        """Check whether a journal file exists for the given context.

        Args:
            ctx_id: The context identifier.
        """
        return self._path(ctx_id).exists()

    def append(self, ctx_id: str, entry: MessageEntry | CompactionEntry) -> None:
        """Append a journal entry to the context's JSONL file.

        Args:
            ctx_id: The context identifier.
            entry: The journal entry to append.

        Raises:
            ContextNotFoundError: If no journal file exists for the context.
        """
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)

        line = entry.model_dump_json() + "\n"
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

        logger.debug("appended %s entry to context %s", entry.t, ctx_id)

    def replay(self, ctx_id: str) -> list[dict[str, Any]]:
        """Replay the journal from the last compaction marker.

        Scans backwards for the most recent compaction entry and builds
        an Anthropic-format ``messages[]`` array from that point forward.

        Args:
            ctx_id: The context identifier.

        Returns:
            List of message dicts suitable for the Anthropic Messages API.

        Raises:
            ContextNotFoundError: If no journal file exists for the context.
        """
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []

        parsed = [json.loads(line) for line in lines]

        cursor = 0
        for i in range(len(parsed) - 1, -1, -1):
            if parsed[i]["t"] == "compact":
                cursor = i
                break

        messages: list[dict[str, Any]] = []
        for entry in parsed[cursor:]:
            if entry["t"] == "compact":
                messages.append({
                    "role": "assistant",
                    "content": entry["content"],
                })
            elif entry["t"] == "msg":
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })

        return messages
