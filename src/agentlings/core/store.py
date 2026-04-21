"""Append-only JSONL conversation journal with compaction-aware replay.

Layout (v2 task-based execution):

- Parent journal: ``{data_dir}/{ctx_id}/journal.jsonl``
- Sub-journal:    ``{data_dir}/{ctx_id}/tasks/{task_id}.jsonl``

Legacy parent journals at ``{data_dir}/{ctx_id}.jsonl`` are auto-migrated
to the new layout on first access so existing deployments continue to work.
"""

from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from agentlings.core.models import (
    AUDIT_MARKER_TYPES,
    CompactionEntry,
    MergeCommitted,
    MergeStarted,
    MessageEntry,
    TaskCancelled,
    TaskDispatched,
    TaskFailed,
)

logger = logging.getLogger(__name__)


# Parent journal entry types that can be appended directly to a context journal.
ParentJournalEntry = (
    MessageEntry
    | CompactionEntry
    | TaskDispatched
    | TaskCancelled
    | TaskFailed
    | MergeStarted
    | MergeCommitted
)


class ContextNotFoundError(Exception):
    """Raised when a context ID does not correspond to an existing journal file."""


def _append_jsonl(path: Path, line: str) -> None:
    """Append a single JSONL line under an exclusive fcntl lock.

    Callers must ensure ``path``'s parent directory exists. ``line`` must end
    with a newline. The lock is released before the file handle is closed.
    """
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(line)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all JSONL entries from a file, returning an empty list if missing."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines()]


class JournalStore:
    """Manages per-context JSONL journal files for conversation persistence.

    Each context gets a directory at ``{data_dir}/{ctx_id}/`` containing
    ``journal.jsonl`` (the parent journal) and a ``tasks/`` subdirectory
    for per-task sub-journals. Entries are append-only.
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        """Root data directory."""
        return self._data_dir

    def context_dir(self, ctx_id: str) -> Path:
        """Directory holding a context's journal and task sub-journals."""
        return self._data_dir / ctx_id

    def _path(self, ctx_id: str) -> Path:
        """Path to a context's parent journal."""
        return self.context_dir(ctx_id) / "journal.jsonl"

    def _legacy_path(self, ctx_id: str) -> Path:
        """Pre-v2 parent journal path."""
        return self._data_dir / f"{ctx_id}.jsonl"

    def task_path(self, ctx_id: str, task_id: str) -> Path:
        """Path to a task's sub-journal."""
        return self.context_dir(ctx_id) / "tasks" / f"{task_id}.jsonl"

    def _maybe_migrate_legacy(self, ctx_id: str) -> None:
        """Move a pre-v2 journal into the new per-context directory layout.

        Idempotent: does nothing if the new path already exists or the legacy
        path does not.
        """
        legacy = self._legacy_path(ctx_id)
        new = self._path(ctx_id)
        if not legacy.exists() or new.exists():
            return
        new.parent.mkdir(parents=True, exist_ok=True)
        legacy.rename(new)
        logger.info("migrated legacy journal for context %s", ctx_id)

    def create(self, ctx_id: str) -> None:
        """Create an empty journal file for a new context.

        Args:
            ctx_id: The context identifier.
        """
        self._maybe_migrate_legacy(ctx_id)
        path = self._path(ctx_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        logger.info("created context %s", ctx_id)

    def exists(self, ctx_id: str) -> bool:
        """Check whether a journal file exists for the given context.

        Args:
            ctx_id: The context identifier.
        """
        self._maybe_migrate_legacy(ctx_id)
        return self._path(ctx_id).exists()

    def append(self, ctx_id: str, entry: ParentJournalEntry) -> None:
        """Append a journal entry to the context's JSONL file.

        Args:
            ctx_id: The context identifier.
            entry: The journal entry to append.

        Raises:
            ContextNotFoundError: If no journal file exists for the context.
        """
        self._maybe_migrate_legacy(ctx_id)
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)

        _append_jsonl(path, entry.model_dump_json() + "\n")
        logger.debug("appended %s entry to context %s", entry.t, ctx_id)

    def append_many(self, ctx_id: str, entries: Iterable[ParentJournalEntry]) -> None:
        """Append several entries under a single lock acquisition.

        Used by merge-back to keep the wrapper + contents atomic under the
        per-context lock. Write order matches iteration order.

        Args:
            ctx_id: The context identifier.
            entries: Ordered entries to append.

        Raises:
            ContextNotFoundError: If no journal file exists for the context.
        """
        self._maybe_migrate_legacy(ctx_id)
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)

        lines = [e.model_dump_json() + "\n" for e in entries]
        if not lines:
            return
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.writelines(lines)
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def read_entries(self, ctx_id: str) -> list[dict[str, Any]]:
        """Return every parsed journal entry for a context in insertion order.

        Raises:
            ContextNotFoundError: If no journal file exists for the context.
        """
        self._maybe_migrate_legacy(ctx_id)
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)
        return _read_jsonl(path)

    def replay(self, ctx_id: str) -> list[dict[str, Any]]:
        """Replay the journal from the last compaction marker as LLM messages.

        Audit markers (``task_dispatch``, ``task_cancel``, ``task_fail``,
        ``merge_start``, ``merge_commit``) are stripped unconditionally —
        they carry no conversational content and must never reach the LLM.

        Args:
            ctx_id: The context identifier.

        Returns:
            List of Anthropic-format message dicts from the latest compaction
            onwards. Returns an empty list if the journal is empty.

        Raises:
            ContextNotFoundError: If no journal file exists for the context.
        """
        parsed = self.read_entries(ctx_id)
        if not parsed:
            return []

        cursor = 0
        for i in range(len(parsed) - 1, -1, -1):
            if parsed[i]["t"] == "compact":
                cursor = i
                break

        messages: list[dict[str, Any]] = []
        for entry in parsed[cursor:]:
            t = entry["t"]
            if t in AUDIT_MARKER_TYPES:
                continue
            if t == "compact":
                messages.append({
                    "role": "assistant",
                    "content": entry["content"],
                })
            elif t == "msg":
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })
        return messages

    def iter_context_ids(self) -> list[str]:
        """Return all context IDs under the data directory.

        Includes both the new layout (``{ctx_id}/journal.jsonl``) and legacy
        flat files (``{ctx_id}.jsonl``). Legacy hits are migrated to the new
        layout as a side effect.
        """
        ids: set[str] = set()
        for path in self._data_dir.glob("*/journal.jsonl"):
            ids.add(path.parent.name)
        for path in self._data_dir.glob("*.jsonl"):
            ctx_id = path.stem
            # Migrate lazily so the next access uses the new layout.
            self._maybe_migrate_legacy(ctx_id)
            ids.add(ctx_id)
        return sorted(ids)


class TaskJournal:
    """Append-only JSONL sub-journal for a single task's execution trace.

    Sub-journals are task-local (only the worker writes; the engine tail-reads).
    fcntl locking protects concurrent readers from torn writes.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Filesystem path of this sub-journal."""
        return self._path

    def exists(self) -> bool:
        """Whether the sub-journal file exists."""
        return self._path.exists()

    def create(self) -> None:
        """Create the sub-journal's parent directory and the (empty) file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    def append(self, entry: Any) -> None:
        """Append an entry (any Pydantic ``BaseModel`` with ``model_dump_json``)."""
        if not self._path.exists():
            self.create()
        _append_jsonl(self._path, entry.model_dump_json() + "\n")

    def read_entries(self) -> list[dict[str, Any]]:
        """Return all parsed entries in insertion order."""
        return _read_jsonl(self._path)

    def tail(self) -> dict[str, Any] | None:
        """Return the last entry as a dict, or ``None`` if empty."""
        entries = self.read_entries()
        return entries[-1] if entries else None

    def latest_compaction(self) -> dict[str, Any] | None:
        """Return the most recent compaction entry as a dict, or ``None``."""
        entries = self.read_entries()
        for entry in reversed(entries):
            if entry.get("t") == "compact":
                return entry
        return None

    def terminal_entry(self) -> dict[str, Any] | None:
        """Return a terminal entry (done/fail/cancel) if the task is done."""
        from agentlings.core.models import TASK_TERMINAL_TYPES
        entries = self.read_entries()
        for entry in reversed(entries):
            if entry.get("t") in TASK_TERMINAL_TYPES:
                return entry
        return None
