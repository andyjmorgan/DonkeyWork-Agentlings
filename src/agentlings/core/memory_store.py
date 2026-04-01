"""File-backed memory store with atomic writes."""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from agentlings.core.memory_models import MemoryEntry, MemoryStore

logger = logging.getLogger(__name__)


class MemoryFileStore:
    """Manages the agent's persistent memory on disk.

    Memory lives at ``{data_dir}/memory/memory.json``. Reads and writes
    go through Pydantic models. Writes are atomic (temp file + rename).
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "memory.json"

    @property
    def path(self) -> Path:
        """Path to the memory file."""
        return self._path

    def load(self) -> MemoryStore:
        """Load memory from disk, returning an empty store if the file is missing."""
        if not self._path.exists():
            return MemoryStore()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return MemoryStore.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("corrupt memory file at %s, starting fresh", self._path)
            return MemoryStore()

    def save(self, store: MemoryStore) -> None:
        """Atomically write the memory store to disk."""
        import os

        content = store.model_dump_json(indent=2)
        fd, tmp_path = tempfile.mkstemp(
            dir=self._dir, prefix=".memory_", suffix=".tmp"
        )
        os.close(fd)
        try:
            Path(tmp_path).write_text(content, encoding="utf-8")
            Path(tmp_path).rename(self._path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        logger.debug("memory saved: %d entries", len(store.entries))

    def set(self, key: str, value: str) -> MemoryStore:
        """Upsert a memory entry by key. Returns the updated store."""
        store = self.load()
        now = datetime.now(timezone.utc)
        for entry in store.entries:
            if entry.key == key:
                entry.value = value
                entry.recorded = now
                self.save(store)
                return store

        store.entries.append(MemoryEntry(key=key, value=value, recorded=now))
        self.save(store)
        return store

    def remove(self, key: str) -> MemoryStore:
        """Remove a memory entry by key. Returns the updated store."""
        store = self.load()
        store.entries = [e for e in store.entries if e.key != key]
        self.save(store)
        return store

    def list(self) -> list[MemoryEntry]:
        """Return all current memory entries."""
        return self.load().entries
