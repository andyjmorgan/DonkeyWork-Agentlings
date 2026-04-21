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

    ``load()`` caches the parsed store keyed by the file's mtime; repeated
    reads against an unchanged file avoid the JSON parse cost. Any write
    through ``save()`` resets the cache on the next read.
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "memory"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / "memory.json"
        self._cache: MemoryStore | None = None
        self._cache_mtime_ns: int | None = None

    @property
    def path(self) -> Path:
        """Path to the memory file."""
        return self._path

    def load(self) -> MemoryStore:
        """Load memory from disk, returning an empty store if the file is missing.

        Uses the file's mtime as a cache key so callers under heavy traffic
        don't re-parse the same JSON on every call.
        """
        if not self._path.exists():
            self._cache = None
            self._cache_mtime_ns = None
            return MemoryStore()

        mtime_ns = self._path.stat().st_mtime_ns
        if self._cache is not None and self._cache_mtime_ns == mtime_ns:
            return self._cache.model_copy(deep=True)

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            store = MemoryStore.model_validate(data)
        except (json.JSONDecodeError, ValueError):
            logger.warning("corrupt memory file at %s, starting fresh", self._path)
            store = MemoryStore()

        self._cache = store.model_copy(deep=True)
        self._cache_mtime_ns = mtime_ns
        return store

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
            os.replace(tmp_path, self._path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        # Invalidate the cache so the next load() re-reads with the new mtime.
        self._cache = None
        self._cache_mtime_ns = None
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
