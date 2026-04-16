"""Tests for the file-backed memory store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlings.core.memory_models import MemoryStore
from agentlings.core.memory_store import MemoryFileStore


@pytest.fixture
def mem_store(tmp_data_dir: Path) -> MemoryFileStore:
    return MemoryFileStore(tmp_data_dir)


class TestLoad:
    def test_missing_file_returns_empty(self, mem_store: MemoryFileStore) -> None:
        store = mem_store.load()
        assert store.entries == []

    def test_corrupt_file_returns_empty(self, mem_store: MemoryFileStore) -> None:
        mem_store.path.write_text("not json", encoding="utf-8")
        store = mem_store.load()
        assert store.entries == []

    def test_loads_valid_file(self, mem_store: MemoryFileStore) -> None:
        data = {
            "entries": [{
                "key": "k1",
                "value": "v1",
                "recorded": "2026-04-01T00:00:00Z",
            }]
        }
        mem_store.path.write_text(json.dumps(data), encoding="utf-8")
        store = mem_store.load()
        assert len(store.entries) == 1
        assert store.entries[0].key == "k1"


class TestSave:
    def test_creates_file(self, mem_store: MemoryFileStore) -> None:
        store = MemoryStore()
        mem_store.save(store)
        assert mem_store.path.exists()

    def test_atomic_write(self, mem_store: MemoryFileStore) -> None:
        mem_store.set("k1", "v1")
        mem_store.set("k2", "v2")
        store = mem_store.load()
        assert len(store.entries) == 2


class TestSet:
    def test_adds_new_entry(self, mem_store: MemoryFileStore) -> None:
        store = mem_store.set("new-key", "new-value")
        assert len(store.entries) == 1
        assert store.entries[0].key == "new-key"

    def test_upserts_existing_entry(self, mem_store: MemoryFileStore) -> None:
        mem_store.set("k1", "old")
        store = mem_store.set("k1", "new")
        assert len(store.entries) == 1
        assert store.entries[0].value == "new"


class TestRemove:
    def test_removes_entry(self, mem_store: MemoryFileStore) -> None:
        mem_store.set("k1", "v1")
        mem_store.set("k2", "v2")
        store = mem_store.remove("k1")
        assert len(store.entries) == 1
        assert store.entries[0].key == "k2"

    def test_remove_nonexistent_is_noop(self, mem_store: MemoryFileStore) -> None:
        mem_store.set("k1", "v1")
        store = mem_store.remove("missing")
        assert len(store.entries) == 1


class TestList:
    def test_empty(self, mem_store: MemoryFileStore) -> None:
        assert mem_store.list() == []

    def test_returns_entries(self, mem_store: MemoryFileStore) -> None:
        mem_store.set("k1", "v1")
        mem_store.set("k2", "v2")
        entries = mem_store.list()
        assert len(entries) == 2


class TestDirectoryCreation:
    def test_creates_memory_subdir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        store = MemoryFileStore(data_dir)
        assert (data_dir / "memory").is_dir()


class TestLoadCaching:
    """``load()`` must avoid re-parsing when the file's mtime is unchanged."""

    def test_repeated_load_does_not_reparse(
        self, mem_store: MemoryFileStore
    ) -> None:
        from unittest.mock import patch

        mem_store.save(MemoryStore())
        mem_store.load()  # warm the cache

        with patch("agentlings.core.memory_store.json.loads") as mocked:
            mocked.return_value = {"entries": []}
            for _ in range(5):
                mem_store.load()
            assert mocked.call_count == 0, (
                "cached load must not re-invoke json.loads"
            )

    def test_save_invalidates_cache(self, mem_store: MemoryFileStore) -> None:
        from agentlings.core.memory_models import MemoryEntry
        from datetime import datetime, timezone

        mem_store.save(MemoryStore())
        mem_store.load()

        mem_store.save(MemoryStore(entries=[
            MemoryEntry(key="k", value="v", recorded=datetime.now(timezone.utc)),
        ]))
        loaded = mem_store.load()
        assert len(loaded.entries) == 1
        assert loaded.entries[0].key == "k"

    def test_external_write_with_new_mtime_invalidates(
        self, mem_store: MemoryFileStore, tmp_data_dir: Path
    ) -> None:
        import os

        mem_store.save(MemoryStore())
        mem_store.load()  # cache warm

        # Overwrite with new content and bump mtime manually to simulate an
        # external writer touching the file.
        new_payload = '{"entries": [{"key": "x", "value": "y", "recorded": "2026-01-01T00:00:00+00:00"}]}'
        mem_store.path.write_text(new_payload, encoding="utf-8")
        now_ns = mem_store.path.stat().st_mtime_ns + 1_000_000_000
        os.utime(mem_store.path, ns=(now_ns, now_ns))

        loaded = mem_store.load()
        assert len(loaded.entries) == 1
        assert loaded.entries[0].key == "x"
