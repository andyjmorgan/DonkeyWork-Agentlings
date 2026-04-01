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
