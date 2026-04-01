"""Tests for memory Pydantic models: serialization round-trips."""

from __future__ import annotations

from datetime import datetime, timezone

from agentlings.core.memory_models import (
    ConsolidatedMemory,
    ConversationSummary,
    MemoryCandidate,
    MemoryEntry,
    MemoryStore,
)


class TestMemoryEntry:
    def test_roundtrip(self) -> None:
        entry = MemoryEntry(
            key="cluster-nodes",
            value="4 nodes in the cluster",
            recorded=datetime(2026, 4, 1, 10, 0, 0, tzinfo=timezone.utc),
        )
        dumped = entry.model_dump_json()
        restored = MemoryEntry.model_validate_json(dumped)
        assert restored.key == "cluster-nodes"
        assert restored.value == "4 nodes in the cluster"
        assert restored.recorded.tzinfo is not None


class TestMemoryStore:
    def test_empty_store(self) -> None:
        store = MemoryStore()
        assert store.entries == []

    def test_roundtrip(self) -> None:
        store = MemoryStore(entries=[
            MemoryEntry(
                key="fact-1", value="value-1",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
            MemoryEntry(
                key="fact-2", value="value-2",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        dumped = store.model_dump_json()
        restored = MemoryStore.model_validate_json(dumped)
        assert len(restored.entries) == 2
        assert restored.entries[0].key == "fact-1"


class TestConversationSummary:
    def test_roundtrip(self) -> None:
        summary = ConversationSummary(
            summary="User asked about cluster health.",
            memory_candidates=[
                MemoryCandidate(key="coredns", value="restarting on node3"),
            ],
        )
        dumped = summary.model_dump_json()
        restored = ConversationSummary.model_validate_json(dumped)
        assert restored.summary == "User asked about cluster health."
        assert len(restored.memory_candidates) == 1

    def test_empty_candidates(self) -> None:
        summary = ConversationSummary(summary="Trivial conversation.")
        assert summary.memory_candidates == []


class TestConsolidatedMemory:
    def test_roundtrip(self) -> None:
        consolidated = ConsolidatedMemory(entries=[
            MemoryEntry(
                key="fact", value="value",
                recorded=datetime(2026, 4, 1, tzinfo=timezone.utc),
            ),
        ])
        dumped = consolidated.model_dump_json()
        restored = ConsolidatedMemory.model_validate_json(dumped)
        assert len(restored.entries) == 1
