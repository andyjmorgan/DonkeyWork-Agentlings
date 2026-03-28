from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.models import CompactionEntry, MessageEntry
from agentlings.store import ContextNotFoundError, JournalStore


@pytest.fixture
def store(tmp_data_dir: Path) -> JournalStore:
    return JournalStore(tmp_data_dir)


class TestCreate:
    def test_creates_file(self, store: JournalStore, tmp_data_dir: Path) -> None:
        store.create("ctx-1")
        assert (tmp_data_dir / "ctx-1.jsonl").exists()

    def test_create_idempotent(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.create("ctx-1")


class TestExists:
    def test_exists_false(self, store: JournalStore) -> None:
        assert store.exists("nonexistent") is False

    def test_exists_true(self, store: JournalStore) -> None:
        store.create("ctx-1")
        assert store.exists("ctx-1") is True


class TestAppend:
    def test_append_grows_file(
        self, store: JournalStore, tmp_data_dir: Path
    ) -> None:
        store.create("ctx-1")
        entry = MessageEntry(
            ctx="ctx-1",
            role="user",
            content=[{"type": "text", "text": "hello"}],
        )
        store.append("ctx-1", entry)

        lines = (tmp_data_dir / "ctx-1.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

        store.append("ctx-1", entry)
        lines = (tmp_data_dir / "ctx-1.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2

    def test_append_missing_context_raises(self, store: JournalStore) -> None:
        entry = MessageEntry(
            ctx="missing",
            role="user",
            content=[{"type": "text", "text": "hello"}],
        )
        with pytest.raises(ContextNotFoundError):
            store.append("missing", entry)


class TestReplay:
    def test_replay_empty(self, store: JournalStore) -> None:
        store.create("ctx-1")
        assert store.replay("ctx-1") == []

    def test_replay_returns_messages(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "hello"}],
            ),
        )
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="assistant",
                content=[{"type": "text", "text": "hi there"}],
            ),
        )
        messages = store.replay("ctx-1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_replay_from_compaction_marker(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "old message"}],
            ),
        )
        store.append(
            "ctx-1",
            CompactionEntry(ctx="ctx-1", content="Summary of old messages."),
        )
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "new message"}],
            ),
        )

        messages = store.replay("ctx-1")
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "Summary of old messages."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == [{"type": "text", "text": "new message"}]

    def test_replay_uses_last_compaction(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append(
            "ctx-1",
            CompactionEntry(ctx="ctx-1", content="First summary."),
        )
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "middle"}],
            ),
        )
        store.append(
            "ctx-1",
            CompactionEntry(ctx="ctx-1", content="Second summary."),
        )
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "latest"}],
            ),
        )

        messages = store.replay("ctx-1")
        assert len(messages) == 2
        assert messages[0]["content"] == "Second summary."
        assert messages[1]["content"] == [{"type": "text", "text": "latest"}]

    def test_replay_missing_context_raises(self, store: JournalStore) -> None:
        with pytest.raises(ContextNotFoundError):
            store.replay("nonexistent")
