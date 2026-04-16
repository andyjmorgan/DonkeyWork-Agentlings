from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.core.models import (
    CompactionEntry,
    MergeCommitted,
    MergeStarted,
    MessageEntry,
    TaskCancelled,
    TaskCompleted,
    TaskDispatched,
    TaskFailed,
    TaskStarted,
)
from agentlings.core.store import ContextNotFoundError, JournalStore, TaskJournal


@pytest.fixture
def store(tmp_data_dir: Path) -> JournalStore:
    return JournalStore(tmp_data_dir)


class TestCreate:
    def test_creates_file_in_context_dir(
        self, store: JournalStore, tmp_data_dir: Path
    ) -> None:
        store.create("ctx-1")
        assert (tmp_data_dir / "ctx-1" / "journal.jsonl").exists()

    def test_create_idempotent(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.create("ctx-1")

    def test_create_under_nested_data_dir(self, tmp_data_dir: Path) -> None:
        nested = tmp_data_dir / "deep" / "down"
        store = JournalStore(nested)
        store.create("ctx-1")
        assert (nested / "ctx-1" / "journal.jsonl").exists()


class TestExists:
    def test_exists_false(self, store: JournalStore) -> None:
        assert store.exists("nonexistent") is False

    def test_exists_true(self, store: JournalStore) -> None:
        store.create("ctx-1")
        assert store.exists("ctx-1") is True


class TestLegacyMigration:
    def test_legacy_flat_file_migrates_on_exists_check(
        self, tmp_data_dir: Path
    ) -> None:
        legacy = tmp_data_dir / "ctx-1.jsonl"
        legacy.write_text('{"t":"msg","ctx":"ctx-1","role":"user","content":[]}\n')

        store = JournalStore(tmp_data_dir)
        assert store.exists("ctx-1") is True
        assert not legacy.exists()
        assert (tmp_data_dir / "ctx-1" / "journal.jsonl").exists()

    def test_legacy_migration_preserves_content(
        self, tmp_data_dir: Path
    ) -> None:
        legacy = tmp_data_dir / "ctx-1.jsonl"
        legacy.write_text(
            '{"t":"msg","ctx":"ctx-1","role":"user","content":[{"type":"text","text":"hi"}]}\n'
        )

        store = JournalStore(tmp_data_dir)
        store.exists("ctx-1")  # triggers migration

        new_path = tmp_data_dir / "ctx-1" / "journal.jsonl"
        assert "hi" in new_path.read_text()

    def test_migration_is_idempotent(self, tmp_data_dir: Path) -> None:
        store = JournalStore(tmp_data_dir)
        store.create("ctx-1")
        # No legacy file to migrate; this should not error.
        store._maybe_migrate_legacy("ctx-1")
        assert store.exists("ctx-1")

    def test_migration_does_not_overwrite_new_path(
        self, tmp_data_dir: Path
    ) -> None:
        (tmp_data_dir / "ctx-1").mkdir()
        new_path = tmp_data_dir / "ctx-1" / "journal.jsonl"
        new_path.write_text("new-content\n")

        legacy = tmp_data_dir / "ctx-1.jsonl"
        legacy.write_text("legacy-content\n")

        store = JournalStore(tmp_data_dir)
        store._maybe_migrate_legacy("ctx-1")

        assert new_path.read_text() == "new-content\n"
        # Legacy file is left alone when there's already a new-layout file.
        assert legacy.exists()


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

        journal = tmp_data_dir / "ctx-1" / "journal.jsonl"
        assert len(journal.read_text().strip().splitlines()) == 1

        store.append("ctx-1", entry)
        assert len(journal.read_text().strip().splitlines()) == 2

    def test_append_missing_context_raises(self, store: JournalStore) -> None:
        entry = MessageEntry(
            ctx="missing",
            role="user",
            content=[{"type": "text", "text": "hello"}],
        )
        with pytest.raises(ContextNotFoundError):
            store.append("missing", entry)

    def test_append_audit_markers(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t1"))
        store.append("ctx-1", TaskCancelled(ctx="ctx-1", task_id="t1", reason="user"))
        store.append("ctx-1", TaskFailed(ctx="ctx-1", task_id="t2", reason="crash"))
        store.append("ctx-1", MergeStarted(ctx="ctx-1", task_id="t3"))
        store.append("ctx-1", MergeCommitted(ctx="ctx-1", task_id="t3"))

        entries = store.read_entries("ctx-1")
        types = [e["t"] for e in entries]
        assert types == [
            "task_dispatch",
            "task_cancel",
            "task_fail",
            "merge_start",
            "merge_commit",
        ]


class TestAppendMany:
    def test_append_many_writes_all_entries(self, store: JournalStore) -> None:
        store.create("ctx-1")
        entries = [
            MergeStarted(ctx="ctx-1", task_id="t1"),
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "hi"}],
                task_id="t1",
            ),
            MessageEntry(
                ctx="ctx-1",
                role="assistant",
                content=[{"type": "text", "text": "hello"}],
                task_id="t1",
            ),
            MergeCommitted(ctx="ctx-1", task_id="t1"),
        ]
        store.append_many("ctx-1", entries)

        result = store.read_entries("ctx-1")
        assert [e["t"] for e in result] == ["merge_start", "msg", "msg", "merge_commit"]

    def test_append_many_empty_is_noop(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append_many("ctx-1", [])
        assert store.read_entries("ctx-1") == []


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
        store.append("ctx-1", CompactionEntry(ctx="ctx-1", content="First summary."))
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "middle"}],
            ),
        )
        store.append("ctx-1", CompactionEntry(ctx="ctx-1", content="Second summary."))
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


class TestReplayFiltersAuditMarkers:
    """Audit markers must never reach the LLM's message array."""

    def test_task_dispatch_stripped(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t1"))
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "hello"}],
                task_id="t1",
            ),
        )
        messages = store.replay("ctx-1")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_task_cancel_stripped(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t1"))
        store.append(
            "ctx-1", TaskCancelled(ctx="ctx-1", task_id="t1", reason="user")
        )
        assert store.replay("ctx-1") == []

    def test_task_fail_stripped(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t1"))
        store.append(
            "ctx-1",
            TaskFailed(ctx="ctx-1", task_id="t1", reason="crash"),
        )
        assert store.replay("ctx-1") == []

    def test_merge_markers_stripped(self, store: JournalStore) -> None:
        store.create("ctx-1")
        store.append("ctx-1", MergeStarted(ctx="ctx-1", task_id="t1"))
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="user",
                content=[{"type": "text", "text": "q"}],
                task_id="t1",
            ),
        )
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1",
                role="assistant",
                content=[{"type": "text", "text": "a"}],
                task_id="t1",
            ),
        )
        store.append("ctx-1", MergeCommitted(ctx="ctx-1", task_id="t1"))

        messages = store.replay("ctx-1")
        assert len(messages) == 2
        assert [m["role"] for m in messages] == ["user", "assistant"]

    def test_replay_mixed_markers_and_messages(self, store: JournalStore) -> None:
        """Interleaved audit markers, messages, and compaction — replay is still clean."""
        store.create("ctx-1")
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t1"))
        store.append("ctx-1", MergeStarted(ctx="ctx-1", task_id="t1"))
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1", role="user", content=[{"type": "text", "text": "old"}], task_id="t1"
            ),
        )
        store.append("ctx-1", MergeCommitted(ctx="ctx-1", task_id="t1"))
        store.append("ctx-1", CompactionEntry(ctx="ctx-1", content="summary"))
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t2"))
        store.append("ctx-1", TaskFailed(ctx="ctx-1", task_id="t2", reason="x"))
        store.append("ctx-1", TaskDispatched(ctx="ctx-1", task_id="t3"))
        store.append("ctx-1", MergeStarted(ctx="ctx-1", task_id="t3"))
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1", role="user", content=[{"type": "text", "text": "new"}], task_id="t3"
            ),
        )
        store.append(
            "ctx-1",
            MessageEntry(
                ctx="ctx-1", role="assistant", content=[{"type": "text", "text": "ack"}], task_id="t3"
            ),
        )
        store.append("ctx-1", MergeCommitted(ctx="ctx-1", task_id="t3"))

        messages = store.replay("ctx-1")
        # From the latest compaction onwards: compaction, user, assistant.
        # Audit markers for t2 and t3 are stripped.
        assert [m["role"] for m in messages] == ["assistant", "user", "assistant"]
        assert messages[0]["content"] == "summary"
        assert messages[1]["content"] == [{"type": "text", "text": "new"}]
        assert messages[2]["content"] == [{"type": "text", "text": "ack"}]


class TestIterContextIds:
    def test_iter_empty(self, store: JournalStore) -> None:
        assert store.iter_context_ids() == []

    def test_iter_new_layout(self, store: JournalStore) -> None:
        store.create("ctx-a")
        store.create("ctx-b")
        assert store.iter_context_ids() == ["ctx-a", "ctx-b"]

    def test_iter_includes_legacy(self, tmp_data_dir: Path) -> None:
        # Seed one legacy flat file and one new-layout context.
        (tmp_data_dir / "legacy.jsonl").write_text("")
        (tmp_data_dir / "fresh").mkdir()
        (tmp_data_dir / "fresh" / "journal.jsonl").write_text("")

        store = JournalStore(tmp_data_dir)
        ids = store.iter_context_ids()
        assert set(ids) == {"legacy", "fresh"}
        # legacy should have been migrated by the iteration.
        assert (tmp_data_dir / "legacy" / "journal.jsonl").exists()


class TestTaskJournal:
    def test_create_and_append(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "task-abc"))
        tj.create()
        assert tj.exists()

        tj.append(
            TaskStarted(
                ctx="ctx-1",
                task_id="task-abc",
                message="hello",
                parent_cursor=0,
            )
        )
        entries = tj.read_entries()
        assert len(entries) == 1
        assert entries[0]["t"] == "task_start"
        assert entries[0]["message"] == "hello"

    def test_terminal_detection_completed(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "t1"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="t1", message="hi"))
        assert tj.terminal_entry() is None

        tj.append(
            TaskCompleted(
                ctx="ctx-1",
                task_id="t1",
                final_response=[{"type": "text", "text": "done"}],
            )
        )
        term = tj.terminal_entry()
        assert term is not None
        assert term["t"] == "task_done"

    def test_terminal_detection_cancelled(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "t1"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="t1", message="hi"))
        tj.append(TaskCancelled(ctx="ctx-1", task_id="t1", reason="user"))
        term = tj.terminal_entry()
        assert term is not None
        assert term["t"] == "task_cancel"

    def test_terminal_detection_failed(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "t1"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="t1", message="hi"))
        tj.append(TaskFailed(ctx="ctx-1", task_id="t1", reason="boom"))
        term = tj.terminal_entry()
        assert term is not None
        assert term["t"] == "task_fail"

    def test_latest_compaction_returns_most_recent(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "t1"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="t1", message="hi"))
        tj.append(CompactionEntry(ctx="ctx-1", content="first"))
        tj.append(CompactionEntry(ctx="ctx-1", content="second"))
        comp = tj.latest_compaction()
        assert comp is not None
        assert comp["content"] == "second"

    def test_latest_compaction_none_when_absent(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "t1"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="t1", message="hi"))
        assert tj.latest_compaction() is None

    def test_path_enforces_context_task_pairing(
        self, store: JournalStore, tmp_data_dir: Path
    ) -> None:
        """Path binding means `(ctx_id, task_id)` must both be correct."""
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "task-abc"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="task-abc", message="hi"))

        # Wrong context: no such file.
        wrong = TaskJournal(store.task_path("ctx-other", "task-abc"))
        assert not wrong.exists()

        # Wrong task id under the correct context: no such file.
        wrong2 = TaskJournal(store.task_path("ctx-1", "task-wrong"))
        assert not wrong2.exists()

    def test_tail_returns_last_entry(self, store: JournalStore) -> None:
        store.create("ctx-1")
        tj = TaskJournal(store.task_path("ctx-1", "t1"))
        tj.create()
        tj.append(TaskStarted(ctx="ctx-1", task_id="t1", message="hi"))
        tj.append(
            MessageEntry(
                ctx="ctx-1",
                role="assistant",
                content=[{"type": "text", "text": "last"}],
                task_id="t1",
            )
        )
        tail = tj.tail()
        assert tail is not None
        assert tail["t"] == "msg"
        assert tail["role"] == "assistant"

    def test_tail_none_when_empty(self, tmp_data_dir: Path) -> None:
        # Path to a file that doesn't exist.
        tj = TaskJournal(tmp_data_dir / "no" / "such" / "file.jsonl")
        assert tj.tail() is None
