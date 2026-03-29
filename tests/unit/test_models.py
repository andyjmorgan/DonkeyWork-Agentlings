from __future__ import annotations

import json

from agentlings.core.models import (
    CompactionEntry,
    MessageEntry,
)


class TestJournalEntries:
    def test_message_entry_roundtrip(self) -> None:
        entry = MessageEntry(
            ctx="ctx-1",
            role="user",
            content=[{"type": "text", "text": "hello"}],
            via="a2a",
        )
        data = json.loads(entry.model_dump_json())
        assert data["t"] == "msg"
        assert data["role"] == "user"
        assert data["via"] == "a2a"
        assert "ts" in data
        restored = MessageEntry.model_validate(data)
        assert restored.ctx == "ctx-1"

    def test_compaction_entry_roundtrip(self) -> None:
        entry = CompactionEntry(ctx="ctx-1", content="A summary.")
        data = json.loads(entry.model_dump_json())
        assert data["t"] == "compact"
        assert data["content"] == "A summary."
        restored = CompactionEntry.model_validate(data)
        assert restored.content == "A summary."
