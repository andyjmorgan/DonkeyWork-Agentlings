"""Pydantic models for JSONL journal entries used in conversation persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class MessageEntry(BaseModel):
    """A user or assistant message recorded in the conversation journal.

    Attributes:
        t: Discriminator field, always "msg".
        ts: ISO-8601 UTC timestamp of when the entry was created.
        ctx: Context ID linking this entry to a conversation.
        role: Whether this message is from the "user" or "assistant".
        content: List of content blocks (text, tool_use, tool_result, etc.).
        via: Protocol the message arrived through.
    """
    t: Literal["msg"] = "msg"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ctx: str
    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]
    via: Literal["a2a", "mcp"] = "a2a"


class CompactionEntry(BaseModel):
    """A compaction marker that summarises earlier conversation history.

    When replaying a journal, replay starts from the last compaction entry
    rather than the beginning, keeping context windows manageable.

    Attributes:
        t: Discriminator field, always "compact".
        ts: ISO-8601 UTC timestamp of when the compaction was created.
        ctx: Context ID linking this entry to a conversation.
        content: The compacted summary text.
    """
    t: Literal["compact"] = "compact"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ctx: str
    content: str


JournalEntry = Annotated[
    MessageEntry | CompactionEntry,
    Field(discriminator="t"),
]
