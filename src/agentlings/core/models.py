"""JSONL journal entry types for conversation persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


class MessageEntry(BaseModel):
    """A user or assistant message recorded in the JSONL journal.

    Attributes:
        t: Discriminator field, always ``"msg"``.
        ts: ISO 8601 timestamp of when the entry was created.
        ctx: Context ID this entry belongs to.
        role: Whether this is a ``"user"`` or ``"assistant"`` message.
        content: Anthropic-format content blocks.
        via: Protocol that originated the message (``"a2a"`` or ``"mcp"``).
    """

    t: Literal["msg"] = "msg"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ctx: str
    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]
    via: Literal["a2a", "mcp"] = "a2a"


class CompactionEntry(BaseModel):
    """A compaction marker in the JSONL journal.

    When the Anthropic API compacts a conversation, the summary is stored
    as a compaction entry. On replay, all entries before the last compaction
    marker are skipped.

    Attributes:
        t: Discriminator field, always ``"compact"``.
        ts: ISO 8601 timestamp of when compaction occurred.
        ctx: Context ID this entry belongs to.
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
