from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# JSONL journal entries
# ---------------------------------------------------------------------------


class MessageEntry(BaseModel):
    t: Literal["msg"] = "msg"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ctx: str
    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]
    via: Literal["a2a", "mcp"] = "a2a"


class CompactionEntry(BaseModel):
    t: Literal["compact"] = "compact"
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ctx: str
    content: str


JournalEntry = Annotated[
    MessageEntry | CompactionEntry,
    Field(discriminator="t"),
]
