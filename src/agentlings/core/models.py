"""JSONL journal entry types for conversation persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MessageEntry(BaseModel):
    """A user or assistant message recorded in the JSONL journal.

    Attributes:
        t: Discriminator field, always ``"msg"``.
        ts: ISO 8601 timestamp of when the entry was created.
        ctx: Context ID this entry belongs to.
        role: Whether this is a ``"user"`` or ``"assistant"`` message.
        content: Anthropic-format content blocks.
        via: Protocol that originated the message (``"a2a"`` or ``"mcp"``).
        task_id: Task ID this message was handled under. ``None`` for pre-task-era
            journals or for sub-journal entries where the task context is implicit.
    """

    t: Literal["msg"] = "msg"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]
    via: Literal["a2a", "mcp"] = "a2a"
    task_id: str | None = None


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
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    content: str


class TaskStarted(BaseModel):
    """First entry of a task sub-journal recording the request that spawned it.

    Attributes:
        t: Discriminator field, always ``"task_start"``.
        ts: ISO 8601 timestamp of when the task began.
        ctx: Parent context ID.
        task_id: Unique task identifier.
        message: The original user message text.
        parent_cursor: Byte offset into the parent journal at which the task
            snapshot was taken, used for audit and reproducibility.
        via: Protocol that originated the request.
    """

    t: Literal["task_start"] = "task_start"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str
    message: str
    parent_cursor: int = 0
    via: Literal["a2a", "mcp"] = "a2a"


class TaskCompleted(BaseModel):
    """Terminal sub-journal entry marking successful task completion.

    Attributes:
        t: Discriminator field, always ``"task_done"``.
        ts: ISO 8601 timestamp of when the task completed.
        ctx: Parent context ID.
        task_id: Task identifier.
        final_response: Anthropic-format content blocks from the final assistant turn.
    """

    t: Literal["task_done"] = "task_done"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str
    final_response: list[dict[str, Any]]


class TaskFailed(BaseModel):
    """Marker for a task that terminated with a failure.

    Appended to both the task sub-journal (terminal state) and the parent
    context journal (audit marker, ignored by replay).

    Attributes:
        t: Discriminator field, always ``"task_fail"``.
        ts: ISO 8601 timestamp of when the failure was recorded.
        ctx: Parent context ID.
        task_id: Task identifier.
        reason: Short reason code (e.g. ``"process_crash_recovery"``).
        error_details: Optional longer human-readable error text.
    """

    t: Literal["task_fail"] = "task_fail"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str
    reason: str
    error_details: str | None = None


class TaskCancelled(BaseModel):
    """Marker for a task that was cancelled before completion.

    Appended to both the task sub-journal (terminal state) and the parent
    context journal (audit marker, ignored by replay).

    Attributes:
        t: Discriminator field, always ``"task_cancel"``.
        ts: ISO 8601 timestamp of when cancellation was recorded.
        ctx: Parent context ID.
        task_id: Task identifier.
        reason: Short reason code or human-readable text.
    """

    t: Literal["task_cancel"] = "task_cancel"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str
    reason: str


class TaskDispatched(BaseModel):
    """Audit marker appended to the parent journal at task ingress.

    Ignored by replay. Provides a durable record that a task was dispatched
    against this context, correlating with the sub-journal even if the task
    later fails or is cancelled.

    Attributes:
        t: Discriminator field, always ``"task_dispatch"``.
        ts: ISO 8601 timestamp of ingress.
        ctx: Parent context ID.
        task_id: Task identifier.
    """

    t: Literal["task_dispatch"] = "task_dispatch"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str


class MergeStarted(BaseModel):
    """Wrapper marker opening an atomic merge-back transaction in the parent journal.

    Paired with ``MergeCommitted`` to let startup crash-recovery detect
    partially-applied merge-backs and complete them idempotently.

    Attributes:
        t: Discriminator field, always ``"merge_start"``.
        ts: ISO 8601 timestamp.
        ctx: Parent context ID.
        task_id: Task whose merge-back is beginning.
    """

    t: Literal["merge_start"] = "merge_start"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str


class MergeCommitted(BaseModel):
    """Wrapper marker closing an atomic merge-back transaction in the parent journal.

    Presence of this entry without a prior ``MergeStarted`` for the same
    ``task_id`` is legal (idempotent repair). Absence of this entry after
    a ``MergeStarted`` signals an incomplete merge-back to be repaired on startup.

    Attributes:
        t: Discriminator field, always ``"merge_commit"``.
        ts: ISO 8601 timestamp.
        ctx: Parent context ID.
        task_id: Task whose merge-back has committed.
    """

    t: Literal["merge_commit"] = "merge_commit"
    ts: str = Field(default_factory=_now_iso)
    ctx: str
    task_id: str


JournalEntry = Annotated[
    MessageEntry
    | CompactionEntry
    | TaskStarted
    | TaskCompleted
    | TaskFailed
    | TaskCancelled
    | TaskDispatched
    | MergeStarted
    | MergeCommitted,
    Field(discriminator="t"),
]


AUDIT_MARKER_TYPES: frozenset[str] = frozenset({
    "task_dispatch",
    "task_cancel",
    "task_fail",
    "merge_start",
    "merge_commit",
})
"""Entry types that are audit-only and must be stripped from replay output."""


TASK_TERMINAL_TYPES: frozenset[str] = frozenset({
    "task_done",
    "task_fail",
    "task_cancel",
})
"""Sub-journal entry types that mark a task's terminal state."""
