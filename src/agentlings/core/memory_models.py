"""Pydantic models for memory persistence and sleep cycle structured outputs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Generate a JSON schema with ``additionalProperties: false`` on all objects.

    The Anthropic structured outputs API requires this. Pydantic's default
    ``model_json_schema()`` does not set it.

    Args:
        model: The Pydantic model class.

    Returns:
        A JSON Schema dict suitable for ``output_config.format.schema``.
    """
    schema = deepcopy(model.model_json_schema())
    _patch_additional_properties(schema)

    if "$defs" in schema:
        for defn in schema["$defs"].values():
            _patch_additional_properties(defn)

    return schema


def _patch_additional_properties(obj: dict[str, Any]) -> None:
    if obj.get("type") == "object" or "properties" in obj:
        obj["additionalProperties"] = False


class MemoryEntry(BaseModel):
    """A single fact in the agent's long-term memory.

    Attributes:
        key: Unique identifier for this memory entry (slug-style).
        value: The knowledge or fact to remember.
        recorded: When this entry was created or last updated.
    """

    key: str
    value: str
    recorded: datetime


class MemoryStore(BaseModel):
    """On-disk format for the agent's memory file (``memory.json``).

    Attributes:
        entries: All current memory entries.
    """

    entries: list[MemoryEntry] = Field(default_factory=list)


class MemoryCandidate(BaseModel):
    """A potential new memory entry extracted during sleep.

    Attributes:
        key: Proposed key for the memory entry.
        value: The candidate fact.
    """

    key: str
    value: str


class ConversationSummary(BaseModel):
    """Structured output from the deep sleep per-conversation summary call.

    Attributes:
        summary: Concise narrative of what happened in the conversation.
        memory_candidates: New facts worth adding to long-term memory.
    """

    summary: str
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)


class ConsolidatedMemory(BaseModel):
    """Structured output from the REM consolidation call.

    The LLM integrates new candidates with existing memory, prunes stale
    entries, and returns the complete updated memory store.

    Attributes:
        entries: The full set of memory entries after consolidation.
    """

    entries: list[MemoryEntry] = Field(default_factory=list)
