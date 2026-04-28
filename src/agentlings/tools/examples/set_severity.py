"""String ``Enum`` parameter — the LLM sees a constrained set of values."""

from __future__ import annotations

from enum import Enum

from agentlings.tools import tool


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@tool
def set_severity(severity: Severity, reason: str) -> str:
    """Record an incident's severity level.

    The model picks one of the enum values; the framework validates the input
    before the function is invoked, so the function body can rely on
    ``severity`` being a valid ``Severity``.
    """
    return f"severity={severity.value} reason={reason}"
