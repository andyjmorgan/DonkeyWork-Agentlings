"""Smallest possible tool — echoes its input back."""

from __future__ import annotations

from agentlings.tools import tool


@tool
def echo(message: str) -> str:
    """Echo a message back unchanged.

    Useful for sanity-checking that the tool plumbing is working end-to-end.
    """
    return message
