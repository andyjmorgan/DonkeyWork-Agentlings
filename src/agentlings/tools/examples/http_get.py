"""Async I/O tool with ``Annotated[..., Field(...)]`` parameter descriptions."""

from __future__ import annotations

from typing import Annotated

import httpx
from pydantic import Field

from agentlings.tools import tool


@tool
async def http_get(
    url: Annotated[str, Field(description="Absolute URL to fetch (must be http/https).")],
    timeout_seconds: Annotated[
        float,
        Field(ge=1.0, le=60.0, description="Per-request timeout in seconds."),
    ] = 10.0,
) -> str:
    """Fetch a URL via HTTP GET and return the response body as text.

    The tool follows redirects and raises if the response status is >= 400 so
    the LLM sees a clear failure signal rather than a misleading empty body.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_seconds) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text
