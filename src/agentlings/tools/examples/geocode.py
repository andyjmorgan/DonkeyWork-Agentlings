"""Env-var-driven config + nested ``BaseModel`` parameter.

Demonstrates the recommended pattern for tools that need credentials or
endpoint config: read them from the process environment inside the tool —
the framework stays out of secret plumbing entirely.
"""

from __future__ import annotations

import os
from typing import Annotated

import httpx
from pydantic import BaseModel, Field

from agentlings.tools import tool


class Address(BaseModel):
    """A postal address. Only the fields the geocoder needs are required."""

    street: str
    city: str
    country: Annotated[str, Field(description="ISO 3166-1 alpha-2 country code, e.g. 'IE'.")]


class Coordinates(BaseModel):
    lat: float
    lng: float


@tool
async def geocode(address: Address) -> Coordinates:
    """Resolve a postal address to latitude/longitude.

    Reads ``GEOCODE_API_KEY`` from the environment. Tools own their own
    secret plumbing; the framework is intentionally not involved.
    """
    api_key = os.environ.get("GEOCODE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEOCODE_API_KEY is not set. Configure the tool's environment "
            "before enabling it on an agent."
        )

    base_url = os.environ.get("GEOCODE_BASE_URL", "https://api.example.com/geocode")
    params = {
        "street": address.street,
        "city": address.city,
        "country": address.country,
        "key": api_key,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(base_url, params=params)
        response.raise_for_status()
        payload = response.json()
        return Coordinates(lat=payload["lat"], lng=payload["lng"])
