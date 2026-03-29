from __future__ import annotations

import httpx
import pytest


class TestAgentCard:
    @pytest.mark.asyncio
    async def test_returns_200_no_auth(self, server_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{server_url}/.well-known/agent-card.json")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_valid_json(self, server_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{server_url}/.well-known/agent-card.json")
        card = resp.json()
        assert card["name"] == "test-agent"
        assert card["description"] == "A test agent"

    @pytest.mark.asyncio
    async def test_capabilities(self, server_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{server_url}/.well-known/agent-card.json")
        card = resp.json()
        assert card["capabilities"]["streaming"] is False
        assert card["capabilities"]["pushNotifications"] is False

    @pytest.mark.asyncio
    async def test_security_schemes(self, server_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{server_url}/.well-known/agent-card.json")
        card = resp.json()
        assert "apiKey" in card["securitySchemes"]
        assert card["security"] == [{"apiKey": []}]

    @pytest.mark.asyncio
    async def test_legacy_path_still_works(self, server_url: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{server_url}/.well-known/agent.json")
        assert resp.status_code == 200
