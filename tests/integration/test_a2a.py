from __future__ import annotations

import httpx
import pytest

from tests.integration.a2a_client import A2ATestClient, A2AResponse


class TestA2ANewConversation:
    @pytest.mark.asyncio
    async def test_send_returns_response(self, a2a_client: A2ATestClient) -> None:
        result = await a2a_client.send("hello")
        assert isinstance(result, A2AResponse)
        assert len(result.text) > 0

    @pytest.mark.asyncio
    async def test_response_has_context_id(self, a2a_client: A2ATestClient) -> None:
        result = await a2a_client.send("what can you do?")
        assert isinstance(result, A2AResponse)
        assert result.context_id is not None
        assert len(result.context_id) > 0


class TestA2AContinuation:
    @pytest.mark.asyncio
    async def test_continuation_with_context_id(self, a2a_client: A2ATestClient) -> None:
        r1 = await a2a_client.send("hello")
        assert isinstance(r1, A2AResponse)
        assert r1.context_id is not None

        r2 = await a2a_client.send("follow up", context_id=r1.context_id)
        assert isinstance(r2, A2AResponse)
        assert r2.context_id == r1.context_id


class TestA2AAuth:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_401(self, a2a_client: A2ATestClient) -> None:
        status = await a2a_client.send_no_auth("hello")
        assert status == 401

    @pytest.mark.asyncio
    async def test_wrong_api_key_returns_401(self, server_url: str) -> None:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{server_url}/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"type": "text", "text": "hello"}],
                        }
                    },
                },
                headers={
                    "X-API-Key": "wrong-key",
                    "Content-Type": "application/json",
                },
            )
        assert resp.status_code == 401
