"""Live API tests against the real Anthropic API.

These tests require a valid ANTHROPIC_API_KEY in .env.test.
Skip automatically if the key is not available.
Run with: pytest tests/unit/test_live_api.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env.test")

API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5-20251001")

pytestmark = pytest.mark.skipif(
    not API_KEY or API_KEY == "not-configured",
    reason="No ANTHROPIC_API_KEY in .env.test",
)


@pytest.fixture
def anthropic_client():
    from agentlings.core.llm import AnthropicLLMClient
    return AnthropicLLMClient(api_key=API_KEY, model=MODEL, max_tokens=1024)


class TestCountTokens:
    async def test_count_tokens_returns_int(self, anthropic_client) -> None:
        count = await anthropic_client.count_tokens("Hello, world!")
        assert isinstance(count, int)
        assert count > 0

    async def test_longer_text_has_more_tokens(self, anthropic_client) -> None:
        short = await anthropic_client.count_tokens("Hi")
        long = await anthropic_client.count_tokens("This is a much longer sentence with many more words in it.")
        assert long > short


class TestComplete:
    async def test_basic_completion(self, anthropic_client) -> None:
        response = await anthropic_client.complete(
            system=[{"type": "text", "text": "You are a test agent. Be very brief."}],
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            tools=[],
        )
        assert len(response.content) > 0
        assert response.content[0]["type"] == "text"
        assert response.stop_reason == "end_turn"

    async def test_structured_output(self, anthropic_client) -> None:
        """Verify output_config works for structured JSON output."""
        from agentlings.core.memory_models import ConversationSummary, strict_json_schema

        import anthropic

        client = anthropic.AsyncAnthropic(api_key=API_KEY)
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": "You summarize conversations."}],
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this conversation:\n"
                    "User: What's the status of node3?\n"
                    "Agent: Node3 has high memory pressure, CoreDNS is restarting."
                ),
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": strict_json_schema(ConversationSummary),
                }
            },
        )

        text = response.content[0].text
        parsed = ConversationSummary.model_validate_json(text)
        assert parsed.summary
        assert isinstance(parsed.memory_candidates, list)


class TestBatchAPI:
    async def test_batch_create_and_retrieve(self, anthropic_client) -> None:
        """Verify the batch API call shapes work end-to-end."""
        from agentlings.core.llm import BatchRequest
        from agentlings.core.memory_models import ConversationSummary, strict_json_schema

        request = BatchRequest(
            custom_id="test-conv-1",
            system=[{"type": "text", "text": "You summarize conversations briefly."}],
            messages=[{
                "role": "user",
                "content": "Summarize: User asked about disk space. Agent checked and reported 80% full.",
            }],
            max_tokens=1024,
            output_schema=strict_json_schema(ConversationSummary),
        )

        batch_ids = await anthropic_client.batch_create([request], model=MODEL)
        assert len(batch_ids) == 1
        batch_id = batch_ids[0]
        assert batch_id

        import asyncio
        for _ in range(60):
            status = await anthropic_client.batch_status(batch_id)
            if status.processing_status == "ended":
                break
            await asyncio.sleep(5)

        assert status.processing_status == "ended"
        assert status.succeeded >= 1

        results = await anthropic_client.batch_results(batch_id)
        assert len(results) == 1
        assert results[0].custom_id == "test-conv-1"
        assert results[0].status == "succeeded"

        text = ""
        for block in results[0].content:
            if block.get("type") == "text":
                text = block["text"]
                break

        parsed = ConversationSummary.model_validate_json(text)
        assert parsed.summary
