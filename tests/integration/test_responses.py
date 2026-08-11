"""Live integration tests for the OpenAI Responses wire format.

These run against an OpenAI-compatible ``/v1/responses`` endpoint — in this
lab, the slipspace gateway — with two model profiles:

* a real OpenAI model (``RESPONSES_OPENAI_MODEL``, default ``gpt-5-mini``),
  which the gateway routes upstream to api.openai.com;
* a local Ollama responses target (``RESPONSES_LOCAL_MODEL``, default
  ``muse-glimmer:latest`` — a reasoning model, so the local profile also
  exercises reasoning-item translation from a stateless backend).

They are gated behind the ``responses`` pytest marker and the
``RESPONSES_API_KEY`` environment variable so they don't run in default CI.
Credentials come from the environment only — never from the repo (mirroring
``test_live_api.py``'s convention).

Run with:

    RESPONSES_API_KEY=... RESPONSES_BASE_URL=https://sluice.donkeywork.dev \
        pytest -m responses tests/integration/test_responses.py

Test layout (mirrors ``test_ollama.py``):

* ``TestResponsesLLMClientLive`` — drives ``ResponsesLLMClient`` directly:
  text completion for both profiles, and a full stateless tool-call round
  trip (function_call → replayed history incl. reasoning items →
  function_call_output → final text) on the OpenAI profile. This is the
  contract the Responses translation actually has to hold.
* ``TestResponsesA2A`` — boots a real agentling with
  ``AGENT_WIRE_FORMAT=responses`` per profile and round-trips an A2A
  message through the full task engine (spawn → merge-back). The OpenAI
  profile also exercises the tool path (spawn → bash tool call →
  merge-back); the local profile runs tool-less — small local models are
  not reliably tool-capable, the same reasoning as the Ollama suite.
"""

from __future__ import annotations

import os

import pytest

from agentlings.config import AgentConfig
from agentlings.core.llm_responses import ResponsesLLMClient
from tests.integration.a2a_client import A2AResponse, A2ATestClient
from tests.integration.conftest import (
    free_port,
    start_agentling_server,
    stop_agentling_server,
)

RESPONSES_BASE_URL = os.environ.get(
    "RESPONSES_BASE_URL", "https://sluice.donkeywork.dev",
)
RESPONSES_API_KEY = os.environ.get("RESPONSES_API_KEY", "")
OPENAI_MODEL = os.environ.get("RESPONSES_OPENAI_MODEL", "gpt-5-mini")
LOCAL_MODEL = os.environ.get("RESPONSES_LOCAL_MODEL", "muse-glimmer:latest")
ALL_MODELS = [OPENAI_MODEL, LOCAL_MODEL]

# Reasoning models spend output budget on thinking before any visible text;
# keep the cap high enough that answers survive.
MAX_TOKENS = 2048

pytestmark = [
    pytest.mark.responses,
    pytest.mark.skipif(
        not RESPONSES_API_KEY,
        reason="RESPONSES_API_KEY not set",
    ),
]


def _client(model: str) -> ResponsesLLMClient:
    return ResponsesLLMClient(
        api_key=RESPONSES_API_KEY,
        model=model,
        max_tokens=MAX_TOKENS,
        base_url=RESPONSES_BASE_URL,
    )


ECHO_TOOL = {
    "name": "echo",
    "description": "Echo back the supplied text.",
    "input_schema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}


# --------------------------------------------------------------------------- #
# Direct LLM client tests — the contract tests for the Responses translation.
# --------------------------------------------------------------------------- #


class TestResponsesLLMClientLive:
    @pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.replace(":", "_"))
    async def test_completion_round_trips_text(self, model: str) -> None:
        """A prompt in, a non-empty text block and usage out — both profiles."""
        client = _client(model)
        try:
            response = await client.complete(
                system=[{"type": "text", "text": "Reply concisely."}],
                messages=[{
                    "role": "user",
                    "content": "Reply with exactly the token RESPONSES-OK and nothing else.",
                }],
                tools=[],
            )
        finally:
            await client.aclose()

        assert response.content, f"{model}: empty content list"
        assert response.stop_reason == "end_turn", (
            f"{model}: unexpected stop_reason {response.stop_reason!r}"
        )
        text_blocks = [b for b in response.content if b.get("type") == "text"]
        assert text_blocks, f"{model}: no text blocks in {response.content!r}"
        joined = " ".join(b.get("text", "") for b in text_blocks)
        assert joined.strip(), f"{model}: text blocks were empty"
        assert response.usage.get("output_tokens", 0) > 0, (
            f"{model}: usage not reported: {response.usage!r}"
        )

    async def test_stateless_tool_round_trip(self) -> None:
        """The full stateless loop against a real OpenAI reasoning model.

        Turn 1 must produce a ``tool_use`` block translated from a
        ``function_call`` item. The entire assistant turn (including any
        reasoning/thinking blocks) is then replayed with the tool result —
        this is exactly what the completion cycle does — and turn 2 must
        yield a terminal text response. Reasoning models reject replayed
        function calls whose reasoning items are missing, so this test
        proves the item-identity preservation works end to end.
        """
        client = _client(OPENAI_MODEL)
        try:
            system = [{"type": "text", "text": "Use the echo tool when asked."}]
            messages: list[dict] = [{
                "role": "user",
                "content": "Call the echo tool with text='ping', then tell me what it returned.",
            }]
            turn1 = await client.complete(system=system, messages=messages, tools=[ECHO_TOOL])

            assert turn1.stop_reason == "tool_use", (
                f"expected tool_use, got {turn1.stop_reason!r}: {turn1.content!r}"
            )
            tool_uses = [b for b in turn1.content if b.get("type") == "tool_use"]
            assert len(tool_uses) == 1
            assert tool_uses[0]["name"] == "echo"
            assert tool_uses[0]["input"].get("text"), (
                f"echo tool input missing text: {tool_uses[0]['input']!r}"
            )

            messages.append({"role": "assistant", "content": turn1.content})
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": tool_uses[0]["id"],
                "content": "ping",
                "is_error": False,
            }]})
            turn2 = await client.complete(system=system, messages=messages, tools=[ECHO_TOOL])
        finally:
            await client.aclose()

        assert turn2.stop_reason == "end_turn", (
            f"expected terminal turn, got {turn2.stop_reason!r}: {turn2.content!r}"
        )
        text = " ".join(
            b.get("text", "") for b in turn2.content if b.get("type") == "text"
        )
        assert "ping" in text.lower(), f"final answer didn't mention the tool result: {text!r}"

    async def test_unknown_model_raises_api_error(self) -> None:
        """The gateway's no-binding rejection surfaces as ResponsesAPIError."""
        from agentlings.core.llm_responses import ResponsesAPIError

        client = _client("definitely-not-a-model")
        try:
            with pytest.raises(ResponsesAPIError):
                await client.complete(
                    system=[],
                    messages=[{"role": "user", "content": "hi"}],
                    tools=[],
                )
        finally:
            await client.aclose()


# --------------------------------------------------------------------------- #
# End-to-end A2A tests — the full task-engine path over the responses wire
# format: spawn → (tool call →) merge-back.
# --------------------------------------------------------------------------- #


def _boot_agentling(tmp_path_factory, model: str, tools_yaml: str):
    api_key = "responses-integration-test-key"

    safe = model.replace(":", "_").replace("/", "_").replace(".", "_")
    data_dir = tmp_path_factory.mktemp(f"data_{safe}")
    agent_yaml = tmp_path_factory.mktemp(f"config_{safe}") / "agent.yaml"
    agent_yaml.write_text(
        "name: responses-test-agent\n"
        "description: Integration test agent on the responses wire format\n"
        f"{tools_yaml}"
        "sleep:\n"
        "  enabled: false\n"
    )

    config = AgentConfig(
        openai_api_key=RESPONSES_API_KEY,
        openai_base_url=RESPONSES_BASE_URL,
        agent_wire_format="responses",
        agent_api_key=api_key,
        agent_data_dir=data_dir,
        agent_llm_backend="anthropic",
        agent_model=model,
        agent_max_tokens=MAX_TOKENS,
        agent_host="127.0.0.1",
        agent_port=free_port(),
        agent_config=str(agent_yaml),
        agent_task_await_seconds=120,
    )
    url, server, thread = start_agentling_server(config)
    return url, api_key, server, thread


@pytest.fixture(scope="module", params=ALL_MODELS, ids=lambda m: m.replace(":", "_"))
def responses_agentling(request, tmp_path_factory):
    """One tool-less agentling per model profile (both must round-trip A2A)."""
    model = request.param
    url, api_key, server, thread = _boot_agentling(
        tmp_path_factory, model, "tools: []\n",
    )
    yield url, api_key, model
    stop_agentling_server(server, thread)


@pytest.fixture(scope="module")
def responses_tool_agentling(tmp_path_factory):
    """A bash-tooled agentling on the OpenAI profile for the tool-path test."""
    url, api_key, server, thread = _boot_agentling(
        tmp_path_factory, OPENAI_MODEL, "tools:\n  - bash\n",
    )
    yield url, api_key
    stop_agentling_server(server, thread)


class TestResponsesA2A:
    async def test_round_trip(self, responses_agentling) -> None:
        """A user message in, an assistant text response out — full task
        engine (spawn → merge-back) over the responses wire format.

        Asserts on shape (non-empty text, present context_id), not content
        — same policy as the Ollama suite.
        """
        url, api_key, model = responses_agentling
        client = A2ATestClient(url, api_key, request_timeout=180.0)
        result = await client.send("Say hi in one short sentence.")
        assert isinstance(result, A2AResponse), f"{model}: {result!r}"
        assert result.context_id, f"{model}: missing context_id"
        assert result.text.strip(), (
            f"{model}: empty assistant response — task likely returned a "
            f"slow-path handle. Raw: {result.raw!r}"
        )

    async def test_multi_turn_follow_up_in_same_context(self, responses_agentling) -> None:
        """A follow-up message in an EXISTING context — the second request
        replays the journaled assistant text through the translation
        layer (assistant-role message with output_text content parts).

        Live verification of the input shape: api.openai.com accepts
        assistant-role input messages with output_text parts (and rejects
        input_text there — 'Supported values are: output_text and
        refusal'), as does Ollama's /v1/responses. A regression in the
        history translation surfaces here as a failed task or empty
        follow-up, which the fresh-context tests can never catch.
        """
        url, api_key, model = responses_agentling
        client = A2ATestClient(url, api_key, request_timeout=180.0)
        first = await client.send(
            "Pick one English word as a codeword and tell it to me plainly."
        )
        assert isinstance(first, A2AResponse), f"{model}: {first!r}"
        assert first.context_id, f"{model}: missing context_id"
        assert first.text.strip(), f"{model}: empty first reply"

        follow_up = await client.send(
            "Repeat the codeword you just told me, and nothing else.",
            context_id=first.context_id,
        )
        assert isinstance(follow_up, A2AResponse), f"{model}: {follow_up!r}"
        assert follow_up.context_id == first.context_id
        assert follow_up.text.strip(), (
            f"{model}: empty follow-up — assistant-history replay likely "
            f"rejected by the backend. First: {first.text!r}"
        )

    async def test_tool_call_merge_back(self, responses_tool_agentling, tmp_path) -> None:
        """spawn → bash tool call → merge-back, end to end over responses.

        The sentinel is a random token written to a file the model never
        sees in-prompt — it cannot be inferred or derived, so the final
        answer containing it proves the completion cycle actually executed
        the bash tool and replayed the (reasoning + function_call +
        output) history through the translation layer.
        """
        from uuid import uuid4

        sentinel = f"RESPONSES-SENTINEL-{uuid4().hex[:16]}"
        sentinel_file = tmp_path / "sentinel.txt"
        sentinel_file.write_text(sentinel + "\n")

        url, api_key = responses_tool_agentling
        client = A2ATestClient(url, api_key, request_timeout=180.0)
        result = await client.send(
            "Use the bash tool to run exactly this command: "
            f"cat {sentinel_file}. "
            "Then report the command's full output back to me verbatim."
        )
        assert isinstance(result, A2AResponse), f"unexpected result: {result!r}"
        assert result.context_id, "missing context_id"
        assert sentinel in result.text, (
            f"final answer missing tool output sentinel: {result.text!r}"
        )
