"""Live integration tests against a real Ollama server.

These tests require an Ollama instance reachable on the network with the
models ``qwen3:4b`` and ``gemma3:4b`` already pulled. They are gated
behind the ``ollama`` pytest marker and the ``OLLAMA_HOST`` environment
variable so they don't run as part of the default CI loop.

Defaults assume the dev cluster at ``192.168.69.21:11434``. Override
with ``OLLAMA_HOST=http://host:port pytest -m ollama``.

Run with:

    pytest -m ollama tests/integration/test_ollama.py

Skip with:

    pytest -m "not ollama"  # the default in most CI configurations

Test layout:

* ``TestOllamaLLMClient`` — drives ``AnthropicLLMClient`` directly against
  Ollama. Proves the ``base_url`` override reaches the right host and
  that prompts come back through the Anthropic-shaped content-block
  parser. This is the contract Ollama compatibility actually has to hold.
* ``TestOllamaA2A`` — boots a real agentling pointed at Ollama and sends
  one A2A message per model. Smaller smoke test — no tools (``gemma3:4b``
  rejects requests with a ``tools`` array) and no memory-recall asserts
  (4B-class models are too unreliable for that within fast-path timeouts).
"""

from __future__ import annotations

import asyncio
import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from agentlings.config import AgentConfig
from agentlings.core.llm import AnthropicLLMClient
from agentlings.server import _create_app
from tests.integration.a2a_client import A2AResponse, A2ATestClient

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.69.21:11434")
OLLAMA_MODELS = ["qwen3:4b", "gemma3:4b"]


def _ollama_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        resp = httpx.get(f"{url}/api/tags", timeout=timeout)
        return resp.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        not _ollama_reachable(OLLAMA_HOST),
        reason=f"Ollama not reachable at {OLLAMA_HOST}",
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- #
# Direct LLM client tests — exercise the AnthropicLLMClient against Ollama
# without the full A2A/MCP stack. These are the contract tests for the
# ``base_url`` override added for Ollama support.
# --------------------------------------------------------------------------- #


class TestOllamaLLMClient:
    @pytest.mark.parametrize("model", OLLAMA_MODELS)
    async def test_completion_reaches_ollama(self, model: str) -> None:
        """``AnthropicLLMClient`` with ``base_url`` should hit the Ollama endpoint
        and return a non-empty text content block.

        The prompt asks for the literal token ``OLLAMA-OK`` so we have a
        concrete signal that the pipeline carried text both ways.
        """
        client = AnthropicLLMClient(
            api_key="ollama",
            model=model,
            max_tokens=256,
            base_url=OLLAMA_HOST,
        )
        response = await client.complete(
            system=[{"type": "text", "text": "Reply concisely."}],
            messages=[{
                "role": "user",
                "content": "Reply with exactly the token OLLAMA-OK and nothing else.",
            }],
            tools=[],
        )

        assert response.content, f"{model}: empty content list"
        assert response.stop_reason, f"{model}: missing stop_reason"

        text_blocks = [b for b in response.content if b.get("type") == "text"]
        assert text_blocks, f"{model}: no text content blocks in {response.content!r}"
        joined = " ".join(b.get("text", "") for b in text_blocks)
        assert joined.strip(), f"{model}: text blocks were empty"

    async def test_qwen_supports_tools(self) -> None:
        """qwen3:4b is the tool-capable model on this Ollama box.

        We pass an Anthropic-shaped tool schema and assert the Ollama
        endpoint accepts the request (i.e. doesn't 400 with
        ``model does not support tools`` the way gemma3:4b does). We
        intentionally don't assert the model *invokes* the tool — small
        models are inconsistent and that's a model behaviour test, not a
        compatibility test.
        """
        client = AnthropicLLMClient(
            api_key="ollama",
            model="qwen3:4b",
            max_tokens=256,
            base_url=OLLAMA_HOST,
        )
        response = await client.complete(
            system=[{"type": "text", "text": "You may use the echo tool."}],
            messages=[{
                "role": "user",
                "content": "Please call the echo tool with text='hi'.",
            }],
            tools=[{
                "name": "echo",
                "description": "Echo back the supplied text.",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            }],
        )
        assert response.content, "qwen3:4b returned no content with tools attached"

    async def test_gemma_rejects_tools(self) -> None:
        """gemma3:4b on Ollama rejects requests that include a tool schema.

        This locks in the operational contract: an agentling configured
        with tools cannot be backed by gemma3:4b. If Ollama later starts
        supporting tools for gemma, this test will start failing — at
        which point we can flip gemma into the tool-capable list above.
        """
        client = AnthropicLLMClient(
            api_key="ollama",
            model="gemma3:4b",
            max_tokens=128,
            base_url=OLLAMA_HOST,
        )
        with pytest.raises(Exception) as exc_info:
            await client.complete(
                system=[{"type": "text", "text": "hi"}],
                messages=[{"role": "user", "content": "hello"}],
                tools=[{
                    "name": "echo",
                    "description": "echo",
                    "input_schema": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                }],
            )
        assert "does not support tools" in str(exc_info.value).lower() or \
               "400" in str(exc_info.value), \
               f"Expected gemma3:4b tool rejection, got: {exc_info.value!r}"

    async def test_completion_without_tools_works_for_gemma(self) -> None:
        """Same gemma3:4b request as above but without a tools array — should succeed."""
        client = AnthropicLLMClient(
            api_key="ollama",
            model="gemma3:4b",
            max_tokens=128,
            base_url=OLLAMA_HOST,
        )
        response = await client.complete(
            system=[{"type": "text", "text": "Reply concisely."}],
            messages=[{"role": "user", "content": "Say hi in one word."}],
            tools=[],
        )
        text_blocks = [b for b in response.content if b.get("type") == "text"]
        assert text_blocks, f"gemma3:4b returned no text without tools: {response.content!r}"


# --------------------------------------------------------------------------- #
# End-to-end A2A smoke test — boots a real agentling pointed at Ollama and
# round-trips a single message. No tools (gemma rejects them, qwen handles
# them flakily on a 4B model). Proves the server lifespan, A2A executor,
# task engine, and Anthropic-compatible client all wire up correctly.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module", params=OLLAMA_MODELS, ids=lambda m: m.replace(":", "_"))
def ollama_agentling(request, tmp_path_factory):
    model = request.param
    api_key = "ollama-integration-test-key"
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    safe = model.replace(":", "_").replace("/", "_")
    data_dir = tmp_path_factory.mktemp(f"data_{safe}")
    agent_yaml = tmp_path_factory.mktemp(f"config_{safe}") / "agent.yaml"
    # No tools: gemma3:4b rejects any request with a tools array, and we
    # don't need tool execution to verify Anthropic-compatibility plumbing.
    agent_yaml.write_text(
        "name: ollama-test-agent\n"
        "description: Integration test agent backed by Ollama\n"
        "tools: []\n"
        "sleep:\n"
        "  enabled: false\n"
    )

    config = AgentConfig(
        anthropic_api_key="ollama",
        anthropic_base_url=OLLAMA_HOST,
        agent_api_key=api_key,
        agent_data_dir=data_dir,
        agent_llm_backend="anthropic",
        agent_model=model,
        # qwen3 is a thinking model that emits <think> blocks before its
        # final answer — a small max_tokens cap truncates the assistant
        # before any user-visible text is produced.
        agent_max_tokens=2048,
        agent_host="127.0.0.1",
        agent_port=port,
        agent_config=str(agent_yaml),
        agent_task_await_seconds=120,
    )
    app = _create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    loop = asyncio.new_event_loop()

    def _run() -> None:
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            resp = httpx.get(f"{url}/.well-known/agent-card.json")
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError(f"agentling failed to start for model {model}")

    yield url, api_key, model

    server.should_exit = True
    thread.join(timeout=5)


class TestOllamaA2A:
    async def test_round_trip(self, ollama_agentling) -> None:
        """A user message in, an assistant text response out — through Ollama.

        Asserts on shape (non-empty text, present context_id), not content.
        Small open-source models are too unpredictable for content asserts
        and that's not what this test exists to check.
        """
        url, api_key, model = ollama_agentling
        # 4B-class models on Ollama can take 30s+ on cold-load — bump the
        # client-side httpx timeout well past the SDK's default.
        client = A2ATestClient(url, api_key, request_timeout=180.0)
        result = await client.send("Say hi in one short sentence.")
        assert isinstance(result, A2AResponse), f"{model}: {result!r}"
        assert result.context_id, f"{model}: missing context_id"
        assert result.text.strip(), (
            f"{model}: empty assistant response — task likely returned a "
            f"slow-path handle. Raw: {result.raw!r}"
        )
