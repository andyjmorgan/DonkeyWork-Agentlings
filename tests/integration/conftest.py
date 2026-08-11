from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from agentlings.config import AgentConfig
from agentlings.server import _create_app
from tests.integration.a2a_client import A2ATestClient
from tests.integration.mcp_client import MCPTestClient


# --------------------------------------------------------------------------- #
# Noise suppression — these messages are framework-level teardown artifacts,
# not test failures. Dropping them keeps the Docker CI output scannable.
# --------------------------------------------------------------------------- #


class _DropTeardownNoise(logging.Filter):
    """Suppress specific framework teardown messages that clutter test output.

    - ``Task was destroyed but it is pending!`` — asyncio GC warning from
      ``sse_starlette._shutdown_watcher`` when the Starlette app is torn
      down from a thread at the end of a test module. Not our code, not a
      test failure.
    - ``Queue is closed. Event will not be dequeued.`` — A2A event-queue
      shutdown warning emitted after the executor already returned.
    """

    _PATTERNS = (
        "Task was destroyed but it is pending",
        "Queue is closed. Event will not be dequeued",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(p in message for p in self._PATTERNS)


# Install once, at import time — fires before any pytest collection so even
# teardown-phase messages from fixtures are filtered.
_teardown_filter = _DropTeardownNoise()
for _name in ("asyncio", "a2a.server.events.event_queue"):
    logging.getLogger(_name).addFilter(_teardown_filter)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Kept under the old name for any external references; new code should use
# ``free_port``.
_free_port = free_port


def start_agentling_server(
    config: AgentConfig,
    readiness_path: str = "/.well-known/agent-card.json",
) -> tuple[str, uvicorn.Server, threading.Thread]:
    """Boot an agentling app in a daemon thread and wait until it is ready.

    The single boot helper shared by every integration fixture. The
    readiness loop sleeps unconditionally per iteration (an early 404/503
    must not burn all attempts in milliseconds), tolerates any httpx
    error — not just ConnectError — and shuts the server thread down
    before raising on any failure.

    Returns ``(url, server, thread)``; stop with ``stop_agentling_server``.
    """
    app = _create_app(config)
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=config.agent_port, log_level="warning",
        )
    )
    loop = asyncio.new_event_loop()

    def _run() -> None:
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{config.agent_port}"
    try:
        for _ in range(50):
            try:
                resp = httpx.get(f"{url}{readiness_path}")
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError(f"agentling failed to start on {url}")
    except BaseException:
        stop_agentling_server(server, thread)
        raise

    return url, server, thread


def stop_agentling_server(server: uvicorn.Server, thread: threading.Thread) -> None:
    """Signal the server to exit and join its thread."""
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def api_key() -> str:
    return os.environ.get("AGENT_API_KEY", "integration-test-key")


@pytest.fixture(scope="session")
def base_url() -> str:
    return os.environ.get("AGENT_URL", "")


@pytest.fixture(scope="session")
def _server(base_url: str, api_key: str, tmp_path_factory):
    if base_url:
        yield base_url
        return

    data_dir = tmp_path_factory.mktemp("data")
    agent_yaml = tmp_path_factory.mktemp("config") / "agent.yaml"
    agent_yaml.write_text(
        "name: test-agent\n"
        "description: A test agent\n"
        "tools:\n"
        "  - bash\n"
        "  - filesystem\n"
    )
    config = AgentConfig(
        anthropic_api_key="not-used",
        agent_api_key=api_key,
        agent_data_dir=data_dir,
        agent_llm_backend="mock",
        agent_host="127.0.0.1",
        agent_port=free_port(),
        agent_config=str(agent_yaml),
    )
    url, server, thread = start_agentling_server(
        config, readiness_path="/.well-known/agent.json",
    )

    yield url

    stop_agentling_server(server, thread)


@pytest.fixture
def server_url(_server) -> str:
    return _server


@pytest.fixture
def a2a_client(server_url: str, api_key: str) -> A2ATestClient:
    return A2ATestClient(server_url, api_key)


@pytest.fixture
def mcp_client(server_url: str, api_key: str) -> MCPTestClient:
    return MCPTestClient(server_url, api_key)
