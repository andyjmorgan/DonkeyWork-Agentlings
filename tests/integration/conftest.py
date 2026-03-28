from __future__ import annotations

import asyncio
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


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    data_dir = tmp_path_factory.mktemp("data")
    config = AgentConfig(
        anthropic_api_key="not-used",
        agent_api_key=api_key,
        agent_data_dir=data_dir,
        agent_llm_backend="mock",
        agent_name="test-agent",
        agent_description="A test agent",
        agent_host="127.0.0.1",
        agent_port=port,
    )
    app = _create_app(config)

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )

    loop = asyncio.new_event_loop()

    def run_server():
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            resp = httpx.get(f"{url}/.well-known/agent.json")
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start")

    yield url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def server_url(_server) -> str:
    return _server


@pytest.fixture
def a2a_client(server_url: str, api_key: str) -> A2ATestClient:
    return A2ATestClient(server_url, api_key)


@pytest.fixture
def mcp_client(server_url: str, api_key: str) -> MCPTestClient:
    return MCPTestClient(server_url, api_key)
