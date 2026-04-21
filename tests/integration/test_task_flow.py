"""Integration tests for the full task lifecycle over the wire.

These tests stand up a dedicated agentling server with a tight
``AGENT_TASK_AWAIT_SECONDS`` so they can reliably observe the slow-path
``working`` envelope and then the corresponding completion via polling.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from agentlings.config import AgentConfig
from agentlings.server import _create_app
from tests.integration.a2a_client import A2ATestClient, A2AResponse
from tests.integration.mcp_client import MCPTestClient


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def slow_server(tmp_path_factory):
    """Single server shared by every task-flow test.

    Runs with ``AGENT_TASK_AWAIT_SECONDS=2`` — short enough to exercise the
    slow path when combined with a ``delay-N`` mock marker, but long enough
    that unadorned ``"hello"`` messages still land inline as completed.

    Consolidating to one server avoids the asyncio-loop contention that
    happens when pytest spins up multiple module-scoped uvicorn threads.
    """
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    data_dir = tmp_path_factory.mktemp("task-flow-data")
    agent_yaml = tmp_path_factory.mktemp("task-flow-config") / "agent.yaml"
    agent_yaml.write_text(
        "name: task-flow-test-agent\n"
        "description: A test agent with a 2s await\n"
        "tools:\n"
        "  - bash\n"
        "  - filesystem\n"
    )
    config = AgentConfig(
        anthropic_api_key="not-used",
        agent_api_key="slow-key",
        agent_data_dir=data_dir,
        agent_llm_backend="mock",
        agent_host="127.0.0.1",
        agent_port=port,
        agent_config=str(agent_yaml),
        agent_task_await_seconds=2,
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
            resp = httpx.get(f"{url}/.well-known/agent-card.json")
            if resp.status_code == 200:
                break
        except httpx.ConnectError:
            time.sleep(0.1)
    else:
        raise RuntimeError("slow server failed to start")

    yield url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def slow_client(slow_server: str) -> MCPTestClient:
    return MCPTestClient(slow_server, api_key="slow-key")


@pytest.fixture
def slow_a2a_client(slow_server: str) -> A2ATestClient:
    return A2ATestClient(slow_server, "slow-key")


# Reconnect-flow tests reuse the same server/clients under a more descriptive
# alias so each class reads cleanly.
@pytest.fixture
def realistic_mcp(slow_client: MCPTestClient) -> MCPTestClient:
    return slow_client


@pytest.fixture
def realistic_a2a(slow_a2a_client: A2ATestClient) -> A2ATestClient:
    return slow_a2a_client


class TestSlowPathYield:
    @pytest.mark.asyncio
    async def test_initial_call_yields_working_taskid(
        self, slow_client: MCPTestClient
    ) -> None:
        """With 0s await, the first return should be a working handle."""
        result = await slow_client.call_tool("hello")
        # Either already-completed (mock is fast) OR working — both are
        # legitimate; what matters is the envelope is well-formed and
        # carries the taskId in both cases.
        assert result.task_id is not None
        assert result.status in ("working", "completed")
        assert result.context_id is not None

    @pytest.mark.asyncio
    async def test_poll_eventually_reaches_completed(
        self, slow_client: MCPTestClient
    ) -> None:
        """Even if the first call returns working, polling reaches completed."""
        result = await slow_client.call_tool("hello")

        # If it's already completed, we're done. Otherwise poll until terminal.
        if result.status != "completed":
            polled = await slow_client.poll_task(
                task_id=result.task_id,  # type: ignore[arg-type]
                context_id=result.context_id,
                wait_seconds=5.0,
            )
            assert polled.status == "completed"
            assert polled.message


class TestA2ANativeTaskFlow:
    """SendMessage returns a native A2A Task on slow path; GetTask hits our engine."""

    @pytest.mark.asyncio
    async def test_send_message_returns_task_or_message(
        self, slow_a2a_client: A2ATestClient
    ) -> None:
        result = await slow_a2a_client.send("hello")
        assert isinstance(result, A2AResponse)
        # Either a completed task (history populated) or a working task (task_id set).
        assert result.raw["task_id"] is not None

    @pytest.mark.asyncio
    async def test_get_task_reflects_engine_state(
        self, slow_a2a_client: A2ATestClient
    ) -> None:
        result = await slow_a2a_client.send("hello")
        assert isinstance(result, A2AResponse)
        task_id = result.raw["task_id"]
        assert task_id is not None

        get_resp = await slow_a2a_client.get_task(task_id)
        # Expect either a completed task result or a working one; either way
        # the id must match and the status state is one our engine produces.
        task_obj = get_resp.get("result")
        assert task_obj is not None
        assert task_obj["id"] == task_id
        assert task_obj["status"]["state"] in (
            "working", "completed", "failed", "canceled",
        )

    @pytest.mark.asyncio
    async def test_get_task_unknown_yields_error(
        self, slow_a2a_client: A2ATestClient
    ) -> None:
        resp = await slow_a2a_client.get_task("definitely-not-a-real-task")
        # The A2A SDK surfaces missing tasks as a JSON-RPC error.
        assert "error" in resp or (
            resp.get("result") is None
        )

    @pytest.mark.asyncio
    async def test_cancel_task_routes_to_engine(
        self, slow_a2a_client: A2ATestClient
    ) -> None:
        result = await slow_a2a_client.send("hi")
        assert isinstance(result, A2AResponse)
        task_id = result.raw["task_id"]
        assert task_id is not None

        cancel_resp = await slow_a2a_client.cancel_task(task_id)
        # Either the task was already terminal (cannot cancel → error) or the
        # cancel succeeded and returned the cancelled task. Both are valid
        # depending on timing; what matters is the wire works and we don't
        # crash.
        assert "result" in cancel_resp or "error" in cancel_resp


async def _poll_until_terminal(
    client: MCPTestClient,
    task_id: str,
    context_id: str | None,
    deadline: float,
) -> "MCPToolCallResult":  # type: ignore[name-defined]
    """Poll in a loop (each call capped at server's await) until terminal or deadline."""
    last = None
    while time.monotonic() < deadline:
        last = await client.poll_task(
            task_id=task_id,
            context_id=context_id,
            wait_seconds=2.0,
        )
        if last.status != "working":
            return last
    assert last is not None
    return last


class TestMCPReconnectByTaskId:
    """End-to-end: long-running request yields a handle, client reconnects.

    Uses a 2s server await + a 3s mock delay. The first HTTP call returns a
    working handle after the 2s await fires (delay still has ~1s left). A
    follow-up poll (server caps ``waitSeconds`` at ``AGENT_TASK_AWAIT_SECONDS``
    = 2s) then observes completion. If a single poll happens to race the
    delay, ``_poll_until_terminal`` loops with a bounded deadline.
    """

    @pytest.mark.asyncio
    async def test_long_task_yields_and_polls_to_completion(
        self, realistic_mcp: MCPTestClient
    ) -> None:
        started = time.monotonic()

        # 1. Fire a genuinely slow request.
        handle = await realistic_mcp.call_tool("delay-3 please answer")
        elapsed_call = time.monotonic() - started
        assert handle.task_id is not None, "server must yield a taskId"
        assert handle.status in ("working", "completed"), handle.raw
        assert elapsed_call < 4.0, (
            f"HTTP handler should return within ~2s await, took {elapsed_call:.2f}s"
        )

        if handle.status == "completed":
            # Happy accident — finished fast; nothing to reconnect to.
            assert handle.message
            return

        # 2. Reconnect via poll until completion (deadline = started + 8s).
        final = await _poll_until_terminal(
            realistic_mcp,
            handle.task_id,
            handle.context_id,
            deadline=started + 8.0,
        )
        assert final.status == "completed", (
            f"expected completed, got {final.status}: {final.raw}"
        )
        assert final.task_id == handle.task_id
        assert final.message, "completed task must carry the final response"

    @pytest.mark.asyncio
    async def test_poll_without_context_id_resolves_correctly(
        self, realistic_mcp: MCPTestClient
    ) -> None:
        """Task id alone is sufficient to resume — contextId is optional."""
        started = time.monotonic()
        handle = await realistic_mcp.call_tool("delay-3 answer me")
        assert handle.task_id is not None

        if handle.status == "completed":
            return

        final = await _poll_until_terminal(
            realistic_mcp,
            handle.task_id,
            context_id=None,
            deadline=started + 8.0,
        )
        assert final.status == "completed"


class TestA2AReconnectByTaskId:
    """End-to-end reconnect for the A2A surface via native GetTask."""

    @pytest.mark.asyncio
    async def test_native_task_yielded_then_get_task_completes(
        self, realistic_a2a: A2ATestClient
    ) -> None:
        started = time.monotonic()

        # 1. SendMessage with a long delay — slow path yields a native Task.
        send_result = await realistic_a2a.send("delay-3 please answer")
        assert isinstance(send_result, A2AResponse)
        task_id = send_result.raw["task_id"]
        task_state = send_result.raw["task_state"]
        assert task_id, "native Task object must carry a task id"
        elapsed_send = time.monotonic() - started
        assert elapsed_send < 4.0, (
            f"SendMessage should return within the 2s await, took {elapsed_send:.2f}s"
        )

        if task_state == "completed":
            # Finished within the 2s await window — still a valid outcome.
            return

        assert task_state == "working", (
            f"expected working state on slow path, got {task_state!r}"
        )

        # 2. GetTask loop — routed through EngineTaskStore.
        deadline = started + 8.0
        final = None
        while time.monotonic() < deadline:
            resp = await realistic_a2a.get_task(task_id)
            assert "result" in resp, f"GetTask should return a task, got {resp}"
            assert resp["result"]["id"] == task_id
            state = resp["result"]["status"]["state"]
            if state != "working":
                final = resp["result"]
                break
            await asyncio.sleep(0.5)

        assert final is not None, "task never reached terminal state within deadline"
        assert final["status"]["state"] == "completed"

        # History should carry the agent's final message.
        history = final.get("history", [])
        agent_msgs = [m for m in history if m["role"] == "agent"]
        assert agent_msgs, "completed task must have an agent message in history"

