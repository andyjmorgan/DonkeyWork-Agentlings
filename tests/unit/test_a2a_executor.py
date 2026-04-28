"""Tests for ``AgentlingExecutor`` covering the A2A SendMessage entrypoint.

Focus is on protocol-level wiring: that ``configuration.return_immediately``
is honored as a per-request opt-out of the await window, and that the
default behavior continues to block until ``AGENT_TASK_AWAIT_SECONDS``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from a2a.helpers.proto_helpers import new_text_message
from a2a.server.agent_execution.context import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    Task,
)
from a2a.types import TaskState as A2ATaskState

from agentlings.config import AgentConfig
from agentlings.core.llm import LLMResponse
from agentlings.core.store import JournalStore
from agentlings.core.task import TaskEngine
from agentlings.protocol.a2a import AgentlingExecutor
from agentlings.tools.registry import ToolRegistry
from tests.unit.test_task import ControllableLLM


class _LoopShim:
    """Minimal stand-in for ``MessageLoop`` exposing ``.engine`` only.

    The executor's constructor reads ``loop.engine`` once at init and never
    touches the loop again. Building the real ``MessageLoop`` would force
    it to spin up its own internal ``TaskEngine``, defeating the point of
    sharing the test-controlled engine.
    """

    def __init__(self, engine: TaskEngine) -> None:
        self.engine = engine


def _make_request_context(
    *,
    text: str,
    context_id: str,
    task_id: str,
    return_immediately: bool | None = None,
) -> RequestContext:
    msg = new_text_message(text, context_id=context_id, role=Role.ROLE_USER)
    if return_immediately is None:
        configuration = None
    else:
        configuration = SendMessageConfiguration(
            return_immediately=return_immediately,
        )
    request = SendMessageRequest(message=msg, configuration=configuration)
    return RequestContext(
        call_context=ServerCallContext(),
        request=request,
        context_id=context_id,
        task_id=task_id,
    )


async def _execute_and_drain(
    executor: AgentlingExecutor,
    ctx: RequestContext,
    queue: EventQueue,
    *,
    timeout: float,
) -> tuple[list[object], float]:
    """Run ``executor.execute`` and drain enqueued events concurrently.

    Returns the list of events enqueued and the wall-clock seconds taken.
    A concurrent consumer is required because ``EventQueue.close()`` blocks
    on ``queue.join()`` until every enqueued event has been ``task_done``'d.
    """

    events: list[object] = []

    async def consumer() -> None:
        while True:
            try:
                event = await queue.dequeue_event()
            except Exception:  # queue closed
                return
            events.append(event)
            queue.task_done()
            if queue.is_closed() and queue.queue.empty():
                return

    start = asyncio.get_event_loop().time()
    consumer_task = asyncio.create_task(consumer())
    try:
        await asyncio.wait_for(executor.execute(ctx, queue), timeout=timeout)
    finally:
        try:
            await asyncio.wait_for(consumer_task, timeout=1.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()
    elapsed = asyncio.get_event_loop().time() - start
    return events, elapsed


@pytest.fixture
def slow_engine(
    tmp_data_dir: Path, test_config: AgentConfig
) -> tuple[TaskEngine, ControllableLLM]:
    """Engine wired to a ControllableLLM that never replies until told to.

    This guarantees that any call to ``engine.spawn`` with a non-zero
    ``await_seconds`` will burn the whole window — perfect for proving that
    ``return_immediately`` collapses it to zero.
    """
    llm = ControllableLLM()
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)
    return engine, llm


class TestReturnImmediately:
    @pytest.mark.asyncio
    async def test_return_immediately_yields_task_without_blocking(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        engine, llm = slow_engine
        # Configure a long await window — if the executor honored it, the
        # test would hang for ~30s. With return_immediately=True it should
        # return ~immediately.
        test_config.agent_task_await_seconds = 30
        loop = _LoopShim(engine)
        executor = AgentlingExecutor(loop, test_config)

        ctx = _make_request_context(
            text="hello",
            context_id="ctx-immediate",
            task_id="task-immediate",
            return_immediately=True,
        )
        queue = EventQueue()

        # If the executor blocked on await_seconds (30), the 2s wall-clock
        # cap would expire and fail the test.
        events, elapsed = await _execute_and_drain(
            executor, ctx, queue, timeout=2.0,
        )
        assert elapsed < 1.0, f"executor did not return immediately ({elapsed:.3f}s)"
        assert len(events) == 1
        event = events[0]
        assert isinstance(event, Task), f"expected Task, got {type(event).__name__}"
        assert event.id == "task-immediate"
        assert event.context_id == "ctx-immediate"
        assert event.status.state in (
            A2ATaskState.TASK_STATE_WORKING,
            A2ATaskState.TASK_STATE_SUBMITTED,
        )

        # Cleanup: let the still-running worker finish so the test doesn't
        # leak a background task.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get("task-immediate")
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

    @pytest.mark.asyncio
    async def test_default_blocking_uses_configured_await(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        """Without return_immediately, the executor blocks up to the
        configured await window and surfaces a working Task when the worker
        hasn't finished by then."""
        engine, llm = slow_engine
        test_config.agent_task_await_seconds = 0.2
        loop = _LoopShim(engine)
        executor = AgentlingExecutor(loop, test_config)

        ctx = _make_request_context(
            text="hello",
            context_id="ctx-default",
            task_id="task-default",
            # configuration absent — represents a stock client that hasn't
            # opted out of blocking.
            return_immediately=None,
        )
        queue = EventQueue()

        events, elapsed = await _execute_and_drain(
            executor, ctx, queue, timeout=2.0,
        )
        # Must have waited approximately the configured window.
        assert elapsed >= 0.15, f"executor returned too early ({elapsed:.3f}s)"
        assert len(events) == 1
        assert isinstance(events[0], Task)

        # Cleanup.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get("task-default")
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

    @pytest.mark.asyncio
    async def test_return_immediately_false_still_blocks(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        """An explicit ``return_immediately=False`` must behave like the
        default — i.e. honor the configured await window."""
        engine, llm = slow_engine
        test_config.agent_task_await_seconds = 0.2
        loop = _LoopShim(engine)
        executor = AgentlingExecutor(loop, test_config)

        ctx = _make_request_context(
            text="hello",
            context_id="ctx-explicit-blocking",
            task_id="task-explicit-blocking",
            return_immediately=False,
        )
        queue = EventQueue()

        events, elapsed = await _execute_and_drain(
            executor, ctx, queue, timeout=2.0,
        )
        assert elapsed >= 0.15, (
            f"executor returned too early with return_immediately=False "
            f"({elapsed:.3f}s)"
        )
        assert isinstance(events[0], Task)

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get("task-explicit-blocking")
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

    @pytest.mark.asyncio
    async def test_fast_path_with_immediate_completion_returns_message(
        self,
        tmp_data_dir: Path,
        test_config: AgentConfig,
    ) -> None:
        """If the LLM is fast (mock backend completes synchronously),
        the default path still returns a Message, not a Task — proving
        we haven't accidentally forced everything onto the slow path."""
        from agentlings.core.llm import MockLLMClient

        tools = ToolRegistry()
        tools.register_tools(["bash", "filesystem"])
        llm = MockLLMClient(tool_names=tools.tool_names())
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config, store=store, llm=llm, tools=tools,
        )
        loop = _LoopShim(engine)
        executor = AgentlingExecutor(loop, test_config)

        ctx = _make_request_context(
            text="hello",
            context_id="ctx-fast",
            task_id="task-fast",
            return_immediately=None,
        )
        queue = EventQueue()
        events, _elapsed = await _execute_and_drain(
            executor, ctx, queue, timeout=5.0,
        )

        assert len(events) == 1
        assert isinstance(events[0], Message), (
            f"fast path should return Message, got {type(events[0]).__name__}"
        )
