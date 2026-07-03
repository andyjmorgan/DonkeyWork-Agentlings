"""Tests for ``AgentlingExecutor`` covering the A2A SendMessage entrypoint.

Focus is on protocol-level wiring: that ``configuration.return_immediately``
is honored as a per-request opt-out of the await window, that the default
behavior continues to block until ``AGENT_TASK_AWAIT_SECONDS``, and that
``CancelTask`` waits for the cooperative cancel to land so the SDK sees a
``TASK_STATE_CANCELED`` task rather than rejecting the cancel.
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
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)
from a2a.types import TaskState as A2ATaskState

from agentlings.config import A2AConfig, AgentConfig
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


async def _cancel_and_drain(
    executor: AgentlingExecutor,
    ctx: RequestContext,
    queue: EventQueue,
    *,
    timeout: float,
    while_waiting=None,
) -> tuple[list[object], float]:
    """Run ``executor.cancel`` and drain enqueued events concurrently.

    ``while_waiting`` is an optional coroutine function invoked after the
    cancel call has started — used to unblock the worker's in-flight LLM
    call so the cooperative cancel can be observed.
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
    cancel_task = asyncio.create_task(executor.cancel(ctx, queue))
    try:
        if while_waiting is not None:
            await while_waiting()
        await asyncio.wait_for(cancel_task, timeout=timeout)
    finally:
        try:
            await asyncio.wait_for(consumer_task, timeout=1.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()
    elapsed = asyncio.get_event_loop().time() - start
    return events, elapsed


def _make_cancel_context(*, context_id: str, task_id: str) -> RequestContext:
    """Mirror how the SDK's ``on_cancel_task`` builds the RequestContext."""
    return RequestContext(
        call_context=ServerCallContext(),
        request=None,
        context_id=context_id,
        task_id=task_id,
    )


class TestCancelTask:
    @pytest.mark.asyncio
    async def test_cancel_running_task_yields_canceled_status(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        """Cancelling a working task must enqueue a status update with
        TASK_STATE_CANCELED — the SDK's ``on_cancel_task`` raises
        ``TaskNotCancelableError`` for anything else, so enqueueing the
        immediate ``cancelling`` snapshot (mapped to WORKING) breaks the
        whole A2A cancel path."""
        engine, llm = slow_engine
        test_config.agent_task_await_seconds = 5
        executor = AgentlingExecutor(_LoopShim(engine), test_config)

        # Start a task that blocks inside the LLM call.
        await engine.spawn(
            message="hello",
            context_id="ctx-cancel",
            via="a2a",
            await_seconds=0,
            task_id="task-cancel",
        )
        await asyncio.wait_for(llm.before_call_event.wait(), timeout=2.0)

        async def unblock_worker() -> None:
            # The worker only observes the cancel flag after the in-flight
            # LLM call returns, so wait for the flag then release the LLM.
            for _ in range(200):
                rec = engine.registry.get("task-cancel")
                if rec is not None and rec.cancel_flag:
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("cancel flag was never raised on the record")
            llm.push(LLMResponse(
                content=[{"type": "text", "text": "too late"}],
                stop_reason="end_turn",
            ))

        ctx = _make_cancel_context(
            context_id="ctx-cancel", task_id="task-cancel",
        )
        queue = EventQueue()
        events, _elapsed = await _cancel_and_drain(
            executor, ctx, queue, timeout=5.0, while_waiting=unblock_worker,
        )

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, TaskStatusUpdateEvent), (
            f"expected TaskStatusUpdateEvent, got {type(event).__name__}"
        )
        assert event.task_id == "task-cancel"
        assert event.context_id == "ctx-cancel"
        assert event.status.state == A2ATaskState.TASK_STATE_CANCELED

    @pytest.mark.asyncio
    async def test_cancel_terminal_task_returns_state_without_waiting(
        self,
        tmp_data_dir: Path,
        test_config: AgentConfig,
    ) -> None:
        """Cancelling an already-completed task must return its terminal
        state immediately, not burn the await window."""
        from agentlings.core.llm import MockLLMClient

        tools = ToolRegistry()
        tools.register_tools(["bash", "filesystem"])
        llm = MockLLMClient(tool_names=tools.tool_names())
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config, store=store, llm=llm, tools=tools,
        )
        test_config.agent_task_await_seconds = 30
        executor = AgentlingExecutor(_LoopShim(engine), test_config)

        state = await engine.spawn(
            message="hello",
            context_id="ctx-cancel-done",
            via="a2a",
            await_seconds=5,
            task_id="task-cancel-done",
        )
        assert state.status.value == "completed"

        ctx = _make_cancel_context(
            context_id="ctx-cancel-done", task_id="task-cancel-done",
        )
        queue = EventQueue()
        events, elapsed = await _cancel_and_drain(
            executor, ctx, queue, timeout=5.0,
        )
        assert elapsed < 1.0, (
            f"cancel of terminal task blocked on the await window ({elapsed:.3f}s)"
        )
        assert len(events) == 1
        assert isinstance(events[0], TaskStatusUpdateEvent)
        assert events[0].status.state == A2ATaskState.TASK_STATE_COMPLETED

    @pytest.mark.asyncio
    async def test_cancel_unknown_task_reports_not_found(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        engine, _llm = slow_engine
        executor = AgentlingExecutor(_LoopShim(engine), test_config)

        ctx = _make_cancel_context(
            context_id="ctx-missing", task_id="task-missing",
        )
        queue = EventQueue()
        events, _elapsed = await _cancel_and_drain(
            executor, ctx, queue, timeout=5.0,
        )
        assert len(events) == 1
        assert isinstance(events[0], Message)
        assert "not found" in events[0].parts[0].text

    @pytest.mark.asyncio
    async def test_sdk_on_cancel_task_returns_canceled_task(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        """Full A2A path: the SDK's ``on_cancel_task`` (the ``tasks/cancel``
        handler) must resolve to a canceled Task instead of raising
        ``TaskNotCancelableError``."""
        from a2a.server.request_handlers import DefaultRequestHandler
        from a2a.types import CancelTaskRequest

        from agentlings.protocol.a2a_task_store import EngineTaskStore
        from agentlings.protocol.agent_card import generate_agent_card

        engine, llm = slow_engine
        test_config.agent_task_await_seconds = 5
        executor = AgentlingExecutor(_LoopShim(engine), test_config)
        handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=EngineTaskStore(engine),
            agent_card=generate_agent_card(test_config),
        )

        await engine.spawn(
            message="hello",
            context_id="ctx-sdk-cancel",
            via="a2a",
            await_seconds=0,
            task_id="task-sdk-cancel",
        )
        await asyncio.wait_for(llm.before_call_event.wait(), timeout=2.0)

        cancel_op = asyncio.create_task(
            handler.on_cancel_task(
                CancelTaskRequest(id="task-sdk-cancel"), ServerCallContext(),
            )
        )
        for _ in range(200):
            rec = engine.registry.get("task-sdk-cancel")
            if rec is not None and rec.cancel_flag:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("cancel flag was never raised on the record")
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "too late"}],
            stop_reason="end_turn",
        ))

        result = await asyncio.wait_for(cancel_op, timeout=5.0)
        assert isinstance(result, Task)
        assert result.id == "task-sdk-cancel"
        assert result.status.state == A2ATaskState.TASK_STATE_CANCELED


class TestStreamingExecution:
    @pytest.mark.asyncio
    async def test_streaming_yields_working_task_then_final_task(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        """A2A ``message/stream`` skips the await timer and observes the
        durable task until terminal state."""
        engine, llm = slow_engine
        test_config.agent_task_await_seconds = 30
        test_config._definition.a2a = A2AConfig(streaming=True)  # noqa: SLF001
        loop = _LoopShim(engine)
        executor = AgentlingExecutor(loop, test_config)

        ctx = _make_request_context(
            text="hello",
            context_id="ctx-stream",
            task_id="task-stream",
            return_immediately=None,
        )
        ctx.call_context.state["method"] = "SendStreamingMessage"
        queue = EventQueue()

        start = asyncio.get_event_loop().time()
        execute_task = asyncio.create_task(executor.execute(ctx, queue))

        first = await asyncio.wait_for(queue.dequeue_event(), timeout=2.0)
        queue.task_done()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed < 1.0, (
            f"streaming did not yield the task immediately ({elapsed:.3f}s)"
        )
        assert isinstance(first, TaskStatusUpdateEvent)
        assert first.task_id == "task-stream"
        assert first.context_id == "ctx-stream"
        assert first.status.state == A2ATaskState.TASK_STATE_WORKING
        assert not execute_task.done()

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "stream done"}],
            stop_reason="end_turn",
        ))

        artifact = await asyncio.wait_for(queue.dequeue_event(), timeout=3.0)
        queue.task_done()
        assert isinstance(artifact, TaskArtifactUpdateEvent)
        assert artifact.task_id == "task-stream"
        assert artifact.context_id == "ctx-stream"
        assert artifact.artifact.parts[0].text == "stream done"
        assert artifact.last_chunk is True

        final = await asyncio.wait_for(queue.dequeue_event(), timeout=3.0)
        queue.task_done()
        assert isinstance(final, TaskStatusUpdateEvent)
        assert final.task_id == "task-stream"
        assert final.context_id == "ctx-stream"
        assert final.status.state == A2ATaskState.TASK_STATE_COMPLETED

        await asyncio.wait_for(execute_task, timeout=3.0)

    @pytest.mark.asyncio
    async def test_streaming_emits_tool_progress_status_with_metadata(
        self,
        slow_engine: tuple[TaskEngine, ControllableLLM],
        test_config: AgentConfig,
    ) -> None:
        engine, llm = slow_engine
        test_config._definition.a2a = A2AConfig(  # noqa: SLF001
            streaming=True,
            tool_progress_summaries=True,
        )
        loop = _LoopShim(engine)
        executor = AgentlingExecutor(loop, test_config)

        llm.push(LLMResponse(
            content=[{
                "type": "tool_use",
                "id": "toolu_progress",
                "name": "missing_tool",
                "input": {
                    "agentling_action_summary": "Checking the remote agent status",
                },
            }],
            stop_reason="tool_use",
        ))

        ctx = _make_request_context(
            text="use a tool",
            context_id="ctx-progress",
            task_id="task-progress",
            return_immediately=None,
        )
        ctx.call_context.state["method"] = "SendStreamingMessage"
        queue = EventQueue()
        execute_task = asyncio.create_task(executor.execute(ctx, queue))

        first = await asyncio.wait_for(queue.dequeue_event(), timeout=2.0)
        queue.task_done()
        assert isinstance(first, TaskStatusUpdateEvent)
        assert first.status.state == A2ATaskState.TASK_STATE_WORKING

        progress = await asyncio.wait_for(queue.dequeue_event(), timeout=2.0)
        queue.task_done()
        assert isinstance(progress, TaskStatusUpdateEvent)
        assert progress.task_id == "task-progress"
        assert progress.context_id == "ctx-progress"
        assert progress.status.message.parts[0].text == "Checking the remote agent status"
        metadata = progress.metadata["agentling"]
        assert metadata["type"] == "tool_call"
        assert metadata["tool_call_id"] == "toolu_progress"
        assert metadata["status"] == "EXECUTING"
        assert metadata["tool_name"] == "missing_tool"
        assert metadata["description"] == "Checking the remote agent status"

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "done after progress"}],
            stop_reason="end_turn",
        ))

        # A completed tool progress event and artifact may arrive before terminal status.
        while True:
            event = await asyncio.wait_for(queue.dequeue_event(), timeout=3.0)
            queue.task_done()
            if (
                isinstance(event, TaskStatusUpdateEvent)
                and event.status.state == A2ATaskState.TASK_STATE_COMPLETED
            ):
                final = event
                break

        assert final.status.state == A2ATaskState.TASK_STATE_COMPLETED
        await asyncio.wait_for(execute_task, timeout=3.0)
