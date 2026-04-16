"""Tests for ``EngineTaskStore`` and the A2A Task translation helper."""

from __future__ import annotations

from pathlib import Path

import pytest
from a2a.types import TaskState as A2ATaskState

from agentlings.config import AgentConfig
from agentlings.core.llm import MockLLMClient
from agentlings.core.store import JournalStore
from agentlings.core.task import TaskEngine, TaskState, TaskStatus
from agentlings.protocol.a2a_task_store import EngineTaskStore, task_state_to_a2a_task
from agentlings.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Translation helper
# --------------------------------------------------------------------------- #


class TestTaskStateTranslation:
    def test_completed_maps_to_completed(self) -> None:
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.COMPLETED,
            content=[{"type": "text", "text": "hello"}],
        )
        a2a_task = task_state_to_a2a_task(state)
        assert a2a_task.id == "t1"
        assert a2a_task.context_id == "c1"
        assert a2a_task.status.state == A2ATaskState.completed

    def test_completed_history_contains_final_response(self) -> None:
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.COMPLETED,
            content=[{"type": "text", "text": "the final answer"}],
        )
        a2a_task = task_state_to_a2a_task(state)
        assert len(a2a_task.history) == 1
        msg = a2a_task.history[0]
        assert msg.role.value == "agent"
        # Part is wrapped in Part root.
        assert msg.parts[0].root.text == "the final answer"

    def test_working_maps_to_working(self) -> None:
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.WORKING,
        )
        a2a_task = task_state_to_a2a_task(state)
        assert a2a_task.status.state == A2ATaskState.working
        assert a2a_task.history == []

    def test_cancelling_maps_to_working(self) -> None:
        """CANCELLING is an internal state; clients see `working` until terminal."""
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.CANCELLING,
        )
        a2a_task = task_state_to_a2a_task(state)
        assert a2a_task.status.state == A2ATaskState.working

    def test_cancelled_maps_to_canceled(self) -> None:
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.CANCELLED,
            error="user requested",
        )
        a2a_task = task_state_to_a2a_task(state)
        assert a2a_task.status.state == A2ATaskState.canceled
        assert a2a_task.status.message is not None
        assert a2a_task.status.message.parts[0].root.text == "user requested"

    def test_failed_maps_to_failed(self) -> None:
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.FAILED,
            error="boom",
        )
        a2a_task = task_state_to_a2a_task(state)
        assert a2a_task.status.state == A2ATaskState.failed
        assert a2a_task.status.message is not None
        assert a2a_task.status.message.parts[0].root.text == "boom"

    def test_empty_content_produces_empty_history(self) -> None:
        state = TaskState(
            task_id="t1",
            context_id="c1",
            status=TaskStatus.COMPLETED,
            content=[],
        )
        a2a_task = task_state_to_a2a_task(state)
        assert a2a_task.history == []


# --------------------------------------------------------------------------- #
# EngineTaskStore
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine(tmp_data_dir: Path, test_config: AgentConfig) -> TaskEngine:
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    tools.register_tools(["bash", "filesystem"])
    llm = MockLLMClient(tool_names=tools.tool_names())
    return TaskEngine(config=test_config, store=store, llm=llm, tools=tools)


@pytest.fixture
def task_store(engine: TaskEngine) -> EngineTaskStore:
    return EngineTaskStore(engine)


class TestEngineTaskStore:
    @pytest.mark.asyncio
    async def test_get_returns_completed_task(
        self, engine: TaskEngine, task_store: EngineTaskStore
    ) -> None:
        state = await engine.spawn("hello")
        assert state.status == TaskStatus.COMPLETED

        a2a_task = await task_store.get(state.task_id)
        assert a2a_task is not None
        assert a2a_task.id == state.task_id
        assert a2a_task.status.state == A2ATaskState.completed
        assert len(a2a_task.history) == 1

    @pytest.mark.asyncio
    async def test_get_unknown_returns_none(
        self, task_store: EngineTaskStore
    ) -> None:
        a2a_task = await task_store.get("nonexistent")
        assert a2a_task is None

    @pytest.mark.asyncio
    async def test_save_is_noop(
        self, engine: TaskEngine, task_store: EngineTaskStore
    ) -> None:
        """The engine is authoritative — ``save()`` must not disturb state."""
        state = await engine.spawn("hello")
        a2a_task = await task_store.get(state.task_id)
        assert a2a_task is not None

        # Saving a mutated version must be ignored.
        # (We just pass the existing task back; engine state doesn't change.)
        await task_store.save(a2a_task)

        a2a_task2 = await task_store.get(state.task_id)
        assert a2a_task2 is not None
        assert a2a_task2.status.state == A2ATaskState.completed

    @pytest.mark.asyncio
    async def test_delete_is_noop(
        self, engine: TaskEngine, task_store: EngineTaskStore
    ) -> None:
        state = await engine.spawn("hello")
        await task_store.delete(state.task_id)
        # Engine still has the task.
        a2a_task = await task_store.get(state.task_id)
        assert a2a_task is not None

    @pytest.mark.asyncio
    async def test_get_reflects_cancellation(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        """After cancel, GetTask must report the cancelled state."""
        from tests.unit.test_task import ControllableLLM
        from a2a.types import TaskState as A2ATaskState
        from agentlings.core.llm import LLMResponse

        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(
            config=test_config, store=store, llm=llm, tools=tools,
        )
        task_store = EngineTaskStore(engine)

        state = await engine.spawn("work", await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        # Before cancel, store reports working.
        a2a_task = await task_store.get(state.task_id)
        assert a2a_task is not None
        assert a2a_task.status.state == A2ATaskState.working

        await engine.cancel(state.task_id)
        # Drain LLM so worker wakes at checkpoint.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            import asyncio
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

        a2a_task2 = await task_store.get(state.task_id)
        assert a2a_task2 is not None
        assert a2a_task2.status.state == A2ATaskState.canceled
