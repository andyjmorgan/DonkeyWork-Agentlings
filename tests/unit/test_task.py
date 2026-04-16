"""Tests for the task engine: registry, worker, merge-back, cancel, crash recovery."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from agentlings.config import AgentConfig
from agentlings.core.llm import BaseLLMClient, LLMResponse, MockLLMClient
from agentlings.core.models import (
    CompactionEntry,
    MergeCommitted,
    MergeStarted,
    MessageEntry,
    TaskCompleted,
    TaskFailed,
    TaskStarted,
)
from agentlings.core.store import JournalStore, TaskJournal
from agentlings.core.task import (
    ContextBusyError,
    InvalidTaskInputError,
    TaskContextMismatchError,
    TaskEngine,
    TaskNotFoundError,
    TaskRecord,
    TaskRegistry,
    TaskStatus,
    TaskWorker,
)
from agentlings.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine(tmp_data_dir: Path, test_config: AgentConfig) -> TaskEngine:
    store = JournalStore(tmp_data_dir)
    tools = ToolRegistry()
    tools.register_tools(["bash", "filesystem"])
    llm = MockLLMClient(tool_names=tools.tool_names())
    return TaskEngine(config=test_config, store=store, llm=llm, tools=tools)


@pytest.fixture
def store(tmp_data_dir: Path) -> JournalStore:
    return JournalStore(tmp_data_dir)


# --------------------------------------------------------------------------- #
# Controllable mock LLM — lets tests block, fail, or emit custom responses.
# --------------------------------------------------------------------------- #


class ControllableLLM(BaseLLMClient):
    """LLM client under test control.

    Waits for tests to push responses (or set the fail flag) so we can
    reliably reproduce race-y behavior like cancellation mid-turn.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[LLMResponse | Exception] = asyncio.Queue()
        self.call_count = 0
        self.before_call_event = asyncio.Event()

    def push(self, response: LLMResponse) -> None:
        self._queue.put_nowait(response)

    def push_exception(self, exc: Exception) -> None:
        self._queue.put_nowait(exc)

    async def complete(self, system, messages, tools, output_schema=None) -> LLMResponse:  # noqa: ANN001
        self.call_count += 1
        self.before_call_event.set()
        item = await self._queue.get()
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(self, system, messages, tools):  # noqa: ANN001
        raise NotImplementedError
        yield  # pragma: no cover

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def batch_create(self, requests, model=None):
        return []

    async def batch_status(self, batch_id: str):
        raise NotImplementedError

    async def batch_results(self, batch_id: str):
        return []


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class TestTaskRegistry:
    async def test_register_creates_entry(self) -> None:
        reg = TaskRegistry()
        rec = await reg.register("t1", "ctx1", "hello", via="mcp")
        assert rec.task_id == "t1"
        assert rec.context_id == "ctx1"
        assert rec.message == "hello"
        assert rec.via == "mcp"
        assert rec.status == TaskStatus.WORKING
        assert reg.active_count() == 1

    async def test_register_rejects_busy_context(self) -> None:
        reg = TaskRegistry()
        await reg.register("t1", "ctx1", "hello")
        with pytest.raises(ContextBusyError) as exc_info:
            await reg.register("t2", "ctx1", "second")
        assert exc_info.value.context_id == "ctx1"
        assert exc_info.value.active_task_id == "t1"

    async def test_register_allows_different_contexts_concurrently(self) -> None:
        reg = TaskRegistry()
        await reg.register("t1", "ctxA", "hi")
        await reg.register("t2", "ctxB", "hi")
        assert reg.active_count() == 2

    async def test_unregister_frees_context(self) -> None:
        reg = TaskRegistry()
        await reg.register("t1", "ctx1", "hi")
        await reg.unregister("t1")
        assert reg.active_count() == 0
        # Same context can now register again.
        await reg.register("t2", "ctx1", "hi")

    async def test_unregister_missing_is_noop(self) -> None:
        reg = TaskRegistry()
        await reg.unregister("never-registered")

    async def test_get_returns_record(self) -> None:
        reg = TaskRegistry()
        await reg.register("t1", "ctx1", "hi")
        rec = reg.get("t1")
        assert rec is not None
        assert rec.task_id == "t1"

    async def test_get_returns_none_for_unknown(self) -> None:
        reg = TaskRegistry()
        assert reg.get("no-such-task") is None

    async def test_get_by_context_returns_live_task(self) -> None:
        reg = TaskRegistry()
        await reg.register("t1", "ctx1", "hi")
        rec = reg.get_by_context("ctx1")
        assert rec is not None and rec.task_id == "t1"

    async def test_request_cancel_flips_flag(self) -> None:
        reg = TaskRegistry()
        rec = await reg.register("t1", "ctx1", "hi")
        assert rec.cancel_flag is False
        ok = reg.request_cancel("t1")
        assert ok is True
        assert rec.cancel_flag is True
        assert rec.status == TaskStatus.CANCELLING

    async def test_request_cancel_unknown_returns_false(self) -> None:
        reg = TaskRegistry()
        assert reg.request_cancel("unknown") is False


# --------------------------------------------------------------------------- #
# Engine: spawn + fast path (completion inside await window)
# --------------------------------------------------------------------------- #


class TestSpawnFastPath:
    async def test_spawn_creates_context_and_task(self, engine: TaskEngine) -> None:
        state = await engine.spawn("hello", via="mcp")
        assert state.context_id
        assert state.task_id
        assert state.status == TaskStatus.COMPLETED
        assert state.content  # mock returns some text

    async def test_spawn_uses_supplied_context_id(self, engine: TaskEngine) -> None:
        ctx = str(uuid4())
        state = await engine.spawn("hi", context_id=ctx, via="mcp")
        assert state.context_id == ctx

    async def test_spawn_records_dispatch_marker_in_parent(
        self, engine: TaskEngine, store: JournalStore
    ) -> None:
        state = await engine.spawn("hello")
        entries = store.read_entries(state.context_id)
        assert any(
            e["t"] == "task_dispatch" and e["task_id"] == state.task_id
            for e in entries
        )

    async def test_spawn_writes_task_started_in_subjournal(
        self, engine: TaskEngine, store: JournalStore
    ) -> None:
        state = await engine.spawn("hi there")
        tj = TaskJournal(store.task_path(state.context_id, state.task_id))
        entries = tj.read_entries()
        assert entries[0]["t"] == "task_start"
        assert entries[0]["message"] == "hi there"

    async def test_spawn_merge_back_writes_user_and_assistant(
        self, engine: TaskEngine, store: JournalStore
    ) -> None:
        state = await engine.spawn("hello world")
        entries = store.read_entries(state.context_id)
        msg_entries = [e for e in entries if e["t"] == "msg"]
        roles = [e["role"] for e in msg_entries]
        assert "user" in roles
        assert "assistant" in roles

    async def test_spawn_wraps_merge_back(
        self, engine: TaskEngine, store: JournalStore
    ) -> None:
        state = await engine.spawn("hi")
        entries = store.read_entries(state.context_id)
        has_start = any(
            e["t"] == "merge_start" and e["task_id"] == state.task_id
            for e in entries
        )
        has_commit = any(
            e["t"] == "merge_commit" and e["task_id"] == state.task_id
            for e in entries
        )
        assert has_start and has_commit

    async def test_replay_shows_only_conversation(
        self, engine: TaskEngine, store: JournalStore
    ) -> None:
        """Audit markers never leak into replay output."""
        state = await engine.spawn("hello")
        messages = store.replay(state.context_id)
        # mock echo → user + assistant, nothing else.
        assert [m["role"] for m in messages] == ["user", "assistant"]

    async def test_registry_cleared_after_success(self, engine: TaskEngine) -> None:
        state = await engine.spawn("hello")
        # give the finally-block a moment to run after completion_event fires
        await asyncio.sleep(0.01)
        assert engine.registry.get(state.task_id) is None
        assert engine.registry.active_count() == 0


# --------------------------------------------------------------------------- #
# Engine: spawn + slow path (timeout yields working handle)
# --------------------------------------------------------------------------- #


class TestSpawnSlowPath:
    async def test_timeout_returns_working_handle(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        # await_seconds = 0.1; don't push a response → worker blocks on LLM.
        state = await engine.spawn("do work", await_seconds=0.1)

        assert state.status == TaskStatus.WORKING
        assert not state.content
        assert state.task_id

        # Clean up — push a terminal response so the worker exits.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "late"}],
            stop_reason="end_turn",
        ))
        # Let worker finish.
        record = engine.registry.get(state.task_id)
        if record is not None:
            await asyncio.wait_for(record.completion_event.wait(), timeout=2.0)


# --------------------------------------------------------------------------- #
# Engine: busy-context rejection
# --------------------------------------------------------------------------- #


class TestContextBusy:
    async def test_second_spawn_rejects(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        ctx = str(uuid4())
        first = await engine.spawn("first", context_id=ctx, await_seconds=0.05)
        assert first.status == TaskStatus.WORKING  # stuck waiting for LLM

        with pytest.raises(ContextBusyError) as exc_info:
            await engine.spawn("second", context_id=ctx, await_seconds=0.05)
        assert exc_info.value.active_task_id == first.task_id
        assert exc_info.value.context_id == ctx

        # Clean up.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(first.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=2.0)

    async def test_after_completion_context_is_reusable(
        self, engine: TaskEngine
    ) -> None:
        ctx = str(uuid4())
        first = await engine.spawn("first", context_id=ctx)
        assert first.status == TaskStatus.COMPLETED
        await asyncio.sleep(0.02)
        second = await engine.spawn("second", context_id=ctx)
        assert second.status == TaskStatus.COMPLETED
        assert second.context_id == ctx


# --------------------------------------------------------------------------- #
# Engine: polling
# --------------------------------------------------------------------------- #


class TestPoll:
    async def test_poll_returns_terminal_state(self, engine: TaskEngine) -> None:
        state = await engine.spawn("hi")
        # Already terminal since mock is synchronous.
        polled = await engine.poll(state.task_id, state.context_id)
        assert polled.status == TaskStatus.COMPLETED
        assert polled.content

    async def test_poll_unknown_task_raises(self, engine: TaskEngine) -> None:
        with pytest.raises(TaskNotFoundError):
            await engine.poll("never-existed")

    async def test_poll_wait_returns_early_on_completion(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        ctx = str(uuid4())
        state = await engine.spawn("hi", context_id=ctx, await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        # Begin a long poll in parallel.
        poll_task = asyncio.create_task(
            engine.poll(state.task_id, wait_seconds=5.0, cap_seconds=5.0)
        )
        await asyncio.sleep(0.05)

        # Push the LLM response — worker completes.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
        ))

        polled = await asyncio.wait_for(poll_task, timeout=3.0)
        assert polled.status == TaskStatus.COMPLETED
        assert any(b.get("text") == "done" for b in polled.content)

    async def test_poll_wait_timeout_returns_working(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("hi", await_seconds=0.05)
        polled = await engine.poll(state.task_id, wait_seconds=0.05)
        assert polled.status == TaskStatus.WORKING

        # Clean up.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=2.0)

    async def test_poll_context_assertion_mismatch_live_task(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        """For a live task, the registry catches context mismatches explicitly."""
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("hi", await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        with pytest.raises(TaskContextMismatchError):
            await engine.poll(state.task_id, context_id="wrong-ctx")

        # Clean up.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=2.0)

    async def test_poll_terminal_task_with_wrong_context_not_found(
        self, engine: TaskEngine
    ) -> None:
        """Terminal task + wrong contextId behaves as task_not_found.

        The filesystem path binding (data/{ctx}/tasks/{task}.jsonl) IS the
        access check — the wrong context yields a different path that simply
        doesn't exist.
        """
        state = await engine.spawn("hi")
        await asyncio.sleep(0.02)
        with pytest.raises(TaskNotFoundError):
            await engine.poll(state.task_id, context_id="different-ctx")

    async def test_poll_terminal_task_without_registry_entry(
        self, engine: TaskEngine
    ) -> None:
        """Tasks purged from the registry must still be pollable via filesystem."""
        state = await engine.spawn("hi")
        # small delay so registry clears in worker's finally
        await asyncio.sleep(0.02)
        assert engine.registry.get(state.task_id) is None

        # Poll without contextId — engine must filesystem-scan.
        polled = await engine.poll(state.task_id)
        assert polled.status == TaskStatus.COMPLETED


# --------------------------------------------------------------------------- #
# Engine: cancellation
# --------------------------------------------------------------------------- #


class TestCancel:
    async def test_cancel_marks_task_cancelled(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("work", await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        await engine.cancel(state.task_id, state.context_id)

        # Drain LLM response so the worker wakes at its next checkpoint.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "partial"}],
            stop_reason="end_turn",
        ))

        rec = engine.registry.get(state.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

        # Sub-journal should have TaskCancelled as its terminal entry.
        tj = TaskJournal(store.task_path(state.context_id, state.task_id))
        term = tj.terminal_entry()
        assert term is not None
        assert term["t"] == "task_cancel"

    async def test_cancel_skips_merge_back(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("work", await_seconds=0.05)
        await engine.cancel(state.task_id, state.context_id)

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

        entries = store.read_entries(state.context_id)
        # Parent must have task_dispatch + task_cancel, but no merge_* entries
        # and no msg entries for this taskId.
        kinds = [e["t"] for e in entries]
        assert "task_dispatch" in kinds
        assert "task_cancel" in kinds
        assert "merge_start" not in kinds
        assert "merge_commit" not in kinds
        assert not any(
            e.get("t") == "msg" and e.get("task_id") == state.task_id
            for e in entries
        )

    async def test_cancel_pristine_context_replay(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        """A cancelled task's message must never appear in replay."""
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("secret", await_seconds=0.05)
        await engine.cancel(state.task_id, state.context_id)

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)

        messages = store.replay(state.context_id)
        assert all(
            "secret" not in str(m.get("content", "")) for m in messages
        )

    async def test_cancel_unknown_raises(self, engine: TaskEngine) -> None:
        with pytest.raises(TaskNotFoundError):
            await engine.cancel("no-such")

    async def test_cancel_context_mismatch_raises(
        self, engine: TaskEngine,
    ) -> None:
        state = await engine.spawn("hi", await_seconds=0)
        # Terminal already — even a terminal task must honour the mismatch check.
        await asyncio.sleep(0.02)
        # For a terminal task, engine.cancel returns state (no mismatch possible)
        # since there's no registry entry to cross-check. We verify the correct
        # path: a live task with mismatch should raise.
        llm = ControllableLLM()
        store = JournalStore(engine._store.data_dir)  # noqa: SLF001 — test helper
        tools = ToolRegistry()
        engine2 = TaskEngine(
            config=engine._config,  # noqa: SLF001
            store=store,
            llm=llm,
            tools=tools,
        )
        live = await engine2.spawn("wait", await_seconds=0.02)
        with pytest.raises(TaskContextMismatchError):
            await engine2.cancel(live.task_id, context_id="wrong")
        # Clean up.
        llm.push(LLMResponse(
            content=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
        ))
        rec = engine2.registry.get(live.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=2.0)


# --------------------------------------------------------------------------- #
# Engine: failure paths
# --------------------------------------------------------------------------- #


class TestFailure:
    async def test_llm_exception_marks_failed(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        # await briefly so the worker kicks off and then blocks on get().
        spawn_task = asyncio.create_task(engine.spawn("boom", await_seconds=2.0))
        await asyncio.sleep(0.05)

        llm.push_exception(RuntimeError("llm down"))
        state = await spawn_task
        assert state.status == TaskStatus.FAILED
        assert state.error and "llm down" in state.error

    async def test_failure_pristine_context_replay(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        spawn_task = asyncio.create_task(
            engine.spawn("sensitive", await_seconds=2.0)
        )
        await asyncio.sleep(0.05)
        llm.push_exception(RuntimeError("oops"))
        state = await spawn_task

        messages = store.replay(state.context_id)
        assert messages == []  # nothing committed to parent on failure


# --------------------------------------------------------------------------- #
# Engine: crash recovery
# --------------------------------------------------------------------------- #


class TestCrashRecovery:
    async def test_orphaned_subjournal_gets_task_failed(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(
            config=test_config,
            store=store,
            llm=MockLLMClient(),
            tools=tools,
        )

        # Manually seed: a sub-journal with TaskStarted but no terminal.
        ctx_id = "ctx-1"
        task_id = "t-orphan"
        store.create(ctx_id)
        tj = TaskJournal(store.task_path(ctx_id, task_id))
        tj.create()
        tj.append(TaskStarted(ctx=ctx_id, task_id=task_id, message="hi"))

        engine.recover_on_startup()

        term = tj.terminal_entry()
        assert term is not None
        assert term["t"] == "task_fail"
        assert term["reason"] == "process_crash_recovery"

        # Parent also got the audit marker mirrored.
        parent_entries = store.read_entries(ctx_id)
        assert any(
            e["t"] == "task_fail" and e["task_id"] == task_id
            for e in parent_entries
        )

    async def test_orphaned_merge_start_without_task_done_closes_wrapper(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config,
            store=store,
            llm=MockLLMClient(),
            tools=ToolRegistry(),
        )

        ctx = "ctx-1"
        tid = "t-partial"
        store.create(ctx)
        # Seed: dispatch + merge_start, but no merge_commit, and no sub-journal.
        store.append_many(ctx, [
            TaskStarted.model_construct(ctx=ctx, task_id=tid, message="hi"),
        ]) if False else None  # noqa — documentary; skip
        from agentlings.core.models import TaskDispatched
        store.append(ctx, TaskDispatched(ctx=ctx, task_id=tid))
        store.append(ctx, MergeStarted(ctx=ctx, task_id=tid))

        engine.recover_on_startup()

        entries = store.read_entries(ctx)
        commits = [e for e in entries if e["t"] == "merge_commit" and e["task_id"] == tid]
        assert len(commits) == 1

    async def test_orphaned_merge_with_task_done_applies_missing_messages(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config,
            store=store,
            llm=MockLLMClient(),
            tools=ToolRegistry(),
        )

        ctx = "ctx-1"
        tid = "t-complete-but-merge-died"
        store.create(ctx)

        # Parent had merge_start and dispatch but nothing else.
        from agentlings.core.models import TaskDispatched
        store.append(ctx, TaskDispatched(ctx=ctx, task_id=tid))
        store.append(ctx, MergeStarted(ctx=ctx, task_id=tid))

        # Sub-journal is complete.
        tj = TaskJournal(store.task_path(ctx, tid))
        tj.create()
        tj.append(TaskStarted(ctx=ctx, task_id=tid, message="hello", via="mcp"))
        tj.append(TaskCompleted(
            ctx=ctx, task_id=tid,
            final_response=[{"type": "text", "text": "done"}],
        ))

        engine.recover_on_startup()

        msgs = [e for e in store.read_entries(ctx) if e["t"] == "msg" and e["task_id"] == tid]
        roles = [m["role"] for m in msgs]
        assert "user" in roles and "assistant" in roles

        # Replay shows the recovered conversation without leakage.
        replay = store.replay(ctx)
        assert [m["role"] for m in replay] == ["user", "assistant"]
        # User message text was reconstructed from TaskStarted.
        assert any(
            "hello" in str(m.get("content", "")) for m in replay if m["role"] == "user"
        )

    async def test_recovery_is_idempotent(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config,
            store=store,
            llm=MockLLMClient(),
            tools=ToolRegistry(),
        )

        ctx = "ctx-1"
        tid = "t-partial"
        store.create(ctx)

        from agentlings.core.models import TaskDispatched
        store.append(ctx, TaskDispatched(ctx=ctx, task_id=tid))
        store.append(ctx, MergeStarted(ctx=ctx, task_id=tid))
        tj = TaskJournal(store.task_path(ctx, tid))
        tj.create()
        tj.append(TaskStarted(ctx=ctx, task_id=tid, message="hi"))
        tj.append(TaskCompleted(
            ctx=ctx, task_id=tid, final_response=[{"type": "text", "text": "x"}],
        ))

        engine.recover_on_startup()
        first = store.read_entries(ctx)

        engine.recover_on_startup()
        second = store.read_entries(ctx)

        # Second recovery must not add duplicate user/assistant msgs — but it
        # does close with another merge_commit because each partial pass sees a
        # fresh "orphaned merge_start" only on the first run. A no-op second
        # run is the expected behavior.
        assert len(second) == len(first)


# --------------------------------------------------------------------------- #
# Engine: compaction port
# --------------------------------------------------------------------------- #


class TestCompactionPort:
    async def test_latest_compaction_ported_on_successful_merge_back(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        # First turn: compaction block + terminal text.
        llm.push(LLMResponse(
            content=[
                {"type": "compaction", "content": "summary-of-parent"},
                {"type": "text", "text": "final answer"},
            ],
            stop_reason="end_turn",
        ))

        state = await engine.spawn("big context", await_seconds=5.0)
        assert state.status == TaskStatus.COMPLETED

        entries = store.read_entries(state.context_id)
        compaction_entries = [e for e in entries if e["t"] == "compact"]
        assert len(compaction_entries) == 1
        assert compaction_entries[0]["content"] == "summary-of-parent"

        # And replay sees it as the cursor (with assistant after it).
        replay = store.replay(state.context_id)
        assert replay[0]["role"] == "assistant"
        assert replay[0]["content"] == "summary-of-parent"


# --------------------------------------------------------------------------- #
# Engine: input validation
# --------------------------------------------------------------------------- #


class TestInputValidation:
    async def test_context_busy_error_surface_has_active_task(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        ctx = "ctx-1"
        first = await engine.spawn("1", context_id=ctx, await_seconds=0.05)
        try:
            await engine.spawn("2", context_id=ctx, await_seconds=0.05)
        except ContextBusyError as e:
            assert e.active_task_id == first.task_id
        finally:
            llm.push(LLMResponse(
                content=[{"type": "text", "text": "ok"}],
                stop_reason="end_turn",
            ))
            rec = engine.registry.get(first.task_id)
            if rec is not None:
                await asyncio.wait_for(rec.completion_event.wait(), timeout=2.0)


# --------------------------------------------------------------------------- #
# Engine: via protocol is preserved
# --------------------------------------------------------------------------- #


class TestViaPreservation:
    async def test_via_mcp_tagged_on_journal(
        self, engine: TaskEngine, store: JournalStore,
    ) -> None:
        state = await engine.spawn("hi", via="mcp")
        entries = store.read_entries(state.context_id)
        for e in entries:
            if e["t"] == "msg":
                assert e["via"] == "mcp"
                return
        pytest.fail("no msg entry recorded")

    async def test_via_a2a_tagged_on_journal(
        self, engine: TaskEngine, store: JournalStore,
    ) -> None:
        state = await engine.spawn("hi", via="a2a")
        entries = store.read_entries(state.context_id)
        for e in entries:
            if e["t"] == "msg":
                assert e["via"] == "a2a"
                return
        pytest.fail("no msg entry recorded")


# --------------------------------------------------------------------------- #
# Enterprise hardening
# --------------------------------------------------------------------------- #


class TestInputValidation:
    async def test_empty_message_rejected(self, engine: TaskEngine) -> None:
        with pytest.raises(InvalidTaskInputError):
            await engine.spawn("")

    async def test_whitespace_message_rejected(self, engine: TaskEngine) -> None:
        with pytest.raises(InvalidTaskInputError):
            await engine.spawn("   \n\t  ")


class TestTaskNotFoundError:
    async def test_error_carries_searched_breadcrumb(
        self, engine: TaskEngine
    ) -> None:
        with pytest.raises(TaskNotFoundError) as exc_info:
            await engine.poll("nonexistent-task")
        assert exc_info.value.searched is not None
        assert "registry+filesystem" in exc_info.value.searched

    async def test_error_names_searched_context(self, engine: TaskEngine) -> None:
        with pytest.raises(TaskNotFoundError) as exc_info:
            await engine.poll("nonexistent", context_id="some-ctx")
        assert exc_info.value.searched is not None
        assert "some-ctx" in exc_info.value.searched


class TestContextLockGC:
    """The ``_context_locks`` dict must shrink as contexts go idle."""

    async def test_lock_released_after_task_completes(
        self, engine: TaskEngine
    ) -> None:
        ctx = str(uuid4())
        state = await engine.spawn("hello", context_id=ctx)
        assert state.status == TaskStatus.COMPLETED
        # Give the asyncio done-callback a tick to fire.
        await asyncio.sleep(0.05)
        assert ctx not in engine._context_locks, (  # noqa: SLF001 — test helper
            "idle context lock must be garbage collected"
        )

    async def test_lock_retained_while_task_active(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        from tests.unit.test_task import ControllableLLM

        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        ctx = str(uuid4())
        state = await engine.spawn("wait", context_id=ctx, await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        assert ctx in engine._context_locks  # noqa: SLF001 — test helper

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "ok"}],
            stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            await asyncio.wait_for(rec.completion_event.wait(), timeout=3.0)
        await asyncio.sleep(0.05)

        assert ctx not in engine._context_locks  # noqa: SLF001

    async def test_locks_capped_across_many_contexts(
        self, engine: TaskEngine
    ) -> None:
        for _ in range(20):
            await engine.spawn("hi", context_id=str(uuid4()))
        await asyncio.sleep(0.1)
        # Worst case: 1 lock held during inflight final-poll callback
        # ordering, but after settling, all should be gone.
        assert len(engine._context_locks) == 0  # noqa: SLF001


class TestShutdown:
    async def test_shutdown_is_idempotent(self, engine: TaskEngine) -> None:
        await engine.shutdown()
        await engine.shutdown()

    async def test_shutdown_rejects_new_spawns(self, engine: TaskEngine) -> None:
        from agentlings.core.task import TaskError

        await engine.shutdown()
        with pytest.raises(TaskError):
            await engine.spawn("hi")

    async def test_shutdown_drains_inflight_workers(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        from tests.unit.test_task import ControllableLLM

        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("work", await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        shutdown_task = asyncio.create_task(engine.shutdown(grace_seconds=3.0))

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
        ))

        await asyncio.wait_for(shutdown_task, timeout=5.0)

        assert engine.registry.active_count() == 0
        assert len(engine._workers) == 0  # noqa: SLF001

    async def test_shutdown_hard_cancels_past_grace(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        from tests.unit.test_task import ControllableLLM

        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        state = await engine.spawn("forever", await_seconds=0.05)
        assert state.status == TaskStatus.WORKING

        # LLM never answers; shutdown grace expires and hard-cancels.
        await asyncio.wait_for(engine.shutdown(grace_seconds=0.2), timeout=2.0)

        assert len(engine._workers) == 0  # noqa: SLF001


class TestCrashRecoveryRegressions:
    """Three recovery bugs that Codex flagged on review."""

    async def test_task_done_without_merge_start_is_recovered(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        """Crash between sub-journal TaskCompleted and parent MergeStarted.

        Prior bug: ``_recover_orphaned_subjournals`` skipped the sub-journal
        because it had a terminal, and ``_recover_partial_mergebacks`` found
        no MergeStarted to repair, so the successful exchange was lost
        from parent replay forever.
        """
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config, store=store,
            llm=MockLLMClient(), tools=ToolRegistry(),
        )

        ctx = "ctx-1"
        tid = "t-completed-but-unmerged"
        store.create(ctx)
        from agentlings.core.models import TaskDispatched
        store.append(ctx, TaskDispatched(ctx=ctx, task_id=tid))

        tj = TaskJournal(store.task_path(ctx, tid))
        tj.create()
        tj.append(TaskStarted(ctx=ctx, task_id=tid, message="hello", via="mcp"))
        tj.append(TaskCompleted(
            ctx=ctx, task_id=tid,
            final_response=[{"type": "text", "text": "world"}],
        ))

        engine.recover_on_startup()

        entries = store.read_entries(ctx)
        msgs = [e for e in entries if e["t"] == "msg" and e["task_id"] == tid]
        roles = [m["role"] for m in msgs]
        assert "user" in roles and "assistant" in roles, (
            "recovery must merge the completed task's content to parent"
        )

        replay = store.replay(ctx)
        assert [m["role"] for m in replay] == ["user", "assistant"]
        assert any("hello" in str(m.get("content", "")) for m in replay if m["role"] == "user")
        assert any("world" in str(m.get("content", "")) for m in replay if m["role"] == "assistant")

    async def test_repair_does_not_duplicate_compaction_after_assistant(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        """Crash after assistant write but before MergeCommitted.

        Prior bug: ``_repair_merge`` unconditionally appended a compaction
        entry even when the assistant message was already in parent. That
        new compaction became the replay cursor and hid the final reply.
        """
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config, store=store,
            llm=MockLLMClient(), tools=ToolRegistry(),
        )

        ctx = "ctx-1"
        tid = "t-partial-merge-with-compaction"
        store.create(ctx)

        from agentlings.core.models import TaskDispatched
        # Parent has everything except MergeCommitted.
        store.append(ctx, TaskDispatched(ctx=ctx, task_id=tid))
        store.append(ctx, MergeStarted(ctx=ctx, task_id=tid))
        store.append(ctx, MessageEntry(
            ctx=ctx, role="user",
            content=[{"type": "text", "text": "hi"}],
            via="mcp", task_id=tid,
        ))
        store.append(ctx, CompactionEntry(ctx=ctx, content="summary-v1"))
        store.append(ctx, MessageEntry(
            ctx=ctx, role="assistant",
            content=[{"type": "text", "text": "final reply"}],
            via="mcp", task_id=tid,
        ))

        # Sub-journal is complete with the same compaction.
        tj = TaskJournal(store.task_path(ctx, tid))
        tj.create()
        tj.append(TaskStarted(ctx=ctx, task_id=tid, message="hi", via="mcp"))
        tj.append(CompactionEntry(ctx=ctx, content="summary-v1"))
        tj.append(TaskCompleted(
            ctx=ctx, task_id=tid,
            final_response=[{"type": "text", "text": "final reply"}],
        ))

        engine.recover_on_startup()

        entries = store.read_entries(ctx)
        compactions = [e for e in entries if e["t"] == "compact"]
        assert len(compactions) == 1, (
            "recovery must not duplicate the compaction; replay would "
            "otherwise lose the final assistant reply"
        )

        replay = store.replay(ctx)
        # Latest compact = the original one, before the assistant. Replay
        # should still expose the final reply.
        texts = [
            str(m.get("content", ""))
            for m in replay
            if m["role"] == "assistant"
        ]
        assert any("final reply" in t for t in texts), (
            "final assistant reply must remain visible after recovery"
        )

    async def test_repair_preserves_existing_content_without_duplication(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        """Full merge-back but missing MergeCommitted — idempotent close."""
        store = JournalStore(tmp_data_dir)
        engine = TaskEngine(
            config=test_config, store=store,
            llm=MockLLMClient(), tools=ToolRegistry(),
        )

        ctx = "ctx-1"
        tid = "t-missing-commit"
        store.create(ctx)

        from agentlings.core.models import TaskDispatched
        store.append(ctx, TaskDispatched(ctx=ctx, task_id=tid))
        store.append(ctx, MergeStarted(ctx=ctx, task_id=tid))
        store.append(ctx, MessageEntry(
            ctx=ctx, role="user", content=[{"type": "text", "text": "q"}],
            via="mcp", task_id=tid,
        ))
        store.append(ctx, MessageEntry(
            ctx=ctx, role="assistant",
            content=[{"type": "text", "text": "a"}],
            via="mcp", task_id=tid,
        ))

        tj = TaskJournal(store.task_path(ctx, tid))
        tj.create()
        tj.append(TaskStarted(ctx=ctx, task_id=tid, message="q", via="mcp"))
        tj.append(TaskCompleted(
            ctx=ctx, task_id=tid,
            final_response=[{"type": "text", "text": "a"}],
        ))

        engine.recover_on_startup()

        entries = store.read_entries(ctx)
        # No duplicate msgs for this taskId.
        user_count = sum(
            1 for e in entries
            if e["t"] == "msg" and e["task_id"] == tid and e["role"] == "user"
        )
        assistant_count = sum(
            1 for e in entries
            if e["t"] == "msg" and e["task_id"] == tid and e["role"] == "assistant"
        )
        assert user_count == 1
        assert assistant_count == 1
        # MergeCommitted now present.
        assert any(
            e["t"] == "merge_commit" and e["task_id"] == tid for e in entries
        )


class TestTracebackCapture:
    async def test_failed_task_stores_traceback(
        self, tmp_data_dir: Path, test_config: AgentConfig
    ) -> None:
        from tests.unit.test_task import ControllableLLM

        llm = ControllableLLM()
        store = JournalStore(tmp_data_dir)
        tools = ToolRegistry()
        engine = TaskEngine(config=test_config, store=store, llm=llm, tools=tools)

        spawn_task = asyncio.create_task(
            engine.spawn("boom", await_seconds=2.0)
        )
        await asyncio.sleep(0.05)
        llm.push_exception(RuntimeError("deep failure in tool"))
        state = await spawn_task

        assert state.status == TaskStatus.FAILED
        tj = TaskJournal(store.task_path(state.context_id, state.task_id))
        term = tj.terminal_entry()
        assert term is not None
        assert term["t"] == "task_fail"
        details = term.get("error_details") or ""
        # Traceback capture should include the call site, not just the message.
        assert "RuntimeError" in details
        assert "deep failure in tool" in details
        assert "Traceback" in details or "File " in details


# --------------------------------------------------------------------------- #
# Pytest config: mark async tests
# --------------------------------------------------------------------------- #


pytestmark = pytest.mark.asyncio
