"""Task execution engine: every request becomes a task with its own sub-journal.

Core responsibilities:

- Registry tracks live (in-flight) tasks, one per context.
- Workers run the message loop in a per-task sub-journal.
- Merge-back writes successful results to the parent journal under a per-context lock.
- Cancellation is cooperative; the worker checks a flag between checkpoints.
- Startup recovery closes out orphaned sub-journals and merge-backs.

The engine is the only public entry point protocol adapters should talk to.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from agentlings.config import AgentConfig
from agentlings.core.completion import (
    CancellationRequested,
    CompletionResult,
    run_completion,
)
from agentlings.core.llm import BaseLLMClient
from agentlings.core.memory_store import MemoryFileStore
from agentlings.core.models import (
    CompactionEntry,
    MergeCommitted,
    MergeStarted,
    MessageEntry,
    TaskCancelled,
    TaskCompleted,
    TaskDispatched,
    TaskFailed,
    TaskStarted,
)
from agentlings.core.prompt import build_system_prompt
from agentlings.core.skills import SkillRef
from agentlings.core.store import JournalStore, TaskJournal
from agentlings.core.telemetry import (
    attach_context,
    capture_context,
    get_meter,
    otel_span,
)
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Cap on the stored traceback. Preserves enough stack context for
# post-mortem diagnosis without bloating the journal when a deep error chain
# serializes dozens of KB.
_FAILURE_DETAIL_MAX = 4096

# Grace window given to in-flight workers during ``TaskEngine.shutdown``.
# Beyond this the worker's asyncio task is hard-cancelled.
_SHUTDOWN_GRACE_SECONDS = 5.0


@dataclass
class _TaskMetrics:
    """Lifecycle counters and gauges surfaced via OpenTelemetry.

    Enterprise operators rely on these to alert on failure-rate spikes,
    track active concurrency, and size capacity. Lookup is cached at
    module init so individual call sites pay no repeat cost.
    """

    tasks_spawned: Any
    tasks_completed: Any
    tasks_failed: Any
    tasks_cancelled: Any
    tasks_active: Any
    context_busy_rejections: Any
    crash_recovery_repaired: Any
    crash_recovery_failed: Any


@functools.lru_cache(maxsize=1)
def _build_metrics() -> _TaskMetrics:
    m = get_meter()
    return _TaskMetrics(
        tasks_spawned=m.create_counter(
            "agentling.tasks.spawned_total",
            description="Total tasks ingressed into the engine.",
        ),
        tasks_completed=m.create_counter(
            "agentling.tasks.completed_total",
            description="Total tasks that reached completed terminal state.",
        ),
        tasks_failed=m.create_counter(
            "agentling.tasks.failed_total",
            description="Total tasks that failed (exception or asyncio cancel).",
        ),
        tasks_cancelled=m.create_counter(
            "agentling.tasks.cancelled_total",
            description="Total tasks that reached cancelled terminal state.",
        ),
        tasks_active=m.create_up_down_counter(
            "agentling.tasks.active",
            description="Tasks currently registered (gauge via up/down counter).",
        ),
        context_busy_rejections=m.create_counter(
            "agentling.tasks.context_busy_rejections_total",
            description="Spawn attempts rejected because the context was busy.",
        ),
        crash_recovery_repaired=m.create_counter(
            "agentling.tasks.crash_recovery_repaired_total",
            description="Orphaned sub-journals or partial merges repaired on startup.",
        ),
        crash_recovery_failed=m.create_counter(
            "agentling.tasks.crash_recovery_failed_total",
            description="Recovery passes that raised an exception for a context.",
        ),
    )


_METRICS = _build_metrics()


class TaskStatus(str, Enum):
    """Lifecycle states tracked by the registry for a live task.

    Sub-journal terminal markers (``task_done``/``task_fail``/``task_cancel``)
    mirror the corresponding enum values. ``working`` means the task exists
    and may still do more work; ``cancelling`` means a cancel flag was raised
    and the worker has not yet observed the terminal checkpoint.
    """

    WORKING = "working"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset({
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
})


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TaskError(Exception):
    """Base class for engine-raised exceptions."""


class ContextBusyError(TaskError):
    """Raised when a new request targets a context that already has a live task."""

    def __init__(self, context_id: str, active_task_id: str) -> None:
        super().__init__(
            f"context {context_id} is busy with task {active_task_id}"
        )
        self.context_id = context_id
        self.active_task_id = active_task_id


class TaskNotFoundError(TaskError):
    """Raised when a poll/cancel targets a task the engine cannot locate.

    Carries a ``searched`` breadcrumb so operators can tell whether the
    lookup exhausted the in-memory registry, a specific context directory,
    or both.
    """

    def __init__(self, task_id: str, searched: str | None = None) -> None:
        detail = f" (searched: {searched})" if searched else ""
        super().__init__(f"task {task_id} not found{detail}")
        self.task_id = task_id
        self.searched = searched


class TaskContextMismatchError(TaskError):
    """Raised when the caller asserts a contextId that doesn't match the task's."""

    def __init__(self, task_id: str, expected: str, actual: str) -> None:
        super().__init__(
            f"task {task_id}: expected context {expected}, actual context {actual}"
        )
        self.task_id = task_id
        self.expected = expected
        self.actual = actual


class InvalidTaskInputError(TaskError):
    """Raised on malformed inputs (e.g. both ``message`` and ``taskId`` present)."""


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


@dataclass
class TaskRecord:
    """Registry entry for a live task.

    Attributes:
        task_id: Unique task identifier.
        context_id: Parent context this task belongs to.
        status: Current lifecycle state.
        cancel_flag: Set to ``True`` to request cooperative cancellation.
        completion_event: Fires when the task reaches a terminal state.
        owner: Placeholder for future multi-tenant auth binding.
        message: The original user message text.
        via: Protocol surface that created this task.
    """

    task_id: str
    context_id: str
    message: str
    via: str = "a2a"
    status: TaskStatus = TaskStatus.WORKING
    cancel_flag: bool = False
    completion_event: asyncio.Event = field(default_factory=asyncio.Event)
    owner: str | None = None


class TaskRegistry:
    """In-memory store of live tasks with per-context admission control.

    One task per context at a time — attempting to register a second task
    for the same context raises ``ContextBusyError``. Terminal tasks are
    removed immediately via ``unregister``.

    All mutations are synchronous. asyncio is single-threaded and there
    are no ``await`` points inside the critical sections here, so the
    "check then set" in ``register`` runs atomically without needing an
    ``asyncio.Lock``. Keeping these methods synchronous also means
    ``unregister`` in a worker's ``finally`` block cannot be interrupted
    by a re-delivered ``CancelledError`` during shutdown — the registry
    always releases state cleanly.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._by_context: dict[str, str] = {}

    def register(
        self,
        task_id: str,
        context_id: str,
        message: str,
        via: str = "a2a",
        owner: str | None = None,
    ) -> TaskRecord:
        """Atomically check context-busy and register a new task.

        Raises:
            ContextBusyError: If ``context_id`` already has a live task.
        """
        if context_id in self._by_context:
            raise ContextBusyError(
                context_id=context_id,
                active_task_id=self._by_context[context_id],
            )
        record = TaskRecord(
            task_id=task_id,
            context_id=context_id,
            message=message,
            via=via,
            owner=owner,
        )
        self._tasks[task_id] = record
        self._by_context[context_id] = task_id
        return record

    def unregister(self, task_id: str) -> None:
        """Remove a task from the registry. Idempotent."""
        record = self._tasks.pop(task_id, None)
        if record is not None:
            self._by_context.pop(record.context_id, None)

    def get(self, task_id: str) -> TaskRecord | None:
        """Return the record for ``task_id`` or ``None`` if unknown."""
        return self._tasks.get(task_id)

    def get_by_context(self, context_id: str) -> TaskRecord | None:
        """Return the live task for a context, if any."""
        tid = self._by_context.get(context_id)
        return self._tasks.get(tid) if tid else None

    def request_cancel(self, task_id: str) -> bool:
        """Flip the cancel flag on a live task. Returns ``False`` if unknown."""
        record = self._tasks.get(task_id)
        if record is None:
            return False
        record.cancel_flag = True
        record.status = TaskStatus.CANCELLING
        return True

    def active_count(self) -> int:
        """Number of currently-registered tasks."""
        return len(self._tasks)


# --------------------------------------------------------------------------- #
# Task state returned to callers
# --------------------------------------------------------------------------- #


@dataclass
class TaskState:
    """Snapshot of a task returned to protocol handlers.

    Attributes:
        task_id: Task identifier.
        context_id: Parent context.
        status: Current lifecycle state.
        content: Final response content blocks (populated when status is
            ``completed``; otherwise empty).
        error: Short reason/details string for failed or cancelled states.
    """

    task_id: str
    context_id: str
    status: TaskStatus
    content: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #


class TaskWorker:
    """Executes one task end-to-end: snapshot → loop → terminal → merge-back.

    Workers never touch the parent journal until merge-back, which is guarded
    by a per-context lock held by the engine. Sub-journal writes happen
    continuously as the worker runs so ops can inspect progress and crash
    recovery has something to read.
    """

    def __init__(
        self,
        record: TaskRecord,
        config: AgentConfig,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        store: JournalStore,
        task_journal: TaskJournal,
        registry: TaskRegistry,
        context_lock: asyncio.Lock,
        memory_store: MemoryFileStore | None = None,
        skills: list[SkillRef] | None = None,
        otel_parent_context: Any = None,
    ) -> None:
        self._record = record
        self._config = config
        self._llm = llm
        self._tools = tools
        self._store = store
        self._journal = task_journal
        self._registry = registry
        self._context_lock = context_lock
        self._memory_store = memory_store
        self._skills = skills or []
        self._otel_parent_context = otel_parent_context
        self._completion_result: CompletionResult | None = None

    async def run(self) -> None:
        """Drive the task to a terminal state.

        The worker writes exactly one terminal sub-journal entry before
        returning, regardless of success, failure, or cancellation.

        The worker span is attached to the OTel context captured at
        ``engine.spawn`` time so traces stitch through the
        request-handler → engine → worker → completion → llm boundary.
        """
        record = self._record
        with attach_context(self._otel_parent_context), otel_span("agentling.task.worker", {
            "task.id": record.task_id,
            "task.context_id": record.context_id,
            "task.via": record.via,
        }) as span:
            try:
                await self._run_inner()
                span.set_attribute("task.terminal", "completed")
                if self._completion_result is not None:
                    self._stamp_tokens(span, self._completion_result.token_usage)
            except CancellationRequested as cancelled:
                span.set_attribute("task.terminal", "cancelled")
                if cancelled.partial is not None:
                    self._stamp_tokens(span, cancelled.partial.token_usage)
                cancel = TaskCancelled(
                    ctx=record.context_id,
                    task_id=record.task_id,
                    reason="cooperative_cancel",
                )
                self._journal.append(cancel)
                self._store.append(record.context_id, cancel)
                record.status = TaskStatus.CANCELLED
                _METRICS.tasks_cancelled.add(1)
                logger.info(
                    "task cancelled",
                    extra={
                        "task_id": record.task_id,
                        "context_id": record.context_id,
                        "reason": "cooperative_cancel",
                    },
                )
            except asyncio.CancelledError:
                # Re-raised so asyncio teardown observes the cancel; the
                # terminal markers below ensure operators can still see
                # what happened post-mortem.
                fail = TaskFailed(
                    ctx=record.context_id,
                    task_id=record.task_id,
                    reason="asyncio_cancel",
                )
                self._journal.append(fail)
                self._store.append(record.context_id, fail)
                record.status = TaskStatus.FAILED
                _METRICS.tasks_failed.add(1)
                raise
            except Exception as exc:  # noqa: BLE001 — recorded as failure
                span.set_attribute("task.terminal", "failed")
                tb = traceback.format_exc()
                logger.error(
                    "task failed",
                    extra={
                        "task_id": record.task_id,
                        "context_id": record.context_id,
                        "exc_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                fail = TaskFailed(
                    ctx=record.context_id,
                    task_id=record.task_id,
                    reason="worker_exception",
                    error_details=tb[-_FAILURE_DETAIL_MAX:],
                )
                self._journal.append(fail)
                self._store.append(record.context_id, fail)
                record.status = TaskStatus.FAILED
                _METRICS.tasks_failed.add(1)
            else:
                _METRICS.tasks_completed.add(1)
            finally:
                record.completion_event.set()
                # Synchronous — no await point, so even a re-delivered
                # CancelledError during shutdown cannot interrupt cleanup.
                self._registry.unregister(record.task_id)

    async def _run_inner(self) -> None:
        """The happy path: snapshot, complete, merge back.

        On cancellation or exception, the outer ``run`` method handles terminal
        marker writes and registry cleanup.
        """
        record = self._record

        messages = list(self._store.replay(record.context_id))
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": record.message}],
        })

        memory = self._memory_store.load() if self._memory_store else None
        memory_config = self._config.memory_config
        system = build_system_prompt(
            config=self._config,
            memory=memory,
            data_dir=self._config.agent_data_dir,
            injection_prompt=memory_config.injection_prompt if memory_config else None,
            token_budget=memory_config.token_budget if memory_config else 2000,
            skills=self._skills,
        )

        async def _on_turn(turn: Any) -> None:
            self._journal.append(MessageEntry(
                ctx=record.context_id,
                role="assistant",
                content=turn.response.content,
                via=record.via,  # type: ignore[arg-type]
                task_id=record.task_id,
            ))
            if turn.tool_results:
                self._journal.append(MessageEntry(
                    ctx=record.context_id,
                    role="user",
                    content=turn.tool_results,
                    via=record.via,  # type: ignore[arg-type]
                    task_id=record.task_id,
                ))
            for block in turn.response.content:
                if block.get("type") == "compaction":
                    self._journal.append(CompactionEntry(
                        ctx=record.context_id,
                        content=block.get("content", ""),
                    ))

        result = await run_completion(
            llm=self._llm,
            system=system,
            messages=messages,
            tools=self._tools,
            turn_callback=_on_turn,
            should_cancel=lambda: record.cancel_flag,
        )
        # Stash so ``run`` can stamp the cycle's token totals onto the
        # worker span before exiting.
        self._completion_result = result

        self._journal.append(TaskCompleted(
            ctx=record.context_id,
            task_id=record.task_id,
            final_response=result.content,
        ))

        await self._merge_back(result)
        record.status = TaskStatus.COMPLETED

    @staticmethod
    def _stamp_tokens(span: Any, totals: dict[str, int]) -> None:
        """Stamp token totals onto the worker span (cheap when telemetry off)."""
        span.set_attribute("task.input_tokens", int(totals.get("input", 0)))
        span.set_attribute("task.output_tokens", int(totals.get("output", 0)))
        span.set_attribute("task.cache_creation_tokens", int(totals.get("cache_creation", 0)))
        span.set_attribute("task.cache_read_tokens", int(totals.get("cache_read", 0)))

    async def _merge_back(self, result: CompletionResult) -> None:
        """Atomically propagate the task's outcome to the parent journal.

        Ordering:
            1. ``MergeStarted`` wrapper.
            2. User ``MessageEntry`` with ``task_id`` meta.
            3. Latest ``CompactionEntry`` from the sub-journal, if any.
            4. Assistant ``MessageEntry`` (final response).
            5. ``MergeCommitted`` wrapper.

        All five writes happen under the context lock in one
        ``append_many`` call so startup crash recovery can detect a partial
        merge via an orphaned ``MergeStarted``.
        """
        record = self._record
        entries: list[Any] = [
            MergeStarted(ctx=record.context_id, task_id=record.task_id),
            MessageEntry(
                ctx=record.context_id,
                role="user",
                content=[{"type": "text", "text": record.message}],
                via=record.via,  # type: ignore[arg-type]
                task_id=record.task_id,
            ),
        ]
        latest_comp = self._journal.latest_compaction()
        if latest_comp is not None:
            entries.append(CompactionEntry(
                ctx=record.context_id,
                content=latest_comp.get("content", ""),
            ))
        entries.append(MessageEntry(
            ctx=record.context_id,
            role="assistant",
            content=result.content,
            via=record.via,  # type: ignore[arg-type]
            task_id=record.task_id,
        ))
        entries.append(MergeCommitted(ctx=record.context_id, task_id=record.task_id))

        async with self._context_lock:
            self._store.append_many(record.context_id, entries)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


class TaskEngine:
    """Top-level orchestrator. Spawns, polls, and cancels tasks.

    Concurrency model:
        - The registry lock serializes admission (one task per context).
        - ``_context_locks_guard`` serializes access to the lock dict itself.
        - Per-context ``asyncio.Lock`` objects serialize merge-back writes to
          the parent journal. Held briefly by workers only.

    Lock acquisition order (never violated to avoid deadlock):
        registry._lock  →  _context_locks_guard  →  per-context lock
    """

    def __init__(
        self,
        config: AgentConfig,
        store: JournalStore,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        memory_store: MemoryFileStore | None = None,
        skills: list[SkillRef] | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._llm = llm
        self._tools = tools
        self._memory_store = memory_store
        self._skills = skills or []
        self._registry = TaskRegistry()
        self._context_locks: dict[str, asyncio.Lock] = {}
        self._context_locks_guard = asyncio.Lock()
        self._workers: dict[str, asyncio.Task[None]] = {}
        self._shutdown_started = False

    @property
    def registry(self) -> TaskRegistry:
        """Underlying registry (for introspection / tests)."""
        return self._registry

    async def _lock_for(self, context_id: str) -> asyncio.Lock:
        async with self._context_locks_guard:
            lock = self._context_locks.get(context_id)
            if lock is None:
                lock = asyncio.Lock()
                self._context_locks[context_id] = lock
            return lock

    def _on_worker_done(self, task_id: str, context_id: str) -> None:
        """Worker post-finish housekeeping fired from the asyncio callback.

        Removes the ``_workers`` entry and — critically — garbage-collects
        the context lock if no task remains on that context. Without this,
        ``_context_locks`` grows unbounded in long-running servers.

        Must not touch asyncio locks directly (we're in a ``done_callback``
        synchronous hook). We only drop keys that are provably safe.
        """
        self._workers.pop(task_id, None)
        _METRICS.tasks_active.add(-1)
        if self._registry.get_by_context(context_id) is not None:
            return
        lock = self._context_locks.get(context_id)
        if lock is None or lock.locked():
            return
        self._context_locks.pop(context_id, None)

    async def spawn(
        self,
        message: str,
        context_id: str | None = None,
        via: str = "a2a",
        await_seconds: float = 60.0,
        owner: str | None = None,
        task_id: str | None = None,
    ) -> TaskState:
        """Start a new task, block up to ``await_seconds`` for completion.

        Returns a ``TaskState`` reflecting the state at the moment the HTTP
        handler should respond. If the task finished within the await window,
        ``status == COMPLETED`` and ``content`` is the final response. If the
        await timed out, ``status == WORKING`` and ``content`` is empty — the
        caller yields the task handle to its client.

        Args:
            task_id: Optional caller-supplied task id. Used by protocol
                adapters (e.g. A2A) whose SDK generates the id themselves and
                require it to match across request/response.
        """
        if self._shutdown_started:
            raise TaskError("engine is shutting down")

        if not message or not message.strip():
            raise InvalidTaskInputError("message must not be empty")

        ctx_id = context_id or str(uuid4())
        task_id = task_id or str(uuid4())

        with otel_span("agentling.engine.spawn", {
            "task.id": task_id,
            "task.context_id": ctx_id,
            "task.via": via,
            "task.await_seconds": await_seconds,
        }) as span:
            if not self._store.exists(ctx_id):
                self._store.create(ctx_id)
                logger.info(
                    "context created",
                    extra={"context_id": ctx_id, "via": via},
                )

            try:
                record = self._registry.register(
                    task_id=task_id,
                    context_id=ctx_id,
                    message=message,
                    via=via,
                    owner=owner,
                )
            except ContextBusyError:
                _METRICS.context_busy_rejections.add(1)
                span.set_attribute("task.outcome", "context_busy")
                raise

            self._store.append(ctx_id, TaskDispatched(ctx=ctx_id, task_id=task_id))

            task_journal = TaskJournal(self._store.task_path(ctx_id, task_id))
            task_journal.create()
            task_journal.append(TaskStarted(
                ctx=ctx_id,
                task_id=task_id,
                message=message,
                via=via,  # type: ignore[arg-type]
            ))

            ctx_lock = await self._lock_for(ctx_id)
            # Capture the calling OTel context here so the worker — which
            # runs in a fresh ``asyncio.create_task`` and would otherwise
            # have no parent span — re-attaches it before opening its
            # own span tree.
            parent_ctx = capture_context()
            worker = TaskWorker(
                record=record,
                config=self._config,
                llm=self._llm,
                tools=self._tools,
                store=self._store,
                task_journal=task_journal,
                registry=self._registry,
                context_lock=ctx_lock,
                memory_store=self._memory_store,
                skills=self._skills,
                otel_parent_context=parent_ctx,
            )
            asyncio_task = asyncio.create_task(worker.run(), name=f"task:{task_id}")
            self._workers[task_id] = asyncio_task
            asyncio_task.add_done_callback(
                lambda _t: self._on_worker_done(task_id, ctx_id)
            )

            _METRICS.tasks_spawned.add(1)
            _METRICS.tasks_active.add(1)
            logger.info(
                "task spawned",
                extra={"task_id": task_id, "context_id": ctx_id, "via": via},
            )

            try:
                if await_seconds > 0:
                    await asyncio.wait_for(
                        record.completion_event.wait(),
                        timeout=await_seconds,
                    )
            except asyncio.TimeoutError:
                pass

            state = self._state_from_task(task_id, ctx_id, record, task_journal)
            span.set_attribute("task.status", state.status.value)
            return state

    async def poll(
        self,
        task_id: str,
        context_id: str | None = None,
        wait_seconds: float = 0.0,
        cap_seconds: float = 60.0,
    ) -> TaskState:
        """Poll a task for its current state.

        Args:
            task_id: Target task.
            context_id: Optional assertion — if provided, must match the task's
                context or a ``TaskContextMismatchError`` is raised.
            wait_seconds: If positive, block up to this many seconds for a
                terminal state (capped at ``cap_seconds``).
            cap_seconds: Server-side maximum for ``wait_seconds``.

        Raises:
            TaskNotFoundError: If no sub-journal and no registry entry exist.
            TaskContextMismatchError: If the assertion doesn't match.
        """
        with otel_span("agentling.engine.poll", {
            "task.id": task_id,
            "task.context_id": context_id or "",
            "task.wait_seconds": wait_seconds,
        }) as span:
            record = self._registry.get(task_id)
            resolved_ctx = context_id or (record.context_id if record else None)

            if resolved_ctx is None:
                found = self._locate_task_context(task_id)
                if found is None:
                    raise TaskNotFoundError(
                        task_id,
                        searched="registry+filesystem",
                    )
                resolved_ctx = found

            if context_id is not None and record is not None and context_id != record.context_id:
                raise TaskContextMismatchError(task_id, context_id, record.context_id)

            task_journal = TaskJournal(self._store.task_path(resolved_ctx, task_id))
            if not task_journal.exists() and record is None:
                raise TaskNotFoundError(
                    task_id,
                    searched=f"context={resolved_ctx}",
                )

            if wait_seconds > 0 and record is not None:
                effective = min(wait_seconds, cap_seconds)
                try:
                    await asyncio.wait_for(
                        record.completion_event.wait(),
                        timeout=effective,
                    )
                except asyncio.TimeoutError:
                    pass

            record = self._registry.get(task_id) or record
            state = self._state_from_task(task_id, resolved_ctx, record, task_journal)
            span.set_attribute("task.status", state.status.value)
            return state

    async def cancel(
        self,
        task_id: str,
        context_id: str | None = None,
    ) -> TaskState:
        """Request cooperative cancellation. Returns the state after the request.

        If the task already terminal, returns its terminal state unchanged.
        If the task is active, flips the cancel flag (the worker observes it
        at its next checkpoint and writes the cancel markers).
        """
        with otel_span("agentling.engine.cancel", {
            "task.id": task_id,
            "task.context_id": context_id or "",
        }) as span:
            record = self._registry.get(task_id)
            if record is None:
                resolved_ctx = context_id or self._locate_task_context(task_id)
                if resolved_ctx is None:
                    raise TaskNotFoundError(
                        task_id,
                        searched="registry+filesystem",
                    )
                tj = TaskJournal(self._store.task_path(resolved_ctx, task_id))
                if not tj.exists():
                    raise TaskNotFoundError(
                        task_id,
                        searched=f"context={resolved_ctx}",
                    )
                state = self._state_from_task(task_id, resolved_ctx, None, tj)
                span.set_attribute("task.cancel_outcome", "already_terminal")
                span.set_attribute("task.status", state.status.value)
                return state

            if context_id is not None and context_id != record.context_id:
                raise TaskContextMismatchError(task_id, context_id, record.context_id)

            self._registry.request_cancel(task_id)
            tj = TaskJournal(self._store.task_path(record.context_id, task_id))
            state = self._state_from_task(task_id, record.context_id, record, tj)
            span.set_attribute("task.cancel_outcome", "requested")
            span.set_attribute("task.status", state.status.value)
            return state

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _locate_task_context(self, task_id: str) -> str | None:
        """Scan the data directory for a sub-journal matching ``task_id``.

        Used as a fallback when a poll arrives without a ``context_id`` and
        the task isn't in the registry (already terminal and unregistered).
        Path layout is ``data/{ctx_id}/tasks/{task_id}.jsonl``.
        """
        matches = list(self._store.data_dir.glob(f"*/tasks/{task_id}.jsonl"))
        if not matches:
            return None
        return matches[0].parent.parent.name

    def _state_from_task(
        self,
        task_id: str,
        context_id: str,
        record: TaskRecord | None,
        journal: TaskJournal,
    ) -> TaskState:
        """Build a ``TaskState`` from the registry + sub-journal tail."""
        terminal = journal.terminal_entry()
        if terminal is not None:
            t = terminal["t"]
            if t == "task_done":
                return TaskState(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus.COMPLETED,
                    content=terminal.get("final_response", []),
                )
            if t == "task_fail":
                return TaskState(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus.FAILED,
                    error=terminal.get("error_details") or terminal.get("reason"),
                )
            if t == "task_cancel":
                return TaskState(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus.CANCELLED,
                    error=terminal.get("reason"),
                )
        # No terminal yet — must still be live.
        status = (
            record.status if record is not None else TaskStatus.WORKING
        )
        return TaskState(
            task_id=task_id,
            context_id=context_id,
            status=status,
        )

    # ------------------------------------------------------------------ #
    # Graceful shutdown
    # ------------------------------------------------------------------ #

    async def shutdown(self, grace_seconds: float = _SHUTDOWN_GRACE_SECONDS) -> None:
        """Stop accepting new spawns and drain in-flight workers.

        1. Mark the engine as shutting down (future ``spawn`` calls raise).
        2. Flip every live task's cancel flag so workers stop at their next
           cooperative checkpoint.
        3. Wait up to ``grace_seconds`` for all workers to finish naturally.
        4. Hard-cancel any that remain, then await their ``CancelledError``
           teardown so the asyncio event loop doesn't emit ``Task was
           destroyed but it is pending`` warnings at process exit.

        Idempotent — subsequent calls are no-ops.
        """
        if self._shutdown_started:
            return
        self._shutdown_started = True
        logger.info(
            "task engine shutdown initiated",
            extra={"active_tasks": len(self._workers)},
        )

        for task_id in list(self._workers):
            self._registry.request_cancel(task_id)

        pending = list(self._workers.values())
        if pending:
            done, still_pending = await asyncio.wait(
                pending,
                timeout=grace_seconds,
                return_when=asyncio.ALL_COMPLETED,
            )
            for worker in still_pending:
                worker.cancel()
            if still_pending:
                # Drain cancellations so teardown is clean even if workers
                # were blocked on I/O when we pulled the plug.
                await asyncio.gather(*still_pending, return_exceptions=True)
            logger.info(
                "task engine shutdown complete",
                extra={
                    "workers_drained": len(done),
                    "workers_hard_cancelled": len(still_pending),
                },
            )

    # ------------------------------------------------------------------ #
    # Crash recovery
    # ------------------------------------------------------------------ #

    def recover_on_startup(self) -> None:
        """Repair orphaned sub-journals and incomplete merge-backs.

        Three crash windows are handled:

        1. Sub-journal has ``TaskStarted`` but no terminal marker — the
           worker never reached a terminal. Append
           ``TaskFailed { reason: "process_crash_recovery" }``.

        2. Sub-journal has ``TaskCompleted`` but the parent never received
           the corresponding merge-back (crash between ``TaskCompleted``
           write and ``MergeStarted`` write, or after ``MergeStarted`` but
           before ``MergeCommitted``). Idempotently apply whatever merge
           entries are missing and close with ``MergeCommitted``.

        3. Parent has ``MergeStarted`` with no matching ``MergeCommitted``
           and the sub-journal is missing or not ``TaskCompleted`` (highly
           unusual — sub-journal might have been purged). Close the wrapper
           with ``MergeCommitted`` alone, no conversational content.
        """
        with otel_span("agentling.engine.recovery"):
            self._recover_orphaned_subjournals()
            self._recover_incomplete_merges()

    def _recover_orphaned_subjournals(self) -> None:
        for sub_path in self._store.data_dir.glob("*/tasks/*.jsonl"):
            ctx_id = sub_path.parent.parent.name
            task_id = sub_path.stem
            tj = TaskJournal(sub_path)
            entries = tj.read_entries()
            if not entries:
                continue
            if any(e.get("t") == "task_start" for e in entries) and tj.terminal_entry() is None:
                marker = TaskFailed(
                    ctx=ctx_id,
                    task_id=task_id,
                    reason="process_crash_recovery",
                )
                tj.append(marker)
                try:
                    self._store.append(ctx_id, marker)
                except Exception:  # pragma: no cover — parent may be missing
                    _METRICS.crash_recovery_failed.add(1)
                    logger.exception(
                        "recovery: could not mirror marker to parent",
                        extra={"context_id": ctx_id, "task_id": task_id},
                    )
                    continue
                _METRICS.crash_recovery_repaired.add(1)
                logger.info(
                    "recovery: closed orphaned task",
                    extra={"context_id": ctx_id, "task_id": task_id},
                )

    def _recover_incomplete_merges(self) -> None:
        """Walk every context and close out merges that never committed."""
        for ctx_id in self._store.iter_context_ids():
            try:
                parent_entries = self._store.read_entries(ctx_id)
            except Exception:  # pragma: no cover
                _METRICS.crash_recovery_failed.add(1)
                logger.exception(
                    "recovery: failed to read parent journal",
                    extra={"context_id": ctx_id},
                )
                continue

            merge_states: dict[str, str] = {}
            for entry in parent_entries:
                t = entry.get("t")
                if t == "merge_start":
                    merge_states[entry["task_id"]] = "started"
                elif t == "merge_commit":
                    merge_states[entry["task_id"]] = "committed"

            repaired_in_ctx: set[str] = set()
            tasks_dir = self._store.context_dir(ctx_id) / "tasks"
            if tasks_dir.exists():
                for sub_path in tasks_dir.glob("*.jsonl"):
                    task_id = sub_path.stem
                    if merge_states.get(task_id) == "committed":
                        continue
                    tj = TaskJournal(sub_path)
                    terminal = tj.terminal_entry()
                    if terminal is None or terminal["t"] != "task_done":
                        continue
                    self._repair_merge(ctx_id, task_id)
                    repaired_in_ctx.add(task_id)

            for task_id, state in merge_states.items():
                if state != "started":
                    continue
                if task_id in repaired_in_ctx:
                    continue
                sub_path = self._store.task_path(ctx_id, task_id)
                if sub_path.exists():
                    # The sub-journal exists but its terminal wasn't task_done
                    # (task_fail or task_cancel). Close the wrapper so this
                    # entry doesn't resurface on every subsequent startup;
                    # no conversational content leaks since _repair_merge
                    # only appends content on task_done.
                    self._repair_merge(ctx_id, task_id)
                    continue
                # Sub-journal missing entirely — close wrapper only.
                self._store.append(ctx_id, MergeCommitted(ctx=ctx_id, task_id=task_id))
                _METRICS.crash_recovery_repaired.add(1)
                logger.info(
                    "recovery: closed orphaned merge_start with no sub-journal",
                    extra={"context_id": ctx_id, "task_id": task_id},
                )

    def _repair_merge(self, ctx_id: str, task_id: str) -> None:
        """Complete a partial merge-back idempotently.

        If the sub-journal has a ``task_done`` marker, re-apply any missing
        message/compaction entries then append ``MergeCommitted``. Otherwise
        just close the wrapper.
        """
        tj = TaskJournal(self._store.task_path(ctx_id, task_id))
        terminal = tj.terminal_entry() if tj.exists() else None

        parent_entries = self._store.read_entries(ctx_id)
        already_msgs = {
            (e.get("role"), e.get("task_id"))
            for e in parent_entries
            if e.get("t") == "msg"
        }

        repair: list[Any] = []
        if terminal is not None and terminal["t"] == "task_done":
            sub_entries = tj.read_entries()
            start = next(
                (e for e in sub_entries if e.get("t") == "task_start"),
                None,
            )
            user_needed = ("user", task_id) not in already_msgs
            assistant_needed = ("assistant", task_id) not in already_msgs

            if start is not None and user_needed:
                repair.append(MessageEntry(
                    ctx=ctx_id,
                    role="user",
                    content=[{"type": "text", "text": start["message"]}],
                    via=start.get("via", "a2a"),
                    task_id=task_id,
                ))
            # Compaction is only re-ported when the assistant message is ALSO
            # being re-ported. Appending a duplicate compaction after an
            # already-written assistant reply would move the replay cursor
            # forward of that reply and shadow it on future turns.
            if assistant_needed:
                latest_comp = tj.latest_compaction()
                if latest_comp is not None:
                    repair.append(CompactionEntry(
                        ctx=ctx_id,
                        content=latest_comp.get("content", ""),
                    ))
                repair.append(MessageEntry(
                    ctx=ctx_id,
                    role="assistant",
                    content=terminal.get("final_response", []),
                    via=start.get("via", "a2a") if start else "a2a",
                    task_id=task_id,
                ))
        repair.append(MergeCommitted(ctx=ctx_id, task_id=task_id))
        self._store.append_many(ctx_id, repair)
        _METRICS.crash_recovery_repaired.add(1)
        logger.info(
            "recovery: closed merge-back",
            extra={"context_id": ctx_id, "task_id": task_id},
        )
