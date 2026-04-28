"""Bridge from the A2A SDK's ``TaskStore`` interface to our ``TaskEngine``.

The A2A SDK's ``DefaultRequestHandler`` consults an injected ``TaskStore``
when a client calls ``GetTask``. Rather than duplicate state in an
``InMemoryTaskStore``, this bridge delegates every lookup to the engine so
the SDK's answers always reflect live state.

- ``get(task_id)`` → polls the engine, translates ``TaskState`` → ``a2a.Task``.
- ``save(task)`` is a no-op; the engine is authoritative.
- ``delete(task_id)`` is a no-op; retention follows the engine's lifecycle.
"""

from __future__ import annotations

import logging
from typing import Any

from a2a.server.context import ServerCallContext
from a2a.server.tasks.task_store import TaskStore
from a2a.types import (
    Artifact,
    ListTasksRequest,
    ListTasksResponse,
    Message,
    Part,
    Role,
    Task,
    TaskState as A2ATaskState,
    TaskStatus as A2ATaskStatus,
)

from agentlings.core.task import (
    TaskEngine,
    TaskNotFoundError,
    TaskState,
    TaskStatus,
)

logger = logging.getLogger(__name__)


_STATE_MAP: dict[TaskStatus, int] = {
    TaskStatus.WORKING: A2ATaskState.TASK_STATE_WORKING,
    TaskStatus.CANCELLING: A2ATaskState.TASK_STATE_WORKING,
    TaskStatus.COMPLETED: A2ATaskState.TASK_STATE_COMPLETED,
    TaskStatus.FAILED: A2ATaskState.TASK_STATE_FAILED,
    TaskStatus.CANCELLED: A2ATaskState.TASK_STATE_CANCELED,
}


def task_state_to_a2a_task(state: TaskState) -> Task:
    """Render an engine ``TaskState`` as an A2A ``Task`` for the wire.

    Completed tasks carry their final assistant message in both ``artifacts``
    (per the A2A spec, which says results SHOULD be returned via Artifacts)
    and ``history`` (kept for backward compatibility with existing clients).
    """
    history: list[Message] = []
    artifacts: list[Artifact] = []
    if state.status == TaskStatus.COMPLETED and state.content:
        text = _extract_text(state.content)
        if text:
            history.append(
                Message(
                    role=Role.ROLE_AGENT,
                    parts=[Part(text=text)],
                    context_id=state.context_id,
                    task_id=state.task_id,
                    message_id=f"{state.task_id}-final",
                )
            )
            artifacts.append(
                Artifact(
                    artifact_id=f"{state.task_id}-result",
                    parts=[Part(text=text)],
                )
            )

    status_kwargs: dict[str, Any] = {"state": _STATE_MAP[state.status]}
    if state.error and state.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
        status_kwargs["message"] = Message(
            role=Role.ROLE_AGENT,
            parts=[Part(text=state.error)],
            context_id=state.context_id,
            task_id=state.task_id,
            message_id=f"{state.task_id}-status",
        )

    return Task(
        id=state.task_id,
        context_id=state.context_id,
        status=A2ATaskStatus(**status_kwargs),
        artifacts=artifacts,
        history=history,
    )


def _extract_text(content: list[dict[str, Any]]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


class EngineTaskStore(TaskStore):
    """A2A SDK ``TaskStore`` backed by our ``TaskEngine``.

    GetTask requests reaching the SDK's ``DefaultRequestHandler`` are routed
    through this store. The handler calls ``get`` and returns the resulting
    ``Task`` to the client.
    """

    def __init__(self, engine: TaskEngine) -> None:
        self._engine = engine

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        """Translate ``engine.poll(task_id)`` into an A2A ``Task``.

        Returns ``None`` if the engine has no record of the task (so the SDK
        produces its own ``task_not_found`` error).
        """
        try:
            state = await self._engine.poll(task_id=task_id, wait_seconds=0)
        except TaskNotFoundError:
            return None
        except Exception:  # noqa: BLE001
            logger.exception("EngineTaskStore.get failed for task %s", task_id)
            return None
        return task_state_to_a2a_task(state)

    async def save(
        self, task: Task, context: ServerCallContext | None = None
    ) -> None:
        """No-op. The engine is the source of truth; SDK saves are advisory."""
        logger.debug("EngineTaskStore.save ignored for task %s", task.id)

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        """No-op. Retention follows the engine's lifecycle, not SDK calls."""
        logger.debug("EngineTaskStore.delete ignored for task %s", task_id)

    async def list(
        self,
        params: ListTasksRequest,
        context: ServerCallContext | None = None,
    ) -> ListTasksResponse:
        """Return an empty listing — the engine doesn't expose a task index.

        The SDK's ``ListTasks`` RPC is optional; agentling doesn't advertise
        the capability, so callers shouldn't invoke this. We return an empty
        response rather than raise so the handler stays well-behaved if a
        request slips through.
        """
        logger.debug("EngineTaskStore.list returning empty response")
        return ListTasksResponse()
