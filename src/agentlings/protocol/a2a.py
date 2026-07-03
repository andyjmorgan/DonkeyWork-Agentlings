"""A2A protocol executor bridging incoming requests into the task engine.

SendMessage spawns a task in the shared engine and awaits up to
``AGENT_TASK_AWAIT_SECONDS``. If the task completes in the await window, the
final response is returned as a plain agent ``Message``. Otherwise a native
A2A ``Task`` object (``status.state == working``) is enqueued so the caller
can poll via ``GetTask`` — those GetTask calls are routed back to our engine
by ``EngineTaskStore`` so the SDK's answers always reflect live state.

When A2A streaming is enabled, ``message/stream`` skips the await window,
enqueues the working Task immediately, then observes the durable engine task
until terminal state and enqueues the final Task. Client disconnects only drop
the stream observer; the underlying task keeps running.

Clients can opt out of the await window per-request by setting
``configuration.return_immediately = true`` on ``message/send``; in that
case the executor passes ``await_seconds=0`` to the engine and a ``Task``
object is enqueued without blocking.

Cancellation (``CancelTask``) hits the engine's cancel path by task id.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from a2a.helpers.proto_helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Role,
    TaskArtifactUpdateEvent,
    TaskState as A2ATaskState,
    TaskStatus as A2ATaskStatus,
    TaskStatusUpdateEvent,
)
from google.protobuf.struct_pb2 import Struct

from agentlings.config import AgentConfig
from agentlings.core.loop import MessageLoop
from agentlings.core.task import (
    ContextBusyError,
    TERMINAL_STATUSES,
    TaskNotFoundError,
    TaskState,
    TaskStatus,
)
from agentlings.core.telemetry import otel_span
from agentlings.protocol.a2a_task_store import task_state_to_a2a_task

logger = logging.getLogger(__name__)


def _agent_text_message(
    text: str,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
) -> Message:
    """Build an agent-role ``Message`` proto carrying a single text part.

    Replaces the removed ``a2a.utils.new_agent_text_message`` helper. The
    proto ``Message`` requires a ``message_id`` and has no field presence on
    string fields, so empty-string defaults are used when ids are unset.
    """
    return new_text_message(
        text,
        context_id=context_id or "",
        task_id=task_id or "",
        role=Role.ROLE_AGENT,
    )


def _progress_status_update(progress: Any) -> TaskStatusUpdateEvent:
    """Render internal tool progress as an A2A task status update."""
    metadata = Struct()
    if progress.data:
        metadata.update({
            "agentling": {
                "extension": "https://donkeywork.dev/a2a/extensions/tool-progress/v1",
                **progress.data,
            }
        })
    return TaskStatusUpdateEvent(
        task_id=progress.task_id,
        context_id=progress.context_id,
        status=A2ATaskStatus(
            state=A2ATaskState.TASK_STATE_WORKING,
            message=_agent_text_message(
                progress.text or "Task progress updated",
                context_id=progress.context_id,
                task_id=progress.task_id,
            ),
        ),
        metadata=metadata,
    )


def _terminal_stream_events(
    state: TaskState,
) -> list[TaskArtifactUpdateEvent | TaskStatusUpdateEvent]:
    """Render a terminal engine state as A2A streaming task update events."""
    task = task_state_to_a2a_task(state)
    events: list[TaskArtifactUpdateEvent | TaskStatusUpdateEvent] = []
    for artifact in task.artifacts:
        events.append(
            TaskArtifactUpdateEvent(
                task_id=task.id,
                context_id=task.context_id,
                artifact=artifact,
                last_chunk=True,
            )
        )
    events.append(
        TaskStatusUpdateEvent(
            task_id=task.id,
            context_id=task.context_id,
            status=task.status,
        )
    )
    return events


class AgentlingExecutor(AgentExecutor):
    """Executes A2A requests by forwarding user input through the shared task engine."""

    def __init__(self, loop: MessageLoop, config: AgentConfig) -> None:
        self._loop = loop
        self._engine = loop.engine
        self._await_seconds = float(config.agent_task_await_seconds)
        self._streaming_enabled = config.a2a_config.streaming

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Process an incoming A2A request and enqueue the agent's response.

        Enqueues either a ``Message`` (fast path, completed) or a native
        ``Task`` object (slow path, still working) depending on whether the
        task finished within the configured await window.
        """
        if self._is_streaming_request(context) and self._streaming_enabled:
            await self._execute_streaming(context, event_queue)
            return

        user_text = context.get_user_input()
        context_id = context.context_id
        # The A2A SDK generates a task_id for every inbound SendMessage. We
        # must use that same id inside the engine so GetTask lookups (routed
        # through EngineTaskStore) and the Task object we enqueue all agree.
        sdk_task_id = context.task_id

        # A2A 1.0: clients can opt out of blocking by setting
        # ``configuration.return_immediately``. When set, skip the await
        # window so a Task handle is enqueued immediately.
        return_immediately = bool(
            getattr(context.configuration, "return_immediately", False)
        )
        await_seconds = 0.0 if return_immediately else self._await_seconds

        logger.debug(
            "a2a execute: context_id=%s task_id=%s return_immediately=%s text=%r",
            context_id,
            sdk_task_id,
            return_immediately,
            (user_text or "")[:100],
        )

        with otel_span("agentling.a2a.execute", {
            "agent.via": "a2a",
            "task.context_id": context_id or "",
            "task.id": sdk_task_id or "",
            "a2a.return_immediately": return_immediately,
            "a2a.message_chars": len(user_text or ""),
        }) as span:
            try:
                state = await self._engine.spawn(
                    message=user_text,
                    context_id=context_id,
                    via="a2a",
                    await_seconds=await_seconds,
                    task_id=sdk_task_id,
                )
            except ContextBusyError as e:
                # Surface as a plain agent message — there is no live Task on this
                # context that the caller can latch onto (it's someone else's).
                span.set_attribute("a2a.outcome", "context_busy")
                await event_queue.enqueue_event(
                    _agent_text_message(
                        _format_busy(e),
                        context_id=context_id,
                    )
                )
                await event_queue.close()
                return
            except Exception:  # noqa: BLE001
                span.set_attribute("a2a.outcome", "exception")
                logger.exception("error processing A2A message")
                await event_queue.enqueue_event(
                    _agent_text_message(
                        "Internal error processing request.",
                        context_id=context_id,
                    )
                )
                await event_queue.close()
                return

            span.set_attribute("task.status", state.status.value)

            if state.status == TaskStatus.COMPLETED:
                span.set_attribute("a2a.outcome", "completed")
                # Fast path — return the final response text as a Message event.
                response_text = _extract_text(state.content)
                await event_queue.enqueue_event(
                    _agent_text_message(
                        response_text,
                        context_id=state.context_id,
                        task_id=state.task_id,
                    )
                )
            else:
                span.set_attribute("a2a.outcome", "task_handle")
                # Slow path or terminal-with-error — enqueue a native A2A Task so
                # the client's GetTask/CancelTask flows reach our engine.
                a2a_task = task_state_to_a2a_task(state)
                await event_queue.enqueue_event(a2a_task)

            logger.debug(
                "a2a response: context_id=%s status=%s task_id=%s",
                state.context_id, state.status.value, state.task_id,
            )
            await event_queue.close()

    def _is_streaming_request(self, context: RequestContext) -> bool:
        """Whether this executor call came from A2A ``message/stream``."""
        return (
            context.call_context.state.get("method") == "SendStreamingMessage"
        )

    async def _execute_streaming(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Handle A2A ``message/stream`` as a live observer of an engine task.

        The stream path deliberately uses ``await_seconds=0`` and then waits on
        internal progress notifications. That keeps the durable task as the
        source of truth and prevents client disconnect from owning task
        lifecycle; cancellation remains explicit via ``tasks/cancel``.
        """
        user_text = context.get_user_input()
        context_id = context.context_id
        sdk_task_id = context.task_id

        logger.debug(
            "a2a streaming execute: context_id=%s task_id=%s text=%r",
            context_id,
            sdk_task_id,
            (user_text or "")[:100],
        )

        subscription = (
            self._engine.subscribe(sdk_task_id) if sdk_task_id else None
        )
        with otel_span("agentling.a2a.execute", {
            "agent.via": "a2a",
            "task.context_id": context_id or "",
            "task.id": sdk_task_id or "",
            "a2a.streaming": True,
            "a2a.message_chars": len(user_text or ""),
        }) as span:
            try:
                state = await self._engine.spawn(
                    message=user_text,
                    context_id=context_id,
                    via="a2a",
                    await_seconds=0.0,
                    task_id=sdk_task_id,
                )
            except ContextBusyError as e:
                span.set_attribute("a2a.outcome", "context_busy")
                if subscription is not None:
                    subscription.close()
                await event_queue.enqueue_event(
                    _agent_text_message(
                        _format_busy(e),
                        context_id=context_id,
                    )
                )
                await event_queue.close()
                return
            except Exception:  # noqa: BLE001
                span.set_attribute("a2a.outcome", "exception")
                logger.exception("error processing streaming A2A message")
                if subscription is not None:
                    subscription.close()
                await event_queue.enqueue_event(
                    _agent_text_message(
                        "Internal error processing request.",
                        context_id=context_id,
                    )
                )
                await event_queue.close()
                return

            try:
                span.set_attribute("task.status", state.status.value)
                if subscription is None:
                    subscription = self._engine.subscribe(state.task_id)
                await event_queue.enqueue_event(task_state_to_a2a_task(state))

                if state.status in TERMINAL_STATUSES:
                    span.set_attribute("a2a.outcome", "completed")
                    return

                if subscription is not None:
                    async for progress in subscription:
                        if progress.kind.startswith("tool_"):
                            await event_queue.enqueue_event(
                                _progress_status_update(progress)
                            )
                            continue
                        if progress.kind != "task_terminal":
                            continue
                        final_state = await self._engine.poll(
                            task_id=state.task_id,
                            context_id=state.context_id,
                            wait_seconds=0,
                        )
                        span.set_attribute("task.status", final_state.status.value)
                        span.set_attribute("a2a.outcome", final_state.status.value)
                        for event in _terminal_stream_events(final_state):
                            await event_queue.enqueue_event(event)
                        return
            finally:
                if subscription is not None:
                    subscription.close()
                await event_queue.close()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Handle A2A ``CancelTask`` by routing to the engine's cancel path.

        ``RequestContext.task_id`` is populated from the inbound request,
        which (for our clients) is the engine's task_id — the Task object
        enqueued on the slow path carries that id directly.
        """
        task_id = context.task_id
        with otel_span("agentling.a2a.cancel", {
            "agent.via": "a2a",
            "task.context_id": context.context_id or "",
            "task.id": task_id or "",
        }) as span:
            if not task_id:
                span.set_attribute("a2a.outcome", "missing_task_id")
                await event_queue.enqueue_event(
                    _agent_text_message(
                        "CancelTask requires a task_id.",
                        context_id=context.context_id,
                    )
                )
                await event_queue.close()
                return

            try:
                state = await self._engine.cancel(task_id=task_id)
            except TaskNotFoundError:
                span.set_attribute("a2a.outcome", "not_found")
                await event_queue.enqueue_event(
                    _agent_text_message(
                        f"Task {task_id} not found.",
                        context_id=context.context_id,
                    )
                )
                await event_queue.close()
                return
            except Exception:  # noqa: BLE001
                span.set_attribute("a2a.outcome", "exception")
                logger.exception("cancel failed for task %s", task_id)
                await event_queue.enqueue_event(
                    _agent_text_message(
                        "Internal error during cancel.",
                        context_id=context.context_id,
                    )
                )
                await event_queue.close()
                return

            span.set_attribute("task.status", state.status.value)
            span.set_attribute("a2a.outcome", "cancelled")
            # Enqueue the updated Task so the SDK relays the cancelled state.
            await event_queue.enqueue_event(task_state_to_a2a_task(state))
            await event_queue.close()


def _format_busy(e: ContextBusyError) -> str:
    envelope = {
        "status": "busy",
        "error": "context_busy",
        "contextId": e.context_id,
        "activeTaskId": e.active_task_id,
        "message": (
            f"Context {e.context_id} is busy with task {e.active_task_id}. "
            "Retry shortly."
        ),
    }
    return json.dumps(envelope)


def _extract_text(content: list[dict]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def _format_state(state: TaskState, await_seconds: float) -> str:
    """Legacy helper kept for backwards compatibility in tests."""
    if state.status == TaskStatus.COMPLETED:
        return _extract_text(state.content)
    envelope = {
        "status": state.status.value,
        "taskId": state.task_id,
        "contextId": state.context_id,
    }
    if state.error:
        envelope["error"] = state.error
    return json.dumps(envelope)
