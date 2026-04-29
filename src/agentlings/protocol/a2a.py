"""A2A protocol executor bridging incoming requests into the task engine.

SendMessage spawns a task in the shared engine and awaits up to
``AGENT_TASK_AWAIT_SECONDS``. If the task completes in the await window, the
final response is returned as a plain agent ``Message``. Otherwise a native
A2A ``Task`` object (``status.state == working``) is enqueued so the caller
can poll via ``GetTask`` — those GetTask calls are routed back to our engine
by ``EngineTaskStore`` so the SDK's answers always reflect live state.

Clients can opt out of the await window per-request by setting
``configuration.return_immediately = true`` on ``message/send``; in that
case the executor passes ``await_seconds=0`` to the engine and a ``Task``
object is enqueued without blocking.

Cancellation (``CancelTask``) hits the engine's cancel path by task id.
"""

from __future__ import annotations

import json
import logging

from a2a.helpers.proto_helpers import new_text_message
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.types import Message, Role

from agentlings.config import AgentConfig
from agentlings.core.loop import MessageLoop
from agentlings.core.task import (
    ContextBusyError,
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


class AgentlingExecutor(AgentExecutor):
    """Executes A2A requests by forwarding user input through the shared task engine."""

    def __init__(self, loop: MessageLoop, config: AgentConfig) -> None:
        self._loop = loop
        self._engine = loop.engine
        self._await_seconds = float(config.agent_task_await_seconds)

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Process an incoming A2A request and enqueue the agent's response.

        Enqueues either a ``Message`` (fast path, completed) or a native
        ``Task`` object (slow path, still working) depending on whether the
        task finished within the configured await window.
        """
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
