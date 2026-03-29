from __future__ import annotations

import logging

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from agentlings.loop import MessageLoop

logger = logging.getLogger(__name__)


class AgentlingExecutor(AgentExecutor):
    def __init__(self, loop: MessageLoop) -> None:
        self._loop = loop

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_text = context.get_user_input()
        context_id = context.context_id

        logger.debug(
            "execute called: context_id=%s, user_text=%s",
            context_id,
            user_text[:100] if user_text else "",
        )

        try:
            result = await self._loop.process_message(
                text=user_text,
                context_id=context_id,
                via="a2a",
            )
        except Exception:
            logger.exception("error processing A2A message")
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "Internal error processing request.",
                    context_id=context_id,
                )
            )
            await event_queue.close()
            return

        response_text = _extract_text(result.content)
        response_msg = new_agent_text_message(
            response_text, context_id=result.context_id
        )

        logger.debug(
            "returning response: context_id=%s, text=%s",
            result.context_id,
            response_text[:100] if response_text else "",
        )

        await event_queue.enqueue_event(response_msg)
        await event_queue.close()

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        await event_queue.enqueue_event(
            new_agent_text_message(
                "Cancellation is not supported.",
                context_id=context.context_id,
            )
        )
        await event_queue.close()


def _extract_text(content: list[dict]) -> str:
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "\n".join(parts)
