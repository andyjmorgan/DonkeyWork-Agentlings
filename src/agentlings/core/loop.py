"""Message loop — the single entrance point for all agent interactions.

In v2, the loop is a thin backwards-compatible facade over ``TaskEngine``.
Every call to ``process_message`` spawns a task, blocks until it reaches a
terminal state, and returns the final response as a ``LoopResult``.

Protocol adapters that want the full task surface (poll, cancel, slow-path
yield) should call ``TaskEngine`` directly via the ``engine`` property.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentlings.config import AgentConfig
from agentlings.core.llm import BaseLLMClient
from agentlings.core.memory_store import MemoryFileStore
from agentlings.core.skills import SkillRef
from agentlings.core.store import JournalStore
from agentlings.core.task import (
    TaskEngine,
    TaskState,
    TaskStatus,
)
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


# "Effectively unlimited" await for callers that expect synchronous semantics.
# Real deployments rely on ``TaskEngine.spawn`` directly with the configured
# ``AGENT_TASK_AWAIT_SECONDS`` cap.
_SYNCHRONOUS_AWAIT_SECONDS = 3600.0


class LoopError(RuntimeError):
    """Raised when the synchronous message-loop facade cannot deliver a result.

    Occurs when the underlying task failed or was cancelled. The ``state``
    attribute carries the ``TaskState`` for callers that need to inspect.
    """

    def __init__(self, state: TaskState) -> None:
        super().__init__(
            f"task {state.task_id} ended with status {state.status.value}: {state.error or ''}"
        )
        self.state = state


@dataclass
class LoopResult:
    """Result of processing a message through the loop.

    Attributes:
        context_id: The conversation context identifier.
        content: Anthropic-format content blocks from the final LLM response.
        task_id: The task identifier used to execute this message (v2).
    """

    context_id: str
    content: list[dict[str, Any]]
    task_id: str | None = None


class MessageLoop:
    """Synchronous facade over ``TaskEngine`` for legacy callers.

    Attributes:
        engine: The underlying task engine. Protocol adapters should prefer
            its API (``spawn``, ``poll``, ``cancel``) when they need the
            task-based surface.
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
        self._engine = TaskEngine(
            config=config,
            store=store,
            llm=llm,
            tools=tools,
            memory_store=memory_store,
            skills=skills,
        )

    @property
    def engine(self) -> TaskEngine:
        """The underlying task engine."""
        return self._engine

    async def process_message(
        self,
        text: str,
        context_id: str | None = None,
        stream: bool = False,
        via: str = "a2a",
    ) -> LoopResult:
        """Process a user message and return the agent's response.

        Backward-compatible synchronous API. Internally spawns a task and
        waits for it to reach a terminal state.

        Args:
            text: The user's message text.
            context_id: Existing context ID to continue, or ``None`` for a new one.
            stream: Legacy flag; currently ignored.
            via: Protocol that originated the request (``"a2a"`` or ``"mcp"``).

        Returns:
            The context ID, the final response content blocks, and the task ID.

        Raises:
            LoopError: If the underlying task failed or was cancelled.
        """
        state = await self._engine.spawn(
            message=text,
            context_id=context_id,
            via=via,
            await_seconds=_SYNCHRONOUS_AWAIT_SECONDS,
        )
        if state.status != TaskStatus.COMPLETED:
            raise LoopError(state)
        return LoopResult(
            context_id=state.context_id,
            content=state.content,
            task_id=state.task_id,
        )
