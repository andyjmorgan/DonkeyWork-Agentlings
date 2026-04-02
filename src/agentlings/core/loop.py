"""Message loop — the single entrance point for all agent interactions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agentlings.config import AgentConfig
from agentlings.core.completion import CompletionResult, run_completion
from agentlings.core.llm import BaseLLMClient
from agentlings.core.memory_store import MemoryFileStore
from agentlings.core.models import CompactionEntry, MessageEntry
from agentlings.core.prompt import build_system_prompt
from agentlings.core.store import JournalStore
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Result of processing a message through the loop.

    Attributes:
        context_id: The conversation context identifier.
        content: Anthropic-format content blocks from the final LLM response.
    """

    context_id: str
    content: list[dict[str, Any]]


class MessageLoop:
    """Orchestrates the append-replay-complete-execute cycle.

    Both A2A and MCP protocol handlers feed into this single entrance.
    The loop appends user input to the JSONL journal, replays the conversation,
    calls the LLM via the shared completion cycle, and journals the results.
    """

    def __init__(
        self,
        config: AgentConfig,
        store: JournalStore,
        llm: BaseLLMClient,
        tools: ToolRegistry,
        memory_store: MemoryFileStore | None = None,
    ) -> None:
        self._config = config
        self._store = store
        self._llm = llm
        self._tools = tools
        self._memory_store = memory_store

    async def process_message(
        self,
        text: str,
        context_id: str | None = None,
        stream: bool = False,
        via: str = "a2a",
    ) -> LoopResult:
        """Process a user message and return the agent's response.

        Creates a new context if none is provided, or continues an existing one.
        Delegates LLM interaction to ``run_completion`` and journals every turn.

        Args:
            text: The user's message text.
            context_id: Existing context ID to continue, or ``None`` for a new conversation.
            stream: Whether to use streaming (not yet implemented).
            via: Protocol that originated the request (``"a2a"`` or ``"mcp"``).

        Returns:
            The context ID and the final response content blocks.
        """
        if context_id is None:
            context_id = str(uuid4())
            self._store.create(context_id)
            logger.info("new context: %s", context_id)
        elif not self._store.exists(context_id):
            self._store.create(context_id)
            logger.info("new context (external id): %s", context_id)

        self._store.append(
            context_id,
            MessageEntry(
                ctx=context_id,
                role="user",
                content=[{"type": "text", "text": text}],
                via=via,
            ),
        )

        memory = self._memory_store.load() if self._memory_store else None
        memory_config = self._config.definition.memory
        system = build_system_prompt(
            config=self._config,
            memory=memory,
            data_dir=self._config.agent_data_dir,
            injection_prompt=memory_config.injection_prompt if memory_config else None,
            token_budget=memory_config.token_budget if memory_config else 2000,
        )

        messages = self._store.replay(context_id)
        result = await run_completion(
            llm=self._llm,
            system=system,
            messages=messages,
            tools=self._tools,
        )

        self._journal_completion(context_id, result, via)

        return LoopResult(context_id=context_id, content=result.content)

    def _journal_completion(
        self,
        context_id: str,
        result: CompletionResult,
        via: str,
    ) -> None:
        """Persist all turns from a completion cycle into the JSONL journal."""
        for turn in result.turns:
            self._store.append(
                context_id,
                MessageEntry(
                    ctx=context_id,
                    role="assistant",
                    content=turn.response.content,
                    via=via,
                ),
            )

            if turn.tool_results:
                self._store.append(
                    context_id,
                    MessageEntry(
                        ctx=context_id,
                        role="user",
                        content=turn.tool_results,
                        via=via,
                    ),
                )

        for block in result.content:
            if block.get("type") == "compaction":
                self._store.append(
                    context_id,
                    CompactionEntry(
                        ctx=context_id,
                        content=block.get("content", ""),
                    ),
                )
                logger.info("compaction marker stored for context %s", context_id)
