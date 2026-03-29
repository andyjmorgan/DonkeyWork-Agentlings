"""Core message loop that drives LLM completion and tool execution cycles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agentlings.config import AgentConfig
from agentlings.llm import BaseLLMClient
from agentlings.models import CompactionEntry, MessageEntry
from agentlings.prompt import build_system_prompt
from agentlings.store import ContextNotFoundError, JournalStore
from agentlings.tools import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class LoopResult:
    """Result of processing a message through the loop.

    Attributes:
        context_id: The conversation context identifier.
        content: Response content blocks from the LLM.
    """
    context_id: str
    content: list[dict[str, Any]]


class MessageLoop:
    """Orchestrates the LLM completion and tool-use cycle for a conversation.

    Appends all messages to a JSONL journal, replays context for each LLM call,
    and loops on tool-use responses until the model produces a terminal text reply.
    """

    def __init__(
        self,
        config: AgentConfig,
        store: JournalStore,
        llm: BaseLLMClient,
        tools: ToolRegistry,
    ) -> None:
        self._config = config
        self._store = store
        self._llm = llm
        self._tools = tools
        self._system = build_system_prompt(config)

    async def process_message(
        self,
        text: str,
        context_id: str | None = None,
        stream: bool = False,
        via: str = "a2a",
    ) -> LoopResult:
        """Process a user message and return the agent's final response.

        Creates a new context if none is provided. Runs the LLM in a loop,
        executing any requested tools, until the model returns a text response.

        Args:
            text: The user's message text.
            context_id: Existing conversation context ID, or None to create one.
            stream: Whether to use streaming (currently unused).
            via: Protocol the message arrived through ("a2a" or "mcp").

        Returns:
            The context ID and final response content blocks.
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

        messages = self._store.replay(context_id)
        tool_schemas = self._tools.list_schemas()

        while True:
            response = await self._llm.complete(self._system, messages, tool_schemas)

            has_tool_use = any(
                block.get("type") == "tool_use" for block in response.content
            )

            if has_tool_use:
                self._store.append(
                    context_id,
                    MessageEntry(
                        ctx=context_id,
                        role="assistant",
                        content=response.content,
                        via=via,
                    ),
                )

                tool_results = []
                for block in response.content:
                    if block.get("type") == "tool_use":
                        result = await self._tools.execute(
                            block["name"], block.get("input", {})
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block["id"],
                            "content": result.output,
                            "is_error": result.is_error,
                        })

                self._store.append(
                    context_id,
                    MessageEntry(
                        ctx=context_id,
                        role="user",
                        content=tool_results,
                        via=via,
                    ),
                )

                messages = self._store.replay(context_id)
                continue

            for block in response.content:
                if block.get("type") == "compaction":
                    self._store.append(
                        context_id,
                        CompactionEntry(
                            ctx=context_id,
                            content=block.get("content", ""),
                        ),
                    )
                    logger.info("compaction marker stored for context %s", context_id)

            self._store.append(
                context_id,
                MessageEntry(
                    ctx=context_id,
                    role="assistant",
                    content=response.content,
                    via=via,
                ),
            )

            return LoopResult(context_id=context_id, content=response.content)
