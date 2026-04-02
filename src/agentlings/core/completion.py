"""Reusable LLM completion cycle: call model, execute tools, repeat until terminal response."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agentlings.core.llm import BaseLLMClient, LLMResponse
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class CompletionTurn:
    """A single assistant response and its corresponding tool results (if any).

    Attributes:
        response: The LLM response for this turn.
        tool_results: Tool result content blocks returned to the LLM, empty for terminal turns.
    """

    response: LLMResponse
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CompletionResult:
    """Result of running the LLM completion cycle.

    Attributes:
        content: Anthropic-format content blocks from the final LLM response.
        stop_reason: Why the model stopped generating.
        turns: Every turn in the cycle, each pairing an LLM response with its tool results.
    """

    content: list[dict[str, Any]]
    stop_reason: str | None = None
    turns: list[CompletionTurn] = field(default_factory=list)


async def run_completion(
    llm: BaseLLMClient,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
) -> CompletionResult:
    """Run the LLM in a loop, executing tool calls until a terminal text response.

    This is the core interaction cycle extracted from the message loop so that
    both conversation handling and the sleep cycle can reuse it.

    Args:
        llm: The LLM client to use for completions.
        system: System prompt blocks.
        messages: Conversation messages (mutated in place with tool results).
        tools: Registry of available tools.

    Returns:
        The final response content and all intermediate responses.
    """
    tool_schemas = tools.list_schemas()
    turns: list[CompletionTurn] = []

    while True:
        response = await llm.complete(system, messages, tool_schemas)

        has_tool_use = any(
            block.get("type") == "tool_use" for block in response.content
        )

        if not has_tool_use:
            turns.append(CompletionTurn(response=response))
            return CompletionResult(
                content=response.content,
                stop_reason=response.stop_reason,
                turns=turns,
            )

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.get("type") == "tool_use":
                result = await tools.execute(block["name"], block.get("input", {}))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result.output,
                    "is_error": result.is_error,
                })

        turns.append(CompletionTurn(response=response, tool_results=tool_results))
        messages.append({"role": "user", "content": tool_results})
