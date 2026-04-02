"""Reusable LLM completion cycle: call model, execute tools, repeat until terminal response."""

from __future__ import annotations

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agentlings.core.llm import BaseLLMClient, LLMResponse
from agentlings.core.telemetry import get_meter, otel_span
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def _get_metrics() -> dict[str, Any]:
    m = get_meter()
    return {
        "duration": m.create_histogram(
            "agentling.completion.duration_seconds",
            description="End-to-end completion cycle duration",
        ),
        "turns": m.create_histogram(
            "agentling.completion.turns",
            description="Number of LLM turns per completion cycle",
        ),
        "tool_calls": m.create_counter(
            "agentling.tool.calls",
            description="Total tool invocations",
        ),
        "tool_errors": m.create_counter(
            "agentling.tool.errors",
            description="Tool execution errors",
        ),
    }


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
    cycle_start = time.monotonic()
    metrics = _get_metrics()

    with otel_span("agentling.completion") as cycle_span:
        while True:
            turn_number = len(turns) + 1

            with otel_span("agentling.completion.llm_call", {"completion.turn": turn_number}):
                response = await llm.complete(system, messages, tool_schemas)

            has_tool_use = any(
                block.get("type") == "tool_use" for block in response.content
            )

            if not has_tool_use:
                turns.append(CompletionTurn(response=response))
                elapsed = time.monotonic() - cycle_start
                logger.info(
                    "completion finished: %d turn(s) in %.1fs, stop_reason=%s",
                    len(turns), elapsed, response.stop_reason,
                )
                cycle_span.set_attribute("completion.turns", len(turns))
                cycle_span.set_attribute("completion.stop_reason", response.stop_reason or "unknown")
                cycle_span.set_attribute("completion.duration_s", round(elapsed, 2))
                metrics["duration"].record(elapsed)
                metrics["turns"].record(len(turns))
                return CompletionResult(
                    content=response.content,
                    stop_reason=response.stop_reason,
                    turns=turns,
                )

            messages.append({"role": "assistant", "content": response.content})

            tool_calls = [b for b in response.content if b.get("type") == "tool_use"]
            logger.info(
                "turn %d: executing %d tool(s): %s",
                turn_number, len(tool_calls),
                ", ".join(b["name"] for b in tool_calls),
            )

            tool_results = []
            for block in tool_calls:
                with otel_span("agentling.completion.tool_exec", {
                    "tool.name": block["name"],
                    "completion.turn": turn_number,
                }) as tool_span:
                    result = await tools.execute(block["name"], block.get("input", {}))
                    tool_span.set_attribute("tool.is_error", result.is_error)

                tool_attrs = {"tool.name": block["name"]}
                metrics["tool_calls"].add(1, tool_attrs)
                if result.is_error:
                    metrics["tool_errors"].add(1, tool_attrs)
                    logger.warning("tool %s returned error: %.200s", block["name"], result.output)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": result.output,
                    "is_error": result.is_error,
                })

            turns.append(CompletionTurn(response=response, tool_results=tool_results))
            messages.append({"role": "user", "content": tool_results})
