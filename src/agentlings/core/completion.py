"""Reusable LLM completion cycle: call model, execute tools, repeat until terminal response."""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agentlings.core.llm import BaseLLMClient, LLMResponse
from agentlings.core.telemetry import get_meter, otel_span
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class CancellationRequested(Exception):
    """Raised inside ``run_completion`` when a cancel callback returns ``True``.

    Callers are expected to catch this and apply their own cancel terminal
    semantics (e.g. writing a ``TaskCancelled`` marker). The ``partial``
    attribute carries a ``CompletionResult`` with turns completed so far.
    """

    def __init__(self, message: str, partial: "CompletionResult") -> None:
        super().__init__(message)
        self.partial = partial

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
    turn_callback: Callable[[CompletionTurn], Awaitable[None]] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> CompletionResult:
    """Run the LLM in a loop, executing tool calls until a terminal text response.

    This is the core interaction cycle extracted from the message loop so that
    both conversation handling and the sleep cycle can reuse it.

    Args:
        llm: The LLM client to use for completions.
        system: System prompt blocks.
        messages: Conversation messages (mutated in place with tool results).
        tools: Registry of available tools.
        turn_callback: Optional async callback invoked after each completed turn
            (with the turn that just concluded). Used by the task engine to
            journal per-turn progress into the sub-journal.
        should_cancel: Optional synchronous predicate checked at cooperative
            checkpoints — after each LLM call, before each tool batch, and
            after each tool batch. When it returns ``True``,
            ``CancellationRequested`` is raised and all partial progress so
            far is available in ``CompletionResult`` via a raised exception
            attribute.

    Returns:
        The final response content and all intermediate responses.

    Raises:
        CancellationRequested: If ``should_cancel`` returns ``True`` at a
            checkpoint. The exception carries a ``partial`` attribute with a
            ``CompletionResult`` containing turns completed so far.
    """
    tool_schemas = tools.list_schemas()
    turns: list[CompletionTurn] = []
    cycle_start = time.monotonic()
    metrics = _get_metrics()

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            partial = CompletionResult(
                content=turns[-1].response.content if turns else [],
                stop_reason="cancelled",
                turns=turns,
            )
            raise CancellationRequested("cancellation requested", partial)

    with otel_span("agentling.completion") as cycle_span:
        while True:
            turn_number = len(turns) + 1

            with otel_span("agentling.completion.llm_call", {"completion.turn": turn_number}):
                response = await llm.complete(system, messages, tool_schemas)

            _check_cancel()

            has_tool_use = any(
                block.get("type") == "tool_use" for block in response.content
            )

            if not has_tool_use:
                terminal_turn = CompletionTurn(response=response)
                turns.append(terminal_turn)
                if turn_callback is not None:
                    await turn_callback(terminal_turn)
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

            _check_cancel()

            async def _exec_tool(block: dict[str, Any]) -> dict[str, Any]:
                try:
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

                    return {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": result.output,
                        "is_error": result.is_error,
                    }
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("unhandled exception in tool %s", block["name"])
                    metrics["tool_calls"].add(1, {"tool.name": block["name"]})
                    metrics["tool_errors"].add(1, {"tool.name": block["name"]})
                    return {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": f"Tool {block['name']} failed with an internal error",
                        "is_error": True,
                    }

            tool_results = list(await asyncio.gather(*(_exec_tool(b) for b in tool_calls)))

            _check_cancel()

            turn = CompletionTurn(response=response, tool_results=tool_results)
            turns.append(turn)
            if turn_callback is not None:
                await turn_callback(turn)
            messages.append({"role": "user", "content": tool_results})
