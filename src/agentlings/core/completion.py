"""Reusable LLM completion cycle: call model, execute tools, repeat until terminal response."""

from __future__ import annotations

import asyncio
import copy
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from agentlings.core.llm import BaseLLMClient, LLMResponse
from agentlings.core.telemetry import get_meter, otel_span, record_tool_duration
from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Default cap on a single tool result's character length before it is truncated
# on its way back to the model. A single runaway command (e.g. ``grep -r`` over
# ``$HOME`` sweeping up minified assets) can otherwise produce a multi-MB result
# that pushes the whole request body past the API's hard size limit, failing the
# turn with a 413 instead of letting the model recover. Overridable per-agent via
# ``AgentDefinition.max_tool_result_chars``; ``<= 0`` disables truncation.
DEFAULT_MAX_TOOL_RESULT_CHARS = 32_000

TOOL_ACTION_SUMMARY_FIELD = "agentling_action_summary"
TOOL_ACTION_SUMMARY_DESCRIPTION = (
    "A 5-15 word human-readable description of this action."
)


def _truncate_tool_output(output: str, limit: int) -> str:
    """Cap an oversized tool result and tell the model it was truncated.

    Keeps the first ``limit`` characters and appends a marker stating the
    original size and how much was dropped, so the model knows the output is
    incomplete and can re-run with a narrower command rather than silently
    losing data. ``limit <= 0`` disables truncation and returns ``output``
    unchanged.
    """
    if limit <= 0 or len(output) <= limit:
        return output
    dropped = len(output) - limit
    notice = (
        f"\n\n[agentling: tool result truncated — it was {len(output):,} chars, "
        f"over the {limit:,}-char limit, so {dropped:,} chars were dropped. "
        f"The text above is the first {limit:,} chars. Re-run with a more "
        f"targeted command (narrower path, a grep filter, head -c, etc.) to see "
        f"the rest.]"
    )
    return output[:limit] + notice


def _tool_schemas_with_action_summaries(
    schemas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return tool schemas requiring Agentlings' progress-summary field.

    The underlying tool implementations never see this field; ``run_completion``
    strips it before registry execution. The schema wrapper is opt-in so
    existing agents keep their current tool contract unless configured
    otherwise.
    """
    wrapped = copy.deepcopy(schemas)
    for schema in wrapped:
        input_schema = schema.setdefault("input_schema", {})
        input_schema.setdefault("type", "object")
        properties = input_schema.setdefault("properties", {})
        properties[TOOL_ACTION_SUMMARY_FIELD] = {
            "type": "string",
            "description": TOOL_ACTION_SUMMARY_DESCRIPTION,
            "minLength": 12,
            "maxLength": 120,
        }
        required = input_schema.setdefault("required", [])
        if TOOL_ACTION_SUMMARY_FIELD not in required:
            required.append(TOOL_ACTION_SUMMARY_FIELD)
    return wrapped


def _summary_from_tool_input(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Extract a model-authored action summary with a conservative fallback."""
    raw = tool_input.get(TOOL_ACTION_SUMMARY_FIELD)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return f"Running {tool_name}"


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
class ToolExecutionEvent:
    """Progress event emitted around an individual tool execution."""

    phase: str
    tool_name: str
    tool_use_id: str
    summary: str
    turn_number: int
    is_error: bool = False


@dataclass
class CompletionResult:
    """Result of running the LLM completion cycle.

    Attributes:
        content: Anthropic-format content blocks from the final LLM response.
        stop_reason: Why the model stopped generating.
        turns: Every turn in the cycle, each pairing an LLM response with its tool results.
        token_usage: Sum of token usage across every turn in the cycle.
            Keys: ``input``, ``output``, ``cache_creation``, ``cache_read``.
            Callers (e.g. ``TaskWorker``) stamp these onto parent spans so
            a single trace surfaces the full token bill for a task.
    """

    content: list[dict[str, Any]]
    stop_reason: str | None = None
    turns: list[CompletionTurn] = field(default_factory=list)
    token_usage: dict[str, int] = field(default_factory=lambda: {
        "input": 0,
        "output": 0,
        "cache_creation": 0,
        "cache_read": 0,
    })


async def run_completion(
    llm: BaseLLMClient,
    system: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    tools: ToolRegistry,
    turn_callback: Callable[[CompletionTurn], Awaitable[None]] | None = None,
    tool_event_callback: Callable[[ToolExecutionEvent], Awaitable[None]] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    context_id: str | None = None,
    task_id: str | None = None,
    max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
    tool_progress_summaries: bool = False,
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
        tool_event_callback: Optional async callback invoked before and after
            each individual tool execution. Used by streaming protocol
            adapters for live progress updates.
        should_cancel: Optional synchronous predicate checked at cooperative
            checkpoints — after each LLM call, before each tool batch, and
            after each tool batch. When it returns ``True``,
            ``CancellationRequested`` is raised and all partial progress so
            far is available in ``CompletionResult`` via a raised exception
            attribute.
        context_id: The conversation this cycle belongs to. Forwarded to every
            ``llm.complete`` call so each request carries the session header.
        task_id: The task execution this cycle belongs to. Forwarded to every
            ``llm.complete`` call so each request carries the task header.
        max_tool_result_chars: Maximum character length of a single tool
            result before it is truncated (with a marker) on its way back to
            the model. Guards against a runaway tool blowing the request body
            past the API's hard size limit. ``<= 0`` disables truncation.
        tool_progress_summaries: When ``True``, require an internal
            ``agentling_action_summary`` field on every model-facing tool
            schema and strip it before executing the real tool.

    Returns:
        The final response content and all intermediate responses.

    Raises:
        CancellationRequested: If ``should_cancel`` returns ``True`` at a
            checkpoint. The exception carries a ``partial`` attribute with a
            ``CompletionResult`` containing turns completed so far.
    """
    tool_schemas = tools.list_schemas()
    if tool_progress_summaries:
        tool_schemas = _tool_schemas_with_action_summaries(tool_schemas)
    turns: list[CompletionTurn] = []
    cycle_start = time.monotonic()
    metrics = _get_metrics()
    token_totals: dict[str, int] = {
        "input": 0,
        "output": 0,
        "cache_creation": 0,
        "cache_read": 0,
    }

    def _accumulate_usage(response: LLMResponse) -> None:
        """Fold a single response's token usage into the cycle running totals."""
        usage = response.usage or {}
        token_totals["input"] += int(usage.get("input_tokens", 0) or 0)
        token_totals["output"] += int(usage.get("output_tokens", 0) or 0)
        token_totals["cache_creation"] += int(usage.get("cache_creation_input_tokens", 0) or 0)
        token_totals["cache_read"] += int(usage.get("cache_read_input_tokens", 0) or 0)

    def _stamp_tokens(span: Any) -> None:
        """Set the running token totals as span attributes."""
        span.set_attribute("completion.input_tokens", token_totals["input"])
        span.set_attribute("completion.output_tokens", token_totals["output"])
        span.set_attribute("completion.cache_creation_tokens", token_totals["cache_creation"])
        span.set_attribute("completion.cache_read_tokens", token_totals["cache_read"])

    def _check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            partial = CompletionResult(
                content=turns[-1].response.content if turns else [],
                stop_reason="cancelled",
                turns=turns,
                token_usage=dict(token_totals),
            )
            raise CancellationRequested("cancellation requested", partial)

    with otel_span("agentling.completion") as cycle_span:
        while True:
            turn_number = len(turns) + 1

            with otel_span("agentling.completion.llm_call", {"completion.turn": turn_number}) as llm_span:
                response = await llm.complete(
                    system, messages, tool_schemas,
                    context_id=context_id, task_id=task_id,
                )
                _accumulate_usage(response)
                usage = response.usage or {}
                llm_span.set_attribute("llm.input_tokens", int(usage.get("input_tokens", 0) or 0))
                llm_span.set_attribute("llm.output_tokens", int(usage.get("output_tokens", 0) or 0))
                llm_span.set_attribute(
                    "llm.cache_read_input_tokens",
                    int(usage.get("cache_read_input_tokens", 0) or 0),
                )
                if response.model:
                    llm_span.set_attribute("llm.model", response.model)
                if response.stop_reason:
                    llm_span.set_attribute("llm.stop_reason", response.stop_reason)

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
                _stamp_tokens(cycle_span)
                metrics["duration"].record(elapsed)
                metrics["turns"].record(len(turns))
                return CompletionResult(
                    content=response.content,
                    stop_reason=response.stop_reason,
                    turns=turns,
                    token_usage=dict(token_totals),
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
                tool_name = block["name"]
                tool_input = dict(block.get("input", {}) or {})
                summary = _summary_from_tool_input(tool_name, tool_input)
                tool_input.pop(TOOL_ACTION_SUMMARY_FIELD, None)
                tool_start = time.monotonic()
                try:
                    if tool_event_callback is not None:
                        await tool_event_callback(ToolExecutionEvent(
                            phase="started",
                            tool_name=tool_name,
                            tool_use_id=block["id"],
                            summary=summary,
                            turn_number=turn_number,
                        ))
                    with otel_span("agentling.completion.tool_exec", {
                        "tool.name": tool_name,
                        "completion.turn": turn_number,
                    }) as tool_span:
                        result = await tools.execute(tool_name, tool_input)
                        duration = time.monotonic() - tool_start
                        tool_span.set_attribute("tool.is_error", result.is_error)
                        tool_span.set_attribute("tool.duration_seconds", round(duration, 4))

                    tool_attrs = {"tool.name": tool_name}
                    metrics["tool_calls"].add(1, tool_attrs)
                    record_tool_duration(tool_name, duration, result.is_error)
                    if result.is_error:
                        metrics["tool_errors"].add(1, tool_attrs)
                        logger.warning("tool %s returned error: %.200s", tool_name, result.output)

                    if tool_event_callback is not None:
                        await tool_event_callback(ToolExecutionEvent(
                            phase="completed",
                            tool_name=tool_name,
                            tool_use_id=block["id"],
                            summary=summary,
                            turn_number=turn_number,
                            is_error=result.is_error,
                        ))

                    return {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": _truncate_tool_output(result.output, max_tool_result_chars),
                        "is_error": result.is_error,
                    }
                except asyncio.CancelledError:
                    raise
                except Exception:
                    duration = time.monotonic() - tool_start
                    logger.exception("unhandled exception in tool %s", tool_name)
                    metrics["tool_calls"].add(1, {"tool.name": tool_name})
                    metrics["tool_errors"].add(1, {"tool.name": tool_name})
                    record_tool_duration(tool_name, duration, True)
                    if tool_event_callback is not None:
                        await tool_event_callback(ToolExecutionEvent(
                            phase="failed",
                            tool_name=tool_name,
                            tool_use_id=block["id"],
                            summary=summary,
                            turn_number=turn_number,
                            is_error=True,
                        ))
                    return {
                        "type": "tool_result",
                        "tool_use_id": block["id"],
                        "content": f"Tool {tool_name} failed with an internal error",
                        "is_error": True,
                    }

            tool_results = list(await asyncio.gather(*(_exec_tool(b) for b in tool_calls)))

            _check_cancel()

            turn = CompletionTurn(response=response, tool_results=tool_results)
            turns.append(turn)
            if turn_callback is not None:
                await turn_callback(turn)
            messages.append({"role": "user", "content": tool_results})
