"""Tests for the completion cycle, with emphasis on parallel tool execution safety.

The completion loop is the hot path for every agent interaction. These tests
verify both the happy path and the failure modes that matter in production:
exception isolation under asyncio.gather, result ordering guarantees, and
correct error propagation back to the LLM.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest

from agentlings.core.completion import run_completion
from agentlings.core.llm import LLMResponse, MockLLMClient
from agentlings.tools.registry import ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Helpers — shared LLM stubs and registry builders
# ---------------------------------------------------------------------------

def _stub_registry(execute_fn: Any) -> ToolRegistry:
    """Build a ToolRegistry with a mock execute function and no real tools."""
    registry = ToolRegistry()
    registry.list_schemas = lambda: []
    registry.execute = execute_fn
    return registry


def _tool_calls(*names: str) -> list[dict[str, Any]]:
    """Build a list of tool_use content blocks with unique IDs."""
    return [
        {"type": "tool_use", "id": f"t_{name}_{uuid4().hex[:6]}", "name": name, "input": {}}
        for name in names
    ]


def _tool_calls_stable(*names: str) -> list[dict[str, Any]]:
    """Build tool_use blocks with deterministic IDs for order assertions."""
    return [
        {"type": "tool_use", "id": f"t_{name}", "name": name, "input": {}}
        for name in names
    ]


class _OneShotToolLLM(MockLLMClient):
    """LLM that returns tool calls on the first turn, then a text response.

    This is the most common pattern: the model requests tools once, gets
    results, and produces a final answer. Reused across most parallel tests.
    """

    def __init__(self, tool_content: list[dict[str, Any]]) -> None:
        super().__init__()
        self._tool_content = tool_content
        self._called = False

    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        if not self._called:
            self._called = True
            return LLMResponse(content=self._tool_content, stop_reason="tool_use")
        return LLMResponse(
            content=[{"type": "text", "text": "done"}],
            stop_reason="end_turn",
        )


@pytest.fixture
def tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register_tools(["bash"])
    return registry


# ---------------------------------------------------------------------------
# Basic completion cycle
# ---------------------------------------------------------------------------

class TestRunCompletion:
    """Core completion loop behaviour — no tool parallelism involved."""

    async def test_simple_text_response(self, tools: ToolRegistry) -> None:
        llm = MockLLMClient()
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = await run_completion(llm, [], messages, tools)
        assert result.content[0]["type"] == "text"
        assert len(result.turns) == 1
        assert result.turns[0].tool_results == []

    async def test_tool_use_produces_multiple_turns(self, tools: ToolRegistry) -> None:
        llm = MockLLMClient(tool_names=["bash"])
        messages = [{"role": "user", "content": [{"type": "text", "text": "run bash please"}]}]
        result = await run_completion(llm, [], messages, tools)
        assert len(result.turns) >= 2
        assert result.turns[0].tool_results != []
        assert result.content[0]["type"] == "text"

    async def test_messages_mutated_in_place(self, tools: ToolRegistry) -> None:
        llm = MockLLMClient(tool_names=["bash"])
        messages = [{"role": "user", "content": [{"type": "text", "text": "run bash"}]}]
        original_len = len(messages)
        await run_completion(llm, [], messages, tools)
        assert len(messages) > original_len


# ---------------------------------------------------------------------------
# Parallel tool execution
# ---------------------------------------------------------------------------

class TestParallelToolExecution:
    """Verify tools within a single LLM turn run concurrently via asyncio.gather.

    These tests exist because moving from sequential to parallel execution
    changes the failure semantics: a single exception could cancel siblings,
    result ordering could drift, and error isolation requires defence in depth.
    """

    async def test_tools_run_concurrently(self) -> None:
        """Two slow tools should complete in ~1x delay, not ~2x.

        This is the fundamental proof that gather is working. If this regresses
        to sequential execution, the wall-clock assertion will catch it.
        """
        delay = 0.3
        call_log: list[tuple[str, float]] = []

        async def _slow_tool(name: str, input_dict: dict[str, Any]) -> ToolResult:
            start = time.monotonic()
            await asyncio.sleep(delay)
            call_log.append((name, start))
            return ToolResult(output=f"{name} done", is_error=False)

        llm = _OneShotToolLLM(_tool_calls("tool_a", "tool_b"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]

        t0 = time.monotonic()
        result = await run_completion(llm, [], messages, _stub_registry(_slow_tool))
        elapsed = time.monotonic() - t0

        assert result.content[0]["text"] == "done"
        assert len(call_log) == 2
        assert elapsed < delay * 2, f"Tools ran sequentially ({elapsed:.2f}s >= {delay * 2}s)"

    async def test_result_order_matches_call_order(self) -> None:
        """asyncio.gather preserves input order even when tasks finish out of order.

        The Anthropic API requires tool_results to be ordered consistently with
        tool_use blocks. If the fast tool's result appeared first, the LLM
        would see mismatched tool_use_id / tool_result pairs.
        """
        async def _ordered_tool(name: str, input_dict: dict[str, Any]) -> ToolResult:
            await asyncio.sleep(0.2 if name == "slow" else 0.0)
            return ToolResult(output=name, is_error=False)

        llm = _OneShotToolLLM(_tool_calls_stable("slow", "fast"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_ordered_tool))

        tool_results = result.turns[0].tool_results
        assert tool_results[0]["tool_use_id"] == "t_slow"
        assert tool_results[0]["content"] == "slow"
        assert tool_results[1]["tool_use_id"] == "t_fast"
        assert tool_results[1]["content"] == "fast"

    async def test_single_tool_degenerates_cleanly(self) -> None:
        """gather with a single coroutine should behave identically to a bare await.

        Ensures no overhead or edge-case breakage when the LLM only requests
        one tool — the common case for simple interactions.
        """
        call_count = 0

        async def _single_tool(name: str, input_dict: dict[str, Any]) -> ToolResult:
            nonlocal call_count
            call_count += 1
            return ToolResult(output="result", is_error=False)

        llm = _OneShotToolLLM(_tool_calls("only_tool"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_single_tool))

        assert call_count == 1
        assert result.content[0]["text"] == "done"
        assert len(result.turns[0].tool_results) == 1


# ---------------------------------------------------------------------------
# Error isolation in parallel batches
# ---------------------------------------------------------------------------

class TestParallelToolErrorIsolation:
    """Verify that failures in one tool do not corrupt or cancel siblings.

    The ToolRegistry.execute() catches exceptions and returns ToolResult(is_error=True),
    but defence in depth means _exec_tool in the completion loop also catches —
    because a subclass, decorator, or middleware could bypass the registry's guard.
    These tests exercise both the expected path (is_error result) and the
    catastrophic path (unhandled exception escaping execute).
    """

    async def test_error_result_does_not_poison_sibling_tools(self) -> None:
        """A tool returning is_error=True must not prevent other tools from completing.

        All three results should reach the LLM: two successes and one error.
        The LLM decides how to recover — the framework must not swallow results.
        """
        results_collected: list[str] = []

        async def _mixed_tool(name: str, input_dict: dict[str, Any]) -> ToolResult:
            if name == "fail_tool":
                return ToolResult(output="something went wrong", is_error=True)
            results_collected.append(name)
            return ToolResult(output=f"{name} ok", is_error=False)

        llm = _OneShotToolLLM(_tool_calls("good_tool", "fail_tool", "another_good"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_mixed_tool))

        assert result.content[0]["text"] == "done"
        assert set(results_collected) == {"good_tool", "another_good"}

        tool_turn = result.turns[0]
        assert len(tool_turn.tool_results) == 3
        error_results = [r for r in tool_turn.tool_results if r["is_error"]]
        ok_results = [r for r in tool_turn.tool_results if not r["is_error"]]
        assert len(error_results) == 1
        assert error_results[0]["content"] == "something went wrong"
        assert len(ok_results) == 2

    async def test_all_tools_fail_still_produces_complete_results(self) -> None:
        """Even if every tool in a batch fails, the LLM must see every error.

        A batch of all-errors is a valid scenario (e.g. network partition).
        The completion loop must not short-circuit or discard any result.
        """
        async def _all_fail(name: str, input_dict: dict[str, Any]) -> ToolResult:
            return ToolResult(output=f"{name} failed", is_error=True)

        llm = _OneShotToolLLM(_tool_calls("a", "b", "c"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_all_fail))

        tool_turn = result.turns[0]
        assert len(tool_turn.tool_results) == 3
        assert all(r["is_error"] for r in tool_turn.tool_results)
        assert result.content[0]["type"] == "text"

    async def test_unhandled_exception_in_execute_is_caught(self) -> None:
        """If execute() raises instead of returning ToolResult, the loop must not crash.

        The ToolRegistry.execute() has its own try/except, but a subclass or
        middleware might bypass it. The _exec_tool wrapper in the completion
        loop is the last line of defence — without it, asyncio.gather propagates
        the exception and cancels all sibling tasks.
        """
        async def _explosive_tool(name: str, input_dict: dict[str, Any]) -> ToolResult:
            if name == "bomb":
                raise RuntimeError("unexpected kaboom")
            return ToolResult(output=f"{name} ok", is_error=False)

        llm = _OneShotToolLLM(_tool_calls_stable("safe", "bomb", "also_safe"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_explosive_tool))

        tool_turn = result.turns[0]
        assert len(tool_turn.tool_results) == 3

        safe_result = next(r for r in tool_turn.tool_results if r["tool_use_id"] == "t_safe")
        bomb_result = next(r for r in tool_turn.tool_results if r["tool_use_id"] == "t_bomb")
        also_safe_result = next(r for r in tool_turn.tool_results if r["tool_use_id"] == "t_also_safe")

        assert not safe_result["is_error"]
        assert safe_result["content"] == "safe ok"
        assert bomb_result["is_error"]
        assert "internal error" in bomb_result["content"]
        assert not also_safe_result["is_error"]
        assert also_safe_result["content"] == "also_safe ok"

    async def test_all_tools_throw_exceptions(self) -> None:
        """Total catastrophe: every tool raises. The loop must still return all errors.

        This is the worst-case scenario for asyncio.gather — without
        return_exceptions or per-task try/except, the first exception would
        cancel everything and the LLM would see nothing.
        """
        async def _all_throw(name: str, input_dict: dict[str, Any]) -> ToolResult:
            raise ValueError(f"{name} exploded")

        llm = _OneShotToolLLM(_tool_calls("x", "y"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_all_throw))

        tool_turn = result.turns[0]
        assert len(tool_turn.tool_results) == 2
        assert all(r["is_error"] for r in tool_turn.tool_results)
        assert all("internal error" in r["content"] for r in tool_turn.tool_results)

    async def test_exception_does_not_cancel_slow_sibling(self) -> None:
        """A fast tool that raises must not cancel a slow tool still in-flight.

        This is the specific asyncio.gather hazard: without per-task exception
        handling, gather's default behaviour cancels pending tasks on first
        failure. We need the slow tool to complete and return its result.
        """
        slow_completed = False

        async def _race_tool(name: str, input_dict: dict[str, Any]) -> ToolResult:
            nonlocal slow_completed
            if name == "fast_bomb":
                raise RuntimeError("instant failure")
            await asyncio.sleep(0.2)
            slow_completed = True
            return ToolResult(output="slow finished", is_error=False)

        llm = _OneShotToolLLM(_tool_calls_stable("slow_worker", "fast_bomb"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        result = await run_completion(llm, [], messages, _stub_registry(_race_tool))

        assert slow_completed, "Slow tool was cancelled by fast tool's exception"
        tool_turn = result.turns[0]
        slow_result = next(r for r in tool_turn.tool_results if r["tool_use_id"] == "t_slow_worker")
        assert not slow_result["is_error"]
        assert slow_result["content"] == "slow finished"


# ---------------------------------------------------------------------------
# Message contract — what the LLM sees after tool execution
# ---------------------------------------------------------------------------

class TestToolResultMessageContract:
    """Verify the shape and content of tool_result messages appended to the conversation.

    The Anthropic Messages API is strict about the structure of tool results.
    Malformed results cause 400 errors that are hard to diagnose in production
    because they surface as generic API failures, not tool-specific errors.
    """

    async def test_tool_results_have_required_fields(self) -> None:
        """Every tool result must have type, tool_use_id, content, and is_error."""
        async def _noop(name: str, input_dict: dict[str, Any]) -> ToolResult:
            return ToolResult(output="ok", is_error=False)

        llm = _OneShotToolLLM(_tool_calls("t1", "t2"))
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        await run_completion(llm, [], messages, _stub_registry(_noop))

        tool_result_msg = messages[2]
        assert tool_result_msg["role"] == "user"
        for result in tool_result_msg["content"]:
            assert result["type"] == "tool_result"
            assert "tool_use_id" in result
            assert "content" in result
            assert "is_error" in result

    async def test_tool_use_ids_round_trip_correctly(self) -> None:
        """Each tool_result.tool_use_id must match the corresponding tool_use.id.

        A mismatch here means the LLM cannot correlate which result belongs
        to which call, leading to hallucinated tool outputs.
        """
        async def _echo(name: str, input_dict: dict[str, Any]) -> ToolResult:
            return ToolResult(output=name, is_error=False)

        tool_blocks = _tool_calls_stable("alpha", "beta")
        llm = _OneShotToolLLM(tool_blocks)
        messages = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
        await run_completion(llm, [], messages, _stub_registry(_echo))

        assistant_msg = messages[1]
        tool_result_msg = messages[2]

        sent_ids = [b["id"] for b in assistant_msg["content"] if b["type"] == "tool_use"]
        recv_ids = [r["tool_use_id"] for r in tool_result_msg["content"]]
        assert sent_ids == recv_ids
