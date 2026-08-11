"""Unit tests for the OpenAI Responses wire-format backend.

Mirrors the Anthropic-backend unit coverage in ``test_llm.py`` — headers,
factory wiring, thinking translation, batches behaviour — plus the
Responses-specific translation layers (request, response, usage, stop
reasons) and the error taxonomy, all hermetically via ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest

from agentlings.config import ThinkingConfig
from agentlings.core.llm import (
    CONTEXT_ID_HEADER,
    NAME_HEADER,
    TASK_ID_HEADER,
    create_llm_client,
)
from agentlings.core.llm_responses import (
    ResponsesAPIError,
    ResponsesBatchesUnsupportedError,
    ResponsesConnectionError,
    ResponsesLLMClient,
    build_reasoning_param,
    extract_responses_usage,
    map_stop_reason,
    messages_to_input,
    output_to_blocks,
    system_to_instructions,
    tools_to_responses,
)


def _body(
    output: list[dict[str, Any]] | None = None,
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "status": status,
        "output": output if output is not None else [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
        "usage": usage,
        "incomplete_details": (
            {"reason": incomplete_reason} if incomplete_reason else None
        ),
        "error": error,
        "model": "gpt-test",
    }


def _client(
    handler: Callable[[httpx.Request], httpx.Response],
    **kwargs: Any,
) -> ResponsesLLMClient:
    kwargs.setdefault("api_key", "sk-test")
    kwargs.setdefault("model", "gpt-test")
    kwargs.setdefault("max_tokens", 256)
    return ResponsesLLMClient(transport=httpx.MockTransport(handler), **kwargs)


def _capture_client(
    body: dict[str, Any] | None = None, **kwargs: Any,
) -> tuple[ResponsesLLMClient, list[httpx.Request]]:
    """A client whose transport records every request and returns ``body``."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=body if body is not None else _body())

    return _client(handler, **kwargs), requests


# --------------------------------------------------------------------------- #
# Request translation
# --------------------------------------------------------------------------- #


class TestSystemToInstructions:
    def test_joins_text_blocks(self) -> None:
        system = [
            {"type": "text", "text": "You are helpful."},
            {"type": "text", "text": "Be terse."},
        ]
        assert system_to_instructions(system) == "You are helpful.\n\nBe terse."

    def test_empty_system_returns_none(self) -> None:
        assert system_to_instructions([]) is None

    def test_non_text_blocks_ignored(self) -> None:
        assert system_to_instructions([{"type": "other", "x": 1}]) is None


class TestToolsToResponses:
    def test_flattened_function_shape(self) -> None:
        tools = [{
            "name": "echo",
            "description": "Echo back the supplied text.",
            "input_schema": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        }]
        out = tools_to_responses(tools)
        assert out == [{
            "type": "function",
            "name": "echo",
            "description": "Echo back the supplied text.",
            "parameters": tools[0]["input_schema"],
        }]

    def test_missing_schema_gets_empty_object(self) -> None:
        out = tools_to_responses([{"name": "noop"}])
        assert out[0]["parameters"] == {"type": "object", "properties": {}}
        assert "description" not in out[0]


class TestMessagesToInput:
    def test_string_content_passes_through(self) -> None:
        items = messages_to_input([{"role": "user", "content": "hello"}])
        assert items == [{"role": "user", "content": "hello"}]

    def test_assistant_string_content_becomes_output_text(self) -> None:
        """Compaction entries replay assistant turns as plain strings —
        they must translate to the same output_text shape as list
        content, not pass through as an ambiguous raw string."""
        items = messages_to_input([
            {"role": "assistant", "content": "compacted summary of the chat"},
        ])
        assert items == [{
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": "compacted summary of the chat",
            }],
        }]

    def test_user_text_blocks_become_input_text(self) -> None:
        items = messages_to_input([
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        ])
        assert items == [
            {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        ]

    def test_assistant_text_blocks_become_output_text(self) -> None:
        items = messages_to_input([
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
        ])
        assert items == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "yo"}]},
        ]

    def test_tool_use_becomes_function_call(self) -> None:
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "echo",
                 "input": {"text": "hi"}, "item_id": "fc_1"},
            ]},
        ])
        assert items == [{
            "type": "function_call",
            "call_id": "call_1",
            "name": "echo",
            "arguments": json.dumps({"text": "hi"}),
            "id": "fc_1",
        }]

    def test_tool_use_without_item_id_omits_id(self) -> None:
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "echo", "input": {}},
            ]},
        ])
        assert "id" not in items[0]
        assert items[0]["call_id"] == "call_1"

    def test_tool_result_becomes_function_call_output(self) -> None:
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "ok", "is_error": False},
            ]},
        ])
        assert items == [{
            "type": "function_call_output", "call_id": "call_1", "output": "ok",
        }]

    def test_tool_result_error_gets_marker_prefix(self) -> None:
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": "boom", "is_error": True},
            ]},
        ])
        assert items[0]["output"] == "[tool error] boom"

    def test_tool_result_list_content_flattened_verbatim(self) -> None:
        """Text parts join with NO separator — the messages path passes
        blocks untouched, so verbatim tool output must not gain
        characters."""
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
            ]},
        ])
        assert items[0]["output"] == "ab"

    def test_tool_result_typeless_dict_not_silently_empty(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A dict part without 'type' is structured content — it must go
        through the JSON-with-warning branch, not contribute ''."""
        with caplog.at_level("WARNING"):
            items = messages_to_input([
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1",
                     "content": [{"data": "mystery"}]},
                ]},
            ])
        assert items[0]["output"] == [
            {"type": "input_text", "text": json.dumps({"data": "mystery"})},
        ]
        assert any("no Responses analogue" in r.message for r in caplog.records)

    def test_tool_result_image_preserved_as_input_image(self) -> None:
        """An image-producing tool must show the model the image — not an
        empty string it would fabricate an analysis over."""
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": [
                     {"type": "text", "text": "screenshot:"},
                     {"type": "image", "source": {
                         "type": "base64", "media_type": "image/png",
                         "data": "aGk=",
                     }},
                 ]},
            ]},
        ])
        assert items[0]["output"] == [
            {"type": "input_text", "text": "screenshot:"},
            {"type": "input_image", "image_url": "data:image/png;base64,aGk="},
        ]

    def test_tool_result_url_image_preserved(self) -> None:
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": [{"type": "image", "source": {
                     "type": "url", "url": "https://example.com/x.png",
                 }}]},
            ]},
        ])
        assert items[0]["output"] == [
            {"type": "input_image", "image_url": "https://example.com/x.png"},
        ]

    def test_tool_result_unknown_block_serialised_not_dropped(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Structured blocks with no Responses analogue are passed as JSON
        text (lossy but loud) — never silently reduced to an empty
        string. Presence of the structured block switches the output to
        the parts form."""
        block = {"type": "resource", "resource": {"uri": "file:///x"}}
        with caplog.at_level("WARNING"):
            items = messages_to_input([
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "call_1",
                     "content": [block]},
                ]},
            ])
        assert items[0]["output"] == [
            {"type": "input_text", "text": json.dumps(block)},
        ]
        assert any("no Responses analogue" in r.message for r in caplog.records)

    def test_tool_result_error_prefix_with_structured_content(self) -> None:
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "is_error": True,
                 "content": [
                     {"type": "text", "text": "boom"},
                     {"type": "image", "source": {
                         "type": "base64", "media_type": "image/png", "data": "eA==",
                     }},
                 ]},
            ]},
        ])
        assert items[0]["output"][0] == {"type": "input_text", "text": "[tool error]"}
        assert items[0]["output"][1] == {"type": "input_text", "text": "boom"}

    def test_thinking_with_item_id_replayed_as_reasoning(self) -> None:
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "pondering",
                 "signature": "enc123", "item_id": "rs_1"},
                {"type": "tool_use", "id": "call_1", "name": "echo",
                 "input": {}, "item_id": "fc_1"},
            ]},
        ])
        assert items[0] == {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc123",
            "summary": [{"type": "summary_text", "text": "pondering"}],
        }
        assert items[1]["type"] == "function_call"

    def test_trailing_reasoning_not_replayed(self) -> None:
        """A reasoning item is only valid with a following item — a
        thinking-only assistant message (old journals) must not replay a
        dangling reasoning item that 400s every subsequent turn."""
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "pondering",
                 "signature": "enc123", "item_id": "rs_1"},
            ]},
            {"role": "user", "content": "follow-up"},
        ])
        assert items == [{"role": "user", "content": "follow-up"}]

    def test_thinking_without_signature_dropped(self) -> None:
        """With store:false the server cannot recover reasoning state from
        an id alone — replay is gated on the encrypted payload, so blocks
        from backends that never return encrypted_content are not
        replayed (instead of 400ing the next turn)."""
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "local", "signature": "",
                 "item_id": "rs_local"},
                {"type": "text", "text": "answer"},
            ]},
        ])
        assert items == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        ]

    def test_thinking_without_item_id_dropped(self) -> None:
        # A fabricated reasoning-item id would be rejected upstream, so
        # blocks without preserved identity are not replayed.
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "anon", "signature": ""},
                {"type": "text", "text": "answer"},
            ]},
        ])
        assert items == [
            {"role": "assistant", "content": [{"type": "output_text", "text": "answer"}]},
        ]

    def test_thinking_empty_text_gets_empty_summary(self) -> None:
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "enc",
                 "item_id": "rs_2"},
                {"type": "text", "text": "answer"},
            ]},
        ])
        assert items[0]["summary"] == []
        assert items[0]["encrypted_content"] == "enc"

    def test_block_order_preserved_across_kinds(self) -> None:
        # A realistic assistant turn: reasoning → tool call → text, then the
        # user's tool result. Order must survive translation.
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "t", "signature": "e", "item_id": "rs_1"},
                {"type": "tool_use", "id": "call_1", "name": "echo", "input": {}},
                {"type": "text", "text": "calling echo"},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "done"},
            ]},
        ])
        assert [i.get("type", i.get("role")) for i in items] == [
            "reasoning", "function_call", "assistant", "function_call_output",
        ]

    def test_consecutive_text_blocks_stay_separate_parts(self) -> None:
        """Anthropic text blocks stay distinct — one message item, one
        content part per block. Joining with '' glued the replay-injected
        '[task <id>]' tag onto the message text."""
        items = messages_to_input([
            {"role": "user", "content": [
                {"type": "text", "text": "[task abc123]"},
                {"type": "text", "text": "hello"},
            ]},
        ])
        assert len(items) == 1
        assert items[0]["content"] == [
            {"type": "input_text", "text": "[task abc123]"},
            {"type": "input_text", "text": "hello"},
        ]

    def test_empty_text_blocks_skipped(self) -> None:
        """Strict endpoints reject empty text items — [''] must not emit one."""
        items = messages_to_input([
            {"role": "user", "content": [{"type": "text", "text": ""}]},
        ])
        assert items == []

    def test_empty_text_among_real_text_dropped(self) -> None:
        items = messages_to_input([
            {"role": "assistant", "content": [
                {"type": "text", "text": ""},
                {"type": "text", "text": "real"},
            ]},
        ])
        assert items == [{
            "role": "assistant",
            "content": [{"type": "output_text", "text": "real"}],
        }]

    def test_unknown_block_types_dropped(self) -> None:
        items = messages_to_input([
            {"role": "user", "content": [{"type": "mystery", "x": 1}]},
        ])
        assert items == []


class TestBuildReasoningParam:
    """Mirror of ``TestThinkingHelper`` for the responses wire format."""

    def test_off_and_none_return_none(self) -> None:
        assert build_reasoning_param(None) is None
        assert build_reasoning_param(ThinkingConfig(mode="off")) is None

    def test_adaptive_without_effort_returns_none(self) -> None:
        assert build_reasoning_param(ThinkingConfig(mode="adaptive")) is None

    @pytest.mark.parametrize("effort,expected", [
        ("low", "low"), ("medium", "medium"), ("high", "high"),
        ("xhigh", "high"), ("max", "high"),
    ])
    def test_adaptive_effort_mapping(self, effort: str, expected: str) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort=effort)  # type: ignore[arg-type]
        assert build_reasoning_param(cfg) == {"effort": expected}

    def test_display_summarized_adds_summary_auto(self) -> None:
        cfg = ThinkingConfig(mode="adaptive", effort="low", display="summarized")
        assert build_reasoning_param(cfg) == {"effort": "low", "summary": "auto"}

    def test_budget_mode_returns_none(self) -> None:
        cfg = ThinkingConfig(mode="budget", budget_tokens=2048)
        assert build_reasoning_param(cfg) is None

    def test_budget_mode_warns_at_construction(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Mirror of the messages-path model/mode mismatch warning: loud at
        # startup, not a refusal.
        with caplog.at_level("WARNING"):
            _client(
                lambda r: httpx.Response(200, json=_body()),
                thinking=ThinkingConfig(mode="budget", budget_tokens=2048),
            )
        assert any("budget" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# Response translation
# --------------------------------------------------------------------------- #


class TestOutputToBlocks:
    def test_output_text_becomes_text_block(self) -> None:
        blocks, tool, refusal = output_to_blocks([
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hi"}]},
        ])
        assert blocks == [{"type": "text", "text": "hi"}]
        assert not tool and not refusal

    def test_function_call_becomes_tool_use(self) -> None:
        blocks, tool, _ = output_to_blocks([
            {"type": "function_call", "id": "fc_1", "call_id": "call_1",
             "name": "echo", "arguments": '{"text": "hi"}'},
        ])
        assert tool
        assert blocks == [{
            "type": "tool_use", "id": "call_1", "name": "echo",
            "input": {"text": "hi"}, "item_id": "fc_1",
        }]

    def test_malformed_arguments_fail_the_turn(self) -> None:
        """Invalid JSON must never be rewritten into an executable
        empty-input call — a tool whose defaults have side effects could
        fire. The turn fails loudly instead."""
        with pytest.raises(ResponsesAPIError) as exc_info:
            output_to_blocks([
                {"type": "function_call", "call_id": "c", "name": "echo",
                 "arguments": "not json"},
            ])
        assert exc_info.value.error_type == "invalid_function_call_arguments"

    def test_non_object_arguments_fail_the_turn(self) -> None:
        with pytest.raises(ResponsesAPIError):
            output_to_blocks([
                {"type": "function_call", "call_id": "c", "name": "echo",
                 "arguments": '["a", "list"]'},
            ])

    _FOLLOWING_MESSAGE = {
        "type": "message", "role": "assistant",
        "content": [{"type": "output_text", "text": "done"}],
    }

    def test_reasoning_becomes_thinking_block(self) -> None:
        blocks, _, _ = output_to_blocks([
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
             "summary": [{"type": "summary_text", "text": "hmm"}]},
            self._FOLLOWING_MESSAGE,
        ])
        assert blocks[0] == {
            "type": "thinking", "thinking": "hmm", "signature": "enc",
            "item_id": "rs_1",
        }

    def test_reasoning_content_parts_also_extracted(self) -> None:
        # Some local backends emit reasoning text under ``content`` instead
        # of ``summary``.
        blocks, _, _ = output_to_blocks([
            {"type": "reasoning", "id": "rs_1", "summary": [],
             "content": [{"type": "reasoning_text", "text": "local thought"}]},
            self._FOLLOWING_MESSAGE,
        ])
        assert blocks[0]["thinking"] == "local thought"

    def test_completed_trailing_reasoning_dropped(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A COMPLETED response ending on reasoning (no following item)
        must not journal the orphan — its replay 400s every later turn."""
        with caplog.at_level("WARNING"):
            blocks, _, _ = output_to_blocks([
                self._FOLLOWING_MESSAGE,
                {"type": "reasoning", "id": "rs_tail",
                 "encrypted_content": "enc", "summary": []},
            ])
        assert blocks == [{"type": "text", "text": "done"}]
        assert any("trailing reasoning" in r.message for r in caplog.records)

    def test_trailing_reasoning_after_call_reordered_not_dropped(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A lax backend ordering [function_call, reasoning] must keep the
        pair together: dropping the reasoning while the tool executes
        would journal a call whose replay is rejected forever. The
        reasoning is reordered before its sibling call instead."""
        with caplog.at_level("WARNING"):
            blocks, tool, _ = output_to_blocks([
                {"type": "function_call", "call_id": "call_1", "name": "echo",
                 "arguments": '{"text": "hi"}'},
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
                 "summary": []},
            ])
        assert tool
        assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
        assert blocks[0]["item_id"] == "rs_1"
        assert any("reordered" in r.message for r in caplog.records)

    def test_parallel_calls_reorder_before_first_call(self) -> None:
        """With PARALLEL function calls, trailing reasoning must land
        before the FIRST call — otherwise the earlier calls journal
        without a preceding reasoning item and every replay 400s after
        their side effects."""
        blocks, _, _ = output_to_blocks([
            {"type": "function_call", "call_id": "call_a", "name": "f",
             "arguments": "{}"},
            {"type": "function_call", "call_id": "call_b", "name": "g",
             "arguments": "{}"},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "e",
             "summary": []},
        ])
        assert [b["type"] for b in blocks] == ["thinking", "tool_use", "tool_use"]
        assert blocks[1]["id"] == "call_a"
        assert blocks[2]["id"] == "call_b"

    def test_multiple_trailing_reasoning_keep_order_when_reordered(self) -> None:
        blocks, _, _ = output_to_blocks([
            {"type": "function_call", "call_id": "call_1", "name": "echo",
             "arguments": "{}"},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "e1", "summary": []},
            {"type": "reasoning", "id": "rs_2", "encrypted_content": "e2", "summary": []},
        ])
        assert [b.get("item_id") for b in blocks] == ["rs_1", "rs_2", ""]
        assert blocks[-1]["type"] == "tool_use"

    def test_empty_output_text_parts_never_journaled(self) -> None:
        blocks, _, _ = output_to_blocks([
            {"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": ""},
                {"type": "output_text", "text": "real"},
            ]},
        ])
        assert blocks == [{"type": "text", "text": "real"}]

    def test_refusal_content_flagged(self) -> None:
        blocks, _, refusal = output_to_blocks([
            {"type": "message", "role": "assistant",
             "content": [{"type": "refusal", "refusal": "no can do"}]},
        ])
        assert refusal
        assert blocks == [{"type": "text", "text": "no can do"}]

    def test_unknown_item_types_dropped(self) -> None:
        blocks, tool, refusal = output_to_blocks([
            {"type": "web_search_call", "id": "ws_1"},
        ])
        assert blocks == [] and not tool and not refusal

    def test_incomplete_drops_malformed_function_call(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A half-emitted function call (truncated response) must be
        dropped, never executed."""
        with caplog.at_level("WARNING"):
            blocks, tool, _ = output_to_blocks(
                [{"type": "function_call", "call_id": "c", "name": "echo",
                  "arguments": '{"text": "hal'}],
                incomplete=True,
            )
        assert blocks == []
        assert tool is False
        assert any("incomplete" in r.message for r in caplog.records)

    def test_incomplete_drops_even_parseable_function_calls(self) -> None:
        """The invariant is absolute: an incomplete turn never returns
        tool_use blocks, even for calls that fully parsed — a journaled
        call that will never get an output would 400 every later replay,
        and crash recovery could mistake it for a pending executable
        call."""
        blocks, tool, _ = output_to_blocks(
            [{"type": "function_call", "call_id": "c", "name": "echo",
              "arguments": '{"text": "hi"}'}],
            incomplete=True,
        )
        assert blocks == []
        assert tool is False

    def test_incomplete_drops_orphaned_reasoning(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When an incomplete turn's function calls are dropped, the
        sibling reasoning items must go too — a journaled reasoning item
        without its required following item poisons every later replay."""
        with caplog.at_level("WARNING"):
            blocks, tool, _ = output_to_blocks(
                [
                    {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
                     "summary": []},
                    {"type": "function_call", "call_id": "c", "name": "echo",
                     "arguments": '{"text": "hi"}'},
                ],
                incomplete=True,
            )
        assert blocks == []
        assert not tool
        assert any("reasoning" in r.message for r in caplog.records)

    def test_incomplete_keeps_text(self) -> None:
        blocks, _, _ = output_to_blocks(
            [{"type": "message", "role": "assistant",
              "content": [{"type": "output_text", "text": "partial answ"}]}],
            incomplete=True,
        )
        assert blocks == [{"type": "text", "text": "partial answ"}]

    def test_missing_call_id_fails_before_execution(self) -> None:
        """No fabricated call ids: without a real call_id the tool result
        could never be correlated on replay, so the call is refused
        before any tool_use block exists to execute."""
        with pytest.raises(ResponsesAPIError) as exc_info:
            output_to_blocks([
                {"type": "function_call", "id": "fc_1", "name": "echo",
                 "arguments": "{}"},
            ])
        assert exc_info.value.error_type == "invalid_function_call"

    def test_missing_name_fails(self) -> None:
        with pytest.raises(ResponsesAPIError):
            output_to_blocks([
                {"type": "function_call", "call_id": "call_1",
                 "arguments": "{}"},
            ])

    @pytest.mark.parametrize("bad_args", [None, "", "   ", 42, {"a": 1}])
    def test_absent_or_non_string_arguments_fail(self, bad_args) -> None:
        """Only a JSON-object string is a valid arguments field. Absent,
        empty, or non-string values must not silently become {} — an
        empty-input call could fire a tool's side-effecting defaults."""
        item = {"type": "function_call", "call_id": "call_1", "name": "restart"}
        if bad_args is not None:
            item["arguments"] = bad_args
        with pytest.raises(ResponsesAPIError) as exc_info:
            output_to_blocks([item])
        assert exc_info.value.error_type == "invalid_function_call_arguments"

    def test_literal_empty_object_arguments_valid(self) -> None:
        blocks, tool, _ = output_to_blocks([
            {"type": "function_call", "call_id": "call_1", "name": "noop",
             "arguments": "{}"},
        ])
        assert tool and blocks[0]["input"] == {}

    def test_function_call_without_item_id_still_marked_foreign(self) -> None:
        """tool_use blocks stamp item_id unconditionally (empty when the
        backend gave no output-item id) — consistent with thinking blocks,
        so the messages-path sanitiser always recognises responses-origin
        history after a wire-format switch."""
        blocks, _, _ = output_to_blocks([
            {"type": "function_call", "call_id": "call_1", "name": "echo",
             "arguments": "{}"},
        ])
        assert blocks[0]["item_id"] == ""

    def test_reasoning_without_id_still_marked_foreign(self) -> None:
        """A backend reasoning item without an id must not journal as a
        native-looking Anthropic thinking block — the item_id marker is
        stamped (empty) so the messages-path sanitiser recognises it."""
        blocks, _, _ = output_to_blocks([
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "anon thought"},
            ]},
            self._FOLLOWING_MESSAGE,
        ])
        assert blocks[0] == {
            "type": "thinking", "thinking": "anon thought",
            "signature": "", "item_id": "",
        }


class TestMapStopReason:
    def test_tool_call_wins_when_completed(self) -> None:
        assert map_stop_reason("completed", None, True, False) == "tool_use"

    def test_completed_is_end_turn(self) -> None:
        assert map_stop_reason("completed", None, False, False) == "end_turn"

    def test_incomplete_max_output_tokens(self) -> None:
        assert map_stop_reason("incomplete", "max_output_tokens", False, False) == "max_tokens"

    def test_truncation_beats_tool_call(self) -> None:
        """A response cut off by max_output_tokens must surface max_tokens
        even when a (possibly half-emitted) function call is present."""
        assert map_stop_reason("incomplete", "max_output_tokens", True, False) == "max_tokens"

    def test_content_filter_beats_tool_call(self) -> None:
        assert map_stop_reason("incomplete", "content_filter", True, False) == "refusal"

    def test_incomplete_content_filter(self) -> None:
        assert map_stop_reason("incomplete", "content_filter", False, False) == "refusal"

    def test_incomplete_unknown_reason_end_turn(
        self, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("WARNING"):
            assert map_stop_reason("incomplete", "mystery", False, False) == "end_turn"
        assert any("unmapped" in r.message for r in caplog.records)

    def test_incomplete_unknown_reason_never_tool_use(self) -> None:
        """A genuinely unknown incomplete reason maps to end_turn and
        never presents as a tool turn."""
        assert map_stop_reason("incomplete", "mystery", True, False) == "end_turn"

    @pytest.mark.parametrize("reason", ["max_output_tokens", "max_tokens", "length"])
    def test_truncation_synonyms_map_to_max_tokens(self, reason: str) -> None:
        """Compatible backends spell truncation differently — known
        synonyms must surface as max_tokens, not present a truncated turn
        as a clean completion."""
        assert map_stop_reason("incomplete", reason, False, False) == "max_tokens"
        assert map_stop_reason("incomplete", reason, True, False) == "max_tokens"

    def test_refusal_maps_to_refusal(self) -> None:
        assert map_stop_reason("completed", None, False, True) == "refusal"


class TestExtractResponsesUsage:
    def test_cached_tokens_subtracted_from_input(self) -> None:
        usage = extract_responses_usage({
            "input_tokens": 100,
            "output_tokens": 20,
            "input_tokens_details": {"cached_tokens": 60},
            "output_tokens_details": {"reasoning_tokens": 5},
        })
        assert usage == {
            "input_tokens": 40,
            "output_tokens": 20,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 60,
        }

    def test_missing_usage_returns_empty(self) -> None:
        assert extract_responses_usage(None) == {}

    def test_partial_usage_degrades_to_zero(self) -> None:
        usage = extract_responses_usage({"output_tokens": 3})
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 3
        assert usage["cache_read_input_tokens"] == 0

    def test_cached_exceeding_input_floors_at_zero(self) -> None:
        usage = extract_responses_usage({
            "input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 10},
        })
        assert usage["input_tokens"] == 0
        assert usage["cache_read_input_tokens"] == 10


# --------------------------------------------------------------------------- #
# The client end-to-end over MockTransport
# --------------------------------------------------------------------------- #


class TestResponsesComplete:
    @pytest.mark.asyncio
    async def test_simple_completion(self) -> None:
        client, requests = _capture_client(usage_body := _body(
            usage={
                "input_tokens": 10, "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 4},
            },
        ))
        response = await client.complete(
            system=[{"type": "text", "text": "Be terse."}],
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
        )
        assert response.content == [{"type": "text", "text": "hello"}]
        assert response.stop_reason == "end_turn"
        assert response.model == "gpt-test"
        assert response.usage == {
            "input_tokens": 6, "output_tokens": 5,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 4,
        }

        payload = json.loads(requests[0].content)
        assert payload["model"] == "gpt-test"
        assert payload["instructions"] == "Be terse."
        assert payload["input"] == [{"role": "user", "content": "hi"}]
        assert payload["max_output_tokens"] == 256
        assert payload["store"] is False
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert "tools" not in payload

    @pytest.mark.asyncio
    async def test_request_hits_v1_responses_path(self) -> None:
        client, requests = _capture_client(base_url="https://gateway.example")
        await client.complete(system=[], messages=[], tools=[])
        assert str(requests[0].url) == "https://gateway.example/v1/responses"

    @pytest.mark.asyncio
    async def test_base_url_ending_in_v1_not_duplicated(self) -> None:
        client, requests = _capture_client(base_url="https://gateway.example/v1")
        await client.complete(system=[], messages=[], tools=[])
        assert str(requests[0].url) == "https://gateway.example/v1/responses"

    @pytest.mark.asyncio
    async def test_full_endpoint_url_accepted_verbatim(self) -> None:
        """The documented 'endpoint override' wording means a full
        /v1/responses URL must work without path doubling."""
        client, requests = _capture_client(
            base_url="https://gateway.example/v1/responses",
        )
        await client.complete(system=[], messages=[], tools=[])
        assert str(requests[0].url) == "https://gateway.example/v1/responses"

    @pytest.mark.asyncio
    async def test_bearer_auth_header_sent(self) -> None:
        client, requests = _capture_client(api_key="sk-secret")
        await client.complete(system=[], messages=[], tools=[])
        assert requests[0].headers["authorization"] == "Bearer sk-secret"

    @pytest.mark.asyncio
    async def test_empty_api_key_falls_back_to_placeholder(self) -> None:
        # Mirror of the Anthropic client's ``api_key or "unset"`` behaviour
        # for backends that don't validate the key (e.g. Ollama).
        client, requests = _capture_client(api_key="")
        await client.complete(system=[], messages=[], tools=[])
        assert requests[0].headers["authorization"] == "Bearer unset"

    @pytest.mark.asyncio
    async def test_tools_translated_into_payload(self) -> None:
        client, requests = _capture_client()
        await client.complete(
            system=[], messages=[],
            tools=[{"name": "echo", "description": "e",
                    "input_schema": {"type": "object", "properties": {}}}],
        )
        payload = json.loads(requests[0].content)
        assert payload["tools"] == [{
            "type": "function", "name": "echo", "description": "e",
            "parameters": {"type": "object", "properties": {}},
        }]

    @pytest.mark.asyncio
    async def test_output_schema_becomes_text_format(self) -> None:
        client, requests = _capture_client()
        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        await client.complete(system=[], messages=[], tools=[], output_schema=schema)
        payload = json.loads(requests[0].content)
        assert payload["text"] == {
            "format": {
                "type": "json_schema", "name": "output",
                "schema": schema, "strict": False,
            }
        }

    @pytest.mark.asyncio
    async def test_max_tokens_override_per_call(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[], max_tokens=42)
        assert json.loads(requests[0].content)["max_output_tokens"] == 42

    @pytest.mark.asyncio
    async def test_model_override_per_call(self) -> None:
        """The sleep cycle's sleep.model override rides this parameter."""
        client, requests = _capture_client()
        response = await client.complete(
            system=[], messages=[], tools=[], model="gpt-5-nano",
        )
        assert json.loads(requests[0].content)["model"] == "gpt-5-nano"
        assert response.model == "gpt-5-nano"

    @pytest.mark.asyncio
    async def test_default_model_when_no_override(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[])
        assert json.loads(requests[0].content)["model"] == "gpt-test"

    @pytest.mark.asyncio
    async def test_tool_call_response_maps_to_tool_use(self) -> None:
        body = _body(output=[
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
             "summary": []},
            {"type": "function_call", "id": "fc_1", "call_id": "call_1",
             "name": "echo", "arguments": '{"text": "hi"}'},
        ])
        client, _ = _capture_client(body)
        response = await client.complete(system=[], messages=[], tools=[])
        assert response.stop_reason == "tool_use"
        assert [b["type"] for b in response.content] == ["thinking", "tool_use"]
        assert response.content[1]["id"] == "call_1"
        assert response.content[1]["item_id"] == "fc_1"

    @pytest.mark.asyncio
    async def test_incomplete_max_output_tokens_stop_reason(self) -> None:
        body = _body(status="incomplete", incomplete_reason="max_output_tokens")
        client, _ = _capture_client(body)
        response = await client.complete(system=[], messages=[], tools=[])
        assert response.stop_reason == "max_tokens"

    @pytest.mark.asyncio
    async def test_truncated_tool_call_surfaces_max_tokens(self) -> None:
        """A function call cut off by max_output_tokens must not execute:
        the block is dropped and the turn surfaces max_tokens."""
        body = _body(
            status="incomplete",
            incomplete_reason="max_output_tokens",
            output=[{"type": "function_call", "call_id": "call_1",
                     "name": "echo", "arguments": '{"text": "trunc'}],
        )
        client, _ = _capture_client(body)
        response = await client.complete(system=[], messages=[], tools=[])
        assert response.stop_reason == "max_tokens"
        assert not any(b["type"] == "tool_use" for b in response.content)

    @pytest.mark.asyncio
    async def test_incomplete_parseable_tool_call_also_dropped(self) -> None:
        """tool_use blocks are returned iff stop_reason is tool_use: even
        a fully parseable call on an incomplete turn is dropped, so
        journaled turns never contain a dangling function call without an
        output."""
        body = _body(
            status="incomplete",
            incomplete_reason="max_output_tokens",
            output=[{"type": "function_call", "call_id": "call_1",
                     "name": "echo", "arguments": '{"text": "full"}'}],
        )
        client, _ = _capture_client(body)
        response = await client.complete(system=[], messages=[], tools=[])
        assert response.stop_reason == "max_tokens"
        assert not any(b["type"] == "tool_use" for b in response.content)
        # Empty assistant content is invalid to journal — the turn keeps
        # an explanatory text block instead.
        assert response.content == [{
            "type": "text",
            "text": "[the model produced no usable output "
                    "(status='incomplete', reason='max_output_tokens')]",
        }]

    @pytest.mark.asyncio
    async def test_unreplayable_reasoning_on_tool_turn_fails_before_execution(self) -> None:
        """A tool turn whose reasoning lacks encrypted content cannot be
        replayed faithfully — the turn must fail BEFORE any tool executes,
        not succeed once and 400 forever after."""
        body = _body(output=[
            {"type": "reasoning", "id": "rs_1", "summary": []},  # no encrypted_content
            {"type": "function_call", "call_id": "call_1", "name": "echo",
             "arguments": '{"text": "hi"}'},
        ])
        client, _ = _capture_client(body)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert exc_info.value.error_type == "missing_encrypted_reasoning"

    @pytest.mark.asyncio
    async def test_reasoning_with_encryption_but_no_id_fails_tool_turn(self) -> None:
        """The gate uses the exact predicate replay uses (both item id AND
        encrypted content) — encrypted reasoning without an id would pass
        a signature-only check, execute the tool, then be dropped on
        replay while the call is kept: permanent history rejection after
        the side effect."""
        body = _body(output=[
            {"type": "reasoning", "encrypted_content": "enc", "summary": []},  # no id
            {"type": "function_call", "call_id": "call_1", "name": "echo",
             "arguments": '{"text": "hi"}'},
        ])
        client, _ = _capture_client(body)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert exc_info.value.error_type == "missing_encrypted_reasoning"

    def test_gate_and_replay_share_one_predicate(self) -> None:
        """is_replayable_reasoning is the single source of truth."""
        from agentlings.core.llm_responses import is_replayable_reasoning

        assert is_replayable_reasoning({"item_id": "rs_1", "signature": "enc"})
        assert not is_replayable_reasoning({"item_id": "rs_1", "signature": ""})
        assert not is_replayable_reasoning({"item_id": "", "signature": "enc"})
        assert not is_replayable_reasoning({})

    @pytest.mark.asyncio
    async def test_unreplayable_reasoning_on_text_turn_is_fine(self) -> None:
        """Text-only turns don't need reasoning replay — no failure."""
        body = _body(output=[
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "hi"}]},
        ])
        client, _ = _capture_client(body)
        response = await client.complete(system=[], messages=[], tools=[])
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_encrypted_reasoning_tool_turn_succeeds(self) -> None:
        body = _body(output=[
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
             "summary": []},
            {"type": "function_call", "call_id": "call_1", "name": "echo",
             "arguments": '{"text": "hi"}'},
        ])
        client, _ = _capture_client(body)
        response = await client.complete(system=[], messages=[], tools=[])
        assert response.stop_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_completed_with_error_field_raises(self) -> None:
        """status 'completed' alongside a non-null error is a
        contradiction — never an empty success."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "status": "completed", "output": [],
                "error": {"message": "half-broken"},
            })

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "half-broken" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_nonstandard_5xx_retried(self) -> None:
        """The whole 500-599 range is retryable (gateways emit 520/529)."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(520, text="origin error")
            return httpx.Response(200, json=_body())

        client = _client(handler)
        response = await client.complete(system=[], messages=[], tools=[])
        assert calls == 2
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_malformed_arguments_fail_turn_end_to_end(self) -> None:
        body = _body(output=[
            {"type": "function_call", "call_id": "call_1", "name": "restart",
             "arguments": "{broken"},
        ])
        client, _ = _capture_client(body)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert exc_info.value.error_type == "invalid_function_call_arguments"

    @pytest.mark.asyncio
    async def test_failed_status_raises(self) -> None:
        body = _body(status="failed", error={
            "code": "server_error", "message": "boom",
        })
        client, _ = _capture_client(body)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "boom" in str(exc_info.value)
        assert exc_info.value.error_type == "server_error"

    @pytest.mark.asyncio
    async def test_reasoning_param_included_when_configured(self) -> None:
        client, requests = _capture_client(
            thinking=ThinkingConfig(mode="adaptive", effort="medium"),
        )
        await client.complete(system=[], messages=[], tools=[])
        assert json.loads(requests[0].content)["reasoning"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_no_reasoning_param_when_unconfigured(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[])
        assert "reasoning" not in json.loads(requests[0].content)


class TestResponsesContextIdHeader:
    """Mirror of ``TestContextIdHeader`` for the responses backend."""

    @pytest.mark.asyncio
    async def test_context_and_task_id_forwarded_as_headers(self) -> None:
        client, requests = _capture_client()
        await client.complete(
            system=[], messages=[], tools=[],
            context_id="ctx-abc", task_id="task-123",
        )
        assert requests[0].headers[CONTEXT_ID_HEADER] == "ctx-abc"
        assert requests[0].headers[TASK_ID_HEADER] == "task-123"

    @pytest.mark.asyncio
    async def test_only_provided_ids_become_headers(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[], task_id="task-only")
        assert requests[0].headers[TASK_ID_HEADER] == "task-only"
        assert CONTEXT_ID_HEADER not in requests[0].headers

    @pytest.mark.asyncio
    async def test_no_header_when_ids_absent(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[])
        assert CONTEXT_ID_HEADER not in requests[0].headers
        assert TASK_ID_HEADER not in requests[0].headers


class TestResponsesSleepCycleHeader:
    """Parity with the messages path: sleep-cycle traffic is stamped with
    the Agentling-SleepCycle header on the responses wire format too, so
    gateways can isolate it regardless of backend."""

    @pytest.mark.asyncio
    async def test_sleep_cycle_header_stamped(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[], sleep_cycle=True)
        assert requests[0].headers["agentling-sleepcycle"] == "true"

    @pytest.mark.asyncio
    async def test_no_header_on_interactive_traffic(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[])
        assert "agentling-sleepcycle" not in requests[0].headers


class TestResponsesNameHeader:
    """Mirror of ``TestNameHeader`` for the responses backend."""

    @pytest.mark.asyncio
    async def test_name_set_as_default_header(self) -> None:
        client, requests = _capture_client(agent_name="office-k3s")
        await client.complete(system=[], messages=[], tools=[])
        assert requests[0].headers[NAME_HEADER] == "office-k3s"

    @pytest.mark.asyncio
    async def test_no_name_header_when_unset(self) -> None:
        client, requests = _capture_client()
        await client.complete(system=[], messages=[], tools=[])
        assert NAME_HEADER not in requests[0].headers


class TestResponsesErrors:
    @pytest.mark.asyncio
    async def test_400_raises_without_retry(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(400, json={
                "error": {"type": "invalid_request_error", "message": "bad input"},
            })

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert calls == 1
        assert exc_info.value.status_code == 400
        assert exc_info.value.error_type == "invalid_request_error"
        assert "bad input" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_error_type_precedence_matches_in_body_path(self) -> None:
        """The >=400 branch uses _parse_error_field (code-then-type), so
        the same backend error classifies identically regardless of
        transport path — sleep's systemic detection depends on it."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {
                "type": "invalid_request_error",
                "code": "model_not_found",
                "message": "no such model",
            }})

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert exc_info.value.error_type == "model_not_found"

    @pytest.mark.asyncio
    async def test_string_error_body_parsed(self) -> None:
        # Gateways (e.g. slipspace) return {"error": "...", "message": "..."}.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={
                "error": "no_binding", "message": "model not served on this protocol",
            })

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "no_binding" in str(exc_info.value)
        assert "model not served" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_429_retried_then_succeeds(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(429, json={"error": {"message": "slow down"}})
            return httpx.Response(200, json=_body())

        client = _client(handler)
        response = await client.complete(system=[], messages=[], tools=[])
        assert calls == 2
        assert response.stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_500_exhausts_retries_then_raises(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(500, json={"error": {"message": "kaboom"}})

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert calls == 3  # initial + 2 retries
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_non_json_success_body_raises_api_error(self) -> None:
        """A 2xx that isn't JSON (gateway maintenance page) must land in
        the error taxonomy, not escape as a raw parse error."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, text="<html>maintenance</html>",
                headers={"content-type": "text/html"},
            )

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "non-JSON" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_non_object_json_success_body_raises_api_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=["not", "a", "response"])

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "non-object" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_in_body_error_with_200_raises(self) -> None:
        """A gateway returning {'error': ...} with HTTP 200 must not
        become an empty successful end_turn."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"error": {"message": "overloaded"}})

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "overloaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_failed_status_with_string_error(self) -> None:
        """Gateways return plain-string error fields; they must land in
        the taxonomy, not AttributeError through err.get()."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "status": "failed", "error": "backend exploded", "output": [],
            })

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "backend exploded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_unknown_status_raises_contract_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "in_progress", "output": []})

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "contract" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_output_raises_contract_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "completed"})

        client = _client(handler)
        with pytest.raises(ResponsesAPIError) as exc_info:
            await client.complete(system=[], messages=[], tools=[])
        assert "output" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_request_id_header_stable_across_retries(self) -> None:
        """Best-effort dedup hook: every attempt of one logical request
        carries the same x-agentling-request-id; a new logical request
        gets a fresh one."""
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers["x-agentling-request-id"])
            if len(seen) == 1:
                return httpx.Response(503, json={"error": {"message": "brb"}})
            return httpx.Response(200, json=_body())

        client = _client(handler)
        await client.complete(system=[], messages=[], tools=[])
        assert len(seen) == 2 and seen[0] == seen[1]
        await client.complete(system=[], messages=[], tools=[])
        assert seen[2] != seen[0]

    @pytest.mark.asyncio
    async def test_connection_error_retried_then_raises(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused")

        client = _client(handler)
        with pytest.raises(ResponsesConnectionError):
            await client.complete(system=[], messages=[], tools=[])
        assert calls == 3


class TestResponsesBatches:
    """Mirror of the batches coverage: the responses wire format has none.

    The sleep cycle's documented workaround (``sleep.enabled: false``) is
    enforced belt-and-braces by ``supports_batches`` plus loud errors.
    """

    def _client(self) -> ResponsesLLMClient:
        return _client(lambda r: httpx.Response(200, json=_body()))

    def test_supports_batches_flag_false(self) -> None:
        assert ResponsesLLMClient.supports_batches is False
        assert self._client().supports_batches is False

    def test_anthropic_and_mock_flag_true(self) -> None:
        from agentlings.core.llm import AnthropicLLMClient, MockLLMClient

        assert AnthropicLLMClient.supports_batches is True
        assert MockLLMClient.supports_batches is True

    @pytest.mark.asyncio
    async def test_batch_create_raises(self) -> None:
        with pytest.raises(ResponsesBatchesUnsupportedError):
            await self._client().batch_create([])

    @pytest.mark.asyncio
    async def test_batch_status_raises(self) -> None:
        with pytest.raises(ResponsesBatchesUnsupportedError):
            await self._client().batch_status("b1")

    @pytest.mark.asyncio
    async def test_batch_results_raises(self) -> None:
        with pytest.raises(ResponsesBatchesUnsupportedError):
            await self._client().batch_results("b1")

    def test_error_is_not_implemented_subclass(self) -> None:
        assert issubclass(ResponsesBatchesUnsupportedError, NotImplementedError)
        # The message must point at the actual workaround (sleep.batch:
        # false / auto-degrade), not at disabling the sleep cycle.
        assert "sleep.batch" in str(ResponsesBatchesUnsupportedError())

    def test_connection_error_is_connection_error(self) -> None:
        """Systemic-failure classification (sleep live path) relies on this."""
        assert issubclass(ResponsesConnectionError, ConnectionError)


class TestResponsesStream:
    @pytest.mark.asyncio
    async def test_stream_not_implemented(self) -> None:
        # Parity with the Anthropic client.
        client = _client(lambda r: httpx.Response(200, json=_body()))
        with pytest.raises(NotImplementedError):
            async for _ in client.stream([], [], []):
                pass


class TestResponsesCountTokens:
    @pytest.mark.asyncio
    async def test_heuristic_count(self) -> None:
        client = _client(lambda r: httpx.Response(200, json=_body()))
        assert await client.count_tokens("x" * 40) == 10


class TestResponsesFactory:
    """Mirror of ``TestFactory`` for the wire-format selector."""

    def test_wire_format_responses_selects_responses_client(self) -> None:
        client = create_llm_client(
            backend="anthropic", api_key="sk-test",
            model="gpt-5-mini", wire_format="responses",
        )
        assert isinstance(client, ResponsesLLMClient)

    def test_default_wire_format_selects_anthropic(self) -> None:
        from agentlings.core.llm import AnthropicLLMClient

        client = create_llm_client(backend="anthropic", api_key="sk-test")
        assert isinstance(client, AnthropicLLMClient)

    def test_mock_backend_ignores_wire_format(self) -> None:
        from agentlings.core.llm import MockLLMClient

        client = create_llm_client(backend="mock", wire_format="responses")
        assert isinstance(client, MockLLMClient)

    def test_base_url_reaches_endpoint(self) -> None:
        """Core wiring for gateway/Ollama support — if this regresses,
        every request silently goes to api.openai.com."""
        client = create_llm_client(
            backend="anthropic", api_key="k", model="gemma-4-e4b",
            base_url="http://localhost:11434", wire_format="responses",
        )
        assert isinstance(client, ResponsesLLMClient)
        assert client._endpoint == "http://localhost:11434/v1/responses"

    def test_default_base_url_is_openai(self) -> None:
        client = create_llm_client(
            backend="anthropic", api_key="k", wire_format="responses",
        )
        assert client._endpoint == "https://api.openai.com/v1/responses"  # type: ignore[union-attr]

    def test_factory_threads_thinking(self) -> None:
        """Mirror of ``TestThinkingFactory`` for the responses client."""
        cfg = ThinkingConfig(mode="adaptive", effort="medium")
        client = create_llm_client(
            backend="anthropic", api_key="k", model="gpt-5-mini",
            wire_format="responses", thinking=cfg,
        )
        assert client._thinking == cfg  # type: ignore[union-attr]

    def test_factory_drops_off_mode_thinking(self) -> None:
        client = create_llm_client(
            backend="anthropic", api_key="k", wire_format="responses",
            thinking=ThinkingConfig(mode="off"),
        )
        assert client._thinking is None  # type: ignore[union-attr]

    def test_unknown_wire_format_raises(self) -> None:
        """A typo must not silently fall through to the Anthropic path and
        send the caller's OpenAI key to api.anthropic.com."""
        with pytest.raises(ValueError, match="wire_format"):
            create_llm_client(
                backend="anthropic", api_key="k", wire_format="response",
            )
        with pytest.raises(ValueError, match="wire_format"):
            create_llm_client(backend="mock", wire_format="bogus")


class TestServerBootsResponsesBackend:
    """The full server wiring accepts the responses wire format, including
    a sleep block — deep-sleep degrades to live summaries (covered in
    ``test_sleep.py``) instead of failing on the missing batches API.
    """

    def test_create_app_with_responses_and_sleep(self, tmp_path) -> None:
        from agentlings.config import AgentConfig
        from agentlings.server import _create_app

        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: responses-boot-test\n"
            "tools: []\n"
            "memory: {}\n"
            "sleep:\n"
            "  enabled: true\n"
            "  batch: false\n"
        )
        config = AgentConfig(
            openai_api_key="sk-test",
            agent_api_key="k",
            agent_data_dir=tmp_path / "data",
            agent_llm_backend="anthropic",
            agent_wire_format="responses",
            agent_model="gpt-5-mini",
            agent_config=str(agent_yaml),
        )
        app = _create_app(config)
        assert app is not None

    def test_mock_backend_boots_without_openai_credentials(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AGENT_LLM_BACKEND=mock uses no credentials — the responses
        wire-format key validation must not block a mock deployment."""
        from agentlings.config import AgentConfig
        from agentlings.core.llm import MockLLMClient  # noqa: F401
        from agentlings.server import _create_app

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        config = AgentConfig(
            agent_api_key="k",
            agent_data_dir=tmp_path / "data",
            agent_llm_backend="mock",
            agent_wire_format="responses",
            _env_file=None,
        )
        app = _create_app(config)
        assert app is not None
