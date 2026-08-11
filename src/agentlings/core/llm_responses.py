"""OpenAI Responses API backend for the LLM client abstraction.

Implements ``BaseLLMClient`` over the OpenAI Responses wire format
(``POST /v1/responses``), selected via ``AGENT_WIRE_FORMAT=responses``.
The rest of agentlings keeps speaking Anthropic-shaped content blocks
(``text`` / ``tool_use`` / ``tool_result`` / ``thinking``); this module
translates that dialect to and from Responses shapes at the HTTP boundary.

Design notes (see ``docs/responses-wire-format.md`` for the full rationale):

* **Stateless.** Every request sends the full input array with
  ``store: false``; ``previous_response_id`` is never used, because local
  Ollama ``/v1/responses`` implementations are stateless.
* **Faithful replay.** Responses reasoning models reject a replayed
  ``function_call`` whose sibling ``reasoning`` item is missing, so the
  translation preserves original item identity: ``tool_use`` and
  ``thinking`` blocks carry an extra ``item_id`` key (and ``thinking``
  keeps ``encrypted_content`` in ``signature``) which round-trips through
  the journal and back into the input array. Anthropic-shaped consumers
  ignore the extra keys.
* **No batches.** The Anthropic batches API has no Responses analogue
  here; ``batch_*`` raise ``ResponsesBatchesUnsupportedError`` and
  ``supports_batches`` is ``False``. The sleep cycle keeps working: it
  degrades deep-sleep summaries to sequential live completion calls
  automatically (``sleep.batch: false`` selects that path explicitly) —
  the same workaround used for Ollama's Anthropic-compatibility layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx

from agentlings.config import ThinkingConfig
from agentlings.core.llm import (
    CONTEXT_ID_HEADER,
    NAME_HEADER,
    SLEEP_CYCLE_HEADER,
    TASK_ID_HEADER,
    BaseLLMClient,
    BatchItemResult,
    BatchRequest,
    BatchStatus,
    LLMResponse,
)
from agentlings.core.telemetry import (
    llm_complete_span,
    record_llm_usage,
    stamp_llm_completion,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com"

# Stable per-logical-request id, identical across retry attempts of the same
# POST. The Responses API has no idempotency-key mechanism, so this is a
# best-effort dedup hook for gateways.
REQUEST_ID_HEADER = "x-agentling-request-id"

# Mirror the Anthropic SDK's default of 2 retries on transient failures.
MAX_RETRIES = 2
_RETRY_BACKOFF_SECONDS = 0.5


def _is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status warrants a retry.

    429 (rate limit) and the full 500-599 range — gateways emit
    non-standard 5xx codes (520, 529, ...) for transient conditions, so
    no 5xx is excluded.
    """
    return status_code == 429 or 500 <= status_code < 600

# ThinkingEffort (Anthropic scale) → Responses reasoning.effort. OpenAI's
# scale tops out at "high", so the two upper Anthropic notches clamp down.
_EFFORT_MAP = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class ResponsesError(Exception):
    """Base class for Responses-backend errors."""


class ResponsesAPIError(ResponsesError):
    """The Responses endpoint rejected or failed the request.

    Attributes:
        status_code: HTTP status code (``0`` for in-body ``status: failed``).
        error_type: The API's error type/code string, if provided.
        message: Human-readable error message.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 0,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.message = message


class ResponsesConnectionError(ResponsesError, ConnectionError):
    """The Responses endpoint could not be reached after retries.

    Also a ``ConnectionError`` so callers (e.g. the sleep cycle's live
    summary path) can classify it as a systemic failure generically.
    """


class ResponsesBatchesUnsupportedError(ResponsesError, NotImplementedError):
    """Raised by ``batch_*`` — the Responses wire format has no batches path.

    The sleep cycle does not hit this in practice: it degrades deep-sleep
    summaries to live completion calls automatically for batch-less
    backends, and ``sleep.batch: false`` selects that path explicitly —
    the same workaround used for other backends without the Anthropic
    batches API (e.g. Ollama).
    """

    def __init__(self) -> None:
        super().__init__(
            "the OpenAI Responses wire format does not support the batches "
            "API; use sleep.batch: false (live summaries) — the sleep "
            "cycle also degrades to that path automatically"
        )


# --------------------------------------------------------------------------- #
# Request translation: Anthropic-shaped internals → Responses request fields
# --------------------------------------------------------------------------- #


def is_replayable_reasoning(block: dict[str, Any]) -> bool:
    """Whether a ``thinking`` block can be replayed as a reasoning item.

    Replay needs BOTH the original item id and the encrypted payload
    (``signature``): with ``store: false`` the server cannot recover
    reasoning state from an id alone, and a fabricated id would be
    rejected. This single predicate is shared by the replay side
    (``messages_to_input``) and the pre-execution gate in ``complete()``
    so the two can never drift — a tool turn is only allowed to execute
    when its reasoning will replay exactly as required.
    """
    return bool(block.get("item_id")) and bool(block.get("signature"))


def system_to_instructions(system: list[dict[str, Any]]) -> str | None:
    """Join Anthropic system text blocks into a Responses ``instructions`` string."""
    parts = [
        b.get("text", "")
        for b in system
        if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
    ]
    return "\n\n".join(parts) if parts else None


def tools_to_responses(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate Anthropic tool schemas into Responses function tools.

    ``{name, description, input_schema}`` becomes the flattened
    ``{type: "function", name, description, parameters}`` shape.
    """
    out: list[dict[str, Any]] = []
    for tool in tools:
        entry: dict[str, Any] = {
            "type": "function",
            "name": tool["name"],
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            entry["description"] = tool["description"]
        out.append(entry)
    return out


def _tool_result_output(block: dict[str, Any]) -> str | list[dict[str, Any]]:
    """Render a ``tool_result`` block's content for ``function_call_output``.

    Text-only results become a plain string (universally accepted,
    including by minimal local backends). Results carrying non-text
    blocks are preserved instead of silently discarded — mirroring the
    messages path, which passes the block list verbatim:

    * Anthropic ``image`` blocks (base64 or url source) become
      ``input_image`` parts, so an image-producing tool still shows the
      model the image.
    * Any other structured block is serialised to JSON inside an
      ``input_text`` part (with a warning) — lossy but loud, never an
      empty string.

    ``is_error`` has no Responses analogue, so errors are prefixed with a
    ``[tool error]`` marker the model can recognise.
    """
    error_prefix = "[tool error] " if block.get("is_error") else ""
    content = block.get("content", "")

    if not isinstance(content, list):
        return f"{error_prefix}{content}"

    # Plain-text parts are non-dict entries and dicts explicitly typed
    # "text". A typeless dict is structured content, not text — it must
    # not silently contribute an empty string.
    has_non_text = any(
        isinstance(part, dict) and part.get("type") != "text"
        for part in content
    )
    if not has_non_text:
        # Join with no separator: the messages path passes blocks
        # untouched, so verbatim tool output must not gain characters.
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
        return f"{error_prefix}{text}"

    parts: list[dict[str, Any]] = []
    if error_prefix:
        parts.append({"type": "input_text", "text": error_prefix.strip()})
    for part in content:
        if not isinstance(part, dict):
            parts.append({"type": "input_text", "text": str(part)})
        elif part.get("type") == "text":
            if part.get("text"):
                parts.append({"type": "input_text", "text": part["text"]})
        elif part.get("type") == "image":
            source = part.get("source") or {}
            if source.get("type") == "base64" and source.get("data"):
                media = source.get("media_type", "image/png")
                parts.append({
                    "type": "input_image",
                    "image_url": f"data:{media};base64,{source['data']}",
                })
            elif source.get("type") == "url" and source.get("url"):
                parts.append({"type": "input_image", "image_url": source["url"]})
            else:
                logger.warning(
                    "tool_result image block with unsupported source %r; "
                    "passing as JSON text", source.get("type"),
                )
                parts.append({"type": "input_text", "text": json.dumps(part)})
        else:
            logger.warning(
                "tool_result block type %r has no Responses analogue; "
                "passing as JSON text", part.get("type"),
            )
            parts.append({"type": "input_text", "text": json.dumps(part)})
    return parts


def messages_to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate the Anthropic-shaped message history into Responses input items.

    Block order within each message is preserved. Consecutive ``text``
    blocks become separate content parts of a single message item —
    matching the messages path, where each Anthropic text block stays its
    own block (e.g. the replay-injected ``[task <id>]`` tag must not glue
    onto the message text). Empty text blocks are skipped entirely: strict
    endpoints reject empty text items. ``tool_result`` blocks become
    top-level ``function_call_output`` items; ``tool_use`` becomes
    ``function_call``; ``thinking`` blocks are replayed as ``reasoning``
    items only when they carry a preserved ``item_id`` (fabricated
    reasoning-item ids would be rejected upstream).
    """
    items: list[dict[str, Any]] = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, str):
            if content:
                if role == "assistant":
                    # Compaction entries replay assistant turns as plain
                    # strings; keep the translated shape uniform with
                    # list-content assistant history (output_text parts).
                    items.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    })
                else:
                    items.append({"role": role, "content": content})
            continue

        message_start = len(items)
        pending_texts: list[str] = []
        part_type = "output_text" if role == "assistant" else "input_text"

        def _flush_text() -> None:
            if not pending_texts:
                return
            items.append({
                "role": role,
                "content": [
                    {"type": part_type, "text": text} for text in pending_texts
                ],
            })
            pending_texts.clear()

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    pending_texts.append(text)
            elif btype == "tool_use":
                _flush_text()
                item: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                }
                if block.get("item_id"):
                    item["id"] = block["item_id"]
                items.append(item)
            elif btype == "tool_result":
                _flush_text()
                items.append({
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id", ""),
                    "output": _tool_result_output(block),
                })
            elif btype in ("thinking", "redacted_thinking"):
                _flush_text()
                if not is_replayable_reasoning(block):
                    # With store:false the server cannot recover reasoning
                    # state from an id alone — a reasoning item is only
                    # replayable with both its original id and encrypted
                    # payload. The pre-execution gate in ``complete()``
                    # uses the same predicate, so a tool turn never lands
                    # in the journal unless its reasoning replays; blocks
                    # skipped here are text-turn reasoning, which degrades
                    # safely.
                    continue
                reasoning: dict[str, Any] = {
                    "type": "reasoning",
                    "id": block["item_id"],
                    "encrypted_content": block["signature"],
                }
                thinking_text = block.get("thinking", "")
                reasoning["summary"] = (
                    [{"type": "summary_text", "text": thinking_text}]
                    if thinking_text
                    else []
                )
                items.append(reasoning)
            # Unknown block types are dropped — same tolerance the rest of
            # the pipeline extends to unrecognised content.

        _flush_text()

        # Replay-side hardening for journals written before the ingest
        # guard existed: a reasoning item is only valid with a following
        # item, so never let one end a message's group.
        while (
            len(items) > message_start
            and items[-1].get("type") == "reasoning"
        ):
            logger.warning(
                "not replaying trailing reasoning item %s without a "
                "following item", items[-1].get("id") or "?",
            )
            items.pop()

    return items


def build_reasoning_param(thinking: ThinkingConfig | None) -> dict[str, Any] | None:
    """Translate a ThinkingConfig into the Responses ``reasoning`` parameter.

    * ``off``/``None`` → no parameter (model default applies).
    * ``adaptive`` → ``{"effort": <mapped>}`` when an effort is set;
      ``display: "summarized"`` adds ``{"summary": "auto"}``.
    * ``budget`` → not representable on this wire format; the client logs
      one warning at construction and sends nothing.
    """
    if thinking is None or thinking.mode == "off":
        return None
    if thinking.mode == "budget":
        return None
    reasoning: dict[str, Any] = {}
    if thinking.effort is not None:
        reasoning["effort"] = _EFFORT_MAP[thinking.effort]
    if thinking.display == "summarized":
        reasoning["summary"] = "auto"
    return reasoning or None


# --------------------------------------------------------------------------- #
# Response translation: Responses output items → Anthropic-shaped blocks
# --------------------------------------------------------------------------- #


def _reasoning_text(item: dict[str, Any]) -> str:
    """Extract human-readable text from a reasoning item.

    OpenAI puts summaries in ``summary`` (``summary_text`` parts); some
    local backends emit ``content`` with ``reasoning_text`` parts. Join
    whatever is present.
    """
    parts: list[str] = []
    for part in list(item.get("summary") or []) + list(item.get("content") or []):
        if isinstance(part, dict) and part.get("text"):
            parts.append(part["text"])
    return "\n\n".join(parts)


def output_to_blocks(
    output: list[dict[str, Any]],
    incomplete: bool = False,
) -> tuple[list[dict[str, Any]], bool, bool]:
    """Translate a Responses ``output`` array into Anthropic-shaped blocks.

    Invariant: a ``tool_use`` block is returned **iff** the response
    completed and the call's arguments parse as a JSON object. Two rules
    enforce it:

    * ``incomplete=True`` (any non-completed status — truncation, content
      filter, unknown reasons) drops **every** function call, parseable or
      not, and every reasoning item: a cut-off turn must surface its stop
      reason, never journal an assistant tool call that will have no
      output on replay, never execute a call the model didn't finish
      deciding on, and never journal an orphaned reasoning item whose
      required following item was removed (reasoning models reject such
      replays permanently).
    * On a completed response, a function call must carry a non-empty
      ``call_id``, a non-empty ``name``, and a string ``arguments`` field
      that parses to a JSON object — anything else (absent/empty/
      non-string arguments, missing identity) raises ``ResponsesAPIError``
      before any ``tool_use`` block exists to execute. Failing loudly
      beats silently rewriting the call (empty-input defaults or a
      fabricated call id could fire side effects that can never be
      replayed to completion).

    Returns ``(blocks, has_tool_call, has_refusal)``.
    """
    blocks: list[dict[str, Any]] = []
    has_tool_call = False
    has_refusal = False

    for item in output:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "output_text":
                    # Empty text blocks are invalid on the messages path
                    # and stripped on our own replay side — never journal
                    # them in the first place.
                    if part.get("text"):
                        blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "refusal":
                    has_refusal = True
                    if part.get("refusal"):
                        blocks.append({"type": "text", "text": part["refusal"]})
        elif itype == "function_call":
            if incomplete:
                logger.warning(
                    "dropping function_call %s from incomplete response "
                    "(a cut-off turn must not execute or journal tool "
                    "calls): %.200s",
                    item.get("name", "?"), item.get("arguments") or "",
                )
                continue
            call_id = item.get("call_id")
            name = item.get("name")
            if not call_id or not isinstance(call_id, str):
                raise ResponsesAPIError(
                    f"model emitted function_call {name!r} without a "
                    "call_id — the tool result could never be correlated "
                    "on replay, so the call is refused before execution",
                    error_type="invalid_function_call",
                )
            if not name or not isinstance(name, str):
                raise ResponsesAPIError(
                    f"model emitted function_call {call_id!r} without a "
                    "tool name",
                    error_type="invalid_function_call",
                )
            raw_args = item.get("arguments")
            if not isinstance(raw_args, str) or not raw_args.strip():
                raise ResponsesAPIError(
                    f"model emitted function_call {name!r} with missing, "
                    f"empty, or non-string arguments ({raw_args!r:.100}) — "
                    "only a JSON object string (e.g. \"{}\") is valid",
                    error_type="invalid_function_call_arguments",
                )
            try:
                parsed = json.loads(raw_args)
                if not isinstance(parsed, dict):
                    raise ValueError("arguments not an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise ResponsesAPIError(
                    f"model emitted function_call {name!r} with arguments "
                    f"that are not a valid JSON object: "
                    f"{str(raw_args)[:200]!r}",
                    error_type="invalid_function_call_arguments",
                ) from exc
            has_tool_call = True
            # ``item_id`` is stamped unconditionally (empty when the
            # backend gave no output-item id): its presence marks the
            # block as responses-origin so the messages-path sanitiser
            # recognises the history as foreign after a wire-format
            # switch — consistent with thinking blocks.
            blocks.append({
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": parsed,
                "item_id": item.get("id") or "",
            })
        elif itype == "reasoning":
            if incomplete:
                # The reasoning item's required following item (function
                # call or message) is being dropped/cut off — journaling
                # the orphan would poison every later replay.
                logger.warning(
                    "dropping reasoning item %s from incomplete response",
                    item.get("id", "?"),
                )
                continue
            # ``item_id`` is stamped unconditionally (empty when the
            # backend gave no id): its presence marks the block as
            # responses-origin so the Anthropic client's sanitiser can
            # recognise and drop it after a wire-format switch — an
            # OpenAI encrypted payload is not an Anthropic signature.
            blocks.append({
                "type": "thinking",
                "thinking": _reasoning_text(item),
                "signature": item.get("encrypted_content") or "",
                "item_id": item.get("id") or "",
            })
        # Other item types (built-in tool calls etc.) are out of scope and
        # dropped; agentlings only registers function tools.

    # A reasoning item is only valid with a following message/function
    # call, so thinking blocks must never end the turn. Collect the tail:
    trailing: list[dict[str, Any]] = []
    while blocks and blocks[-1].get("type") == "thinking":
        trailing.append(blocks.pop())
    if trailing:
        tool_use_indices = [
            i for i, b in enumerate(blocks) if b.get("type") == "tool_use"
        ]
        if tool_use_indices:
            # A lax backend ordered [function_call, reasoning]. Dropping
            # the reasoning while keeping its paired call would let the
            # tool execute and then journal a call whose replay is
            # rejected forever — the pair must stay together. Reinsert
            # the reasoning before the last tool call (the canonical
            # [reasoning, function_call] order), preserving its own
            # relative order.
            idx = tool_use_indices[-1]
            blocks[idx:idx] = list(reversed(trailing))
            logger.warning(
                "reordered %d trailing reasoning item(s) before their "
                "sibling tool call (backend emitted reasoning after the "
                "call)", len(trailing),
            )
        else:
            # Text-only turn: a trailing orphan can simply be dropped.
            for block in trailing:
                logger.warning(
                    "dropping trailing reasoning without a following item "
                    "(item_id=%s)", block.get("item_id") or "?",
                )

    return blocks, has_tool_call, has_refusal


# Incomplete-reason spellings that all mean "output token cap reached".
# api.openai.com uses ``max_output_tokens``; compatible backends commonly
# reuse chat-completions vocabulary.
_TRUNCATION_REASONS = frozenset({"max_output_tokens", "max_tokens", "length"})


def map_stop_reason(
    status: str | None,
    incomplete_reason: str | None,
    has_tool_call: bool,
    has_refusal: bool,
) -> str:
    """Map Responses completion status onto Anthropic stop reasons.

    Incomplete status is checked before tool calls, and any incomplete
    response has already had its function calls dropped by
    ``output_to_blocks(incomplete=True)`` — so ``tool_use`` is only ever
    reported for completed responses, upholding the invariant that a
    ``tool_use`` block always arrives with ``stop_reason="tool_use"``.
    Known truncation synonyms used by compatible backends (``length``,
    ``max_tokens``) map to ``max_tokens`` like the canonical
    ``max_output_tokens``; genuinely unknown incomplete reasons map to
    ``end_turn`` with a warning.
    """
    if status == "incomplete":
        if incomplete_reason in _TRUNCATION_REASONS:
            return "max_tokens"
        if incomplete_reason == "content_filter":
            return "refusal"
        logger.warning(
            "responses status=incomplete with unmapped reason %r; "
            "treating as end_turn", incomplete_reason,
        )
        return "end_turn"
    if has_tool_call:
        return "tool_use"
    if has_refusal:
        return "refusal"
    return "end_turn"


def extract_responses_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Normalise Responses usage into the Anthropic-semantics usage dict.

    OpenAI's ``input_tokens`` *includes* cached tokens; the internal
    contract (Anthropic semantics) counts uncached input only, with cache
    reads reported separately. Missing fields degrade to zero — local
    backends report partial usage.
    """
    if not isinstance(usage, dict):
        return {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    details = usage.get("input_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0) if isinstance(details, dict) else 0
    return {
        "input_tokens": max(input_tokens - cached, 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def _responses_endpoint(base_url: str | None) -> str:
    """Resolve the full ``/v1/responses`` URL from a base URL override.

    Accepts a bare host (``https://gw.example``), a versioned base
    (``https://gw.example/v1``), or the full endpoint URL
    (``https://gw.example/v1/responses``) — the documented "endpoint
    override" wording means all three must work without path doubling.
    """
    base = (base_url or DEFAULT_BASE_URL).rstrip("/")
    if base.endswith("/responses"):
        return base
    if base.endswith("/v1"):
        return f"{base}/responses"
    return f"{base}/v1/responses"


def _parse_error_field(error: Any) -> tuple[str, str | None]:
    """Extract ``(message, code)`` from a Responses ``error`` field.

    The field is a dict upstream but gateways return plain strings too —
    both must land in the error taxonomy rather than raise AttributeError.
    """
    if isinstance(error, dict):
        return (
            str(error.get("message") or error) or "unknown error",
            error.get("code") or error.get("type"),
        )
    if error:
        return str(error), None
    return "unknown error", None


class ResponsesLLMClient(BaseLLMClient):
    """LLM client backed by the OpenAI Responses API.

    Works against api.openai.com and any OpenAI-compatible endpoint that
    implements ``/v1/responses`` (e.g. a gateway, or Ollama's Responses
    layer) via ``base_url``. Operates statelessly: the full input array is
    sent on every call with ``store: false``.
    """

    supports_batches = False

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int,
        base_url: str | None = None,
        agent_name: str | None = None,
        thinking: ThinkingConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint = _responses_endpoint(base_url)
        self._model = model
        self._max_tokens = max_tokens
        self._agent_name = agent_name
        self._thinking = thinking if thinking is not None and thinking.mode != "off" else None
        if self._thinking is not None and self._thinking.mode == "budget":
            logger.warning(
                "thinking.mode='budget' is not representable on the "
                "responses wire format; no reasoning parameter will be "
                "sent — use mode 'adaptive' (effort maps to "
                "reasoning.effort) or 'off'"
            )
        headers = {
            "Authorization": f"Bearer {api_key or 'unset'}",
            "Content-Type": "application/json",
        }
        if agent_name:
            headers[NAME_HEADER] = agent_name
        client_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": httpx.Timeout(600.0, connect=10.0),
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._http = httpx.AsyncClient(**client_kwargs)

    def _build_payload(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: dict[str, Any] | None,
        max_tokens: int,
        model: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": messages_to_input(messages),
            "max_output_tokens": max_tokens,
            # Stateless operation: never rely on server-side response state.
            "store": False,
            # Ask for encrypted reasoning payloads so reasoning items can be
            # replayed verbatim on later turns (required by OpenAI reasoning
            # models when a function_call is replayed). If a backend ignores
            # `include` and returns no encrypted_content, the translation
            # layer does not replay its reasoning items at all (replay is
            # gated on the encrypted payload) — degraded but safe.
            "include": ["reasoning.encrypted_content"],
        }
        instructions = system_to_instructions(system)
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = tools_to_responses(tools)
        if output_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": output_schema,
                    # Strict mode demands additionalProperties:false and
                    # all-required on every level; agentling callers pass
                    # plain JSON Schemas, so stay permissive.
                    "strict": False,
                }
            }
        reasoning = build_reasoning_param(getattr(self, "_thinking", None))
        if reasoning:
            payload["reasoning"] = reasoning
        return payload

    async def _post_with_retries(
        self, payload: dict[str, Any], extra_headers: dict[str, str],
    ) -> dict[str, Any]:
        """POST the payload, retrying transient failures, and return the body.

        Retries are NOT idempotent at the protocol level: the Responses
        API has no idempotency-key mechanism, so a timeout after the
        server accepted (and billed) a request re-issues the whole POST
        and only the final completion is observed in usage accounting.
        As a best-effort dedup hook every attempt of one logical request
        carries the same ``x-agentling-request-id`` header, which
        gateways can use to collapse duplicates.

        Raises:
            ResponsesAPIError: On non-retryable HTTP errors, retries
                exhausted, an in-body error, or a body that does not
                match the Responses contract.
            ResponsesConnectionError: When the endpoint stays unreachable.
        """
        last_error: Exception | None = None
        request_headers = dict(extra_headers)
        request_headers[REQUEST_ID_HEADER] = uuid4().hex
        for attempt in range(MAX_RETRIES + 1):
            if attempt:
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)))
            try:
                response = await self._http.post(
                    self._endpoint, json=payload, headers=request_headers,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "responses request failed (attempt %d/%d): %s",
                    attempt + 1, MAX_RETRIES + 1, exc,
                )
                continue

            if _is_retryable_status(response.status_code) and attempt < MAX_RETRIES:
                logger.warning(
                    "responses endpoint returned %d (attempt %d/%d); retrying",
                    response.status_code, attempt + 1, MAX_RETRIES + 1,
                )
                continue

            if response.status_code >= 400:
                # Same parser as the in-body failure path
                # (_parse_error_field, code-then-type precedence) so an
                # identical backend error classifies identically
                # regardless of which transport path delivered it — the
                # sleep cycle's systemic detection depends on that.
                error_type: str | None = None
                message = response.text[:2000]
                try:
                    body = response.json()
                except ValueError:
                    body = None
                if isinstance(body, dict) and body.get("error") is not None:
                    err = body["error"]
                    message, error_type = _parse_error_field(err)
                    if isinstance(err, str) and body.get("message"):
                        # Gateway shape: {"error": "code", "message": "..."}
                        message = f"{err}: {body['message']}"
                raise ResponsesAPIError(
                    f"responses endpoint returned {response.status_code}: {message}",
                    status_code=response.status_code,
                    error_type=error_type,
                )

            try:
                body = response.json()
            except ValueError as exc:
                # A 2xx that isn't JSON (gateway maintenance page, proxy
                # error body) must land in the error taxonomy, not escape
                # as a raw parse error.
                raise ResponsesAPIError(
                    "responses endpoint returned a non-JSON success body: "
                    f"{response.text[:200]!r}",
                    status_code=response.status_code,
                ) from exc
            if not isinstance(body, dict):
                raise ResponsesAPIError(
                    "responses endpoint returned a non-object JSON body: "
                    f"{str(body)[:200]!r}",
                    status_code=response.status_code,
                )
            # Contract validation — a 2xx body must actually be a
            # Responses object. An in-body error (some gateways return
            # {"error": ...} with HTTP 200), a failed status, an unknown
            # status, or a missing/non-list output must all land in the
            # error taxonomy instead of becoming an empty end_turn.
            status = body.get("status")
            error_field = body.get("error")
            if status == "failed" or error_field is not None:
                # Any non-null error field makes the body a failure —
                # a "completed" status alongside an error is a
                # contradiction that must not pass as success.
                message, code = _parse_error_field(error_field)
                raise ResponsesAPIError(
                    f"response failed: {message}",
                    status_code=0,
                    error_type=code,
                )
            if status not in ("completed", "incomplete"):
                raise ResponsesAPIError(
                    "responses body does not match the Responses contract "
                    f"(status={status!r}): {str(body)[:200]!r}",
                    status_code=response.status_code,
                )
            if not isinstance(body.get("output"), list):
                raise ResponsesAPIError(
                    "responses body has a missing or non-list 'output' "
                    f"field: {str(body)[:200]!r}",
                    status_code=response.status_code,
                )
            return body

        raise ResponsesConnectionError(
            f"responses endpoint unreachable after {MAX_RETRIES + 1} attempts: {last_error}"
        )

    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: dict[str, Any] | None = None,
        context_id: str | None = None,
        task_id: str | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        sleep_cycle: bool = False,
    ) -> LLMResponse:
        effective_max_tokens = max_tokens if max_tokens is not None else self._max_tokens
        use_model = model or self._model
        payload = self._build_payload(
            system, messages, tools, output_schema, effective_max_tokens,
            use_model,
        )
        extra_headers: dict[str, str] = {}
        if context_id:
            extra_headers[CONTEXT_ID_HEADER] = context_id
        if task_id:
            extra_headers[TASK_ID_HEADER] = task_id
        if sleep_cycle:
            extra_headers[SLEEP_CYCLE_HEADER] = "true"

        with llm_complete_span(
            backend="openai-responses",
            model=use_model,
            max_tokens=effective_max_tokens,
            message_count=len(messages),
            tool_count=len(tools or []),
            has_output_schema=bool(output_schema),
            context_id=context_id,
            task_id=task_id,
            sleep_cycle=sleep_cycle,
        ) as span:
            start = time.monotonic()
            body = await self._post_with_retries(payload, extra_headers)
            duration = time.monotonic() - start

            incomplete_details = body.get("incomplete_details") or {}
            incomplete_reason = (
                incomplete_details.get("reason")
                if isinstance(incomplete_details, dict) else None
            )
            blocks, has_tool_call, has_refusal = output_to_blocks(
                body.get("output") or [],
                incomplete=body.get("status") == "incomplete",
            )
            # Replay-safety gate: a tool turn whose reasoning came back
            # without encrypted content cannot be replayed faithfully —
            # the next request would carry the function call without its
            # required reasoning item and reasoning models reject that
            # permanently. Fail the turn BEFORE any tool executes, rather
            # than let a side effect land in an unrecoverable
            # conversation. (Both api.openai.com and the lab's local
            # backends return encrypted_content; a gateway stripping it
            # is a configuration fault worth surfacing loudly.)
            if has_tool_call:
                unreplayable = [
                    b for b in blocks
                    if b.get("type") == "thinking"
                    and not is_replayable_reasoning(b)
                ]
                if unreplayable:
                    raise ResponsesAPIError(
                        "backend returned reasoning that cannot be "
                        "replayed (missing encrypted content and/or item "
                        "id) on a tool turn; stateless replay of the tool "
                        "call would fail — ensure the endpoint honours "
                        "include=[\"reasoning.encrypted_content\"] and "
                        "returns reasoning item ids",
                        error_type="missing_encrypted_reasoning",
                    )
            if not blocks:
                # Everything translatable was dropped (e.g. an incomplete
                # response whose only output was a cut-off function call)
                # or the model produced nothing. Empty assistant content
                # is invalid to journal and replay — surface an
                # explanatory text block instead.
                blocks = [{
                    "type": "text",
                    "text": (
                        "[the model produced no usable output "
                        f"(status={body.get('status')!r}"
                        + (f", reason={incomplete_reason!r}" if incomplete_reason else "")
                        + ")]"
                    ),
                }]
            stop_reason = map_stop_reason(
                body.get("status"),
                incomplete_reason,
                has_tool_call,
                has_refusal,
            )
            usage = extract_responses_usage(body.get("usage"))
            usage_total = record_llm_usage(usage, model=use_model, path="live")
            stamp_llm_completion(span, duration, stop_reason, usage_total)

            return LLMResponse(
                content=blocks,
                stop_reason=stop_reason,
                usage=usage,
                model=use_model,
            )

    async def stream(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("Streaming not yet implemented")
        yield  # pragma: no cover

    async def count_tokens(self, text: str) -> int:
        """Approximate token count.

        The Responses API has no token-counting endpoint, so this uses the
        same ``len(text) // 4`` heuristic as the mock backend. Its only
        in-repo consumer is memory-budget estimation, where an
        approximation is acceptable.
        """
        return len(text) // 4

    async def batch_create(
        self, requests: list[BatchRequest], model: str | None = None,
    ) -> list[str]:
        raise ResponsesBatchesUnsupportedError()

    async def batch_status(self, batch_id: str) -> BatchStatus:
        raise ResponsesBatchesUnsupportedError()

    async def batch_results(self, batch_id: str) -> list[BatchItemResult]:
        raise ResponsesBatchesUnsupportedError()

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
