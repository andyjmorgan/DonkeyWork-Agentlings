# OpenAI Responses wire format

Status: implemented (0.18.0)

Agentlings speaks one internal dialect — Anthropic-shaped content blocks
(`text`, `tool_use`, `tool_result`, `thinking`) — everywhere: the completion
cycle, the task engine, JSONL journals, and merge-back all operate on that
shape. Until now the only live backend was the Anthropic Messages API (plus
Anthropic-compatible endpoints via `ANTHROPIC_BASE_URL`).

This document describes first-class support for the OpenAI Responses API as a
second selectable wire format. The internal dialect does not change; a new
`ResponsesLLMClient` translates it to and from Responses-API shapes at the
HTTP boundary.

## Goals and non-goals

Goals:

- A config-selectable wire format: `AGENT_WIRE_FORMAT=messages|responses`
  (default `messages`, which is the existing behaviour, unchanged).
- Full parity for the live completion path: text, tool calling both
  directions, structured output, usage accounting, reasoning, error handling,
  identity/correlation headers, telemetry.
- Stateless operation: the full input array is sent on every call. We never
  depend on `previous_response_id` server state, because local Ollama
  `/v1/responses` implementations are stateless.
- Works against api.openai.com and against OpenAI-compatible gateways/local
  backends (slipspace, Ollama `/v1/responses`) via `OPENAI_BASE_URL`.

Non-goals (explicitly out of scope):

- **Batches.** The deep-sleep phase of the nightly sleep cycle uses the
  Anthropic batches API. The Responses backend does not implement batching;
  instead it takes the same workaround path used for other batch-less
  backends: `sleep.batch: false` in the agent YAML runs the deep-sleep
  conversation summaries as sequential live `complete()` calls (structured
  output preserved) rather than a batch submission. The client advertises
  `supports_batches = False`, and `SleepCycle` degrades to the live-summary
  path automatically even when `sleep.batch` is left true (with an info
  log), so the sleep cycle keeps working end to end — only the batching
  transport is out of scope. The `sleep.model` override is honoured on both
  paths (via the per-call `model` parameter on `BaseLLMClient.complete`).
  `batch_create`/`batch_status`/`batch_results` raise
  `ResponsesBatchesUnsupportedError` if called directly.

  Live-summary failure semantics require no operator action: per-item
  failures (including 400-class request rejections — one poison
  conversation must not wedge the cycle) degrade, with a
  one-other-conversation probe declaring systemic when the first two
  conversations both fail with request rejections; **transient** systemic
  failures (connection/timeout, 429, 5xx and in-body equivalents) are
  retried with cycle-level patience before giving up; **hard** failures
  (auth, permissions, `model_not_found`) abort immediately. An abort with
  completed work in hand keeps that work and writes the journal; an abort
  with nothing completed raises loudly. The whole live pass runs under a
  cycle-level deadline mirroring the batch path's 7200s poll timeout;
  expiry keeps completed work. All-failed raises.

  Recovery is coverage-based: each daily journal's machine-owned
  first-line marker (`<!-- agentling-coverage: ctx ... -->`) records
  exactly which conversations it summarised — model-authored body text
  can never forge or destroy coverage — and same-day re-runs merge into
  the existing journal (coverage union, sections appended) rather than
  overwriting it. Once journalling has ever succeeded, light sleep
  reviews the whole retention window and selects precisely the
  conversations no journal covers (or that changed after being
  summarised), so partial cycles, multi-day gaps, and deadline expiries
  all self-heal: every conversation is either summarised into some
  journal or still inside the recovery window.
- Streaming. `stream()` raises `NotImplementedError`, exactly like the
  Anthropic client today.
- Server-side conversation state (`previous_response_id`, `store: true`).
- Built-in OpenAI tools (`web_search`, `code_interpreter`, …). Agentling
  tools are function tools only.

## Architecture

```
                        create_llm_client(backend, wire_format, ...)
                                 │
          ┌──────────────────────┼───────────────────────┐
          ▼                      ▼                       ▼
   MockLLMClient        AnthropicLLMClient       ResponsesLLMClient
   (backend=mock)       (wire_format=messages)   (wire_format=responses)
                        core/llm.py — UNCHANGED  core/llm_responses.py — NEW
```

- `ResponsesLLMClient` lives in a new module, `agentlings/core/llm_responses.py`,
  implementing the existing `BaseLLMClient` ABC. `core/llm.py`'s Anthropic and
  mock paths are untouched; only the factory gains a `wire_format` parameter
  and a lazily-imported branch.
- The client is built on `httpx.AsyncClient` (already a dependency). No
  OpenAI SDK dependency: the wire format is small, we control retries and the
  error taxonomy, and `httpx.MockTransport` gives hermetic unit tests.
- `POST {base_url}/v1/responses`, `Authorization: Bearer <key>`. If
  `base_url` already ends in `/v1`, the path is de-duplicated (mirrors how
  the Anthropic SDK treats `ANTHROPIC_BASE_URL`).

## Config surface

New `AgentConfig` settings (env vars):

| Env var             | Field               | Default    | Meaning |
|---------------------|---------------------|------------|---------|
| `AGENT_WIRE_FORMAT` | `agent_wire_format` | `messages` | `messages` = Anthropic Messages API (existing). `responses` = OpenAI Responses API. |
| `OPENAI_API_KEY`    | `openai_api_key`    | `""`       | Bearer key for the Responses endpoint. Reusing `ANTHROPIC_API_KEY` (a gateway serving both protocols behind one key) requires the **explicit `AGENT_SHARED_LLM_KEY=true` opt-in** plus a parseable non-OpenAI `OPENAI_BASE_URL` — credential routing is never inferred from the URL alone, and openai.com hosts (any form, trailing-dot normalised) are always refused. |
| `OPENAI_BASE_URL`   | `openai_base_url`   | `None`     | Endpoint override (gateway or Ollama). `None` = `https://api.openai.com`. Deliberately does *not* fall back to `ANTHROPIC_BASE_URL` — that points at a Messages-shaped endpoint. |

`AGENT_MODEL`, `AGENT_MAX_TOKENS`, the agent YAML, and everything else are
shared across wire formats — but `AGENT_WIRE_FORMAT=responses` requires
`AGENT_MODEL` to be set **explicitly** (the built-in default is a
Messages-format Claude model no Responses endpoint serves; the config fails
fast instead of sending it to api.openai.com). `agent_llm_backend=mock`
continues to select the mock client regardless of wire format, uses no
credentials, and is exempt from the model requirement.

## Request translation (internal → Responses)

| Internal (Anthropic-shaped)                | Responses request |
|--------------------------------------------|-------------------|
| `system` blocks `[{type: text, text}]`     | `instructions` (text blocks joined with blank lines) |
| user msg, string content                   | `{role: "user", content: "<str>"}` |
| user msg, `text` blocks                    | `{role: "user", content: [{type: "input_text", text}]}` |
| user msg, `tool_result` block              | top-level `{type: "function_call_output", call_id: tool_use_id, output: ...}` — one item per block, original order preserved. Text-only content becomes a plain string; content with non-text blocks becomes a parts array: Anthropic `image` blocks map to `input_image` (data URL or url), other structured blocks are serialised to JSON inside `input_text` with a warning (lossy but never a silent empty string). `is_error: true` prefixes the output with `[tool error] `. |
| assistant msg, `text` block                | `{role: "assistant", content: [{type: "output_text", text}]}` — verified against live traffic: api.openai.com accepts this shape and in fact **requires** `output_text`/`refusal` parts for assistant-role input content (it rejects `input_text` there with "Supported values are: 'output_text' and 'refusal'"); Ollama's `/v1/responses` accepts it too. The multi-turn integration test exercises this replay in an existing context. |
| assistant msg, `tool_use` block            | `{type: "function_call", call_id: <block id>, name, arguments: json.dumps(input)}` (+ `id` when the original item id was preserved, see below) |
| assistant msg, `thinking` block            | `{type: "reasoning", id: <item_id>, summary: [...], encrypted_content: <signature>}` — replayed **only when** the block carries both a preserved `item_id` and a non-empty `signature` (encrypted content): with `store: false` the server cannot recover reasoning state from an id alone, so blocks from backends that never return `encrypted_content` are not replayed (safe degradation instead of a 400 on the next turn). |
| tool schema `{name, description, input_schema}` | `{type: "function", name, description, parameters: input_schema}` |
| `max_tokens`                               | `max_output_tokens` |
| `output_schema`                            | `text: {format: {type: "json_schema", name: "output", schema, strict: false}}` (`strict: true` demands `additionalProperties: false` + all-required on every level; agentling callers pass plain JSON Schemas, so we stay permissive) |
| —                                          | `store: false` on every request (stateless), plus `include: ["reasoning.encrypted_content"]` so OpenAI reasoning items survive round-trips |

## Response translation (Responses → internal)

| Responses output item                      | Internal block |
|--------------------------------------------|----------------|
| `message` → `output_text` content          | `{type: "text", text}` |
| `message` → `refusal` content              | `{type: "text", text: <refusal>}` and `stop_reason = "refusal"` |
| `function_call`                            | `{type: "tool_use", id: call_id, name, input: json.loads(arguments)}` + `item_id: <fc id>` preserved for replay. Only on **completed** responses, and only when the call is fully valid: non-empty `call_id` (never fabricated from the item id), non-empty `name`, and a string `arguments` field parsing to a JSON object (`"{}"` is the only valid empty form — absent/empty/non-string arguments raise). Any violation raises `ResponsesAPIError` before a `tool_use` block exists to execute. |
| `reasoning`                                | `{type: "thinking", thinking: <joined summary text>, signature: <encrypted_content or "">, item_id: <rs id or "">}` — `item_id` is stamped unconditionally (empty when the backend gave no id) as the responses-origin marker for the messages-path sanitiser. On a **tool turn**, reasoning that came back without encrypted content fails the turn (`missing_encrypted_reasoning`) *before any tool executes* — a tool call whose reasoning cannot be replayed would succeed once and then 400 on every later turn. On **incomplete** responses, reasoning items are dropped along with the function calls (an orphaned reasoning item without its required following item poisons replay permanently). |

The extra `item_id` key on `tool_use`/`thinking` blocks is what makes
stateless replay faithful: OpenAI reasoning models 400 when a `function_call`
is replayed without its sibling `reasoning` item, so both keep their original
item identity through the journal and back out. Anthropic-shaped consumers
(completion loop, task engine, journals) ignore unknown keys, so this is
invisible to the rest of the system. On the messages path, the Anthropic
client sanitises these artifacts at its own request boundary: `item_id` is
stripped from `tool_use` blocks, and foreign `thinking` blocks (identified
by `item_id`) are dropped entirely — their `signature` is OpenAI
`encrypted_content`, not an Anthropic thinking signature.

**Tool-call invariant:** a `tool_use` block is returned iff `stop_reason`
is `tool_use`, which requires `status: completed` and cleanly-parsed
arguments. Every function call on an *incomplete* response — truncated,
content-filtered, or unknown reason — is dropped (with a warning): a
journaled assistant tool call that never gets an output would 400 every
later replay and could be mistaken by crash recovery for a pending
executable call.

Stop-reason mapping:

| Responses signal                                | `stop_reason` |
|-------------------------------------------------|---------------|
| `status: completed` + `function_call` item(s)   | `tool_use` |
| `status: completed`, no function call           | `end_turn` |
| `status: incomplete`, reason `max_output_tokens`| `max_tokens` (function calls dropped) |
| `status: incomplete`, reason `content_filter`   | `refusal` (function calls dropped) |
| `status: incomplete`, other/unknown reason      | `end_turn` (logged; function calls dropped) |
| `status: failed`                                | raises `ResponsesAPIError` with the response error |
| any other/missing `status`, or missing `output` | raises `ResponsesAPIError` (contract violation — a bare `{"error": ...}` in a 200 body is surfaced as its error, never an empty success) |

## Usage / token accounting

Anthropic semantics are the internal contract (`LLMResponse.usage`,
`record_llm_usage`, `CompletionResult.token_usage`): `input_tokens` counts
*uncached* input. OpenAI's `input_tokens` *includes* cached tokens, so:

| Internal field                | Responses source |
|-------------------------------|------------------|
| `input_tokens`                | `usage.input_tokens - usage.input_tokens_details.cached_tokens` (floor 0) |
| `output_tokens`               | `usage.output_tokens` |
| `cache_read_input_tokens`     | `usage.input_tokens_details.cached_tokens` |
| `cache_creation_input_tokens` | `0` (no analogue — OpenAI caching is implicit and unbilled as a separate class) |

Missing/partial usage objects (some local backends) degrade to zeros, same as
the Anthropic client's `_extract_usage`. Telemetry spans mirror the Anthropic
client's attributes with `llm.backend = "openai-responses"`.

`count_tokens` has no Responses endpoint; the client uses the same
`len(text) // 4` heuristic as the mock backend (its only in-repo consumer is
memory-budget estimation, where a heuristic is acceptable). Documented on the
method.

## Thinking / reasoning mapping

`ThinkingConfig` is Anthropic-flavoured; the Responses client maps it onto
the `reasoning` request parameter:

| ThinkingConfig                   | Responses request |
|----------------------------------|-------------------|
| `mode: off` / absent             | no `reasoning` param (model default applies) |
| `mode: adaptive`, no effort      | no `reasoning` param |
| `mode: adaptive`, `effort: low/medium/high` | `reasoning: {effort: <same>}` |
| `mode: adaptive`, `effort: xhigh/max`       | `reasoning: {effort: "high"}` (clamped — OpenAI's scale tops out at high) |
| `mode: adaptive`, `display: summarized`     | adds `reasoning: {summary: "auto"}` |
| `mode: budget`                   | not representable; one warning at client construction, no `reasoning` param sent |

## Error taxonomy

The Anthropic path surfaces `anthropic.APIStatusError` and friends; the task
engine treats any exception from `llm.complete` as a task failure. The
Responses client raises a small mirrored hierarchy from
`agentlings.core.llm_responses`:

| Condition                          | Exception |
|------------------------------------|-----------|
| HTTP 4xx (except 429)              | `ResponsesAPIError(status_code, error_type, message)` — no retry |
| HTTP 429, or any 500-599           | retried twice with backoff (Anthropic SDK default is 2 retries), then `ResponsesAPIError`. The whole 5xx range is retryable — gateways emit non-standard codes (520, 529) for transient conditions; nothing is excluded |
| network/timeout                    | retried twice, then `ResponsesConnectionError` (also a `ConnectionError`) |
| `status: "failed"` in a 200 body   | `ResponsesAPIError` carrying `response.error` (dict or plain-string error fields both handled) |
| 200 body violating the contract (non-JSON, non-object, unknown/missing `status`, missing `output`, or **any non-null `error` field** — even alongside `status: completed`) | `ResponsesAPIError` |
| model emits `function_call` with non-object arguments on a completed response | `ResponsesAPIError` (`invalid_function_call_arguments`) |
| `batch_*` called                   | `ResponsesBatchesUnsupportedError` (subclass of `NotImplementedError`) |

All inherit `ResponsesError(Exception)` so callers can catch the family.

Retries are not idempotent at the protocol level — the Responses API has no
idempotency-key mechanism, so a timeout after the server accepted (and
billed) a request re-issues the POST, and only the final completion shows
up in usage accounting. As a best-effort dedup hook, every attempt of one
logical request carries the same `x-agentling-request-id` header, which
gateways can use to collapse duplicates.

`OPENAI_BASE_URL` accepts a bare host (`https://gw.example`), a versioned
base (`.../v1`), or the full endpoint URL (`.../v1/responses`) — all three
resolve to the same endpoint without path doubling.

In the sleep cycle's live-summary path, failures classify as **hard**
(401/403, `invalid_api_key`, `authentication_error`, permission errors,
`model_not_found`, `invalid_request_error` — retrying is pointless,
immediate abort) or **transient** (connection/timeout, 429, 5xx,
`server_error`, rate-limit variants, `insufficient_quota` — retried with
cycle-level patience before aborting). Aborts never discard completed
summaries, and aborted nights are recovered automatically by the next
cycle's widened review window (journal-gap detection), so content is only
lost if an outage outlasts `conversation_retention_days`.

`map_stop_reason` treats `length` and `max_tokens` as truncation synonyms
of `max_output_tokens` (compatible backends reuse chat-completions
vocabulary). Turns whose entire output was dropped keep an explanatory
text block — empty assistant content is invalid to journal or replay.

## Identity and correlation headers

Same header names, same semantics as the Anthropic client:
`x-agentling-name` baked in as a default header at construction;
`x-agentling-context-id` / `x-agentling-task-id` per request. Gateways
(slipspace) correlate on these regardless of protocol.

## Test plan

Unit (`tests/unit/test_llm_responses.py` + additions to `test_config.py` /
`test_sleep.py`): every Anthropic-backend unit test has a Responses mirror —

- request translation: system→instructions, each message/block kind, tool
  schemas, tool_result ordering + `is_error` prefix, thinking replay rules
  (with/without `item_id`), `output_schema`, `max_tokens` override,
  `store: false` + `include` always present;
- response translation: text, function_call (incl. malformed arguments),
  reasoning, refusal, every stop-reason row above, `failed` raising;
- usage mapping incl. cached-token subtraction and absent-usage degradation;
- headers (context/task per-request, name as default, none when absent);
- factory: wire-format selection, base-url wiring, key fallback;
- thinking mapping table above, budget-mode warning;
- error taxonomy via `httpx.MockTransport`: 400 no-retry, 429/5xx retry then
  raise, connection error, in-body `failed`;
- batches: all three methods raise, `supports_batches` flags, and the
  sleep-cycle live-summary path (`sleep.batch: false` uses live calls;
  batch-less backends degrade automatically; default keeps the batch path;
  one failed summary degrades instead of aborting; `sleep.model` warning);
- config: `AGENT_WIRE_FORMAT` validation + new env vars + `sleep.batch`.

Integration (`tests/integration/test_responses.py`, marker `responses`,
mirroring `test_ollama.py`): gated on `RESPONSES_API_KEY` being set
(`RESPONSES_BASE_URL` overrides the gateway URL); credentials come from env
only (mirroring `test_live_api.py`'s convention — never committed). Two
model profiles through the slipspace gateway:

- a real OpenAI model (`RESPONSES_OPENAI_MODEL`, default `gpt-5-mini`);
- a local Ollama responses target (`RESPONSES_LOCAL_MODEL`, default
  `muse-glimmer:latest` — a reasoning model, so the local profile also
  exercises reasoning-item translation).

Coverage: direct-client completion for both models; a full stateless
tool-call round trip (function_call → tool_result replay → final text) on
the OpenAI profile; and the complete task-engine path (A2A message → spawn →
tool call → merge-back) against a live agentling booted with
`AGENT_WIRE_FORMAT=responses` for both profiles (tools only on the OpenAI
profile — small local models are not reliably tool-capable, same reasoning
as the Ollama suite).
