# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Agentlings — a lightweight single-process Python framework for running AI agents that expose both A2A (Agent-to-Agent Protocol v1.0) and MCP (Model Context Protocol) on a single HTTP port. Each agentling is a small, focused agent whose identity is defined by configuration (name, description, system prompt, tools), not code.

## Architecture

Two protocol interfaces (A2A JSON-RPC at `/a2a`, MCP Streamable HTTP at `/mcp`) feed into a shared **task engine**. Every request becomes a task with its own sub-journal; the HTTP handler awaits up to `AGENT_TASK_AWAIT_SECONDS` (default 60) and either returns the final result inline (fast path) or yields a task handle the caller polls (slow path). FIFO ordering is preserved per context — only one task can run per context at a time, so concurrent requests are rejected with `context_busy` for the LLM to retry.

Data layout:

- Parent journal: `{data_dir}/{contextId}/journal.jsonl` (conversation history)
- Sub-journal:    `{data_dir}/{contextId}/tasks/{taskId}.jsonl` (task execution trace)
- Legacy `{data_dir}/{contextId}.jsonl` files are auto-migrated on first access.

Package structure under `src/agentlings/`:
- `config.py` — Pydantic BaseSettings, all config via env vars + `.env`
- `log.py` — console-first logging setup
- `server.py` — Starlette app wiring A2A + MCP routes, API key middleware, startup crash-recovery
- `core/` — task engine
  - `task.py` — `TaskEngine`, `TaskRegistry`, `TaskWorker`, merge-back, crash recovery
  - `loop.py` — synchronous facade over `TaskEngine` for legacy callers
  - `llm.py` — LLM client abstraction (Anthropic + mock backends)
  - `store.py` — JSONL append/replay with compaction cursor + `TaskJournal` (sub-journal)
  - `models.py` — journal entry types (messages, compaction, task markers, merge wrappers)
  - `prompt.py` — system prompt builder
  - `completion.py` — shared LLM completion cycle with per-turn callback + cancellation
- `protocol/` — protocol adapters
  - `a2a.py` — `AgentlingExecutor` bridging A2A SDK to the task engine
  - `mcp.py` — MCP server exposing a single task-aware tool
  - `agent_card.py` — Agent Card generation from config
- `tools/` — pluggable tool system
  - `registry.py` — `ToolRegistry` with group-based registration
  - `builtins.py` — bash and filesystem tool implementations

The Agent Card at `/.well-known/agent.json` is the single source of truth — the MCP tool schema is derived from it.

### Task engine lifecycle

1. **Ingress**: `engine.spawn(message, contextId?)` atomically admits one task per context, writes a `TaskDispatched` audit marker to the parent journal, creates a sub-journal with `TaskStarted`, and spawns a worker.
2. **Execution**: the worker snapshots the parent journal (replay from latest compaction cursor), runs the LLM loop via `compact_20260112`, writes each turn to the sub-journal.
3. **Merge-back (success)**: under the per-context lock, the engine atomically writes `MergeStarted` → user message → latest sub-journal compaction (if any) → assistant final response → `MergeCommitted` to the parent journal.
4. **Cancel/fail**: only a `TaskCancelled` or `TaskFailed` audit marker lands on the parent. No conversational content leaks.
5. **Replay**: audit markers (`task_dispatch`, `task_cancel`, `task_fail`, `merge_start`, `merge_commit`) are stripped by the store's `replay()` — they never reach the LLM.
6. **Crash recovery**: on startup, orphaned sub-journals get `TaskFailed { reason: "process_crash_recovery" }` and partial merge-backs are completed idempotently.

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run the agent
agentling

# Run with overrides
AGENT_NAME="my-agent" AGENT_PORT=9000 agentling

# Build container
docker build -t agentling:latest .

# Run container
docker run -e ANTHROPIC_API_KEY=... -e AGENT_API_KEY=... -v ./data:/data -p 8420:8420 agentling:latest
```

## Configuration

All via environment variables (loaded from `.env` via python-dotenv):
- `ANTHROPIC_API_KEY` (required when talking to api.anthropic.com; optional when `ANTHROPIC_BASE_URL` points at a backend that ignores it, e.g. Ollama)
- `ANTHROPIC_BASE_URL` (optional) — override the Messages endpoint. Set to `http://localhost:11434` to talk to Ollama's Anthropic-compatible API; combine with `AGENT_MODEL` set to an Ollama-served model (e.g. `qwen3-coder`) and `sleep.enabled: false` in the agent YAML since Ollama doesn't implement the batches API
- `AGENT_API_KEY` (required) — checked via `X-API-Key` header
- `AGENT_MODEL` (default `claude-sonnet-4-6`)
- `AGENT_MAX_TOKENS` (default `4096`)
- `AGENT_HOST` / `AGENT_PORT` (default `0.0.0.0:8420`)
- `AGENT_DATA_DIR` (default `./data`) — JSONL journal storage
- `AGENT_NAME` / `AGENT_DESCRIPTION` — identity for Agent Card + MCP tool
- `AGENT_SYSTEM_PROMPT_FILE` — optional path to override default system prompt
- `AGENT_LOG_LEVEL` (default `INFO`)
- `AGENT_TASK_AWAIT_SECONDS` (default `60`) — HTTP handler await timeout before yielding a task handle to the caller

## Logging

Console-only, format: `datetime - level - path - message`. All modules use `logging.getLogger(__name__)`. No third-party logging libraries.

## Design Constraints

- **Task-enabled, message-shape.** Every request becomes a task with its own sub-journal. Protocol surfaces expose task handles for long-running work, but A2A/MCP interactions are still carried as messages — no artifacts, no push notifications, no streaming.
- **One task per context.** FIFO is architectural, not configurable. Concurrent requests to a busy context are rejected with `context_busy`; the LLM retries.
- **Parent journal records only successful completions.** Cancel/fail leave only audit markers, never conversational content. Replay strips audit markers unconditionally.
- **contextId is the conversation handle; taskId is the execution handle.** ContextId maps to a directory; taskId is UUID4 and path-bound via `{contextId}/tasks/{taskId}.jsonl` — the filesystem enforces `(contextId, taskId)` pairing.
- **Confirmation is conversation** — no `input-required` state, just natural language turns.
- **The LLM is the agent** — the framework records and replays, it does not orchestrate.
- **Tools are pluggable** — register at startup, errors returned to LLM as tool results (not exceptions).
- **JSONL is append-only**, never modified or deleted; compaction markers are replay cursors, task compactions are ported to parent on merge-back.
- **Merge-back is atomic** via `MergeStarted`/`MergeCommitted` wrappers so startup recovery can repair partial writes idempotently.
- **MCP Session ID and contextId are independent concepts**, never cross-referenced.

## Development Methodology

**Write → Test → Continue.** Every piece of code gets tested before moving on.

- Bug workflow: write a failing test that reproduces the bug → verify it fails → fix the code → verify the test passes. No exceptions.
- Each task ends with tests that prove the task works.
- Each milestone ends with rigorous integration testing.
- Tests run in Docker via `docker compose` — the test harness spins up the agentling container and runs an MCP client and A2A client against it.
- Unit tests (pytest) for internal modules; integration tests for protocol-level behaviour.
- All LLM responses are mocked — tests never call the real Anthropic API or require a real key. The agentling container in integration tests runs with a mock LLM backend.

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests (builds + runs container)
docker compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Run all tests
pytest tests/unit/ && docker compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

## Dependencies

- `anthropic` — LLM backend with `compact_20260112`
- `starlette` + `uvicorn` — HTTP server
- `pydantic` — config and models
- `mcp` — MCP SDK (Streamable HTTP)
- `python-dotenv` — env file loading
