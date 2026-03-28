# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Agentlings — a lightweight single-process Python framework for running AI agents that expose both A2A (Agent-to-Agent Protocol v1.0) and MCP (Model Context Protocol) on a single HTTP port. Each agentling is a small, focused agent whose identity is defined by configuration (name, description, system prompt, tools), not code.

## Architecture

Two protocol interfaces (A2A JSON-RPC at `/a2a`, MCP Streamable HTTP at `/mcp`) feed into a shared message loop. The loop appends to a per-context JSONL journal, replays from the last compaction marker, calls the Anthropic Messages API with `compact_20260112`, and executes tools in a loop until the LLM produces a terminal text response.

Key modules under `src/agentlings/`:
- `config.py` — Pydantic BaseSettings, all config via env vars + `.env`
- `models.py` — A2A types, JSON-RPC envelope, JSONL journal entries, Agent Card
- `store.py` — JSONL append/replay with compaction cursor
- `loop.py` — single `process_message(text, context_id, stream)` entrance
- `llm.py` — Anthropic API wrapper (sync + streaming), compaction extraction
- `tools.py` — pluggable tool registry, built-in defaults: `shell`, `read_file`, `write_file`
- `a2a_handler.py` — JSON-RPC dispatch for `message/send` and `message/stream`
- `mcp_handler.py` — MCP Streamable HTTP via `mcp` SDK, single tool derived from Agent Card
- `server.py` — Starlette app, route mounting, API key auth middleware

The Agent Card at `/.well-known/agent.json` is the single source of truth — the MCP tool schema is derived from it.

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
- `ANTHROPIC_API_KEY` (required)
- `AGENT_API_KEY` (required) — checked via `X-API-Key` header
- `AGENT_MODEL` (default `claude-sonnet-4-6`)
- `AGENT_MAX_TOKENS` (default `4096`)
- `AGENT_HOST` / `AGENT_PORT` (default `0.0.0.0:8420`)
- `AGENT_DATA_DIR` (default `./data`) — JSONL journal storage
- `AGENT_NAME` / `AGENT_DESCRIPTION` — identity for Agent Card + MCP tool
- `AGENT_SYSTEM_PROMPT_FILE` — optional path to override default system prompt
- `AGENT_LOG_LEVEL` (default `INFO`)

## Logging

Console-only, format: `datetime - level - path - message`. All modules use `logging.getLogger(__name__)`. No third-party logging libraries.

## Design Constraints

- Message-only A2A agent — no tasks, no artifacts, no push notifications
- contextId is the only conversation handle; server-generated, JSONL-backed
- Confirmation is conversation — no `input-required` state, just natural language turns
- The LLM is the agent — the framework records and replays, it does not orchestrate
- Tools are pluggable — register at startup, errors returned to LLM as tool results (not exceptions)
- JSONL is append-only, never modified or deleted; compaction markers are replay cursors
- MCP Session ID and contextId are independent concepts, never cross-referenced

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
