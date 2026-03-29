<p align="center">
  <img src="logo.png" alt="Agentlings" width="256">
</p>

<h1 align="center">Agentlings</h1>

<p align="center">
  Lightweight single-process agent framework exposing both
  <a href="https://a2a-protocol.org">A2A</a> and
  <a href="https://modelcontextprotocol.io">MCP</a> on a single HTTP port.
</p>

---

Each agentling is a small, focused AI agent whose identity is defined by a YAML config file — name, description, system prompt, tools, and skills. The framework handles protocol compliance, conversation journaling, and context management. The LLM is the agent; the framework records and replays.

## Quick start

```bash
pip install -e ".[dev]"

# Create your agent definition
cp agent.example.yaml agent.yaml

# Run with mock LLM (no API key needed)
AGENT_CONFIG=./agent.yaml AGENT_LLM_BACKEND=mock AGENT_API_KEY=dev agentling

# Run with Anthropic
AGENT_CONFIG=./agent.yaml ANTHROPIC_API_KEY=sk-ant-... AGENT_API_KEY=your-key agentling

# See available tools
agentling --list-tools
```

The agent serves:
- `GET /.well-known/agent-card.json` — A2A Agent Card (public, no auth)
- `POST /a2a` — A2A JSON-RPC endpoint
- `POST /mcp` — MCP Streamable HTTP endpoint

## Agent definition

Agent identity lives in a YAML file (`agent.yaml`):

```yaml
name: k3s-agentling
description: A k3s cluster management agent

tools:
  - bash
  - filesystem

skills:
  - id: k8s-ops
    name: Kubernetes Operations
    description: Manage cluster resources, diagnose issues, apply manifests
    tags: [kubernetes, k3s, devops]
  - id: file-management
    name: File Management
    description: Read, write, and search configuration files
    tags: [files, yaml]

system_prompt: |
  You are a DevOps engineer managing a k3s Kubernetes cluster.

  All configuration changes go through /mnt/lab/k3s as the source of truth.
  Never use kubectl patch/edit/set directly — write manifests and apply them.

  Before any destructive operation, describe the impact and ask for confirmation.
```

Point to it with `AGENT_CONFIG=./agent.yaml`.

### Available tools

| Group | Tools | Description |
|-------|-------|-------------|
| `bash` | `bash` | Shell command execution with timeout |
| `filesystem` | `read_file`, `write_file`, `edit_file`, `list_directory`, `search_files` | File operations with offset/limit, find-and-replace, glob search |

Tools are off by default. Run `agentling --list-tools` for details.

## Docker

```bash
docker build -t agentling:latest .
docker run -e AGENT_API_KEY=your-key -e AGENT_LLM_BACKEND=mock -p 8420:8420 agentling
```

## Environment variables

Secrets and runtime settings stay in env vars (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_CONFIG` | — | Path to agent YAML definition |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (required for real LLM) |
| `AGENT_API_KEY` | — | API key for authenticating clients |
| `AGENT_MODEL` | `claude-sonnet-4-6` | Anthropic model ID |
| `AGENT_MAX_TOKENS` | `4096` | Max tokens per LLM response |
| `AGENT_HOST` | `0.0.0.0` | Bind address |
| `AGENT_PORT` | `8420` | Bind port |
| `AGENT_DATA_DIR` | `./data` | JSONL journal storage directory |
| `AGENT_LOG_LEVEL` | `INFO` | Log level |
| `AGENT_LLM_BACKEND` | `anthropic` | `anthropic` or `mock` |
| `AGENT_EXTERNAL_URL` | — | Public URL for Agent Card (needed in Docker/k8s) |

## Architecture

```mermaid
graph TB
    A2A[A2A Client] -->|POST /a2a| A2ASDK[a2a-sdk Server]
    MCP[MCP Client] -->|POST /mcp| MCPSDK[mcp SDK Server]
    A2ASDK --> Executor[AgentlingExecutor]
    Executor --> Loop[MessageLoop]
    MCPSDK --> Loop
    Loop --> Store[JSONL Store]
    Loop --> LLM[LLM Client]
    Loop --> Tools[Tool Registry]
```

Both protocols feed into a single `MessageLoop.process_message()` entrance. Conversations are persisted as append-only JSONL journals with compaction markers as replay cursors.

## Testing

```bash
# Unit tests (no network, no LLM)
pytest tests/unit/ -v

# Integration tests (starts real server with mock LLM)
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

Integration tests use native SDK clients — `a2a-sdk` `ClientFactory` for A2A and `mcp` `ClientSession` for MCP — talking to a real server over HTTP. All LLM responses are mocked.

## Built with

- [a2a-sdk](https://github.com/a2aproject/a2a-python) — A2A protocol server + client
- [mcp](https://github.com/modelcontextprotocol/python-sdk) — MCP protocol server + client
- [anthropic](https://github.com/anthropics/anthropic-sdk-python) — LLM backend
- [starlette](https://www.starlette.io) + [uvicorn](https://www.uvicorn.org) — HTTP server
