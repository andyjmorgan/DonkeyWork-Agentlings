# Memory & Sleep Cycle

Design document for persistent memory and nightly sleep cycle features in Agentlings.

## Overview

Agentlings are currently stateless between conversations. Memory and sleep transform an agentling from a tool that forgets into an agent that learns. Memory provides a small, curated working context injected into every interaction. The sleep cycle is a nightly process that journals activity, consolidates new knowledge, prunes stale information, and performs retention cleanup.

## Memory

### Storage

Memory is a YAML file stored in the agent's data directory alongside JSONL conversation journals:

```
data/
  conversations/
    abc123.jsonl
    def456.jsonl
  journals/
    2026-04-01.md
    2026-04-02.md
  memory.json
```

### Format

```json
{
  "entries": [
    {
      "key": "cluster-node-count",
      "value": "4 nodes: node1 (control), node2-4 (workers)",
      "recorded": "2026-04-01T10:00:00Z"
    },
    {
      "key": "known-issue-coredns",
      "value": "CoreDNS restarts on node3, related to memory pressure",
      "recorded": "2026-03-28T14:30:00Z"
    },
    {
      "key": "storage-path",
      "value": "/mnt/lab/k3s is the source of truth for all manifests",
      "recorded": "2026-03-25T09:00:00Z"
    }
  ]
}
```

JSON for all runtime data, YAML for config. Conversations are JSONL, batch API responses are JSON, structured outputs return JSON. No serialization boundary to cross. The Pydantic models (see below) serialize directly to and from this file.

### Pydantic Models

All structured data exchanged with the LLM uses Pydantic models and Anthropic's structured outputs (`output_config` with `json_schema`). The API guarantees responses conform to the schema. No prompt-based JSON formatting, no parsing retries, no validation loops.

```python
from pydantic import BaseModel
from datetime import datetime


class MemoryEntry(BaseModel):
    key: str
    value: str
    recorded: datetime


class MemoryStore(BaseModel):
    entries: list[MemoryEntry]


class MemoryCandidate(BaseModel):
    key: str
    value: str


class ConversationSummary(BaseModel):
    summary: str
    memory_candidates: list[MemoryCandidate]


class ConsolidatedMemory(BaseModel):
    entries: list[MemoryEntry]
```

`MemoryStore` is the on-disk format (`memory.json`). `ConversationSummary` is the output of each deep sleep batch call. `ConsolidatedMemory` is the output of the REM consolidation call. The `memory_edit` tool reads and writes `MemoryStore` directly.

### Injection

Context is injected into every LLM call in this order:

1. System prompt (identity, personality, foundational knowledge)
2. Memory block (what the agent has learned)
3. Data directory context (where the agent's own files live)
4. Conversation replay (current interaction)

If memory is empty or the file does not exist, the memory block is omitted. The system prompt is the agent's day-one knowledge. Memory is what it learns over time. No overlap, no ambiguity.

### Data Directory Awareness

The agent is told about its own data directory so it can use existing filesystem tools to access past context beyond what is in memory. The following is appended after the memory block:

```
Your data directory is at {data_dir}. It contains:
- memory.json: your long-term memory (also provided above)
- journals/YYYY-MM-DD.md: daily summaries of your past activity
- conversations/*.jsonl: raw conversation logs

You can read these files using your filesystem tools to recall past context
that is not in your current memory. For example:
- List journals to see which days you were active
- Read a journal to recall what happened on a specific day
- Search across journals to find when an issue first appeared
- Read conversation logs for full detail on a past interaction
```

This requires no new tools. The agent already has `read_file`, `list_directory`, and `search_files` from the filesystem tool group. It just needs to know where to look. The journals and conversations become a searchable archive that extends memory without consuming token budget.

### Token Budget

Memory has a configurable hard token budget (default: 2000 tokens). The agent must stay within this budget. If it approaches the limit, it must decide what to drop before adding new facts. Token counting is performed before injection.

### Tool

The agent gets a `memory_edit` tool with three operations:

| Operation | Description |
|-----------|-------------|
| `set`     | Upsert an entry by key. Sets `recorded` to current timestamp. |
| `remove`  | Delete an entry by key. |
| `list`    | Return all current memory entries. |

The system prompt instructs the agent on what belongs in memory: operational knowledge, patterns, decisions made, environmental facts, known issues, things that changed. Memory is not a knowledge base. It is working context.

### Scope

Memory is per-agent, never shared. If two agents need to exchange knowledge, they do so over A2A or MCP. Shared memory creates coupling, versioning problems, and conflict resolution complexity. Clean protocol boundaries replace shared state.

### First Boot

If memory does not exist or is empty, there is no memory. The system prompt carries all foundational knowledge for day one.

## Sleep Cycle

### Configuration

The sleep cycle is configured in the agent YAML:

```yaml
sleep:
  schedule: "0 2 * * *"           # cron expression, default 2am
  journal_retention_days: 30       # how long to keep journal files
  conversation_retention_days: 14  # how long to keep JSONL files
  memory_max_entries: 50           # hard cap on memory entries
  model: null                      # override model for sleep (null = use agent default)
```

### Scheduling

An asyncio task in the main process evaluates the cron expression and fires the sleep cycle. Since agentlings are already long-running uvicorn processes, this fits naturally. No external scheduler required.

### Phases

The sleep cycle maps to biological sleep phases. Each phase is a distinct function, independently testable, with a clear single responsibility.

#### 1. Light Sleep -- Gate Check

Quick check: were there any conversations today? If not, skip all subsequent phases. No work, no spend. Memory stays as-is until the next active day gives the model something to evaluate against.

Also performs pre-flight validation: is the LLM reachable, is the data directory writable.

#### 2. Deep Sleep -- Replay and File

During biological deep sleep, the hippocampus replays experiences and files them from short-term into long-term storage. This phase does the same.

For each conversation from today:

1. Locate the last compaction marker in the JSONL (the "story so far")
2. Collect messages from the compaction marker forward
3. Submit a summary call to the LLM

All summary calls are submitted as a single **batch request** to the Anthropic Message Batches API. This runs at 50% cost and processes in parallel. The sleep cycle polls for completion with a configurable timeout (default: 30 minutes). If the batch fails or times out, skip to deep sleep housekeeping with whatever results are available.

Each summary call receives:

- The agent's system prompt (provides the lens for judging importance)
- Current memory (so it knows what the agent already knows)
- The conversation content (from last compaction marker forward)

The system prompt is cached across batch calls by the Anthropic API, so it is paid for once at full price and cached for the remainder of the batch.

Each call returns a `ConversationSummary` via Anthropic's structured outputs:

```python
response = client.messages.parse(
    model=sleep_model,
    max_tokens=4096,
    system=system_prompt,
    messages=[
        {"role": "user", "content": f"Current memory:\n{current_memory}\n\nConversation:\n{conversation}"}
    ],
    output_format=ConversationSummary,
)
summary = response.parsed_output  # typed ConversationSummary
```

The API guarantees the response conforms to the `ConversationSummary` schema. No JSON parsing, no validation, no retries.

Note: for batch requests, `output_config` is included in each individual request body within the batch. Results are parsed from the batch response JSONL using `ConversationSummary.model_validate_json()`. The `client.messages.parse()` example above illustrates the schema contract; the actual batch implementation uses the batch API with the same `output_config`.

**Prompt for per-conversation summary:**

```
You are performing a nightly review of a conversation that took place today.

Produce a concise summary of what happened: what was asked, what actions were
taken, what the outcome was, and anything left unresolved.

Extract any facts worth adding to your long-term memory. Only extract NEW facts
not already in your current memory. Focus on operational knowledge, patterns,
decisions, things that changed. Ignore passing context.

If the conversation was trivial or contained nothing new worth remembering,
return an empty memory_candidates list.
```

The prompt focuses purely on the cognitive task. All format concerns are handled by the `ConversationSummary` Pydantic model and `output_config`.

Once all batch results are collected, token-count the summaries. If they fit in a single context window, write the journal directly. If not, batch-reduce the summaries until they fit (this should be rare given compaction keeps individual conversations bounded).

Write the journal to `journals/YYYY-MM-DD.md`.

#### 3. REM -- Integrate and Prune

During biological REM sleep, the brain integrates newly filed memories with existing ones, forms associations, and weakens connections that are not reinforced. This phase does the same.

A single LLM call receives:

- Current memory (all existing entries)
- Today's journal (the consolidated narrative)
- All memory candidates extracted during deep sleep

The call outputs a `ConsolidatedMemory` via structured outputs:

```python
response = client.messages.parse(
    model=sleep_model,
    max_tokens=4096,
    system=agent_system_prompt,
    messages=[
        {"role": "user", "content": f"Current memory:\n{current_memory}\n\nToday's journal:\n{journal}\n\nNew candidates:\n{memory_candidates}"}
    ],
    output_format=ConsolidatedMemory,
)
new_memory = response.parsed_output  # typed ConsolidatedMemory
```

**Prompt for REM consolidation:**

```
You are performing nightly memory maintenance.

Your job:
1. Integrate new candidates that add genuine value. Deduplicate against existing entries.
2. Review every existing entry. Is it still relevant? Has it been superseded by
   something learned today? Would it help you do your job tomorrow?
3. Drop anything that is stale, redundant, or no longer operationally useful.
4. You have a hard limit of {memory_max_entries} entries.

Preserve the recorded timestamp for entries you keep unchanged.
Set recorded to the current date for new or modified entries.
```

The prompt focuses on judgment. The `ConsolidatedMemory` schema enforces the output structure.

The output is written atomically to `memory.json` (write to temp file, rename).

#### 4. Deep Sleep (Housekeeping) -- Retention Cleanup

Delete JSONL conversation files older than `conversation_retention_days`. Delete journal files older than `journal_retention_days`. This is simple filesystem work, no LLM calls.

```
data/conversations/ -- delete where file age > conversation_retention_days
data/journals/      -- delete where filename date > journal_retention_days
```

### Error Handling

Take what succeeds, skip what fails. If the batch returns partial results, journal and consolidate from what is available. If consolidation fails, memory stays unchanged. If housekeeping fails, files are retained an extra day. Telemetry and logging surface failures for iteration. No retries in v1.

### Quiet Days

If light sleep finds no conversations, the entire cycle is skipped. No LLM calls, no cost. Memory and journals are untouched.

### Concurrency

The sleep cycle runs at 2am. If a conversation is in progress (unlikely but possible), the sleep cycle skips that conversation. It processes only conversations that have been idle (no new messages) for a configurable grace period (default: 5 minutes). The memory file is written atomically via temp-file-and-rename.

## Observability

### OpenTelemetry Integration

The sleep cycle emits telemetry to an OpenTelemetry collector via HTTP or gRPC. The collector endpoint and protocol are configurable:

```yaml
telemetry:
  enabled: true
  endpoint: "http://otel-collector:4318"   # HTTP endpoint (default)
  protocol: "http"                          # "http" or "grpc"
  service_name: "agentling"
  insecure: true                            # disable TLS for local collectors
```

### Spans

The sleep cycle produces a hierarchical span tree:

```
agentling.sleep
  agentling.sleep.light_sleep
  agentling.sleep.deep_sleep
    agentling.sleep.deep_sleep.batch_submit
    agentling.sleep.deep_sleep.batch_poll
    agentling.sleep.deep_sleep.journal_write
  agentling.sleep.rem
    agentling.sleep.rem.consolidate
    agentling.sleep.rem.memory_write
  agentling.sleep.housekeeping
    agentling.sleep.housekeeping.conversations
    agentling.sleep.housekeeping.journals
```

### Span Attributes

All sleep spans include:

| Attribute                          | Description                                    |
|------------------------------------|------------------------------------------------|
| `agent.name`                       | Agent name from YAML config                    |
| `sleep.phase`                      | Current phase (light_sleep, deep_sleep, rem, housekeeping) |
| `sleep.date`                       | Date being processed (YYYY-MM-DD)              |

Phase-specific attributes:

**Light Sleep:**

| Attribute                          | Description                                    |
|------------------------------------|------------------------------------------------|
| `sleep.conversations_found`        | Number of conversations from today             |
| `sleep.skipped`                    | Whether the cycle was skipped (no work)        |

**Deep Sleep:**

| Attribute                          | Description                                    |
|------------------------------------|------------------------------------------------|
| `sleep.batch.request_count`        | Number of conversations in the batch           |
| `sleep.batch.id`                   | Anthropic batch ID                             |
| `sleep.batch.status`               | Final batch status                             |
| `sleep.batch.succeeded`            | Count of successful results                    |
| `sleep.batch.failed`               | Count of failed results                        |
| `sleep.batch.duration_ms`          | Time from submit to completion                 |
| `sleep.batch.model`                | Model used for batch calls                     |
| `sleep.journal.token_count`        | Token count of written journal                 |
| `sleep.memory_candidates.count`    | Total memory candidates extracted              |

**REM:**

| Attribute                          | Description                                    |
|------------------------------------|------------------------------------------------|
| `sleep.rem.model`                  | Model used for consolidation                   |
| `sleep.rem.input_token_count`      | Tokens in consolidation prompt                 |
| `sleep.rem.output_token_count`     | Tokens in consolidation response               |
| `sleep.rem.entries_before`         | Memory entry count before consolidation        |
| `sleep.rem.entries_after`          | Memory entry count after consolidation         |
| `sleep.rem.entries_added`          | New entries added                              |
| `sleep.rem.entries_pruned`         | Entries removed during pruning                 |
| `sleep.rem.entries_modified`       | Existing entries that were updated             |

**Housekeeping:**

| Attribute                          | Description                                    |
|------------------------------------|------------------------------------------------|
| `sleep.housekeeping.conversations_deleted` | Conversation files deleted             |
| `sleep.housekeeping.journals_deleted`      | Journal files deleted                  |
| `sleep.housekeeping.bytes_reclaimed`       | Disk space freed                       |

### Metrics

The following metrics are emitted as OpenTelemetry metrics:

| Metric                                    | Type      | Description                              |
|-------------------------------------------|-----------|------------------------------------------|
| `agentling.sleep.duration_seconds`        | Histogram | Total sleep cycle duration               |
| `agentling.sleep.phase.duration_seconds`  | Histogram | Per-phase duration (tagged by phase)     |
| `agentling.sleep.batch.cost_usd`          | Counter   | Estimated batch API cost                 |
| `agentling.sleep.rem.cost_usd`            | Counter   | Estimated consolidation call cost        |
| `agentling.sleep.memory.entry_count`      | Gauge     | Current memory entry count post-sleep    |
| `agentling.sleep.memory.token_count`      | Gauge     | Current memory token count post-sleep    |
| `agentling.sleep.conversations_processed` | Counter   | Conversations processed per cycle        |
| `agentling.sleep.errors`                  | Counter   | Errors per cycle (tagged by phase)       |
| `agentling.sleep.skipped`                 | Counter   | Cycles skipped due to no activity        |

### Events

Sleep phase transitions are logged as OpenTelemetry log events on the root span:

```
[SLEEP:LIGHT]        Checking for today's conversations
[SLEEP:LIGHT]        Found 7 conversations, proceeding
[SLEEP:DEEP]         Submitting batch of 7 summary requests
[SLEEP:DEEP]         Batch msgbatch_abc123 completed: 7 succeeded, 0 failed
[SLEEP:DEEP]         Journal written: journals/2026-04-01.md (847 tokens)
[SLEEP:REM]          Consolidating memory: 12 existing + 5 candidates
[SLEEP:REM]          Memory updated: 14 entries (3 added, 1 pruned, 0 modified)
[SLEEP:HOUSEKEEPING] Deleted 3 conversations, 0 journals, reclaimed 45KB
[SLEEP]              Cycle complete in 47.3s
```

### Memory Tool Telemetry

The `memory_edit` tool (used during normal conversations) also emits spans:

```
agentling.memory.set     -- attributes: key, value_token_count, total_entries, total_token_count
agentling.memory.remove  -- attributes: key, total_entries, total_token_count
agentling.memory.list    -- attributes: total_entries, total_token_count
```

This provides visibility into memory changes happening during the day, complementing the nightly sleep cycle telemetry.

## Agent YAML Changes

The agent YAML gains two new top-level sections:

```yaml
name: k3s-agentling
description: A k3s cluster management agent

tools:
  - bash
  - filesystem
  - memory                          # new: enables memory_edit tool

skills:
  - id: k8s-ops
    name: Kubernetes Operations
    description: Manage cluster resources, diagnose issues, apply manifests
    tags: [kubernetes, k3s, devops]

system_prompt: |
  You are a DevOps engineer managing a k3s Kubernetes cluster.
  ...

sleep:                               # new: sleep cycle configuration
  schedule: "0 2 * * *"
  journal_retention_days: 30
  conversation_retention_days: 14
  memory_max_entries: 50
  model: null                        # null = use agent's default model

telemetry:                           # new: OpenTelemetry configuration
  enabled: false
  endpoint: "http://localhost:4318"
  protocol: "http"
  service_name: "agentling"
  insecure: true

memory:                              # new: memory configuration
  token_budget: 2000
```

## Environment Variables

| Variable               | Default | Description                                |
|------------------------|---------|--------------------------------------------|
| `AGENT_OTEL_ENDPOINT`  | --      | OpenTelemetry collector endpoint           |
| `AGENT_OTEL_PROTOCOL`  | `http`  | Collector protocol (`http` or `grpc`)      |
| `AGENT_OTEL_INSECURE`  | `true`  | Disable TLS for collector connection       |

Environment variables override YAML config for secrets and deployment-specific values.

## Dependencies

| Package                         | Purpose                       |
|---------------------------------|-------------------------------|
| `opentelemetry-api`             | OTel tracing and metrics API  |
| `opentelemetry-sdk`             | OTel SDK implementation       |
| `opentelemetry-exporter-otlp`   | OTLP HTTP/gRPC exporter       |

No other new dependencies. Memory uses JSON (stdlib `json`). Pydantic is already a dependency (required by `a2a-sdk` and `pydantic-settings`). Structured outputs use `client.messages.parse()` from the existing `anthropic` SDK. Token counting uses the `anthropic` SDK's tokenizer. Scheduling uses `asyncio` (stdlib). File operations use `os`/`pathlib` (stdlib).

## CLI

Two new commands:

```bash
# Show current memory
agentling memory show

# Trigger sleep cycle manually (useful for testing)
agentling sleep --date 2026-04-01
```

## Summary

Memory gives the agent durable working context. Sleep gives it a nightly reflection cycle that journals, consolidates, and prunes. OpenTelemetry provides full visibility into both. The design adds three dependencies (all OpenTelemetry), uses the existing file-based storage paradigm, leverages Anthropic's structured outputs with Pydantic models for guaranteed schema compliance, and leans on the batch API for cost-efficient nightly processing.
