"""System prompt construction for the LLM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentlings.config import AgentConfig
from agentlings.core.memory_models import MemoryStore
from agentlings.core.skills import SkillRef, format_skills_block

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_INJECTION = """\
## Memory

The following is your long-term memory — facts you have learned over time:

{entries}"""

DEFAULT_DATA_DIR_AWARENESS = """\
## Data Directory

Your data directory is at {data_dir}. Layout:

- {data_dir}/memory/memory.json — your long-term memory (also provided above)
- {data_dir}/journals/YYYY-MM-DD.md — daily summaries of your past activity
- {data_dir}/<context_id>/journal.jsonl — parent journal for a conversation \
(merged user/assistant turns only)
- {data_dir}/<context_id>/tasks/<task_id>.jsonl — sub-journal for a single \
task's execution trace (every tool call, every response)

Every user and assistant message in the conversation above is prefixed with a \
``[task <id>]`` block indicating which task produced it. Use that id to inspect \
the corresponding sub-journal if you need to know what was actually done — the \
parent journal only carries final responses, never the tool-call trace.

## Inspecting Your Own Tasks

Sub-journals can be large — a long task may write hundreds of KB of tool output. \
Never ``cat`` a sub-journal. Read only what you need:

  Status check (cheap, always safe — last entry is the terminal marker):
      tail -n 5 {data_dir}/<context_id>/tasks/<task_id>.jsonl

    Look at the last entry's ``t`` field:
      task_done    → completed; ``final_response`` holds the answer
      task_fail    → failed; ``reason`` and ``error_details`` on the entry
      task_cancel  → cancelled; ``reason`` on the entry
      (no terminal) → still running or merge-back in progress

  Find what tool calls a task made (bounded output):
      jq -c 'select(.t=="msg" and .role=="assistant")' \
{data_dir}/<context_id>/tasks/<task_id>.jsonl | head -c 8000

  Find errors:
      jq -c 'select(.t=="task_fail")' \
{data_dir}/<context_id>/tasks/<task_id>.jsonl

Ignore entries with ``t`` in ``{{task_dispatch, merge_start, merge_commit}}`` — \
those are operational audit markers, not conversational content."""


def build_system_prompt(
    config: AgentConfig,
    memory: MemoryStore | None = None,
    data_dir: Path | None = None,
    injection_prompt: str | None = None,
    token_budget: int = 2000,
    skills: list[SkillRef] | None = None,
) -> list[dict[str, Any]]:
    """Build the system prompt blocks for the Anthropic Messages API.

    Args:
        config: The agent configuration.
        memory: Current memory store to inject. Omitted if ``None`` or empty.
        data_dir: Path to the agent's data directory for the awareness block.
        injection_prompt: Override for the memory injection template.
            Uses ``DEFAULT_MEMORY_INJECTION`` if not provided.
        skills: Discovered runtime skills (Open Skills spec). Prepended ahead
            of the identity block when non-empty so the agent sees them
            before any operator-defined instructions.

    Returns:
        A list of text blocks with ``cache_control`` set to ephemeral.
    """
    if config.system_prompt:
        text = config.system_prompt
    else:
        text = _default_prompt(config)

    blocks: list[dict[str, Any]] = []

    skills_block = format_skills_block(skills or [])
    if skills_block is not None:
        blocks.append({
            "type": "text",
            "text": skills_block,
            "cache_control": {"type": "ephemeral"},
        })

    blocks.append({
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    })

    if memory and memory.entries:
        template = injection_prompt or DEFAULT_MEMORY_INJECTION
        memory_block = _build_memory_block(template, memory.entries, token_budget)
        if memory_block:
            blocks.append({
                "type": "text",
                "text": memory_block,
                "cache_control": {"type": "ephemeral"},
            })

    if data_dir is not None and config.definition.data_dir_awareness:
        awareness = DEFAULT_DATA_DIR_AWARENESS.format(data_dir=data_dir)
        blocks.append({
            "type": "text",
            "text": awareness,
            "cache_control": {"type": "ephemeral"},
        })

    return blocks


CHARS_PER_TOKEN = 4


def _estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _build_memory_block(
    template: str,
    entries: list,
    token_budget: int,
) -> str | None:
    """Build the memory injection block, truncating entries to fit the token budget."""
    included = list(entries)
    while included:
        entries_text = "\n".join(f"- **{e.key}**: {e.value}" for e in included)
        block = template.format(entries=entries_text)
        if _estimate_tokens(block) <= token_budget:
            return block
        included.pop()
        logger.debug("memory block over budget, dropped to %d entries", len(included))
    return None


def _default_prompt(config: AgentConfig) -> str:
    return f"""\
You are {config.agent_name}, {config.agent_description}.

Use the tools available to you to accomplish tasks. When a tool returns output, \
incorporate it into your response.

Be concise. Use code blocks for command output. Respond in plain text unless \
the user asks for a specific format."""
