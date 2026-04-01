"""System prompt construction for the LLM."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentlings.config import AgentConfig
from agentlings.core.memory_models import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_INJECTION = """\
## Memory

The following is your long-term memory — facts you have learned over time:

{entries}"""

DEFAULT_DATA_DIR_AWARENESS = """\
## Data Directory

Your data directory is at {data_dir}. It contains:
- memory/memory.json: your long-term memory (also provided above)
- journals/YYYY-MM-DD.md: daily summaries of your past activity
- conversations as *.jsonl: raw conversation logs

You can read these files using your filesystem tools to recall past context \
that is not in your current memory. For example:
- List journals to see which days you were active
- Read a journal to recall what happened on a specific day
- Search across journals to find when an issue first appeared
- Read conversation logs for full detail on a past interaction"""


def build_system_prompt(
    config: AgentConfig,
    memory: MemoryStore | None = None,
    data_dir: Path | None = None,
    injection_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build the system prompt blocks for the Anthropic Messages API.

    Args:
        config: The agent configuration.
        memory: Current memory store to inject. Omitted if ``None`` or empty.
        data_dir: Path to the agent's data directory for the awareness block.
        injection_prompt: Override for the memory injection template.
            Uses ``DEFAULT_MEMORY_INJECTION`` if not provided.

    Returns:
        A list of text blocks with ``cache_control`` set to ephemeral.
    """
    if config.system_prompt:
        text = config.system_prompt
    else:
        text = _default_prompt(config)

    blocks = [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    if memory and memory.entries:
        template = injection_prompt or DEFAULT_MEMORY_INJECTION
        entries_text = "\n".join(
            f"- **{e.key}**: {e.value}" for e in memory.entries
        )
        memory_block = template.format(entries=entries_text)
        blocks.append({
            "type": "text",
            "text": memory_block,
            "cache_control": {"type": "ephemeral"},
        })

    if data_dir is not None:
        awareness = DEFAULT_DATA_DIR_AWARENESS.format(data_dir=data_dir)
        blocks.append({
            "type": "text",
            "text": awareness,
        })

    return blocks


def _default_prompt(config: AgentConfig) -> str:
    return f"""\
You are {config.agent_name}, {config.agent_description}.

Use the tools available to you to accomplish tasks. When a tool returns output, \
incorporate it into your response.

Be concise. Use code blocks for command output. Respond in plain text unless \
the user asks for a specific format."""
