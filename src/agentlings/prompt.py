from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentlings.config import AgentConfig

logger = logging.getLogger(__name__)


def build_system_prompt(config: AgentConfig) -> list[dict[str, Any]]:
    if config.agent_system_prompt_file:
        path = Path(config.agent_system_prompt_file)
        text = path.read_text(encoding="utf-8")
        logger.info("loaded system prompt from %s", path)
    else:
        text = _default_prompt(config)

    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _default_prompt(config: AgentConfig) -> str:
    return f"""\
You are {config.agent_name}, {config.agent_description}.

Use the tools available to you to accomplish tasks. When a tool returns output, \
incorporate it into your response.

Be concise. Use code blocks for command output. Respond in plain text unless \
the user asks for a specific format."""
