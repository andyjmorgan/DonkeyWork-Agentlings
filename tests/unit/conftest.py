from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.config import AgentConfig
from agentlings.models import CompactionEntry, MessageEntry


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def test_config(tmp_data_dir: Path) -> AgentConfig:
    return AgentConfig(
        anthropic_api_key="test-key",
        agent_api_key="test-agent-key",
        agent_data_dir=tmp_data_dir,
        agent_llm_backend="mock",
        agent_name="test-agent",
        agent_description="A test agent",
    )


@pytest.fixture
def sample_message_entry() -> MessageEntry:
    return MessageEntry(
        ctx="ctx-123",
        role="user",
        content=[{"type": "text", "text": "hello"}],
        via="a2a",
    )


@pytest.fixture
def sample_compaction_entry() -> CompactionEntry:
    return CompactionEntry(
        ctx="ctx-123",
        content="Summary of prior conversation.",
    )
