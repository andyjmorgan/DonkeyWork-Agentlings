"""Tests for the sleep cycle phases."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agentlings.config import AgentConfig, SleepConfig
from agentlings.core.llm import MockLLMClient
from agentlings.core.memory_models import MemoryStore
from agentlings.core.memory_store import MemoryFileStore
from agentlings.core.models import MessageEntry
from agentlings.core.sleep import SleepCycle
from agentlings.core.store import JournalStore


@pytest.fixture
def sleep_config(tmp_data_dir: Path, tmp_path: Path) -> AgentConfig:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        "name: sleep-test-agent\n"
        "description: A test agent for sleep\n"
        "tools:\n"
        "  - bash\n"
        "sleep:\n"
        "  schedule: '0 2 * * *'\n"
        "  conversation_retention_days: 7\n"
        "  journal_retention_days: 14\n"
        "  memory_max_entries: 10\n"
        "memory:\n"
        "  token_budget: 2000\n"
    )
    return AgentConfig(
        anthropic_api_key="test-key",
        agent_api_key="test-key",
        agent_data_dir=tmp_data_dir,
        agent_llm_backend="mock",
        agent_config=str(agent_yaml),
    )


@pytest.fixture
def sleep_deps(sleep_config: AgentConfig, tmp_data_dir: Path):
    store = JournalStore(tmp_data_dir)
    memory = MemoryFileStore(tmp_data_dir)
    llm = MockLLMClient(tool_names=[])
    cycle = SleepCycle(config=sleep_config, llm=llm, memory_store=memory, store=store)
    return cycle, store, memory, sleep_config


class TestLightSleep:
    def test_no_conversations_returns_empty(self, sleep_deps) -> None:
        cycle, _, _, _ = sleep_deps
        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == []

    def test_finds_yesterdays_conversations(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, store, _, _ = sleep_deps
        store.create("test-ctx")
        store.append("test-ctx", MessageEntry(
            ctx="test-ctx", role="user",
            content=[{"type": "text", "text": "hello"}],
        ))

        path = tmp_data_dir / "test-ctx.jsonl"
        yesterday = datetime.now(timezone.utc) - timedelta(hours=12)
        import os
        os.utime(path, (yesterday.timestamp(), yesterday.timestamp()))

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert len(result) == 1


class TestDeepSleep:
    async def test_writes_journal(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, store, _, _ = sleep_deps

        store.create("ctx-1")
        store.append("ctx-1", MessageEntry(
            ctx="ctx-1", role="user",
            content=[{"type": "text", "text": "hello agent"}],
        ))

        path = tmp_data_dir / "ctx-1.jsonl"
        yesterday = datetime.now(timezone.utc) - timedelta(hours=12)
        import os
        os.utime(path, (yesterday.timestamp(), yesterday.timestamp()))

        conversations = cycle._light_sleep(datetime.now(timezone.utc))
        review_date = datetime.now(timezone.utc) - timedelta(days=1)
        date_str = review_date.strftime("%Y-%m-%d")
        summaries, candidates = await cycle._deep_sleep(conversations, date_str)

        journal_path = tmp_data_dir / "journals" / f"{date_str}.md"
        assert journal_path.exists()
        assert len(summaries) > 0


class TestHousekeeping:
    def test_deletes_old_conversations(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, _, _, _ = sleep_deps

        old_file = tmp_data_dir / "old-ctx.jsonl"
        old_file.write_text('{"t":"msg"}\n')
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        import os
        os.utime(old_file, (old_time.timestamp(), old_time.timestamp()))

        cycle._housekeeping(datetime.now(timezone.utc))
        assert not old_file.exists()

    def test_keeps_recent_conversations(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, store, _, _ = sleep_deps
        store.create("recent-ctx")
        store.append("recent-ctx", MessageEntry(
            ctx="recent-ctx", role="user",
            content=[{"type": "text", "text": "recent"}],
        ))
        cycle._housekeeping(datetime.now(timezone.utc))
        assert (tmp_data_dir / "recent-ctx.jsonl").exists()

    def test_deletes_old_journals(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, _, _, _ = sleep_deps
        journals_dir = tmp_data_dir / "journals"
        journals_dir.mkdir()
        old_journal = journals_dir / "2020-01-01.md"
        old_journal.write_text("old journal")

        cycle._housekeeping(datetime.now(timezone.utc))
        assert not old_journal.exists()


class TestLightSleepTimeWindow:
    """Verify the lookback window catches conversations from the previous day.

    The sleep cycle typically fires at 02:00 UTC to review the previous day's
    work. Conversations from 09:00–18:00 UTC yesterday must be discovered,
    not just those from after midnight.
    """

    def test_02am_discovers_yesterdays_conversations(self, sleep_deps, tmp_data_dir: Path) -> None:
        """The exact scenario that caused the production bug: sleep at 02:00,
        conversations from yesterday afternoon are invisible."""
        cycle, store, _, _ = sleep_deps
        store.create("afternoon-ctx")
        store.append("afternoon-ctx", MessageEntry(
            ctx="afternoon-ctx", role="user",
            content=[{"type": "text", "text": "afternoon work"}],
        ))

        path = tmp_data_dir / "afternoon-ctx.jsonl"
        yesterday_3pm = datetime(2026, 4, 2, 15, 0, tzinfo=timezone.utc)
        import os
        os.utime(path, (yesterday_3pm.timestamp(), yesterday_3pm.timestamp()))

        sleep_time = datetime(2026, 4, 3, 2, 0, tzinfo=timezone.utc)
        result = cycle._light_sleep(sleep_time)
        assert len(result) == 1, "Yesterday afternoon's conversation was not discovered"

    def test_ignores_conversations_older_than_lookback(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, store, _, _ = sleep_deps
        store.create("old-ctx")
        store.append("old-ctx", MessageEntry(
            ctx="old-ctx", role="user",
            content=[{"type": "text", "text": "ancient history"}],
        ))

        path = tmp_data_dir / "old-ctx.jsonl"
        two_days_ago = datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc)
        import os
        os.utime(path, (two_days_ago.timestamp(), two_days_ago.timestamp()))

        sleep_time = datetime(2026, 4, 3, 2, 0, tzinfo=timezone.utc)
        result = cycle._light_sleep(sleep_time)
        assert len(result) == 0, "Conversation from 2 days ago should not be picked up"

    def test_ignores_active_conversations(self, sleep_deps, tmp_data_dir: Path) -> None:
        """Conversations modified within the grace period must be skipped —
        they may still be in-flight."""
        cycle, store, _, _ = sleep_deps
        store.create("active-ctx")
        store.append("active-ctx", MessageEntry(
            ctx="active-ctx", role="user",
            content=[{"type": "text", "text": "still talking"}],
        ))

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert len(result) == 0, "Active conversation should be excluded by grace period"


class TestFullCycle:
    async def test_skips_on_no_conversations(self, sleep_deps) -> None:
        cycle, _, memory, _ = sleep_deps
        await cycle.run()
        assert memory.list() == []
