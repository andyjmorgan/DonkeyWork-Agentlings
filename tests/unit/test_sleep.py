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

        path = tmp_data_dir / "test-ctx" / "journal.jsonl"
        yesterday = datetime.now(timezone.utc) - timedelta(hours=12)
        import os
        os.utime(path, (yesterday.timestamp(), yesterday.timestamp()))

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert len(result) == 1


class TestMissedNightRecovery:
    """Failed nights are re-entered automatically: a gap since the last
    daily journal widens the review window, so an aborted cycle's
    conversations are recovered on the next run with no operator action.
    """

    def _seed_old_conversation(self, store, tmp_data_dir: Path, days_ago: float) -> None:
        import os

        store.create("old-ctx")
        store.append("old-ctx", MessageEntry(
            ctx="old-ctx", role="user",
            content=[{"type": "text", "text": "from a missed night"}],
        ))
        then = datetime.now(timezone.utc) - timedelta(days=days_ago)
        path = tmp_data_dir / "old-ctx" / "journal.jsonl"
        os.utime(path, (then.timestamp(), then.timestamp()))

    def test_journal_gap_widens_window(self, sleep_deps, tmp_data_dir: Path) -> None:
        """A conversation from 3 days ago is outside the default window,
        but with the last journal 4 days old it must be picked up."""
        cycle, store, _, _ = sleep_deps
        self._seed_old_conversation(store, tmp_data_dir, days_ago=3)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        last = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
        (journals / f"{last}.md").write_text("# old journal\n")

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == ["old-ctx"]

    def test_covered_conversation_excluded(self, sleep_deps, tmp_data_dir: Path) -> None:
        """A conversation listed in a journal's coverage marker (written
        after its last modification) stays excluded."""
        cycle, store, _, _ = sleep_deps
        self._seed_old_conversation(store, tmp_data_dir, days_ago=3)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        (journals / f"{yesterday}.md").write_text(
            "<!-- agentling-coverage: old-ctx -->\n# fresh journal\n"
        )

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == []

    def test_uncovered_old_conversation_recovered(
        self, sleep_deps, tmp_data_dir: Path,
    ) -> None:
        """Selection spans the whole retention window: a never-covered
        conversation older than the latest journal is still recovered —
        a recent partial journal must not clip it out (multi-day-gap
        partial-failure scenario)."""
        cycle, store, _, _ = sleep_deps
        self._seed_old_conversation(store, tmp_data_dir, days_ago=3)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        # A RECENT journal exists (partial night) covering something else.
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        (journals / f"{yesterday}.md").write_text(
            "<!-- agentling-coverage: some-other-ctx -->\n# partial journal\n"
        )

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == ["old-ctx"], (
            "an uncovered conversation within retention must re-enter "
            "even when newer journals exist"
        )

    def test_forged_heading_in_body_cannot_cover(
        self, sleep_deps, tmp_data_dir: Path,
    ) -> None:
        """Model-authored body text ('### old-ctx' at column 0, or a fake
        coverage comment) must not create phantom coverage — only the
        machine-owned first line counts."""
        cycle, store, _, _ = sleep_deps
        self._seed_old_conversation(store, tmp_data_dir, days_ago=3)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        (journals / f"{yesterday}.md").write_text(
            "<!-- agentling-coverage: some-other-ctx -->\n"
            "# Journal\n\n"
            "### some-other-ctx\n"
            "The user mentioned context ids in chat:\n"
            "### old-ctx\n"
            "<!-- agentling-coverage: old-ctx -->\n"
        )

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == ["old-ctx"], "body text must never forge coverage"

    def test_no_journals_still_recovers_within_retention(
        self, sleep_deps, tmp_data_dir: Path,
    ) -> None:
        """No journals at all is indistinguishable from 'the first cycle
        failed systemically before writing anything' — the retention-wide
        window applies from day one so that failed day is retried
        (Codex round-6 finding 3)."""
        cycle, store, _, _ = sleep_deps
        self._seed_old_conversation(store, tmp_data_dir, days_ago=3)

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == ["old-ctx"]

    def _seed_named(self, store, tmp_data_dir: Path, ctx_id: str, mtime: datetime) -> None:
        import os

        store.create(ctx_id)
        store.append(ctx_id, MessageEntry(
            ctx=ctx_id, role="user",
            content=[{"type": "text", "text": f"in {ctx_id}"}],
        ))
        path = tmp_data_dir / ctx_id / "journal.jsonl"
        os.utime(path, (mtime.timestamp(), mtime.timestamp()))

    def test_off_by_one_boundary_covers_journal_day_tail(
        self, sleep_deps, tmp_data_dir: Path,
    ) -> None:
        """Codex round-5 worked example. A journal dated X only covers up
        to when its cycle ran — resuming from X+1 strands conversations
        modified on X after that run. Recovery must resume from X itself,
        with the summarised-index preventing duplicates:

        journal day(now-3) written MID-DAY (e.g. manual --date run),
        covering ctx-early; ctx-late modified that evening; the next
        night's cycle fails entirely (no journal). Today's run must pick
        up ctx-late and skip ctx-early.
        """
        import os

        cycle, store, _, _ = sleep_deps
        now = datetime.now(timezone.utc)
        day_x = (now - timedelta(days=3)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

        self._seed_named(store, tmp_data_dir, "ctx-early", day_x + timedelta(hours=9))
        self._seed_named(store, tmp_data_dir, "ctx-late", day_x + timedelta(hours=18))

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        journal = journals / f"{day_x.strftime('%Y-%m-%d')}.md"
        journal.write_text(
            "<!-- agentling-coverage: ctx-early -->\n"
            f"# Journal — {day_x.strftime('%Y-%m-%d')}\n\n"
            "### ctx-early\nSummarised.\n"
        )
        # The journal cycle ran mid-day X — before ctx-late's modification.
        run_time = day_x + timedelta(hours=12)
        os.utime(journal, (run_time.timestamp(), run_time.timestamp()))
        # The following night's cycle failed entirely: no journal for X+1.

        result = cycle._light_sleep(now)
        assert result == ["ctx-late"], (
            "the tail of the last-journaled day must re-enter the window; "
            "already-summarised conversations must not"
        )

    def test_partial_failure_day_reenters_only_skipped(
        self, sleep_deps, tmp_data_dir: Path,
    ) -> None:
        """A partial cycle (systemic abort / deadline expiry) writes a
        journal for the completed subset. That day must re-enter and only
        the skipped conversations be re-processed — journal existence is
        not proof of full coverage; its ### headings are the durable
        per-context record."""
        import os

        cycle, store, _, _ = sleep_deps
        now = datetime.now(timezone.utc)
        day = (now - timedelta(days=2)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )

        self._seed_named(store, tmp_data_dir, "ctx-done", day + timedelta(hours=10))
        self._seed_named(store, tmp_data_dir, "ctx-skipped", day + timedelta(hours=10))

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        journal = journals / f"{day.strftime('%Y-%m-%d')}.md"
        journal.write_text(
            "<!-- agentling-coverage: ctx-done -->\n"
            f"# Journal — {day.strftime('%Y-%m-%d')}\n\n"
            "### ctx-done\nSummarised before the abort.\n"
        )
        run_time = day + timedelta(days=1, hours=2)
        os.utime(journal, (run_time.timestamp(), run_time.timestamp()))

        result = cycle._light_sleep(now)
        assert result == ["ctx-skipped"]

    async def test_deadline_expiry_day_reenters(
        self, sleep_deps, tmp_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End to end: a deadline-expired cycle journals the completed
        subset; the next cycle's light sleep re-enters exactly the
        skipped conversations."""
        import agentlings.core.sleep as sleep_module
        from tests.unit.test_sleep import _SpyLLM  # self-import for clarity

        _, store, memory, config = sleep_deps
        now = datetime.now(timezone.utc)
        yesterday = (now - timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        for i in range(3):
            self._seed_named(
                store, tmp_data_dir, f"ctx-{i}", yesterday + timedelta(hours=10 + i),
            )

        llm = _SpyLLM(delay=0.15)
        cycle = SleepCycle(config=config, llm=llm, memory_store=memory, store=store)
        # Force the live path with a tiny deadline: first summary lands,
        # the rest expire.
        monkeypatch.setattr(sleep_module, "SLEEP_LIVE_DEADLINE_SECONDS", 0.2)
        object.__setattr__(cycle._sleep_config, "batch", False)

        date_str = yesterday.strftime("%Y-%m-%d")
        summaries, _ = await cycle._deep_sleep(["ctx-0", "ctx-1", "ctx-2"], date_str)
        assert 1 <= len(summaries) < 3, "expected a partial cycle"

        summarized = {
            line.removeprefix("### ").strip()
            for line in (tmp_data_dir / "journals" / f"{date_str}.md").read_text().splitlines()
            if line.startswith("### ")
        }
        skipped = {"ctx-0", "ctx-1", "ctx-2"} - summarized

        result = set(cycle._light_sleep(now + timedelta(days=1)))
        assert result == skipped, (
            f"re-entry must process exactly the skipped set; "
            f"summarised={summarized} got={result}"
        )

    def test_window_bounded_by_retention(self, sleep_deps, tmp_data_dir: Path) -> None:
        """The widened window never reaches past conversation retention —
        housekeeping deletes those conversations anyway."""
        cycle, store, _, _ = sleep_deps
        # sleep_config fixture sets conversation_retention_days: 7.
        self._seed_old_conversation(store, tmp_data_dir, days_ago=20)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        last = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        (journals / f"{last}.md").write_text("# ancient journal\n")

        result = cycle._light_sleep(datetime.now(timezone.utc))
        assert result == []


class TestDeepSleep:
    async def test_writes_journal(self, sleep_deps, tmp_data_dir: Path) -> None:
        cycle, store, _, _ = sleep_deps

        store.create("ctx-1")
        store.append("ctx-1", MessageEntry(
            ctx="ctx-1", role="user",
            content=[{"type": "text", "text": "hello agent"}],
        ))

        path = tmp_data_dir / "ctx-1" / "journal.jsonl"
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


class TestDeepSleepNoBatch:
    """With ``sleep.batch: false`` the deep-sleep phase must run sequential live
    ``complete()`` calls and never touch the batches API."""

    @pytest.fixture
    def no_batch_deps(self, tmp_data_dir: Path, tmp_path: Path):
        agent_yaml = tmp_path / "agent.yaml"
        agent_yaml.write_text(
            "name: sleep-test-agent\n"
            "description: A test agent for sleep\n"
            "tools:\n"
            "  - bash\n"
            "sleep:\n"
            "  batch: false\n"
            "  model: claude-haiku-4-5\n"
        )
        config = AgentConfig(
            anthropic_api_key="test-key",
            agent_api_key="test-key",
            agent_data_dir=tmp_data_dir,
            agent_llm_backend="mock",
            agent_config=str(agent_yaml),
        )
        store = JournalStore(tmp_data_dir)
        memory = MemoryFileStore(tmp_data_dir)
        llm = MockLLMClient(tool_names=[])
        cycle = SleepCycle(config=config, llm=llm, memory_store=memory, store=store)
        return cycle, store, llm

    def _seed_conversation(self, store: JournalStore, tmp_data_dir: Path, ctx: str) -> None:
        store.create(ctx)
        store.append(ctx, MessageEntry(
            ctx=ctx, role="user",
            content=[{"type": "text", "text": "hello agent"}],
        ))
        path = tmp_data_dir / ctx / "journal.jsonl"
        yesterday = datetime.now(timezone.utc) - timedelta(hours=12)
        import os
        os.utime(path, (yesterday.timestamp(), yesterday.timestamp()))

    async def test_live_path_calls_complete_not_batch(
        self, no_batch_deps, tmp_data_dir: Path
    ) -> None:
        cycle, store, llm = no_batch_deps
        self._seed_conversation(store, tmp_data_dir, "ctx-1")
        self._seed_conversation(store, tmp_data_dir, "ctx-2")

        conversations = cycle._light_sleep(datetime.now(timezone.utc))
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        summaries, _ = await cycle._deep_sleep(conversations, date_str)

        # One live completion per conversation, and no batch was ever submitted.
        assert llm.complete_calls == 2
        assert llm._batch_store == {}
        # The configured sleep model override is forwarded to complete().
        assert llm.last_model == "claude-haiku-4-5"
        # Journal still produced.
        assert (tmp_data_dir / "journals" / f"{date_str}.md").exists()
        assert len(summaries) == 2

    async def test_batch_default_uses_batch_api(
        self, sleep_deps, tmp_data_dir: Path
    ) -> None:
        """The default config (batch unset) must still go through the batch API
        and not fall back to live calls — guards against flipping the default."""
        cycle, store, _, _ = sleep_deps
        llm = cycle._llm  # the MockLLMClient from the default sleep_deps fixture
        self._seed_conversation(store, tmp_data_dir, "ctx-1")

        conversations = cycle._light_sleep(datetime.now(timezone.utc))
        date_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
        await cycle._deep_sleep(conversations, date_str)

        assert llm.complete_calls == 0
        assert len(llm._batch_store) == 1
class _SpyLLM(MockLLMClient):
    """Mock LLM that records path usage for deep-sleep tests.

    Failure knobs: ``fail_on_call`` fails exactly that call number,
    ``fail_from`` fails every call from that number onward (persistent
    failure), ``fail_all`` fails everything. ``delay`` sleeps per call
    (for deadline tests).
    """

    def __init__(
        self,
        supports_batches: bool = True,
        fail_on_call: int | None = None,
        fail_from: int | None = None,
        fail_exc: Exception | None = None,
        fail_all: bool = False,
        delay: float = 0.0,
    ) -> None:
        super().__init__(tool_names=[])
        self.supports_batches = supports_batches
        self._fail_on_call = fail_on_call
        self._fail_from = fail_from
        self._fail_exc = fail_exc
        self._fail_all = fail_all
        self._delay = delay
        self.call_log: list[dict] = []
        self.batch_create_calls = 0

    async def complete(self, system, messages, tools, output_schema=None,
                       context_id=None, task_id=None, max_tokens=None,
                       model=None):
        self.call_log.append({
            "output_schema": output_schema, "max_tokens": max_tokens,
            "model": model,
        })
        if self._delay:
            import asyncio
            await asyncio.sleep(self._delay)
        n = len(self.call_log)
        if (
            self._fail_all
            or self._fail_on_call == n
            or (self._fail_from is not None and n >= self._fail_from)
        ):
            raise (self._fail_exc or RuntimeError("simulated live-summary failure"))
        return await super().complete(
            system, messages, tools, output_schema=output_schema,
            context_id=context_id, task_id=task_id, max_tokens=max_tokens,
            model=model,
        )

    async def batch_create(self, requests, model=None):
        self.batch_create_calls += 1
        return await super().batch_create(requests, model=model)


@pytest.fixture(autouse=True)
def _no_patience_waits(monkeypatch: pytest.MonkeyPatch):
    """Zero out the cycle-level patience backoffs so tests don't sleep."""
    import agentlings.core.sleep as sleep_module

    monkeypatch.setattr(
        sleep_module, "_LIVE_TRANSIENT_RETRY_BACKOFFS", (0.0, 0.0),
    )


def _make_cycle(
    tmp_path: Path,
    tmp_data_dir: Path,
    llm: MockLLMClient,
    sleep_yaml: str,
) -> tuple[SleepCycle, JournalStore]:
    agent_yaml = tmp_path / "agent-live.yaml"
    agent_yaml.write_text(
        "name: sleep-live-test-agent\n"
        "tools: []\n"
        f"sleep:\n{sleep_yaml}"
        "memory:\n"
        "  token_budget: 2000\n"
    )
    config = AgentConfig(
        anthropic_api_key="test-key",
        agent_api_key="test-key",
        agent_data_dir=tmp_data_dir,
        agent_llm_backend="mock",
        agent_config=str(agent_yaml),
    )
    store = JournalStore(tmp_data_dir)
    memory = MemoryFileStore(tmp_data_dir)
    cycle = SleepCycle(config=config, llm=llm, memory_store=memory, store=store)
    return cycle, store


def _seed_conversations(store: JournalStore, count: int) -> list[str]:
    ctx_ids = [f"ctx-{i}" for i in range(count)]
    for ctx_id in ctx_ids:
        store.create(ctx_id)
        store.append(ctx_id, MessageEntry(
            ctx=ctx_id, role="user",
            content=[{"type": "text", "text": f"hello from {ctx_id}"}],
        ))
    return ctx_ids


class TestLiveSummaryPath:
    """The batches workaround: ``sleep.batch: false`` (or a backend that
    advertises no batch support) runs deep-sleep summaries as sequential
    live completion calls instead of the Anthropic batches API.
    """

    async def test_batch_false_uses_live_calls(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        llm = _SpyLLM()
        cycle, store = _make_cycle(
            tmp_path, tmp_data_dir, llm, "  batch: false\n",
        )
        ctx_ids = _seed_conversations(store, 2)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-01")

        assert llm.batch_create_calls == 0, "batches API must not be used"
        assert len(llm.call_log) == 2, "one live call per conversation"
        assert all(
            c["output_schema"] is not None for c in llm.call_log
        ), "live summaries must keep the structured-output schema"
        assert all(c["max_tokens"] == 4096 for c in llm.call_log)
        assert len(summaries) == 2
        journal_path = tmp_data_dir / "journals" / "2026-01-01.md"
        assert journal_path.exists()

    async def test_batchless_backend_degrades_to_live_path(
        self, tmp_path: Path, tmp_data_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A client with ``supports_batches = False`` (e.g. the responses
        wire format) uses live calls even when ``sleep.batch`` is true."""
        llm = _SpyLLM(supports_batches=False)
        cycle, store = _make_cycle(
            tmp_path, tmp_data_dir, llm, "  batch: true\n",
        )
        ctx_ids = _seed_conversations(store, 1)

        with caplog.at_level("INFO"):
            summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-02")

        assert llm.batch_create_calls == 0
        assert len(llm.call_log) == 1
        assert len(summaries) == 1
        assert any(
            "no batches API support" in r.message for r in caplog.records
        )

    async def test_default_still_uses_batch_path(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """Regression guard: the default config on a batch-capable backend
        must keep using the batches API exactly as before."""
        llm = _SpyLLM()
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  enabled: true\n")
        ctx_ids = _seed_conversations(store, 2)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-03")

        assert llm.batch_create_calls == 1
        assert llm.call_log == []
        assert len(summaries) == 2

    async def test_one_live_failure_degrades_not_aborts(
        self, tmp_path: Path, tmp_data_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """One failing conversation must not kill the cycle — mirroring the
        batch path's per-request error tolerance."""
        llm = _SpyLLM(fail_on_call=1)
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 2)

        with caplog.at_level("WARNING"):
            summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-04")

        assert len(llm.call_log) == 2, "second conversation still summarised"
        assert len(summaries) == 1
        assert any(
            "Live summary failed" in r.message for r in caplog.records
        )
        assert (tmp_data_dir / "journals" / "2026-01-04.md").exists()

    async def test_transient_failure_recovers_with_patience(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A brief transient failure (connection blip, momentary 429) is
        retried with cycle-level patience and must NOT cost the night."""
        from agentlings.core.llm_responses import ResponsesConnectionError

        llm = _SpyLLM(
            fail_on_call=1,
            fail_exc=ResponsesConnectionError("momentary blip"),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 3)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-06")

        # First attempt failed, patience retry succeeded, rest normal.
        assert len(llm.call_log) == 4
        assert len(summaries) == 3
        assert (tmp_data_dir / "journals" / "2026-01-06.md").exists()

    async def test_persistent_transient_failure_aborts_loudly(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A persistent outage exhausts patience and aborts with nothing
        burned — the night auto-recovers via the widened window next
        cycle."""
        from agentlings.core.llm_responses import ResponsesConnectionError

        llm = _SpyLLM(
            fail_from=1,
            fail_exc=ResponsesConnectionError("endpoint unreachable"),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 3)

        with pytest.raises(ResponsesConnectionError):
            await cycle._deep_sleep(ctx_ids, "2026-01-14")

        # 1 initial + 2 patience retries, then abort — no per-item hammering.
        assert len(llm.call_log) == 3
        assert not (tmp_data_dir / "journals" / "2026-01-14.md").exists()

    async def test_auth_error_aborts_immediately_without_patience(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """Hard failures (auth) won't self-heal — no retries, immediate
        loud abort."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_on_call=1,
            fail_exc=ResponsesAPIError("invalid key", status_code=401),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 2)

        with pytest.raises(ResponsesAPIError):
            await cycle._deep_sleep(ctx_ids, "2026-01-07")
        assert len(llm.call_log) == 1, "hard failures must not be retried"
        assert not (tmp_data_dir / "journals" / "2026-01-07.md").exists()

    async def test_model_not_found_is_hard_abort(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A wrong sleep.model (400 model_not_found) is hard — one 400,
        not one per conversation per night."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_on_call=1,
            fail_exc=ResponsesAPIError(
                "model not found", status_code=400, error_type="model_not_found",
            ),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 3)

        with pytest.raises(ResponsesAPIError):
            await cycle._deep_sleep(ctx_ids, "2026-01-15")
        assert len(llm.call_log) == 1

    async def test_persistent_server_error_bounded_calls(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A persistent 5xx outage exhausts patience once and aborts —
        never one full retry cycle per conversation."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_from=1,
            fail_exc=ResponsesAPIError("overloaded", status_code=503),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 5)

        with pytest.raises(ResponsesAPIError):
            await cycle._deep_sleep(ctx_ids, "2026-01-09")
        assert len(llm.call_log) == 3  # 1 + 2 patience retries, then stop
        assert not (tmp_data_dir / "journals" / "2026-01-09.md").exists()

    async def test_in_body_transient_error_type_retried_then_aborts(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """In-body failures (HTTP 200, status_code=0) with a transient
        error type — server_error, rate limits — get patience, then abort
        without hammering every conversation."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_from=1,
            fail_exc=ResponsesAPIError(
                "response failed: backend down",
                status_code=0, error_type="server_error",
            ),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 4)

        with pytest.raises(ResponsesAPIError):
            await cycle._deep_sleep(ctx_ids, "2026-01-11")
        assert len(llm.call_log) == 3  # 1 + 2 patience retries
        assert not (tmp_data_dir / "journals" / "2026-01-11.md").exists()

    async def test_in_body_per_item_error_still_degrades(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A non-systemic in-body error (e.g. one conversation too large)
        keeps the per-item degrade behaviour."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_on_call=1,
            fail_exc=ResponsesAPIError(
                "context length exceeded",
                status_code=0, error_type="context_length_exceeded",
            ),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 2)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-12")
        assert len(llm.call_log) == 2
        assert len(summaries) == 1

    async def test_transient_rate_limit_recovers(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A single 429 must not cost the night — patience absorbs it."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_on_call=1,
            fail_exc=ResponsesAPIError("quota", status_code=429),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 2)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-10")
        assert len(summaries) == 2
        assert len(llm.call_log) == 3  # fail, retry-succeed, second ctx

    async def test_all_items_failing_aborts_cycle(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """Even when each failure looks per-item, every summary failing is
        systemic — the cycle must raise, not log success over an empty
        journal."""
        llm = _SpyLLM(fail_all=True)
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 2)

        with pytest.raises(RuntimeError, match="all 2 live deep-sleep summaries failed"):
            await cycle._deep_sleep(ctx_ids, "2026-01-08")
        assert not (tmp_data_dir / "journals" / "2026-01-08.md").exists()

    async def test_sleep_model_override_honoured_on_live_path(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """sleep.model applies to live summaries exactly like the batch
        path — nightly maintenance must not silently bill the flagship
        default model."""
        llm = _SpyLLM()
        cycle, store = _make_cycle(
            tmp_path, tmp_data_dir, llm,
            "  batch: false\n  model: claude-haiku-4-5\n",
        )
        ctx_ids = _seed_conversations(store, 2)

        await cycle._deep_sleep(ctx_ids, "2026-01-05")

        assert all(
            c["model"] == "claude-haiku-4-5" for c in llm.call_log
        ), f"sleep.model not threaded through: {llm.call_log!r}"

    async def test_late_systemic_failure_keeps_completed_work(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A systemic failure AFTER summaries completed must not discard
        them — the day would otherwise be permanently lost (light sleep
        never revisits it and retention deletes the conversations). The
        cycle stops issuing calls, keeps completed work, and writes the
        journal."""
        from agentlings.core.llm_responses import ResponsesConnectionError

        llm = _SpyLLM(
            fail_from=2,
            fail_exc=ResponsesConnectionError("backend died mid-run"),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 4)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-13")

        # First summary succeeded; the second exhausted its patience
        # retries (calls 2-4); the remaining two were skipped WITHOUT
        # further calls.
        assert len(llm.call_log) == 4
        assert len(summaries) == 1
        journal = tmp_data_dir / "journals" / "2026-01-13.md"
        assert journal.exists(), "completed work must still be journaled"

    async def test_poison_conversation_degrades_per_item(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """One conversation with an oversized context (400) must not
        wedge the cycle — the rest summarise and the journal is written.
        Recovery re-enters the poison conversation first every night, so
        a hard-abort here would be a permanent nightly wedge."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_on_call=1,
            fail_exc=ResponsesAPIError(
                "context too large", status_code=400,
                error_type="invalid_request_error",
            ),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 3)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-18")

        assert len(llm.call_log) == 3, "no retries for per-item 400s"
        assert len(summaries) == 2
        assert (tmp_data_dir / "journals" / "2026-01-18.md").exists()

    async def test_two_leading_bad_requests_declare_systemic(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A leading 400 is ambiguous; a second consecutive one (with
        nothing succeeded) resolves the ambiguity as systemic — abort
        instead of burning a 400 per conversation."""
        from agentlings.core.llm_responses import ResponsesAPIError

        llm = _SpyLLM(
            fail_from=1,
            fail_exc=ResponsesAPIError(
                "bad request", status_code=400,
                error_type="invalid_request_error",
            ),
        )
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 5)

        with pytest.raises(ResponsesAPIError):
            await cycle._deep_sleep(ctx_ids, "2026-01-19")
        assert len(llm.call_log) == 2, "probe exactly one other conversation"
        assert not (tmp_data_dir / "journals" / "2026-01-19.md").exists()

    async def test_deadline_expiry_keeps_completed_work(
        self, tmp_path: Path, tmp_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The cycle-level deadline stops a marathon run but keeps the
        completed summaries and writes the journal."""
        import agentlings.core.sleep as sleep_module

        llm = _SpyLLM(delay=0.15)
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 4)
        monkeypatch.setattr(sleep_module, "SLEEP_LIVE_DEADLINE_SECONDS", 0.2)

        summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-01-16")

        # First call (~0.15s) succeeded; deadline expired during/before
        # the second; remaining conversations skipped without calls.
        assert 1 <= len(summaries) <= 2
        assert len(llm.call_log) <= 2
        assert (tmp_data_dir / "journals" / "2026-01-16.md").exists()

    async def test_deadline_with_no_successes_raises(
        self, tmp_path: Path, tmp_data_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import agentlings.core.sleep as sleep_module

        llm = _SpyLLM()
        cycle, store = _make_cycle(tmp_path, tmp_data_dir, llm, "  batch: false\n")
        ctx_ids = _seed_conversations(store, 2)
        monkeypatch.setattr(sleep_module, "SLEEP_LIVE_DEADLINE_SECONDS", 0.0)

        with pytest.raises(RuntimeError, match="all 2 live deep-sleep summaries failed"):
            await cycle._deep_sleep(ctx_ids, "2026-01-17")
        assert llm.call_log == []
        assert not (tmp_data_dir / "journals" / "2026-01-17.md").exists()


class TestJournalMerge:
    """Same-day re-runs (the recovery flow) merge into the existing
    journal instead of overwriting it — overwriting would destroy the
    coverage record and the summaries it indexed."""

    def _cycle(self, tmp_path: Path, tmp_data_dir: Path):
        cycle, _ = _make_cycle(
            tmp_path, tmp_data_dir, _SpyLLM(), "  batch: false\n",
        )
        return cycle

    def test_merge_unions_coverage_and_keeps_bodies(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        cycle = self._cycle(tmp_path, tmp_data_dir)
        cycle._write_journal("2026-02-01", "### ctx-a\nFirst pass.", {"ctx-a": 100.0})
        cycle._write_journal("2026-02-01", "### ctx-b\nRecovery pass.", {"ctx-b": 200.0})

        path = tmp_data_dir / "journals" / "2026-02-01.md"
        content = path.read_text()
        first_line = content.splitlines()[0]
        assert "ctx-a" in first_line and "ctx-b" in first_line, (
            f"coverage must union, got: {first_line!r}"
        )
        assert "First pass." in content, "prior summaries must survive a merge"
        assert "Recovery pass." in content

        index = cycle._summarized_context_index()
        assert set(index) == {"ctx-a", "ctx-b"}

    def test_fresh_journal_has_marker_first_line(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        cycle = self._cycle(tmp_path, tmp_data_dir)
        cycle._write_journal("2026-02-02", "### ctx-x\nBody.", {"ctx-x": 1700000000.5})
        first_line = (
            tmp_data_dir / "journals" / "2026-02-02.md"
        ).read_text().splitlines()[0]
        assert first_line == "<!-- agentling-coverage: ctx-x:1700000000.500000 -->"


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
        assert (tmp_data_dir / "recent-ctx" / "journal.jsonl").exists()

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

        path = tmp_data_dir / "afternoon-ctx" / "journal.jsonl"
        yesterday_3pm = datetime(2026, 4, 2, 15, 0, tzinfo=timezone.utc)
        import os
        os.utime(path, (yesterday_3pm.timestamp(), yesterday_3pm.timestamp()))

        sleep_time = datetime(2026, 4, 3, 2, 0, tzinfo=timezone.utc)
        result = cycle._light_sleep(sleep_time)
        assert len(result) == 1, "Yesterday afternoon's conversation was not discovered"

    def test_ignores_conversations_older_than_retention(self, sleep_deps, tmp_data_dir: Path) -> None:
        """The window is retention-wide (uncovered work is recovered from
        any day within retention — 7 days in this fixture); only
        conversations older than retention fall outside it."""
        cycle, store, _, _ = sleep_deps
        store.create("old-ctx")
        store.append("old-ctx", MessageEntry(
            ctx="old-ctx", role="user",
            content=[{"type": "text", "text": "ancient history"}],
        ))

        path = tmp_data_dir / "old-ctx" / "journal.jsonl"
        nine_days_ago = datetime(2026, 3, 25, 10, 0, tzinfo=timezone.utc)
        import os
        os.utime(path, (nine_days_ago.timestamp(), nine_days_ago.timestamp()))

        sleep_time = datetime(2026, 4, 3, 2, 0, tzinfo=timezone.utc)
        result = cycle._light_sleep(sleep_time)
        assert len(result) == 0, "Conversation older than retention should not be picked up"

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


class _ResultsFailLLM(MockLLMClient):
    """Batch completes (status=ended) but fetching results always fails.

    Reproduces the sluice ``results_url`` failure: the batch object's
    ``results_url`` points at ``api.anthropic.com`` (the real upstream), so the
    SDK follows it directly, bypassing the gateway, and is rejected with a 401
    ``invalid x-api-key``. The retrieval call raises; the cycle must not.
    """

    async def batch_results(self, batch_id: str):
        raise RuntimeError("401 Unauthorized: invalid x-api-key")


class TestBatchResultsResilience:
    """A failure retrieving batch results must degrade gracefully rather than
    abort the cycle (regression: the run aborted on the results 401)."""

    async def test_poll_batch_swallows_results_failure(
        self, sleep_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
        store = JournalStore(tmp_data_dir)
        memory = MemoryFileStore(tmp_data_dir)
        llm = _ResultsFailLLM(tool_names=[])
        cycle = SleepCycle(config=sleep_config, llm=llm, memory_store=memory, store=store)

        # batch_status reports "ended", so _poll_batch takes the results path
        # where batch_results raises. It must degrade to [] rather than propagate.
        results = await cycle._poll_batch("mock_batch_x")
        assert results == []

    async def test_run_completes_when_results_fetch_fails(
        self, sleep_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
        store = JournalStore(tmp_data_dir)
        memory = MemoryFileStore(tmp_data_dir)
        llm = _ResultsFailLLM(tool_names=[])
        cycle = SleepCycle(config=sleep_config, llm=llm, memory_store=memory, store=store)

        # One conversation in the review window so deep_sleep submits a batch.
        store.create("ctx-1")
        store.append("ctx-1", MessageEntry(
            ctx="ctx-1", role="user",
            content=[{"type": "text", "text": "hello agent"}],
        ))
        path = tmp_data_dir / "ctx-1" / "journal.jsonl"
        yesterday = datetime.now(timezone.utc) - timedelta(hours=12)
        import os
        os.utime(path, (yesterday.timestamp(), yesterday.timestamp()))

        # Must complete without raising; with no results, memory stays untouched.
        await cycle.run()
        assert memory.list() == []


class TestRemTokenBudget:
    """REM consolidation must request enough output tokens to re-emit the full
    memory. The 4096 default truncated the JSON mid-entry and the parse failed,
    silently dropping the consolidation (regression)."""

    async def test_rem_requests_consolidation_max_tokens(
        self, sleep_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
        store = JournalStore(tmp_data_dir)
        memory = MemoryFileStore(tmp_data_dir)
        llm = MockLLMClient(tool_names=[])
        cycle = SleepCycle(config=sleep_config, llm=llm, memory_store=memory, store=store)

        await cycle._rem(
            summaries=["### ctx-1\na summary"], candidates=[], date_str="2026-05-26"
        )

        budget = (sleep_config.sleep_config or SleepConfig()).consolidation_max_tokens
        assert budget >= 16384, "consolidation budget must be well above the 4096 default"
        assert llm.last_max_tokens == budget, (
            "REM must request the configured consolidation_max_tokens, not the default"
        )


class TestRemDateInjection:
    """REM must hand the model the review date so it never guesses (which produced
    future-dated `recorded` timestamps in the wild)."""

    async def test_rem_includes_review_date_in_message(
        self, sleep_config: AgentConfig, tmp_data_dir: Path
    ) -> None:
        store = JournalStore(tmp_data_dir)
        memory = MemoryFileStore(tmp_data_dir)
        llm = MockLLMClient(tool_names=[])
        captured: dict[str, object] = {}
        original = llm.complete

        async def spy(*args, **kwargs):  # noqa: ANN002, ANN003
            captured["messages"] = kwargs.get("messages")
            return await original(*args, **kwargs)

        llm.complete = spy  # type: ignore[method-assign]
        cycle = SleepCycle(config=sleep_config, llm=llm, memory_store=memory, store=store)

        await cycle._rem(
            summaries=["### ctx-1\na summary"], candidates=[], date_str="2026-05-26"
        )

        content = captured["messages"][0]["content"]  # type: ignore[index]
        assert "2026-05-26" in content
        assert "under review" in content.lower()
class TestCoverageSnapshots:
    """Codex round-6: coverage must be validated, snapshot-based, and
    merge-proof — never inferred from the journal file's own mtime."""

    def _seed(self, store, tmp_data_dir: Path, ctx_id: str, mtime: datetime) -> None:
        import os

        store.create(ctx_id)
        store.append(ctx_id, MessageEntry(
            ctx=ctx_id, role="user",
            content=[{"type": "text", "text": f"in {ctx_id}"}],
        ))
        path = tmp_data_dir / ctx_id / "journal.jsonl"
        os.utime(path, (mtime.timestamp(), mtime.timestamp()))

    async def test_empty_summary_payload_not_covered(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """A structurally-successful item whose extracted text is empty
        journals nothing usable — it must count as failed and stay
        uncovered so the next cycle retries it."""
        llm = _SpyLLM()
        cycle, store = _make_cycle(
            tmp_path, tmp_data_dir, llm, "  batch: false\n",
        )
        ctx_ids = _seed_conversations(store, 2)

        real_extract = cycle._extract_structured_text
        calls = {"n": 0}

        def extract(content):
            calls["n"] += 1
            if calls["n"] == 1:
                return ""
            return '{"summary": "fine", "memory_candidates": []}'

        cycle._extract_structured_text = extract  # type: ignore[method-assign]
        try:
            summaries, _ = await cycle._deep_sleep(ctx_ids, "2026-03-01")
        finally:
            cycle._extract_structured_text = real_extract  # type: ignore[method-assign]

        assert len(summaries) == 1, "empty payload must not produce a section"
        first_line = (
            tmp_data_dir / "journals" / "2026-03-01.md"
        ).read_text().splitlines()[0]
        assert ctx_ids[0] not in first_line, (
            "empty-payload conversation must NOT be marked covered"
        )
        assert ctx_ids[1] in first_line

    def test_modification_after_snapshot_reselected(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """Freshness compares the per-entry snapshot, not the journal file
        mtime: a conversation modified after its recorded snapshot re-enters
        the window; one unchanged since its snapshot stays excluded."""
        cycle, store = _make_cycle(
            tmp_path, tmp_data_dir, _SpyLLM(), "  batch: false\n",
        )
        now = datetime.now(timezone.utc)
        stale_read = now - timedelta(hours=12)
        modified = now - timedelta(hours=6)
        self._seed(store, tmp_data_dir, "ctx-stale", modified)
        self._seed(store, tmp_data_dir, "ctx-fresh", modified)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        (journals / f"{date}.md").write_text(
            "<!-- agentling-coverage: "
            f"ctx-stale:{stale_read.timestamp():.6f} "
            f"ctx-fresh:{(now - timedelta(hours=1)).timestamp():.6f} -->\n"
            "# journal\n"
        )

        result = cycle._light_sleep(now)
        assert result == ["ctx-stale"], (
            "snapshot older than the conversation mtime must re-select; "
            f"got {result!r}"
        )

    def test_merge_does_not_advance_legacy_freshness(
        self, tmp_path: Path, tmp_data_dir: Path,
    ) -> None:
        """Merging a recovery pass must not launder older entries' freshness
        through the journal file's new mtime (Codex round-6 finding 2):
        a legacy bare entry is pinned to the PRE-merge journal mtime."""
        import os

        cycle, store = _make_cycle(
            tmp_path, tmp_data_dir, _SpyLLM(), "  batch: false\n",
        )
        now = datetime.now(timezone.utc)
        t_journal = now - timedelta(hours=12)
        t_modified = now - timedelta(hours=6)
        self._seed(store, tmp_data_dir, "ctx-legacy", t_modified)

        journals = tmp_data_dir / "journals"
        journals.mkdir(exist_ok=True)
        date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        jpath = journals / f"{date}.md"
        jpath.write_text("<!-- agentling-coverage: ctx-legacy -->\n# journal\n")
        os.utime(jpath, (t_journal.timestamp(), t_journal.timestamp()))

        cycle._write_journal(
            date, "### ctx-new\nRecovery.", {"ctx-new": now.timestamp()},
        )

        result = cycle._light_sleep(now)
        assert result == ["ctx-legacy"], (
            "merge advanced the legacy entry's freshness past the "
            f"conversation's modification; got {result!r}"
        )
