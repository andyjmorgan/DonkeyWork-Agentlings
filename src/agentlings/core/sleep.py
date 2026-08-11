"""Nightly sleep cycle: journal, consolidate memory, and clean up."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from urllib.parse import quote, unquote
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentlings.config import AgentConfig, SleepConfig
from agentlings.core.llm import BaseLLMClient, BatchItemResult, BatchRequest
from agentlings.core.memory_models import (
    ConsolidatedMemory,
    ConversationSummary,
    MemoryCandidate,
    strict_json_schema,
)
from agentlings.core.telemetry import sleep_span
from agentlings.core.memory_store import MemoryFileStore
from agentlings.core.prompt import build_system_prompt
from agentlings.core.store import JournalStore

logger = logging.getLogger(__name__)

IDLE_GRACE_SECONDS = 300

# First line of every daily journal: the machine-owned coverage record
# naming each context the journal summarises, each with the source
# conversation's mtime (epoch seconds) captured when it was READ for
# summarisation. Only line 1 is ever parsed, so model-authored body text
# cannot forge or shadow coverage entries — and the per-entry snapshot
# (rather than the journal file's own mtime, which same-day merges would
# advance for every older entry) is what freshness checks compare against.
_COVERAGE_MARKER_TEMPLATE = "<!-- agentling-coverage: {ids} -->"
_COVERAGE_MARKER_RE = re.compile(r"^<!--\s*agentling-coverage:\s*(.*?)\s*-->\s*$")


def _parse_coverage_marker(line: str) -> dict[str, float | None] | None:
    """Parse a journal's first-line coverage marker.

    Returns ``{context_id: snapshot_epoch_or_None}``, or ``None`` when the
    line is not a coverage marker (foreign/hand-written file — contributes
    nothing to the coverage index rather than risking a forgeable content
    parse). Entries are ``ctx:epoch``; bare ``ctx`` entries (pre-snapshot
    journals) parse with a ``None`` snapshot and fall back to the journal
    file's mtime at read time.
    """
    match = _COVERAGE_MARKER_RE.match(line.strip())
    if match is None:
        return None
    covered: dict[str, float | None] = {}
    for token in match.group(1).split():
        ctx, sep, ts = token.rpartition(":")
        if sep and ctx:
            try:
                # Context ids are client-supplied and percent-encoded at
                # write time (they may contain whitespace, newlines, or
                # colons that would corrupt the space-joined marker).
                covered[unquote(ctx)] = float(ts)
                continue
            except ValueError:
                pass
        if token:
            covered[unquote(token)] = None
    return covered

# Aggregate wall-clock budget for the live-summary path — mirrors the batch
# path's poll timeout so deep sleep cannot run for hours into daytime. On
# expiry, completed work is kept and the journal is written.
SLEEP_LIVE_DEADLINE_SECONDS = 7200.0

# Cycle-level patience for transient systemic failures on the live path.
# The LLM client already does fast retries; these are the slow, generous
# waits appropriate at 2am — a brief rate-limit or restart should cost
# minutes, not the whole night.
_LIVE_TRANSIENT_RETRY_BACKOFFS: tuple[float, ...] = (30.0, 120.0)

# Error types that will not self-heal overnight — bad credential, missing
# permission, nonexistent model. Deliberately NOT invalid_request_error /
# generic 400s: those can be conversation-specific (one poison conversation
# with an oversized context) and hard-aborting on them would wedge the
# cycle permanently, since recovery re-enters the same conversation first
# every night. They degrade per-item, with a one-other-conversation probe
# (see ``_summarize_live``) to catch the genuinely systemic case.
_HARD_ERROR_TYPES = frozenset({
    "invalid_api_key",
    "authentication_error",
    "permission_denied",
    "permission_error",
    "model_not_found",
})

# Error types that may recover with patience — overloaded/broken backend,
# rate limits, exhausted quota windows.
_TRANSIENT_ERROR_TYPES = frozenset({
    "server_error",
    "internal_server_error",
    "rate_limit",
    "rate_limit_error",
    "rate_limit_exceeded",
    "insufficient_quota",
})


def _llm_failure_class(exc: Exception) -> str | None:
    """Classify an LLM-call exception for the live-summary path.

    Returns ``"hard"`` for failures that will not self-heal (auth,
    permissions, bad model/request — retrying is pointless),
    ``"transient"`` for failures that may recover with patience
    (connection/timeout, 429 rate limits, 5xx server errors), and
    ``None`` for per-item failures (e.g. one oversized conversation)
    that should degrade without affecting the rest of the cycle.
    """
    def _status_class(code: int | None) -> str | None:
        if code in (401, 403):
            return "hard"
        if code == 429 or (code is not None and code >= 500):
            return "transient"
        return None

    if isinstance(exc, (ConnectionError, TimeoutError)):
        return "transient"
    try:
        import httpx

        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return "transient"
    except ImportError:  # pragma: no cover - httpx is a hard dependency
        pass
    try:
        import anthropic

        if isinstance(exc, anthropic.APIConnectionError):
            return "transient"
        if isinstance(exc, anthropic.APIStatusError):
            status_class = _status_class(getattr(exc, "status_code", None))
            if status_class:
                return status_class
    except ImportError:  # pragma: no cover - anthropic is a hard dependency
        pass
    from agentlings.core.llm_responses import ResponsesAPIError

    if isinstance(exc, ResponsesAPIError):
        status_class = _status_class(exc.status_code or None)
        if status_class:
            return status_class
        error_type = exc.error_type or ""
        if error_type in _HARD_ERROR_TYPES:
            return "hard"
        if error_type in _TRANSIENT_ERROR_TYPES:
            return "transient"
    return None


def _is_bad_request(exc: Exception) -> bool:
    """Whether an exception is a request-rejection (400-class) failure.

    Ambiguous on its own: could be one poison conversation (oversized
    context) or a systemic misconfiguration. The live-summary loop
    disambiguates by probing at least one other conversation.
    """
    from agentlings.core.llm_responses import ResponsesAPIError

    if isinstance(exc, ResponsesAPIError):
        return (
            exc.status_code == 400
            or (exc.error_type or "") == "invalid_request_error"
        )
    try:
        import anthropic

        if isinstance(exc, anthropic.APIStatusError):
            return getattr(exc, "status_code", None) == 400
    except ImportError:  # pragma: no cover - anthropic is a hard dependency
        pass
    return False

DEFAULT_SUMMARY_PROMPT = """\
You are performing a nightly review of a conversation that took place today.

Produce a concise summary of what happened: what was asked, what actions were \
taken, what the outcome was, and anything left unresolved.

Extract any facts worth adding to your long-term memory. Only extract NEW facts \
not already in your current memory. Focus on operational knowledge, patterns, \
decisions, things that changed. Ignore passing context.

Each candidate must be ONE self-contained fact of at most ~60 words; split \
multi-topic findings into separate candidates.

NEVER record secret values — API keys, tokens, passwords, access keys. Record \
only WHERE the credential lives (a secret store / vault reference), never the \
literal value.

If the conversation was trivial or contained nothing new worth remembering, \
return an empty memory_candidates list."""

DEFAULT_CONSOLIDATION_PROMPT = """\
You are performing nightly memory maintenance. The date of the conversations \
under review is stated in the message below.

Your job:
1. Integrate new candidates that add genuine value. Deduplicate against existing \
entries, and MERGE multiple entries about the same component into a single entry.
2. Review every existing entry. Is it still relevant? Has it been superseded by \
something learned today? Would it help you do your job tomorrow?
3. Drop anything stale, redundant, or no longer operationally useful. Exclude \
pure implementation trivia (library internals, exact schemas, code paths) unless \
tied to an unresolved incident — that detail lives in the source, not memory.
4. You MUST return at most {memory_max_entries} entries. If integrating new \
candidates would exceed the limit, rank every entry by future operational value \
and drop the lowest-value entries until you are under the cap.

NEVER store secret values — API keys, tokens, passwords, access keys. If an \
existing entry contains a literal secret, REWRITE it in place to a reference \
describing where the credential lives, keeping the rest of the entry intact.

Do NOT invent or advance dates. Preserve the recorded timestamp for entries you \
keep unchanged. For new or modified entries, set recorded to the review date \
stated in the message — never a future or guessed date."""


class SleepCycle:
    """Orchestrates the four-phase nightly sleep cycle."""

    def __init__(
        self,
        config: AgentConfig,
        llm: BaseLLMClient,
        memory_store: MemoryFileStore,
        store: JournalStore,
    ) -> None:
        self._config = config
        self._llm = llm
        self._memory_store = memory_store
        self._store = store
        self._sleep_config = config.sleep_config or SleepConfig()

    async def run(self, date: datetime | None = None) -> None:
        """Execute the full sleep cycle reviewing the previous day's conversations.

        Args:
            date: Reference timestamp (defaults to now UTC). The cycle reviews
                  conversations from the day before this timestamp.
        """
        date = date or datetime.now(timezone.utc)
        review_date = date - timedelta(days=1)
        date_str = review_date.strftime("%Y-%m-%d")
        start = time.monotonic()
        logger.info("[SLEEP] Starting cycle for %s", date_str)

        with sleep_span("agentling.sleep", {"agent.name": self._config.agent_name, "sleep.date": date_str}) as root:
            with sleep_span("agentling.sleep.light_sleep", {"sleep.phase": "light_sleep", "sleep.date": date_str}) as ls:
                context_ids = self._light_sleep(date)
                ls.set_attribute("sleep.conversations_found", len(context_ids))
                if not context_ids:
                    ls.set_attribute("sleep.skipped", True)
                    logger.info("[SLEEP:LIGHT] No conversations found, skipping cycle")
                    return

            logger.info("[SLEEP:LIGHT] Found %d conversations, proceeding", len(context_ids))

            with sleep_span("agentling.sleep.deep_sleep", {"sleep.phase": "deep_sleep", "sleep.date": date_str}):
                summaries, candidates = await self._deep_sleep(context_ids, date_str)

            if summaries:
                with sleep_span("agentling.sleep.rem", {"sleep.phase": "rem", "sleep.date": date_str}):
                    await self._rem(summaries, candidates, date_str)

            with sleep_span("agentling.sleep.housekeeping", {"sleep.phase": "housekeeping", "sleep.date": date_str}):
                self._housekeeping(date)

        elapsed = time.monotonic() - start
        logger.info("[SLEEP] Cycle complete in %.1fs", elapsed)

    def _light_sleep(self, date: datetime) -> list[str]:
        """Phase 1: Discover conversations awaiting review.

        Returns context IDs whose parent journal was modified since yesterday's
        midnight and has been idle long enough to be safe to process. Supports
        both the new per-context directory layout and legacy flat ``*.jsonl``
        files; the store's ``iter_context_ids`` also migrates legacy files as a
        side effect.

        **Missed-work recovery:** the review window opens at the latest
        daily journal's own date (not the day after — a journal dated X
        covers conversations from X midnight only up to whenever that
        cycle actually ran), bounded by ``conversation_retention_days``.
        Conversations already summarised — their context id appears in a
        journal's coverage marker with a snapshot at or after their last
        modification — are then filtered out. Together these give the
        recovery invariant: **every conversation is either summarised
        into some journal or still inside the recovery window**, under
        any interleaving of total failures, partial failures (systemic
        abort or deadline expiry mid-cycle), and restarts — with no
        duplicate re-summarisation and no operator action. The journal
        content itself is the durable per-context coverage record.
        """
        data_dir = self._config.agent_data_dir
        cutoff_start = date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        cutoff_start = self._recovery_cutoff(cutoff_start)
        grace_cutoff = datetime.now(timezone.utc) - timedelta(seconds=IDLE_GRACE_SECONDS)
        summarized = self._summarized_context_index()

        context_ids: list[str] = []
        for ctx_id in self._store.iter_context_ids():
            path = self._store._path(ctx_id)  # noqa: SLF001 — internal path helper
            if not path.exists():
                continue
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if not (cutoff_start <= mtime <= grace_cutoff):
                continue
            if ctx_id in summarized and mtime <= summarized[ctx_id]:
                # Already summarised by a journal written after its last
                # modification — re-entering the window (overlap or
                # recovery) must not duplicate it.
                continue
            context_ids.append(ctx_id)

        return context_ids

    def _recovery_cutoff(self, default_cutoff: datetime) -> datetime:
        """Widen the review-window start to cover missed work.

        Once journalling has ever succeeded, the review window spans the
        **entire retention period** and the per-context coverage index
        does the exclusion work. A narrower resume point (e.g. the latest
        journal's date) cannot be trusted: a partial cycle after a
        multi-day gap writes a *recent* journal while older days remain
        uncovered — clipping the window at that journal would strand
        those conversations until retention deleted them. Selection is
        therefore: within retention AND not covered by any journal.

        The retention-wide window applies from the very first cycle: a
        data dir with no journals gets it too, because "no journal has
        ever succeeded" is indistinguishable from "the first scheduled
        cycle failed systemically before writing anything" — and that
        failed day must be retried before retention deletes it. On a
        genuinely fresh install the sweep is bounded by retention and the
        coverage index keeps it duplicate-free.

        **Pre-upgrade migration:** journals written before the coverage
        marker existed carry no per-context record, so a retention-wide
        window would re-summarize every conversation those journals
        already covered (duplicate spend and duplicate REM input). When
        journals exist but NONE carries a marker, review resumes from the
        latest journal's own date instead — the pre-marker boundary rule.
        Once any marker-bearing journal exists, the retention-wide window
        applies.
        """
        journals_dir = self._config.agent_data_dir / "journals"
        wide_cutoff = default_cutoff - timedelta(
            days=self._sleep_config.conversation_retention_days,
        )
        if not journals_dir.exists():
            return wide_cutoff
        journal_dates: list[datetime] = []
        any_marker = False
        for path in journals_dir.glob("*.md"):
            try:
                journal_dates.append(
                    datetime.strptime(path.stem, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc,
                    )
                )
            except ValueError:
                continue
            if not any_marker:
                try:
                    with path.open(encoding="utf-8") as fh:
                        first_line = fh.readline()
                except OSError:
                    continue
                if _parse_coverage_marker(first_line) is not None:
                    any_marker = True
        if not journal_dates:
            return wide_cutoff
        latest = max(journal_dates)
        if not any_marker:
            # Pre-upgrade data dir: coverage unknown, so trust the old
            # boundary — resume from the day the latest journal covered.
            resume = min(max(latest, wide_cutoff), default_cutoff)
            logger.info(
                "[SLEEP:LIGHT] Journals predate coverage markers; "
                "resuming review from %s (retention-wide recovery begins "
                "once a marker-bearing journal exists)",
                resume.strftime("%Y-%m-%d"),
            )
            return resume
        if latest < default_cutoff - timedelta(days=1):
            # More than the routine one-day overlap: cycles were missed.
            logger.warning(
                "[SLEEP:LIGHT] No journal since %s — reviewing the full "
                "retention window to recover unsummarized conversations "
                "from missed cycles",
                latest.strftime("%Y-%m-%d"),
            )
        return wide_cutoff

    def _summarized_context_index(self) -> dict[str, datetime]:
        """Map context id → mtime of the newest journal summarising it.

        Coverage is read exclusively from each journal's first-line
        machine-owned marker (see ``_write_journal``) — never from body
        content, which is model-authored and could otherwise forge
        phantom coverage (a forged entry would silently exclude a real
        conversation until retention deleted it). A conversation is
        considered covered iff a journal listing it was written after the
        conversation's last modification; anything else re-enters the
        review window (partial cycles, deadline expiries, or
        modifications after summarisation).
        """
        journals_dir = self._config.agent_data_dir / "journals"
        index: dict[str, datetime] = {}
        if not journals_dir.exists():
            return index
        for path in journals_dir.glob("*.md"):
            try:
                with path.open(encoding="utf-8") as fh:
                    first_line = fh.readline()
                journal_mtime = datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc,
                )
            except OSError:
                continue
            covered = _parse_coverage_marker(first_line)
            if not covered:
                continue
            for ctx_id, snapshot in covered.items():
                # Per-entry snapshot (conversation mtime captured at read
                # time) is authoritative; legacy entries without one fall
                # back to the journal file's mtime.
                stamp = (
                    datetime.fromtimestamp(snapshot, tz=timezone.utc)
                    if snapshot is not None
                    else journal_mtime
                )
                if ctx_id not in index or index[ctx_id] < stamp:
                    index[ctx_id] = stamp
        return index

    async def _deep_sleep(
        self,
        context_ids: list[str],
        date_str: str,
    ) -> tuple[list[str], list[MemoryCandidate]]:
        """Phase 2: Replay conversations, submit batch summaries, write journal."""
        system = build_system_prompt(self._config)
        memory = self._memory_store.load()
        memory_text = "\n".join(f"- {e.key}: {e.value}" for e in memory.entries)
        sleep_model = self._sleep_config.model

        summary_prompt = self._sleep_config.summary_prompt or DEFAULT_SUMMARY_PROMPT

        batch_requests: list[BatchRequest] = []
        # Conversation mtimes captured BEFORE reading: a message appended
        # while its summary is in flight lands after this snapshot, so the
        # freshness check (conversation mtime <= recorded snapshot) re-selects
        # the conversation next cycle instead of falsely treating it covered.
        snapshots: dict[str, float] = {}
        for ctx_id in context_ids:
            try:
                snapshots[ctx_id] = (
                    self._store._path(ctx_id).stat().st_mtime  # noqa: SLF001
                )
            except OSError:
                continue
            messages_data = self._store.replay(ctx_id)
            if not messages_data:
                continue

            conversation_text = self._format_conversation(messages_data)
            user_content = (
                f"{summary_prompt}\n\n"
                f"Current memory:\n{memory_text}\n\n"
                f"Conversation:\n{conversation_text}"
            )

            batch_requests.append(BatchRequest(
                custom_id=ctx_id,
                system=system,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=4096,
                output_schema=strict_json_schema(ConversationSummary),
            ))

        if not batch_requests:
            return [], []

        use_batches = self._sleep_config.batch and getattr(
            self._llm, "supports_batches", True,
        )
        if not use_batches:
            if self._sleep_config.batch:
                logger.info(
                    "[SLEEP:DEEP] LLM backend has no batches API support; "
                    "running %d summaries as live calls instead",
                    len(batch_requests),
                )
            all_results: list[BatchItemResult] = await self._summarize_live(
                batch_requests, sleep_model,
            )
        else:
            all_results = await self._summarize_batched(batch_requests, sleep_model)

        summaries: list[str] = []
        all_candidates: list[MemoryCandidate] = []
        covered: dict[str, float] = {}
        succeeded = 0
        failed = 0

        for item in all_results:
            if item.status == "failed":
                failed += 1
                logger.warning("[SLEEP:DEEP] Failed: %s — %s", item.custom_id, item.error)
                continue

            text = self._extract_structured_text(item.content)
            if not text or not text.strip():
                # An empty payload journals nothing usable — marking it
                # covered would silently lose the conversation to retention.
                # Count it as a per-item failure so it is retried next cycle.
                failed += 1
                logger.warning(
                    "[SLEEP:DEEP] Empty summary payload for %s — treating "
                    "as failed (will retry next cycle)", item.custom_id,
                )
                continue

            succeeded += 1
            if item.custom_id in snapshots:
                covered[item.custom_id] = snapshots[item.custom_id]
            try:
                parsed = ConversationSummary.model_validate_json(text)
                summaries.append(f"### {item.custom_id}\n{parsed.summary}")
                all_candidates.extend(parsed.memory_candidates)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("[SLEEP:DEEP] Parse error for %s: %s", item.custom_id, e)
                summaries.append(f"### {item.custom_id}\n{text}")

        logger.info(
            "[SLEEP:DEEP] Batch completed: %d succeeded, %d failed",
            succeeded, failed,
        )

        if all_results and not covered:
            # Nothing usable came back (every item failed or returned an
            # empty payload). Writing an empty journal here would mark the
            # night complete and suppress recovery — fail loudly instead so
            # the cycle re-enters these conversations next run.
            raise RuntimeError(
                f"deep sleep produced no usable summaries for {date_str} "
                f"({failed} of {len(all_results)} items failed)"
            )

        self._write_journal(date_str, "\n\n".join(summaries), covered)

        if all_candidates:
            logger.info("[SLEEP:DEEP] Extracted %d memory candidates", len(all_candidates))

        return summaries, all_candidates

    async def _summarize_batched(
        self,
        batch_requests: list[BatchRequest],
        sleep_model: str | None,
    ) -> list[BatchItemResult]:
        """Summarize via the Anthropic Message Batches API (50% cost, parallel)."""
        logger.info(
            "[SLEEP:DEEP] Submitting batch of %d summary requests", len(batch_requests)
        )
        batch_ids = await self._llm.batch_create(batch_requests, model=sleep_model)
        all_results: list[BatchItemResult] = []
        for batch_id in batch_ids:
            all_results.extend(await self._poll_batch(batch_id))
        return all_results

    async def _rem(
        self,
        summaries: list[str],
        candidates: list[MemoryCandidate],
        date_str: str,
    ) -> None:
        """Phase 3: Consolidate memory with today's learnings."""
        memory = self._memory_store.load()
        memory_text = "\n".join(
            f"- {e.key}: {e.value} (recorded: {e.recorded.isoformat()})"
            for e in memory.entries
        )
        journal_text = "\n\n".join(summaries)
        candidates_text = "\n".join(
            f"- {c.key}: {c.value}" for c in candidates
        ) if candidates else "(none)"

        consolidation_prompt = (
            self._sleep_config.consolidation_prompt or DEFAULT_CONSOLIDATION_PROMPT
        ).format(memory_max_entries=self._sleep_config.memory_max_entries)

        entries_before = len(memory.entries)
        logger.info(
            "[SLEEP:REM] Consolidating memory: %d existing + %d candidates",
            entries_before, len(candidates),
        )

        system = build_system_prompt(self._config)
        user_content = (
            f"{consolidation_prompt}\n\n"
            f"Date of the conversations under review: {date_str}\n\n"
            f"Current memory:\n{memory_text}\n\n"
            f"Today's journal:\n{journal_text}\n\n"
            f"New candidates:\n{candidates_text}"
        )

        response = await self._llm.complete(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            tools=[],
            output_schema=strict_json_schema(ConsolidatedMemory),
            max_tokens=self._sleep_config.consolidation_max_tokens,
            model=self._sleep_config.model,
            sleep_cycle=True,
        )

        text = self._extract_structured_text(response.content)
        try:
            consolidated = ConsolidatedMemory.model_validate_json(text)
            from agentlings.core.memory_models import MemoryStore
            new_store = MemoryStore(entries=consolidated.entries)
            self._memory_store.save(new_store)

            entries_after = len(consolidated.entries)
            logger.info(
                "[SLEEP:REM] Memory updated: %d entries (%+d)",
                entries_after, entries_after - entries_before,
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("[SLEEP:REM] Failed to parse consolidated memory: %s", e)

    def _housekeeping(self, date: datetime) -> None:
        """Phase 4: Delete old conversation and journal files."""
        import shutil

        data_dir = self._config.agent_data_dir
        journals_dir = data_dir / "journals"

        conv_cutoff = date - timedelta(days=self._sleep_config.conversation_retention_days)
        journal_cutoff = date - timedelta(days=self._sleep_config.journal_retention_days)

        conv_deleted = 0
        bytes_reclaimed = 0

        # New layout: delete the whole context directory if idle past retention.
        for ctx_id in self._store.iter_context_ids():
            ctx_dir = self._store.context_dir(ctx_id)
            parent = self._store._path(ctx_id)  # noqa: SLF001
            if parent.exists():
                mtime = datetime.fromtimestamp(parent.stat().st_mtime, tz=timezone.utc)
                if mtime < conv_cutoff:
                    size = sum(p.stat().st_size for p in ctx_dir.rglob("*") if p.is_file())
                    if ctx_dir.exists():
                        shutil.rmtree(ctx_dir)
                    conv_deleted += 1
                    bytes_reclaimed += size

        # Legacy flat files that iter_context_ids may have missed because they
        # were never migrated (e.g. context dir still absent).
        for path in data_dir.glob("*.jsonl"):
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime < conv_cutoff:
                size = path.stat().st_size
                path.unlink()
                conv_deleted += 1
                bytes_reclaimed += size

        journal_deleted = 0
        if journals_dir.exists():
            for path in journals_dir.glob("*.md"):
                try:
                    file_date = datetime.strptime(path.stem, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    if file_date < journal_cutoff:
                        size = path.stat().st_size
                        path.unlink()
                        journal_deleted += 1
                        bytes_reclaimed += size
                except ValueError:
                    continue

        if conv_deleted or journal_deleted:
            logger.info(
                "[SLEEP:HOUSEKEEPING] Deleted %d conversations, %d journals, reclaimed %dKB",
                conv_deleted, journal_deleted, bytes_reclaimed // 1024,
            )

    async def _summarize_live(
        self,
        requests: list[BatchRequest],
        sleep_model: str | None,
    ) -> list[BatchItemResult]:
        """Run deep-sleep summaries as sequential live completion calls.

        The workaround path for backends without a batches API (Ollama's
        Anthropic-compatibility layer, the OpenAI Responses wire format),
        selected via ``sleep.batch: false`` or automatically when the
        client advertises ``supports_batches = False``. The ``sleep.model``
        override is honoured per call, exactly like the batch path.
        Results are shaped as ``BatchItemResult`` so the downstream
        summary parsing is shared with the batch path.

        Failure semantics balance loudness against never losing work.
        Nothing requires operator action: an aborted night's
        conversations are recovered automatically by the next cycle's
        widened review window (see ``_recovery_cutoff``), and content is
        only ever lost if an outage outlasts
        ``conversation_retention_days``.

        * A per-item failure degrades to a failed item; the cycle
          continues — mirroring the batch path's per-request tolerance.
        * A **transient** systemic failure (connection/timeout, 429,
          5xx and equivalents) is retried with cycle-level patience
          (``_LIVE_TRANSIENT_RETRY_BACKOFFS``) — a brief rate limit or
          backend restart costs minutes, not the night. If it persists:
          with nothing yet succeeded the cycle aborts loudly (and the
          night auto-recovers next cycle); with completed work in hand,
          further calls stop but the completed summaries are kept and
          journaled.
        * A **hard** systemic failure (auth, permissions, bad model —
          retrying is pointless) aborts immediately when nothing has
          succeeded, or keeps completed work and skips the rest.
        * The whole pass runs under ``SLEEP_LIVE_DEADLINE_SECONDS``
          (mirroring the batch path's poll timeout): on expiry the
          remaining conversations are skipped, completed work is kept,
          and the skipped days auto-recover next cycle.
        * Every summary failing raises — an empty journal must never
          look like a successful quiet night.

        Raises:
            Exception: The underlying error on an early systemic failure,
                or ``RuntimeError`` when every summary failed.
        """
        results: list[BatchItemResult] = []
        succeeded = 0
        abort_reason: str | None = None
        leading_bad_requests = 0
        probe_far_next = False
        start = time.monotonic()
        queue = deque(requests)

        def _record_failure(req: BatchRequest, error: str) -> None:
            results.append(BatchItemResult(
                custom_id=req.custom_id, status="failed", error=error,
            ))

        while queue:
            # A far probe pulls the LAST pending conversation forward: two
            # leading request rejections must never condemn the whole night
            # on their own evidence (two specific poison conversations,
            # re-entered first every cycle, would wedge the agent forever).
            req = queue.pop() if probe_far_next else queue.popleft()
            probe_far_next = False
            if abort_reason is not None:
                _record_failure(req, f"skipped: {abort_reason}")
                continue

            attempt = 0
            while True:
                remaining = SLEEP_LIVE_DEADLINE_SECONDS - (time.monotonic() - start)
                if remaining <= 0:
                    abort_reason = "cycle deadline exceeded"
                    logger.error(
                        "[SLEEP:DEEP] Live-summary deadline (%.0fs) "
                        "exceeded after %d completed summaries; keeping "
                        "completed work — remaining conversations recover "
                        "next cycle",
                        SLEEP_LIVE_DEADLINE_SECONDS, succeeded,
                    )
                    _record_failure(req, "skipped: cycle deadline exceeded")
                    break
                try:
                    response = await asyncio.wait_for(
                        self._llm.complete(
                            system=req.system,
                            messages=req.messages,
                            tools=[],
                            output_schema=req.output_schema,
                            max_tokens=req.max_tokens,
                            model=sleep_model,
                            sleep_cycle=True,
                        ),
                        timeout=remaining,
                    )
                    succeeded += 1
                    results.append(BatchItemResult(
                        custom_id=req.custom_id,
                        content=response.content,
                        status="succeeded",
                    ))
                    break
                except Exception as e:  # noqa: BLE001 — one bad conversation must not kill the cycle
                    failure_class = _llm_failure_class(e)
                    if (
                        failure_class == "transient"
                        and attempt < len(_LIVE_TRANSIENT_RETRY_BACKOFFS)
                    ):
                        # Recompute the budget — the failed call itself may
                        # have consumed most of it; a stale pre-call value
                        # could sleep well past the deadline.
                        remaining_now = SLEEP_LIVE_DEADLINE_SECONDS - (
                            time.monotonic() - start
                        )
                        delay = min(
                            _LIVE_TRANSIENT_RETRY_BACKOFFS[attempt],
                            max(remaining_now, 0.0),
                        )
                        attempt += 1
                        logger.warning(
                            "[SLEEP:DEEP] Transient failure on %s "
                            "(attempt %d); retrying in %.0fs: %s",
                            req.custom_id, attempt, delay, e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    if failure_class is not None:
                        if succeeded == 0:
                            logger.error(
                                "[SLEEP:DEEP] %s LLM failure on the first "
                                "summary (%s); aborting — this night's "
                                "conversations are recovered automatically "
                                "by the next cycle's widened window: %s",
                                failure_class.capitalize(), req.custom_id, e,
                            )
                            raise
                        abort_reason = f"{failure_class} failure: {e}"
                        logger.error(
                            "[SLEEP:DEEP] %s LLM failure after %d "
                            "completed summaries (at %s); keeping "
                            "completed work — skipped conversations "
                            "recover next cycle: %s",
                            failure_class.capitalize(), succeeded,
                            req.custom_id, e,
                        )
                        _record_failure(req, str(e))
                        break
                    if succeeded == 0 and _is_bad_request(e):
                        # Leading 400s are ambiguous — poison conversations
                        # or systemic misconfiguration. Two in a row
                        # trigger a probe of the FARTHEST pending
                        # conversation; only when that third, distinct
                        # conversation is also rejected is the night
                        # declared systemic. No two specific conversations
                        # can therefore permanently block the cycle.
                        leading_bad_requests += 1
                        if leading_bad_requests >= 3:
                            logger.error(
                                "[SLEEP:DEEP] %d distinct conversations "
                                "(including a far probe) all failed with "
                                "request rejections; treating as systemic "
                                "and aborting (recovered automatically "
                                "next cycle): %s",
                                leading_bad_requests, e,
                            )
                            raise
                        if leading_bad_requests == 2 and queue:
                            logger.warning(
                                "[SLEEP:DEEP] Two leading request "
                                "rejections; probing a later conversation "
                                "before declaring systemic",
                            )
                            probe_far_next = True
                    logger.warning(
                        "[SLEEP:DEEP] Live summary failed for %s: %s",
                        req.custom_id, e,
                    )
                    _record_failure(req, str(e))
                    break

        if results and succeeded == 0:
            raise RuntimeError(
                f"all {len(results)} live deep-sleep summaries failed "
                f"(last error: {results[-1].error}); aborting the sleep cycle"
            )
        return results

    async def _poll_batch(
        self,
        batch_id: str,
        timeout: float = 7200,
        initial_interval: float = 5,
        max_interval: float = 60,
    ) -> list[Any]:
        """Poll a batch until completion or timeout, using exponential backoff."""
        deadline = time.monotonic() + timeout
        interval = initial_interval
        while time.monotonic() < deadline:
            status = await self._llm.batch_status(batch_id)
            if status.processing_status == "ended":
                return await self._fetch_results(batch_id)
            logger.debug(
                "[SLEEP:DEEP] Batch %s: %s (next poll in %.0fs)",
                batch_id, status.processing_status, interval,
            )
            await asyncio.sleep(interval)
            interval = min(interval * 2, max_interval)

        logger.warning("[SLEEP:DEEP] Batch %s timed out after %.0fs", batch_id, timeout)
        return await self._fetch_results(batch_id)

    async def _fetch_results(self, batch_id: str) -> list[Any]:
        """Retrieve batch results, degrading to ``[]`` on any failure.

        A failure here must not abort the sleep cycle. The known case is the
        gateway handing back a ``results_url`` that points straight at the
        upstream API (bypassing the gateway's auth), so the SDK's results fetch
        401s — but any transient error is treated the same way. We log loudly
        and skip this batch's summaries so the run can still consolidate memory
        and run housekeeping rather than crashing.
        """
        try:
            return await self._llm.batch_results(batch_id)
        except Exception:
            logger.exception(
                "[SLEEP:DEEP] Failed to retrieve results for batch %s; "
                "skipping this batch", batch_id,
            )
            return []

    def _write_journal(
        self, date_str: str, sections: str, covered: dict[str, float],
    ) -> None:
        """Write (or merge into) the daily journal.

        The first line is a machine-owned coverage marker naming every
        context this journal summarises, each paired with the source
        conversation's mtime snapshot captured when it was read
        (``ctx:epoch``). It is the ONLY line the coverage index reads:
        model-authored summary text can never forge or destroy coverage,
        because body content cannot occupy line one — and freshness is
        judged per entry from the snapshot, never from the journal file's
        own mtime, which merges advance for unrelated older entries.

        A journal that already exists for the date (a same-day re-run —
        the intended recovery flow after a partial cycle) is **merged**:
        coverage entries union (newest snapshot wins per context) and the
        new sections append after the existing body. Overwriting would
        destroy the prior coverage record and the summaries it indexed.
        """
        journals_dir = self._config.agent_data_dir / "journals"
        journals_dir.mkdir(parents=True, exist_ok=True)
        path = journals_dir / f"{date_str}.md"

        coverage: dict[str, float | None] = dict(covered)
        body = f"# Journal — {date_str}\n\n{sections}"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            first_line, _, rest = existing.partition("\n")
            existing_entries = _parse_coverage_marker(first_line)
            if existing_entries is not None:
                journal_mtime = path.stat().st_mtime
                for ctx_id, snapshot in existing_entries.items():
                    prior = journal_mtime if snapshot is None else snapshot
                    mine = coverage.get(ctx_id)
                    if mine is None or mine < prior:
                        coverage[ctx_id] = prior
                existing_body = rest
            else:
                existing_body = existing
            body = (
                existing_body.rstrip("\n")
                + f"\n\n## Recovery pass\n\n{sections}"
            )
            logger.info(
                "[SLEEP:DEEP] Merging into existing journal for %s "
                "(%d newly covered conversations)", date_str, len(covered),
            )
        # Context ids are client-supplied (A2A/MCP contextId) and may
        # contain whitespace, newlines, or colons — percent-encode each id
        # so the single-line space-joined marker survives any id.
        entries = " ".join(
            f"{quote(ctx, safe='')}:{stamp:.6f}"
            if stamp is not None else quote(ctx, safe="")
            for ctx, stamp in sorted(coverage.items())
        )
        marker = _COVERAGE_MARKER_TEMPLATE.format(ids=entries)
        path.write_text(f"{marker}\n{body}\n", encoding="utf-8")
        logger.info("[SLEEP:DEEP] Journal written: %s", path)

    @staticmethod
    def _format_conversation(messages: list[dict[str, Any]]) -> str:
        """Format replayed messages into readable text for the summary prompt."""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str):
                lines.append(f"{role}: {content}")
            elif isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            parts.append(f"[tool: {block.get('name', '?')}]")
                        elif block.get("type") == "tool_result":
                            parts.append(f"[result: {str(block.get('content', ''))[:200]}]")
                if parts:
                    lines.append(f"{role}: {' '.join(parts)}")
        return "\n".join(lines)

    @staticmethod
    def _extract_structured_text(content: list[dict[str, Any]]) -> str:
        """Extract text from content blocks (structured output comes as text blocks)."""
        for block in content:
            if block.get("type") == "text":
                return block.get("text", "")
        return ""
