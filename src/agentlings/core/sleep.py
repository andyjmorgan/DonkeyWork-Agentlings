"""Nightly sleep cycle: journal, consolidate memory, and clean up."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from agentlings.config import AgentConfig, SleepConfig
from agentlings.core.llm import BaseLLMClient, BatchRequest
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

DEFAULT_SUMMARY_PROMPT = """\
You are performing a nightly review of a conversation that took place today.

Produce a concise summary of what happened: what was asked, what actions were \
taken, what the outcome was, and anything left unresolved.

Extract any facts worth adding to your long-term memory. Only extract NEW facts \
not already in your current memory. Focus on operational knowledge, patterns, \
decisions, things that changed. Ignore passing context.

If the conversation was trivial or contained nothing new worth remembering, \
return an empty memory_candidates list."""

DEFAULT_CONSOLIDATION_PROMPT = """\
You are performing nightly memory maintenance.

Your job:
1. Integrate new candidates that add genuine value. Deduplicate against existing entries.
2. Review every existing entry. Is it still relevant? Has it been superseded by \
something learned today? Would it help you do your job tomorrow?
3. Drop anything that is stale, redundant, or no longer operationally useful.
4. You have a hard limit of {memory_max_entries} entries.

Preserve the recorded timestamp for entries you keep unchanged.
Set recorded to the current date for new or modified entries."""


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
                conversations = self._light_sleep(date)
                ls.set_attribute("sleep.conversations_found", len(conversations))
                if not conversations:
                    ls.set_attribute("sleep.skipped", True)
                    logger.info("[SLEEP:LIGHT] No conversations found, skipping cycle")
                    return

            logger.info("[SLEEP:LIGHT] Found %d conversations, proceeding", len(conversations))

            with sleep_span("agentling.sleep.deep_sleep", {"sleep.phase": "deep_sleep", "sleep.date": date_str}):
                summaries, candidates = await self._deep_sleep(conversations, date_str)

            if summaries:
                with sleep_span("agentling.sleep.rem", {"sleep.phase": "rem", "sleep.date": date_str}):
                    await self._rem(summaries, candidates, date_str)

            with sleep_span("agentling.sleep.housekeeping", {"sleep.phase": "housekeeping", "sleep.date": date_str}):
                self._housekeeping(date)

        elapsed = time.monotonic() - start
        logger.info("[SLEEP] Cycle complete in %.1fs", elapsed)

    def _light_sleep(self, date: datetime) -> list[Path]:
        """Phase 1: Discover conversations from the previous 24 hours.

        Returns JSONL files that were modified since yesterday's midnight and
        have been idle long enough to be safe to process.
        """
        data_dir = self._config.agent_data_dir
        cutoff_start = date.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        grace_cutoff = datetime.now(timezone.utc) - timedelta(seconds=IDLE_GRACE_SECONDS)

        conversations = []
        for path in data_dir.glob("*.jsonl"):
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            if mtime >= cutoff_start and mtime <= grace_cutoff:
                conversations.append(path)

        return conversations

    async def _deep_sleep(
        self,
        conversations: list[Path],
        date_str: str,
    ) -> tuple[list[str], list[MemoryCandidate]]:
        """Phase 2: Replay conversations, submit batch summaries, write journal."""
        system = build_system_prompt(self._config)
        memory = self._memory_store.load()
        memory_text = "\n".join(f"- {e.key}: {e.value}" for e in memory.entries)
        sleep_model = self._sleep_config.model

        summary_prompt = self._sleep_config.summary_prompt or DEFAULT_SUMMARY_PROMPT

        batch_requests: list[BatchRequest] = []
        for path in conversations:
            ctx_id = path.stem
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

        logger.info("[SLEEP:DEEP] Submitting batch of %d summary requests", len(batch_requests))

        batch_ids = await self._llm.batch_create(batch_requests, model=sleep_model)

        all_results = []
        for batch_id in batch_ids:
            results = await self._poll_batch(batch_id)
            all_results.extend(results)

        summaries: list[str] = []
        all_candidates: list[MemoryCandidate] = []
        succeeded = 0
        failed = 0

        for item in all_results:
            if item.status == "failed":
                failed += 1
                logger.warning("[SLEEP:DEEP] Failed: %s — %s", item.custom_id, item.error)
                continue

            succeeded += 1
            text = self._extract_structured_text(item.content)
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

        journal_content = f"# Journal — {date_str}\n\n" + "\n\n".join(summaries)
        self._write_journal(date_str, journal_content)

        if all_candidates:
            logger.info("[SLEEP:DEEP] Extracted %d memory candidates", len(all_candidates))

        return summaries, all_candidates

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
            f"Current memory:\n{memory_text}\n\n"
            f"Today's journal:\n{journal_text}\n\n"
            f"New candidates:\n{candidates_text}"
        )

        response = await self._llm.complete(
            system=system,
            messages=[{"role": "user", "content": user_content}],
            tools=[],
            output_schema=strict_json_schema(ConsolidatedMemory),
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
        data_dir = self._config.agent_data_dir
        journals_dir = data_dir / "journals"

        conv_cutoff = date - timedelta(days=self._sleep_config.conversation_retention_days)
        journal_cutoff = date - timedelta(days=self._sleep_config.journal_retention_days)

        conv_deleted = 0
        bytes_reclaimed = 0

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
                return await self._llm.batch_results(batch_id)
            logger.debug(
                "[SLEEP:DEEP] Batch %s: %s (next poll in %.0fs)",
                batch_id, status.processing_status, interval,
            )
            await asyncio.sleep(interval)
            interval = min(interval * 2, max_interval)

        logger.warning("[SLEEP:DEEP] Batch %s timed out after %.0fs", batch_id, timeout)
        try:
            return await self._llm.batch_results(batch_id)
        except Exception:
            return []

    def _write_journal(self, date_str: str, content: str) -> None:
        """Write the daily journal to the journals directory."""
        journals_dir = self._config.agent_data_dir / "journals"
        journals_dir.mkdir(parents=True, exist_ok=True)
        path = journals_dir / f"{date_str}.md"
        path.write_text(content, encoding="utf-8")
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
