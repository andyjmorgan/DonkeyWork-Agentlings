from __future__ import annotations

import fcntl
import json
import logging
from pathlib import Path
from typing import Any

from agentlings.models import CompactionEntry, JournalEntry, MessageEntry

logger = logging.getLogger(__name__)


class ContextNotFoundError(Exception):
    pass


class JournalStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ctx_id: str) -> Path:
        return self._data_dir / f"{ctx_id}.jsonl"

    def create(self, ctx_id: str) -> None:
        path = self._path(ctx_id)
        path.touch(exist_ok=True)
        logger.info("created context %s", ctx_id)

    def exists(self, ctx_id: str) -> bool:
        return self._path(ctx_id).exists()

    def append(self, ctx_id: str, entry: MessageEntry | CompactionEntry) -> None:
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)

        line = entry.model_dump_json() + "\n"
        with open(path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

        logger.debug("appended %s entry to context %s", entry.t, ctx_id)

    def replay(self, ctx_id: str) -> list[dict[str, Any]]:
        path = self._path(ctx_id)
        if not path.exists():
            raise ContextNotFoundError(ctx_id)

        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return []

        parsed = [json.loads(line) for line in lines]

        cursor = 0
        for i in range(len(parsed) - 1, -1, -1):
            if parsed[i]["t"] == "compact":
                cursor = i
                break

        messages: list[dict[str, Any]] = []
        for entry in parsed[cursor:]:
            if entry["t"] == "compact":
                messages.append({
                    "role": "assistant",
                    "content": entry["content"],
                })
            elif entry["t"] == "msg":
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"],
                })

        return messages
