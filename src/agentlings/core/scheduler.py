"""Asyncio-based cron scheduler for the sleep cycle."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Coroutine

logger = logging.getLogger(__name__)


def parse_cron_field(field: str, min_val: int, max_val: int) -> set[int]:
    """Parse a single cron field into a set of matching integer values.

    Supports ``*``, ranges (``1-5``), steps (``*/2``), and comma-separated lists.

    Args:
        field: The cron field string.
        min_val: Minimum valid value for this field.
        max_val: Maximum valid value for this field.

    Returns:
        Set of integer values that match the field expression.
    """
    values: set[int] = set()

    def _validate(val: int, label: str) -> int:
        if val < min_val or val > max_val:
            raise ValueError(f"{label} {val} out of bounds [{min_val}, {max_val}] in '{field}'")
        return val

    for part in field.split(","):
        part = part.strip()
        if "/" in part:
            base, step_str = part.split("/", 1)
            step = int(step_str)
            if step <= 0:
                raise ValueError(f"Step must be positive in '{field}', got {step}")
            if base == "*":
                start, end = min_val, max_val
            elif "-" in base:
                lo, hi = base.split("-", 1)
                start = _validate(int(lo), "Range start")
                end = _validate(int(hi), "Range end")
            else:
                start = _validate(int(base), "Value")
                end = max_val
            values.update(range(start, end + 1, step))
        elif part == "*":
            values.update(range(min_val, max_val + 1))
        elif "-" in part:
            lo, hi = part.split("-", 1)
            start = _validate(int(lo), "Range start")
            end = _validate(int(hi), "Range end")
            values.update(range(start, end + 1))
        else:
            values.add(_validate(int(part), "Value"))

    return values


def cron_matches(expression: str, dt: datetime) -> bool:
    """Check if a datetime matches a 5-field cron expression.

    Fields: minute hour day-of-month month day-of-week (0=Sunday).

    Args:
        expression: Standard 5-field cron expression.
        dt: The datetime to check.

    Returns:
        Whether the datetime matches the expression.
    """
    fields = expression.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Expected 5 cron fields, got {len(fields)}: {expression}")

    minute = parse_cron_field(fields[0], 0, 59)
    hour = parse_cron_field(fields[1], 0, 23)
    dom = parse_cron_field(fields[2], 1, 31)
    month = parse_cron_field(fields[3], 1, 12)
    dow = parse_cron_field(fields[4], 0, 6)

    return (
        dt.minute in minute
        and dt.hour in hour
        and dt.day in dom
        and dt.month in month
        and dt.weekday() in _convert_dow(dow)
    )


def _convert_dow(cron_dow: set[int]) -> set[int]:
    """Convert cron day-of-week (0=Sunday) to Python weekday (0=Monday)."""
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    invalid = cron_dow - mapping.keys()
    if invalid:
        raise ValueError(f"Invalid day-of-week values: {sorted(invalid)} (must be 0-6)")
    return {mapping[d] for d in cron_dow}


async def run_scheduler(
    expression: str,
    callback: Callable[[], Coroutine],
    check_interval: float = 60,
) -> None:
    """Run an asyncio loop that fires a callback when the cron expression matches.

    Checks once per ``check_interval`` seconds. Skips if the callback is
    already running. Designed to run as a background ``asyncio.Task``.

    Args:
        expression: 5-field cron expression.
        callback: Async callable to invoke on match.
        check_interval: Seconds between cron checks.
    """
    last_fire: datetime | None = None
    running = False

    while True:
        await asyncio.sleep(check_interval)

        now = datetime.now(timezone.utc)
        if running:
            continue

        if cron_matches(expression, now):
            fire_key = now.replace(second=0, microsecond=0)
            if last_fire == fire_key:
                continue

            last_fire = fire_key
            running = True
            try:
                logger.info("cron match at %s, firing callback", now.isoformat())
                await callback()
            except Exception:
                logger.exception("scheduler callback failed")
            finally:
                running = False
