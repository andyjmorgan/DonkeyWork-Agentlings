"""Tests for the cron scheduler."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agentlings.core.scheduler import cron_matches, parse_cron_field


class TestParseCronField:
    def test_wildcard(self) -> None:
        assert parse_cron_field("*", 0, 59) == set(range(0, 60))

    def test_single_value(self) -> None:
        assert parse_cron_field("5", 0, 59) == {5}

    def test_range(self) -> None:
        assert parse_cron_field("1-5", 0, 59) == {1, 2, 3, 4, 5}

    def test_step(self) -> None:
        assert parse_cron_field("*/15", 0, 59) == {0, 15, 30, 45}

    def test_comma_separated(self) -> None:
        assert parse_cron_field("1,5,10", 0, 59) == {1, 5, 10}

    def test_range_with_step(self) -> None:
        assert parse_cron_field("0-10/5", 0, 59) == {0, 5, 10}


class TestCronMatches:
    def test_every_minute(self) -> None:
        dt = datetime(2026, 4, 1, 12, 30, tzinfo=timezone.utc)
        assert cron_matches("* * * * *", dt)

    def test_specific_time(self) -> None:
        dt = datetime(2026, 4, 1, 2, 0, tzinfo=timezone.utc)
        assert cron_matches("0 2 * * *", dt)

    def test_no_match(self) -> None:
        dt = datetime(2026, 4, 1, 3, 0, tzinfo=timezone.utc)
        assert not cron_matches("0 2 * * *", dt)

    def test_day_of_week(self) -> None:
        dt = datetime(2026, 4, 6, 0, 0, tzinfo=timezone.utc)  # Monday
        assert cron_matches("0 0 * * 1", dt)
        assert not cron_matches("0 0 * * 0", dt)

    def test_invalid_field_count(self) -> None:
        with pytest.raises(ValueError, match="Expected 5 cron fields"):
            cron_matches("* * *", datetime.now(timezone.utc))
