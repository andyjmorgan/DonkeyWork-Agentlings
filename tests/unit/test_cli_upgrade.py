"""Tests for ``agentling upgrade``: version checks, migration log, dry-run."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentlings.cli import _migrations, _version
from agentlings.cli.init import init_agent
from agentlings.cli.upgrade import upgrade_agent


class TestNoOp:
    def test_freshly_initialized_dir_has_no_pending(self, tmp_path: Path) -> None:
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        result = upgrade_agent(target)
        assert result.applied == []
        assert result.pending_count == 0
        assert result.installed_version == _version.installed_version()

    def test_missing_data_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            upgrade_agent(tmp_path)


class TestPendingMigrations:
    def test_empty_log_makes_all_migrations_pending(self, tmp_path: Path) -> None:
        """A dir from before migration logging exists has no .migrations file.

        Upgrade should detect every known migration as pending and apply them.
        """
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        # Simulate a pre-stamping dir by deleting the log.
        log_path = target / "data" / _migrations.LOG_NAME
        log_path.unlink()
        result = upgrade_agent(target)
        assert len(result.applied) >= 1
        assert all(r.status == "applied" for r in result.applied)
        # Log is now populated so a follow-up call is a no-op.
        again = upgrade_agent(target)
        assert again.applied == []

    def test_dry_run_does_not_modify_log(self, tmp_path: Path) -> None:
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        (target / "data" / _migrations.LOG_NAME).unlink()
        before = _migrations.read_log(target / "data")
        result = upgrade_agent(target, dry_run=True)
        after = _migrations.read_log(target / "data")
        assert before == after  # log unchanged
        assert all(r.status == "skipped" for r in result.applied)


class TestVersionDowngradeRejection:
    def test_recorded_newer_than_installed_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        # Pretend this dir was set up by a future framework version.
        _version.write_dir_version(target, "999.0.0")
        with pytest.raises(RuntimeError, match="downgrades are not supported"):
            upgrade_agent(target)


class TestPartialFailureResume:
    def test_failed_migration_keeps_log_consistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A migration raising mid-run must NOT advance the log past it.

        The next upgrade re-runs the failed migration. Earlier migrations stay
        applied.
        """
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        # Reset to "no migrations applied" state.
        (target / "data" / _migrations.LOG_NAME).unlink()

        # Inject a fake migration that raises, ordered after the seed.
        class _Failing:
            ID = "0002_failing"
            DESCRIPTION = "deliberately broken"

            @staticmethod
            def apply(data_dir: Path) -> None:
                raise RuntimeError("boom")

        from agentlings import migrations as migrations_pkg

        real_discover = migrations_pkg.discover
        monkeypatch.setattr(
            migrations_pkg, "discover", lambda: [*real_discover(), _Failing]
        )

        with pytest.raises(RuntimeError, match="boom"):
            upgrade_agent(target)

        # Seed migration applied; failing one is NOT in the log.
        log = _migrations.read_log(target / "data")
        assert "0001_seed" in log
        assert "0002_failing" not in log
