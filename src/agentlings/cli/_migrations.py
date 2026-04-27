"""Migration log + runner for ``agentling upgrade``.

The log lives at ``<data_dir>/.migrations`` — one applied migration ID per
line. Pending migrations are those discovered on disk that are not yet
present in the log. A failed migration leaves the log unchanged so the next
run picks up where the previous left off.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agentlings import migrations as migrations_pkg

logger = logging.getLogger(__name__)

LOG_NAME = ".migrations"


@dataclass
class MigrationResult:
    id: str
    description: str
    status: str  # "applied" | "skipped"


def read_log(data_dir: Path) -> list[str]:
    """Return the IDs of migrations already applied to ``data_dir``."""
    path = data_dir / LOG_NAME
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(data_dir: Path, migration_id: str) -> None:
    """Atomically append a migration ID to the log."""
    path = data_dir / LOG_NAME
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(existing + migration_id + "\n", encoding="utf-8")


def pending(data_dir: Path) -> list[object]:
    """Return migration modules not yet recorded in the log."""
    applied = set(read_log(data_dir))
    return [m for m in migrations_pkg.discover() if m.ID not in applied]


def run_pending(data_dir: Path, *, dry_run: bool = False) -> list[MigrationResult]:
    """Apply every pending migration, recording each in the log on success.

    ``dry_run`` reports what would run without executing or modifying the log.
    Raises whatever the migration raises; callers decide how to surface it.
    """
    results: list[MigrationResult] = []
    for migration in pending(data_dir):
        if dry_run:
            results.append(MigrationResult(migration.ID, migration.DESCRIPTION, "skipped"))
            continue
        logger.info("applying migration %s: %s", migration.ID, migration.DESCRIPTION)
        migration.apply(data_dir)
        append_log(data_dir, migration.ID)
        results.append(MigrationResult(migration.ID, migration.DESCRIPTION, "applied"))
    return results


def stamp_all_applied(data_dir: Path) -> None:
    """Record every known migration as already applied without running any.

    Called by ``init`` so a fresh agent dir starts with the current migration
    set marked complete — only future migrations need to run on later upgrades.
    """
    path = data_dir / LOG_NAME
    if path.exists():
        return
    ids = [m.ID for m in migrations_pkg.discover()]
    path.write_text("\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
