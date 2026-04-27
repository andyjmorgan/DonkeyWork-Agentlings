"""``agentling upgrade`` — reconcile a dir's data against the installed framework.

The CLI never installs or upgrades the package itself — that's the package
manager's job (``uv pip install --upgrade agentlings`` or equivalent). This
command applies any pending data-layout migrations and bumps the dir's
``.framework-version`` stamp once they all succeed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from agentlings.cli import _migrations, _version
from agentlings.cli.init import DATA_DIRNAME

logger = logging.getLogger(__name__)


@dataclass
class UpgradeResult:
    recorded_version: str | None
    installed_version: str
    applied: list[_migrations.MigrationResult]
    pending_count: int


def upgrade_agent(agent_dir: Path | None = None, *, dry_run: bool = False) -> UpgradeResult:
    """Run pending migrations and stamp the new framework version.

    Args:
        agent_dir: Directory containing ``agent.yaml`` and ``data/``. Defaults
            to CWD.
        dry_run: Report what would run without executing migrations or
            advancing ``.framework-version``.

    Raises:
        FileNotFoundError: When the directory has no ``data/``.
        RuntimeError: When the installed framework is older than what
            scaffolded the directory — downgrades are not supported.
    """
    target = (agent_dir or Path.cwd()).resolve()
    data_dir = target / DATA_DIRNAME
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{target} does not look like an agent dir (no {DATA_DIRNAME}/ subdirectory)"
        )

    recorded = _version.read_dir_version(target)
    installed = _version.installed_version()

    if recorded and _is_newer(recorded, installed):
        raise RuntimeError(
            f"installed framework ({installed}) is older than what scaffolded "
            f"this directory ({recorded}); downgrades are not supported. "
            f"Run 'pip install agentlings=={recorded}' to restore."
        )

    pending = _migrations.pending(data_dir)
    applied = _migrations.run_pending(data_dir, dry_run=dry_run)

    if not dry_run and applied:
        _version.write_dir_version(target, installed)
    elif not dry_run and not applied and recorded != installed:
        # No migrations to run, but the version stamp is stale — bump it so
        # the next upgrade has an accurate baseline.
        _version.write_dir_version(target, installed)

    return UpgradeResult(
        recorded_version=recorded,
        installed_version=installed,
        applied=applied,
        pending_count=len(pending),
    )


def _is_newer(a: str, b: str) -> bool:
    """Return True when ``a`` parses to a strictly newer semver than ``b``.

    Falls back to lexicographic compare when either side is unparseable —
    rare in practice (PEP 440 versions always parse) but safe.
    """
    try:
        ta = tuple(int(p) for p in a.split(".")[:3])
        tb = tuple(int(p) for p in b.split(".")[:3])
        return ta > tb
    except ValueError:
        return a > b
