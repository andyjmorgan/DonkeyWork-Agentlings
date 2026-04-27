"""Seed migration — establishes the migration log without changing data.

Acts as a no-op so the runner has at least one migration to find on a fresh
install. Real schema migrations land alongside this one as future framework
versions ship.
"""

from __future__ import annotations

from pathlib import Path

ID = "0001_seed"
DESCRIPTION = "Initialise migration log"


def apply(data_dir: Path) -> None:  # noqa: ARG001
    return None
