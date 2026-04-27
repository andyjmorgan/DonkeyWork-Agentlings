"""Framework version detection and per-agent-dir version stamping."""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

VERSION_FILE = ".framework-version"


def installed_version() -> str:
    """Return the version of the currently-installed ``agentlings`` package."""
    return metadata.version("agentlings")


def read_dir_version(agent_dir: Path) -> str | None:
    """Return the framework version recorded in ``<agent_dir>/.framework-version``.

    Returns ``None`` when the file is missing or empty — callers treat that as
    "this dir was set up before version stamping existed" and fall back to
    running every migration.
    """
    path = agent_dir / VERSION_FILE
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def write_dir_version(agent_dir: Path, version: str) -> None:
    """Stamp ``version`` into ``<agent_dir>/.framework-version``."""
    path = agent_dir / VERSION_FILE
    path.write_text(version + "\n", encoding="utf-8")
