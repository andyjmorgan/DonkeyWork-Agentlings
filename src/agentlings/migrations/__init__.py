"""Forward-only data-layout migrations applied by ``agentling upgrade``.

Each migration is a module exporting:

- ``ID: str`` — a stable identifier (typically the module name)
- ``DESCRIPTION: str`` — one-line human-readable summary
- ``apply(data_dir: Path) -> None`` — idempotent transformation

Migrations are discovered at import time and ordered by their numeric prefix.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Protocol


class Migration(Protocol):
    ID: str
    DESCRIPTION: str

    def apply(self, data_dir: Path) -> None: ...


def discover() -> list[Migration]:
    """Return all migration modules in this package, ordered by ID."""
    migrations: list[Migration] = []
    for info in pkgutil.iter_modules(__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        if hasattr(module, "ID") and hasattr(module, "apply"):
            migrations.append(module)  # type: ignore[arg-type]
    migrations.sort(key=lambda m: m.ID)
    return migrations
