"""Folder-scan loader for ``@tool``-decorated user tools.

Discovery contract:

- The framework scans ``AGENT_TOOLS_DIR`` (a directory) for ``*.py`` files.
- Each file is imported as an isolated module under the synthetic package
  ``agentling_user_tools.<filename_stem>``; the directory is *not* added to
  ``sys.path`` so user tools cannot accidentally shadow installed packages.
- Files whose name begins with ``_`` are skipped (private/helpers).
- Every module-level attribute that is a ``Tool`` instance is registered
  with the supplied ``ToolRegistry``.
- Import failures and registration failures are logged and skipped — one
  broken tool must not crash the agent.

The loader is intentionally permissive about *what* a user file contains:
it only registers ``Tool`` instances and ignores everything else, so a file
can define helpers, classes, env-var reads, or whatever the tool author
needs.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from agentlings.tools.decorator import Tool

if TYPE_CHECKING:
    from agentlings.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_USER_TOOLS_PACKAGE = "agentling_user_tools"


def load_tools_from_directory(
    directory: Path,
    registry: "ToolRegistry",
) -> list[str]:
    """Scan ``directory`` and register every ``Tool`` instance found.

    Args:
        directory: Filesystem path to scan. ``.py`` files at the top level
            are imported (no recursion, no sub-packages).
        registry: The ``ToolRegistry`` to register discovered tools into.

    Returns:
        The names of tools that were successfully registered. The list is in
        directory-walk order, which the caller may sort if it wants stable
        output.

    The function never raises for content issues: missing directory, broken
    imports, and registration failures are logged and the scan continues. It
    only raises for argument-shape problems (wrong types passed in).
    """
    if not directory.exists():
        logger.info(
            "tool directory %s does not exist — skipping scan", directory
        )
        return []
    if not directory.is_dir():
        logger.warning(
            "AGENT_TOOLS_DIR=%s is not a directory — skipping scan", directory
        )
        return []

    registered: list[str] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.suffix != ".py":
            continue
        if path.stem.startswith("_"):
            logger.debug("skipping private tool file: %s", path)
            continue

        module_name = f"{_USER_TOOLS_PACKAGE}.{path.stem}"
        try:
            module = _import_isolated_module(module_name, path)
        except Exception:  # noqa: BLE001 — log and continue to next file
            logger.exception("failed to import tool file %s", path)
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if not isinstance(attr, Tool):
                continue
            try:
                registry.register_tool_object(attr)
                registered.append(attr.name)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to register tool %r from %s", attr.name, path
                )

    if registered:
        logger.info(
            "loaded %d user tool(s) from %s: %s",
            len(registered),
            directory,
            sorted(registered),
        )
    else:
        logger.info("no user tools registered from %s", directory)

    return registered


def _import_isolated_module(module_name: str, path: Path) -> object:
    """Import a single file as ``module_name`` without polluting ``sys.path``.

    Uses ``importlib.util.spec_from_file_location`` so the file is imported
    by absolute path; the parent directory is never injected into the
    interpreter's import paths.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"could not build import spec for {path} as {module_name}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
