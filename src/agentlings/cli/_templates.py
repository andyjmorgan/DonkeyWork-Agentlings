"""Bundled-template lookup for ``agentling init``.

Templates ship as package data under ``src/agentlings/templates/<name>/``.
Each template directory contains an ``agent.yaml`` (with ``{{NAME}}`` and
optional ``{{API_KEY}}`` placeholders substituted at scaffold time) and an
``.env.example`` file. ``--from-git`` is a future milestone; v1 only loads
from this bundled directory.
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable

TEMPLATES_PACKAGE = "agentlings.templates"


def list_templates() -> list[str]:
    """Return the names of all bundled templates.

    A template is any subdirectory of the ``agentlings.templates`` package
    that contains an ``agent.yaml``.
    """
    root = resources.files(TEMPLATES_PACKAGE)
    names = []
    for entry in root.iterdir():
        if entry.is_dir() and (entry / "agent.yaml").is_file():
            names.append(entry.name)
    return sorted(names)


def template_root(name: str) -> Traversable:
    """Return the resource root for a named template, raising if missing."""
    root = resources.files(TEMPLATES_PACKAGE) / name
    if not (root / "agent.yaml").is_file():
        available = ", ".join(list_templates()) or "(none)"
        raise ValueError(
            f"unknown template '{name}'; available templates: {available}"
        )
    return root


def render_yaml(name: str, agent_name: str) -> str:
    """Load ``agent.yaml`` from the named template and substitute placeholders."""
    raw = (template_root(name) / "agent.yaml").read_text(encoding="utf-8")
    return raw.replace("{{NAME}}", agent_name)


def env_example(name: str) -> str:
    """Return the contents of the template's ``.env.example`` file.

    Templates without an ``.env.example`` return a minimal default.
    """
    path = template_root(name) / ".env.example"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return "ANTHROPIC_API_KEY=\nAGENT_API_KEY=\n"
