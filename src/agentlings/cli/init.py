"""``agentling init`` — scaffold a new agent directory.

Produces a self-contained directory the operator can ``cd`` into and run
``agentling run`` against. Does not install the framework — that's already
installed (it's the binary running). Does not touch the network in v1; only
bundled templates are supported.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass
from pathlib import Path

from agentlings.cli import _migrations, _templates, _version

logger = logging.getLogger(__name__)

DATA_DIRNAME = "data"
ENV_FILENAME = ".env"
ENV_EXAMPLE_FILENAME = ".env.example"
YAML_FILENAME = "agent.yaml"


@dataclass
class InitResult:
    agent_dir: Path
    template: str
    framework_version: str
    generated_api_key: str


def init_agent(
    name: str,
    *,
    dir: Path | None = None,
    template: str = "default",
    api_key: str | None = None,
    anthropic_api_key: str | None = None,
    anthropic_base_url: str | None = None,
    force: bool = False,
) -> InitResult:
    """Scaffold a new agent directory.

    Args:
        name: Agent name. Used as the default directory name and substituted
            into the template's ``agent.yaml``.
        dir: Output directory. Defaults to ``./<name>``.
        template: Bundled template to scaffold from.
        api_key: ``AGENT_API_KEY`` value to write into ``.env``. Auto-generated
            if not supplied.
        anthropic_api_key: Optional Anthropic key to pre-populate ``.env``.
        anthropic_base_url: Optional Anthropic base URL to pre-populate.
        force: Allow scaffolding into an existing directory. Files that already
            exist are overwritten *except* ``data/`` and ``.env``, which are
            never touched once they exist.
    """
    target = (dir or Path.cwd() / name).resolve()
    if target.exists() and not force and any(target.iterdir()):
        raise FileExistsError(
            f"{target} already exists and is non-empty (pass --force to overwrite)"
        )

    target.mkdir(parents=True, exist_ok=True)
    (target / DATA_DIRNAME).mkdir(exist_ok=True)

    yaml_path = target / YAML_FILENAME
    if not yaml_path.exists() or force:
        yaml_path.write_text(_templates.render_yaml(template, name), encoding="utf-8")

    env_example = _templates.env_example(template)
    (target / ENV_EXAMPLE_FILENAME).write_text(env_example, encoding="utf-8")

    generated = api_key or _generate_api_key()
    env_path = target / ENV_FILENAME
    if not env_path.exists():
        env_path.write_text(
            _render_env(env_example, generated, anthropic_api_key, anthropic_base_url),
            encoding="utf-8",
        )

    framework_version = _version.installed_version()
    _version.write_dir_version(target, framework_version)
    _migrations.stamp_all_applied(target / DATA_DIRNAME)

    return InitResult(
        agent_dir=target,
        template=template,
        framework_version=framework_version,
        generated_api_key=generated,
    )


def _generate_api_key() -> str:
    """Return a 32-byte URL-safe random string for ``AGENT_API_KEY``."""
    return secrets.token_urlsafe(32)


def _render_env(
    env_example: str,
    agent_api_key: str,
    anthropic_api_key: str | None,
    anthropic_base_url: str | None,
) -> str:
    """Substitute populated values into the ``.env.example`` template."""
    lines = []
    for line in env_example.splitlines():
        stripped = line.strip()
        if stripped.startswith("AGENT_API_KEY=") and not stripped.startswith("#"):
            lines.append(f"AGENT_API_KEY={agent_api_key}")
        elif stripped.startswith("ANTHROPIC_API_KEY=") and not stripped.startswith("#"):
            value = anthropic_api_key or ""
            lines.append(f"ANTHROPIC_API_KEY={value}")
        elif (
            anthropic_base_url
            and stripped.startswith("# ANTHROPIC_BASE_URL=")
        ):
            lines.append(f"ANTHROPIC_BASE_URL={anthropic_base_url}")
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"
