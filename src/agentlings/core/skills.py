"""Skill discovery and prompt rendering.

A skill is a directory under the configured skills root containing a
``SKILL.md`` file with YAML frontmatter (``name``, ``description``) followed by
a Markdown body. Only the metadata is loaded at startup and injected into the
system prompt; the body and any sibling resources (``scripts/``, ``references/``,
``assets/``) are loaded by the agent on demand — the *progressive disclosure*
pattern from the Open Skills specification (https://agentskills.io/specification).

Discovery is intentionally lenient: malformed entries are logged and skipped so
one broken skill does not prevent the agent from booting. It is also strictly
read-only — no mkdir, no writes, no imports, no ``sys.path`` mutation — so a
pre-existing ``./skills`` directory in the user's working tree (e.g. from an
unrelated project) cannot be clobbered by the agent.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

NAME_MAX_CHARS = 64
DESCRIPTION_MAX_CHARS = 1024

_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*(?:\n|\Z)", re.DOTALL
)
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

SKILL_FILENAME = "SKILL.md"


@dataclass(frozen=True)
class SkillRef:
    """Metadata for one discovered skill.

    Attributes:
        name: Validated skill name (matches ``[a-z0-9]+(?:-[a-z0-9]+)*``,
            capped at ``NAME_MAX_CHARS``). Equal to the parent directory name.
        description: Free-form description, capped at ``DESCRIPTION_MAX_CHARS``.
        path: Absolute path to the skill's ``SKILL.md``.
    """

    name: str
    description: str
    path: Path


def discover_skills(skills_dir: Path) -> list[SkillRef]:
    """Walk ``skills_dir`` and return one ``SkillRef`` per valid skill.

    Returns an empty list if the directory does not exist or contains no
    valid skills. Skills are returned sorted by name for deterministic
    prompt ordering across restarts.
    """
    if not skills_dir.exists() or not skills_dir.is_dir():
        return []

    refs: list[SkillRef] = []
    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_md = entry / SKILL_FILENAME
        if not skill_md.is_file():
            logger.debug("skipping %s: no %s", entry, SKILL_FILENAME)
            continue
        ref = _parse_skill(skill_md)
        if ref is not None:
            refs.append(ref)
    return refs


def _parse_skill(skill_md: Path) -> SkillRef | None:
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("skill unreadable: %s (%s)", skill_md, exc)
        return None

    match = _FRONTMATTER_RE.match(text)
    if match is None:
        logger.warning("skill missing YAML frontmatter: %s", skill_md)
        return None

    try:
        meta = yaml.safe_load(match.group("body")) or {}
    except yaml.YAMLError as exc:
        logger.warning("skill frontmatter not valid YAML: %s (%s)", skill_md, exc)
        return None

    if not isinstance(meta, dict):
        logger.warning("skill frontmatter is not a mapping: %s", skill_md)
        return None

    raw_name = meta.get("name")
    raw_desc = meta.get("description")
    if not isinstance(raw_name, str) or not raw_name.strip():
        logger.warning("skill missing required 'name': %s", skill_md)
        return None
    if not isinstance(raw_desc, str) or not raw_desc.strip():
        logger.warning("skill missing required 'description': %s", skill_md)
        return None

    name = raw_name.strip()[:NAME_MAX_CHARS]
    description = raw_desc.strip()[:DESCRIPTION_MAX_CHARS]

    if not _NAME_RE.match(name):
        logger.warning(
            "skill name %r is not spec-compliant (a-z, 0-9, hyphens; no leading/"
            "trailing/consecutive hyphens): %s",
            name, skill_md,
        )
        return None

    parent = skill_md.parent.name
    if name != parent:
        logger.warning(
            "skill name %r does not match parent directory %r: %s",
            name, parent, skill_md,
        )
        return None

    return SkillRef(name=name, description=description, path=skill_md)


_PROGRESSIVE_DISCLOSURE_PREAMBLE = (
    "## Skills\n"
    "\n"
    "You have access to specialized skills below. Each skill is a self-contained "
    "capability stored on disk. Skills follow **progressive disclosure**: only "
    "the name and description are loaded into this prompt. To activate a skill, "
    "read its `SKILL.md` at the listed path — that loads the full instructions "
    "into context. Skills may also bundle `scripts/`, `references/`, or "
    "`assets/` alongside `SKILL.md`; load those on demand only when the task "
    "requires them.\n"
    "\n"
    "Available skills:"
)


def format_skills_block(skills: list[SkillRef]) -> str | None:
    """Render the system-prompt skills block, or ``None`` when no skills exist."""
    if not skills:
        return None
    lines = [_PROGRESSIVE_DISCLOSURE_PREAMBLE]
    for s in skills:
        lines.append(f"- **{s.name}** (`{s.path}`): {s.description}")
    return "\n".join(lines)
