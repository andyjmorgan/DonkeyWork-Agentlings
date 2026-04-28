"""Tests for skill discovery and prompt-block rendering.

Skills are the agent's runtime capability registry — only their metadata
is stapled into the system prompt; the body is loaded by the agent on
demand. These tests verify spec-compliant frontmatter parsing, the 64/1024
length caps, the lenient handling of malformed entries, and the rendered
prompt block.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

from agentlings.core.skills import (
    DESCRIPTION_MAX_CHARS,
    NAME_MAX_CHARS,
    SkillRef,
    discover_skills,
    format_skills_block,
)


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> Path:
    """Write a SKILL.md under ``root/<name>/`` and return the file path."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    md = skill_dir / "SKILL.md"
    content = f"---\n{frontmatter}\n---\n{body}"
    md.write_text(content, encoding="utf-8")
    return md


class TestDiscoverSkillsMissing:
    """When the skills root is missing or empty, discovery is a no-op."""

    def test_missing_dir_returns_empty(self, tmp_path: Path) -> None:
        assert discover_skills(tmp_path / "absent") == []

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "skills").mkdir()
        assert discover_skills(tmp_path / "skills") == []

    def test_dir_with_only_files_returns_empty(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "stray.md").write_text("not a skill", encoding="utf-8")
        assert discover_skills(skills) == []


class TestDiscoverSkillsHappyPath:
    """Valid skills round-trip through discovery into ``SkillRef`` records."""

    def test_single_valid_skill(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(
            skills,
            "pdf-processing",
            "name: pdf-processing\n"
            "description: Extract text from PDFs. Use when handling PDFs.",
            body="Detailed instructions live here.\n",
        )
        refs = discover_skills(skills)
        assert len(refs) == 1
        assert refs[0].name == "pdf-processing"
        assert "Extract text" in refs[0].description
        assert refs[0].path.name == "SKILL.md"

    def test_skills_sorted_by_name(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        for n in ["zeta", "alpha", "mid"]:
            _write_skill(skills, n, f"name: {n}\ndescription: {n} skill")
        names = [r.name for r in discover_skills(skills)]
        assert names == ["alpha", "mid", "zeta"]


class TestDiscoverSkillsCaps:
    """Name and description are truncated to spec-defined hard caps."""

    def test_description_capped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        long_desc = "x" * (DESCRIPTION_MAX_CHARS + 500)
        _write_skill(skills, "long-desc", f"name: long-desc\ndescription: {long_desc}")
        refs = discover_skills(skills)
        assert len(refs[0].description) == DESCRIPTION_MAX_CHARS

    def test_name_cap_constant_matches_spec(self) -> None:
        assert NAME_MAX_CHARS == 64
        assert DESCRIPTION_MAX_CHARS == 1024


class TestDiscoverSkillsMalformed:
    """Broken skills are skipped with a warning so one bad apple doesn't break boot."""

    def test_no_frontmatter_skipped(
        self, tmp_path: Path, caplog: logging.LogCaptureFixture
    ) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "no-fm"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("just a body, no frontmatter\n", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            assert discover_skills(skills) == []
        assert any("frontmatter" in r.message for r in caplog.records)

    def test_invalid_yaml_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "bad-yaml", "name: x\ndescription: : :::")
        assert discover_skills(skills) == []

    def test_missing_name_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "no-name", "description: only desc")
        assert discover_skills(skills) == []

    def test_missing_description_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "no-desc", "name: no-desc")
        assert discover_skills(skills) == []

    def test_invalid_name_chars_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        # Uppercase parent dir is not spec-compliant.
        _write_skill(skills, "Bad_Name", "name: Bad_Name\ndescription: invalid casing")
        assert discover_skills(skills) == []

    def test_name_does_not_match_dir_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(
            skills, "actual-dir",
            "name: declared-name\ndescription: parent mismatch",
        )
        assert discover_skills(skills) == []

    def test_one_bad_does_not_block_others(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, "good", "name: good\ndescription: works fine")
        _write_skill(skills, "bad", "not even yaml: : :::")
        names = [r.name for r in discover_skills(skills)]
        assert names == ["good"]


class TestDiscoverSkillsEdgeCases:
    """Filesystem and structural edge cases that should not crash discovery."""

    def test_skills_path_is_a_file_not_a_dir(self, tmp_path: Path) -> None:
        as_file = tmp_path / "skills"
        as_file.write_text("oops", encoding="utf-8")
        assert discover_skills(as_file) == []

    def test_subdir_without_skill_md_is_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        (skills / "bare-dir").mkdir(parents=True)
        (skills / "bare-dir" / "README.md").write_text("not the right file", encoding="utf-8")
        assert discover_skills(skills) == []

    def test_empty_skill_md_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "empty"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("", encoding="utf-8")
        assert discover_skills(skills) == []

    def test_unterminated_frontmatter_skipped(self, tmp_path: Path) -> None:
        """Opening ``---`` without a matching closing fence is not valid frontmatter."""
        skills = tmp_path / "skills"
        skill_dir = skills / "open"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: open\ndescription: missing close fence\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []

    def test_frontmatter_at_eof_without_trailing_newline(self, tmp_path: Path) -> None:
        """Closing ``---`` at end-of-file (no trailing newline) is still parseable."""
        skills = tmp_path / "skills"
        skill_dir = skills / "eof"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: eof\ndescription: closes at EOF\n---",
            encoding="utf-8",
        )
        refs = discover_skills(skills)
        assert [r.name for r in refs] == ["eof"]

    def test_empty_frontmatter_body_skipped(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "blank"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\n\n---\n", encoding="utf-8")
        assert discover_skills(skills) == []

    def test_frontmatter_yaml_is_a_list_skipped(self, tmp_path: Path) -> None:
        """Top-level YAML must be a mapping; a list is rejected."""
        skills = tmp_path / "skills"
        skill_dir = skills / "as-list"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n- item\n- another\n---\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []

    def test_body_after_frontmatter_is_ignored_at_discovery(self, tmp_path: Path) -> None:
        """Discovery only reads metadata. A 10k-char body is not a problem."""
        skills = tmp_path / "skills"
        body = "instructions\n" * 1000
        _write_skill(
            skills, "with-body",
            "name: with-body\ndescription: ignored body",
            body=body,
        )
        refs = discover_skills(skills)
        assert refs[0].description == "ignored body"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission semantics")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")
    def test_unreadable_skill_md_skipped(
        self, tmp_path: Path, caplog: logging.LogCaptureFixture
    ) -> None:
        skills = tmp_path / "skills"
        path = _write_skill(
            skills, "locked",
            "name: locked\ndescription: blocked from read",
        )
        path.chmod(0o000)
        try:
            with caplog.at_level(logging.WARNING):
                assert discover_skills(skills) == []
        finally:
            path.chmod(0o644)
        assert any("unreadable" in r.message for r in caplog.records)


class TestNameValidation:
    """The ``name`` regex enforces the Open Skills spec character rules."""

    @pytest.mark.parametrize("bad", [
        "-leading",
        "trailing-",
        "with--double",
        "Upper",
        "with_underscore",
        "white space",
        "with.dot",
    ])
    def test_invalid_names_rejected(self, tmp_path: Path, bad: str) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / bad
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {bad}\ndescription: bad name\n---\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []

    @pytest.mark.parametrize("good", [
        "a",
        "abc",
        "with-hyphen",
        "with-many-hyphens-here",
        "alpha9",
        "9starts-with-digit",
    ])
    def test_valid_names_accepted(self, tmp_path: Path, good: str) -> None:
        skills = tmp_path / "skills"
        _write_skill(skills, good, f"name: {good}\ndescription: ok")
        refs = discover_skills(skills)
        assert [r.name for r in refs] == [good]


class TestFieldTypes:
    """Non-string ``name``/``description`` are treated as missing."""

    def test_name_as_int_rejected(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "numeric"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: 42\ndescription: numeric name\n---\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []

    def test_description_as_list_rejected(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "list-desc"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: list-desc\ndescription:\n  - one\n  - two\n---\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []

    def test_whitespace_only_name_rejected(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "ws"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: '   '\ndescription: whitespace name\n---\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []

    def test_whitespace_only_description_rejected(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "ws-desc"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: ws-desc\ndescription: '   '\n---\n",
            encoding="utf-8",
        )
        assert discover_skills(skills) == []


class TestExtraFrontmatterFields:
    """Optional fields (license, metadata, compatibility) don't disrupt parsing."""

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        _write_skill(
            skills, "rich",
            "name: rich\n"
            "description: many fields\n"
            "license: Apache-2.0\n"
            "compatibility: Designed for Claude Code\n"
            "metadata:\n"
            "  author: example\n"
            "  version: '1.0'\n",
        )
        refs = discover_skills(skills)
        assert len(refs) == 1
        assert refs[0].name == "rich"
        assert refs[0].description == "many fields"


class TestBoundaryLengths:
    """Boundary conditions for the 64/1024 character caps."""

    def test_exactly_64_char_name_accepted(self, tmp_path: Path) -> None:
        name = "a" * 64
        skills = tmp_path / "skills"
        _write_skill(skills, name, f"name: {name}\ndescription: long but valid")
        refs = discover_skills(skills)
        assert refs[0].name == name
        assert len(refs[0].name) == 64

    def test_65_char_dir_truncated_name_mismatches_parent(self, tmp_path: Path) -> None:
        """Once the cap kicks in, the truncated name no longer matches the dir.

        This is the right outcome — the spec requires name == parent dir, and
        a 65-char name is already over the cap, so the skill is rejected
        rather than silently registered with a mangled identity.
        """
        long = "a" * 65
        skills = tmp_path / "skills"
        _write_skill(skills, long, f"name: {long}\ndescription: too long")
        assert discover_skills(skills) == []

    def test_exactly_1024_char_description_accepted(self, tmp_path: Path) -> None:
        desc = "x" * DESCRIPTION_MAX_CHARS
        skills = tmp_path / "skills"
        _write_skill(skills, "max-desc", f"name: max-desc\ndescription: {desc}")
        refs = discover_skills(skills)
        assert len(refs[0].description) == DESCRIPTION_MAX_CHARS


class TestWhitespaceStripping:
    """Leading/trailing whitespace is stripped from name and description before validation."""

    def test_padded_name_and_description(self, tmp_path: Path) -> None:
        skills = tmp_path / "skills"
        skill_dir = skills / "padded"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: \"  padded  \"\ndescription: \"   trim me   \"\n---\n",
            encoding="utf-8",
        )
        refs = discover_skills(skills)
        assert refs[0].name == "padded"
        assert refs[0].description == "trim me"


class TestFormatSkillsBlock:
    """The rendered prompt block teaches progressive disclosure and lists skills."""

    def test_empty_returns_none(self) -> None:
        assert format_skills_block([]) is None

    def test_block_mentions_progressive_disclosure(self, tmp_path: Path) -> None:
        ref = SkillRef(
            name="pdf-processing",
            description="Extract text from PDFs.",
            path=tmp_path / "pdf-processing" / "SKILL.md",
        )
        block = format_skills_block([ref])
        assert block is not None
        assert "progressive disclosure" in block.lower()
        assert "SKILL.md" in block
        assert "pdf-processing" in block
        assert "Extract text from PDFs." in block

    def test_block_lists_each_skill(self, tmp_path: Path) -> None:
        refs = [
            SkillRef(name="a", description="alpha", path=tmp_path / "a" / "SKILL.md"),
            SkillRef(name="b", description="beta", path=tmp_path / "b" / "SKILL.md"),
        ]
        block = format_skills_block(refs)
        assert block is not None
        assert "**a**" in block and "**b**" in block
        assert "alpha" in block and "beta" in block

    def test_block_uses_absolute_path(self, tmp_path: Path) -> None:
        path = (tmp_path / "skills" / "x" / "SKILL.md").resolve()
        block = format_skills_block([SkillRef(name="x", description="d", path=path)])
        assert block is not None
        assert str(path) in block

    def test_block_preserves_skill_order(self, tmp_path: Path) -> None:
        """``format_skills_block`` does not re-sort; it trusts caller order.

        Discovery sorts by name, so the rendered listing is deterministic
        across restarts. But if a future caller passes a custom order
        (e.g. priority), we don't fight it.
        """
        refs = [
            SkillRef(name="z", description="last alphabetically",
                     path=tmp_path / "z" / "SKILL.md"),
            SkillRef(name="a", description="first alphabetically",
                     path=tmp_path / "a" / "SKILL.md"),
        ]
        block = format_skills_block(refs)
        assert block is not None
        assert block.index("**z**") < block.index("**a**")


class TestDiscoverIntegration:
    """End-to-end mix of valid, invalid, and edge cases through one discovery call."""

    def test_mixed_directory(
        self, tmp_path: Path, caplog: logging.LogCaptureFixture
    ) -> None:
        skills = tmp_path / "skills"
        # Two valid skills.
        _write_skill(skills, "pdf-processing",
                     "name: pdf-processing\ndescription: PDFs")
        _write_skill(skills, "data-analysis",
                     "name: data-analysis\ndescription: tables")
        # Several broken neighbours.
        _write_skill(skills, "no-name", "description: orphan")
        _write_skill(skills, "Upper", "name: Upper\ndescription: cased")
        (skills / "stray.txt").write_text("not a dir", encoding="utf-8")
        bare = skills / "bare"
        bare.mkdir()

        with caplog.at_level(logging.WARNING):
            refs = discover_skills(skills)

        assert [r.name for r in refs] == ["data-analysis", "pdf-processing"]
        # We expect at least one warning log for each malformed neighbour.
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 2
