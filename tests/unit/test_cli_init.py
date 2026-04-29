"""Tests for ``agentling init``: scaffold layout, idempotency, key handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentlings import migrations as migrations_pkg
from agentlings.cli import _migrations, _templates, _version
from agentlings.cli.init import init_agent


class TestScaffoldLayout:
    def test_creates_expected_files(self, tmp_path: Path) -> None:
        result = init_agent("my-agent", dir=tmp_path / "my-agent")
        agent_dir = result.agent_dir
        assert (agent_dir / "agent.yaml").is_file()
        assert (agent_dir / ".env").is_file()
        assert (agent_dir / ".env.example").is_file()
        assert (agent_dir / ".framework-version").is_file()
        assert (agent_dir / "data").is_dir()

    def test_creates_skills_and_tools_dirs(self, tmp_path: Path) -> None:
        """Both folder-scan integrations get an empty drop-zone scaffolded.

        Operators uncomment the matching env var in ``.env`` to enable
        scanning. The directories are created empty so the convention is
        self-documenting without auto-activating either integration.
        """
        result = init_agent("my-agent", dir=tmp_path / "my-agent")
        assert (result.agent_dir / "skills").is_dir()
        assert (result.agent_dir / "tools").is_dir()

    def test_env_example_documents_folder_scan_vars(self, tmp_path: Path) -> None:
        result = init_agent("my-agent", dir=tmp_path / "my-agent")
        env_example = (result.agent_dir / ".env.example").read_text()
        assert "AGENT_SKILLS_DIR=./skills" in env_example
        assert "AGENT_TOOLS_DIR=./tools" in env_example

    def test_yaml_substitutes_name(self, tmp_path: Path) -> None:
        result = init_agent("my-agent", dir=tmp_path / "my-agent")
        text = (result.agent_dir / "agent.yaml").read_text(encoding="utf-8")
        assert "name: my-agent" in text
        assert "{{NAME}}" not in text

    def test_framework_version_recorded(self, tmp_path: Path) -> None:
        result = init_agent("a", dir=tmp_path / "a")
        recorded = _version.read_dir_version(result.agent_dir)
        assert recorded == _version.installed_version()
        assert result.framework_version == recorded

    def test_default_dir_is_name_under_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = init_agent("auto-named")
        assert result.agent_dir == (tmp_path / "auto-named").resolve()


class TestApiKeyGeneration:
    def test_auto_generates_strong_key(self, tmp_path: Path) -> None:
        result = init_agent("a", dir=tmp_path / "a")
        env = (result.agent_dir / ".env").read_text(encoding="utf-8")
        assert f"AGENT_API_KEY={result.generated_api_key}" in env
        # token_urlsafe(32) is at least 40 chars; sanity-check we didn't
        # leave the placeholder behind.
        assert len(result.generated_api_key) >= 40

    def test_explicit_key_used_verbatim(self, tmp_path: Path) -> None:
        result = init_agent("a", dir=tmp_path / "a", api_key="explicit-key-123")
        env = (result.agent_dir / ".env").read_text(encoding="utf-8")
        assert "AGENT_API_KEY=explicit-key-123" in env
        assert result.generated_api_key == "explicit-key-123"

    def test_anthropic_key_populated_when_supplied(self, tmp_path: Path) -> None:
        result = init_agent(
            "a", dir=tmp_path / "a", anthropic_api_key="sk-ant-real"
        )
        env = (result.agent_dir / ".env").read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=sk-ant-real" in env

    def test_anthropic_key_blank_when_omitted(self, tmp_path: Path) -> None:
        result = init_agent("a", dir=tmp_path / "a")
        env = (result.agent_dir / ".env").read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=\n" in env

    def test_anthropic_base_url_uncommented_when_supplied(self, tmp_path: Path) -> None:
        result = init_agent(
            "a",
            dir=tmp_path / "a",
            anthropic_base_url="http://localhost:11434",
        )
        env = (result.agent_dir / ".env").read_text(encoding="utf-8")
        assert "ANTHROPIC_BASE_URL=http://localhost:11434" in env


class TestIdempotency:
    def test_refuses_existing_non_empty_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "stray.txt").write_text("hi")
        with pytest.raises(FileExistsError):
            init_agent("a", dir=target)

    def test_force_overwrites_yaml(self, tmp_path: Path) -> None:
        target = tmp_path / "agent"
        init_agent("first", dir=target)
        # Mutate yaml — re-running with --force should restore template content.
        (target / "agent.yaml").write_text("name: tampered\n")
        result = init_agent("second", dir=target, force=True)
        text = (result.agent_dir / "agent.yaml").read_text(encoding="utf-8")
        assert "name: second" in text

    def test_force_preserves_data_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        (target / "data" / "important.txt").write_text("KEEP ME")
        init_agent("a", dir=target, force=True)
        assert (target / "data" / "important.txt").read_text() == "KEEP ME"

    def test_force_preserves_skills_and_tools_content(self, tmp_path: Path) -> None:
        """``--force`` re-scaffolds metadata files but never touches user content
        the operator may have dropped under ``skills/`` or ``tools/``."""
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        (target / "skills" / "pdf-processing").mkdir()
        (target / "skills" / "pdf-processing" / "SKILL.md").write_text("---\nname: pdf-processing\ndescription: pdfs\n---\n")
        (target / "tools" / "weather.py").write_text("# user tool\n")
        init_agent("a", dir=target, force=True)
        assert (target / "skills" / "pdf-processing" / "SKILL.md").is_file()
        assert (target / "tools" / "weather.py").read_text() == "# user tool\n"

    def test_force_preserves_existing_env(self, tmp_path: Path) -> None:
        """Re-running init must never clobber an operator's filled-in .env."""
        target = tmp_path / "agent"
        init_agent("a", dir=target)
        (target / ".env").write_text("ANTHROPIC_API_KEY=sk-real\nAGENT_API_KEY=secret\n")
        init_agent("a", dir=target, force=True)
        env = (target / ".env").read_text()
        assert "sk-real" in env
        assert "secret" in env


class TestMigrationStamping:
    def test_fresh_dir_marks_all_migrations_applied(self, tmp_path: Path) -> None:
        result = init_agent("a", dir=tmp_path / "a")
        log = _migrations.read_log(result.agent_dir / "data")
        assert log == [m.ID for m in migrations_pkg.discover()]


class TestTemplateLookup:
    def test_default_template_present(self) -> None:
        assert "default" in _templates.list_templates()

    def test_unknown_template_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown template"):
            init_agent("a", dir=tmp_path / "a", template="does-not-exist")
