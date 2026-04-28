"""Tests for ``ToolRegistry.register_tool_object`` and the folder-scan loader."""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

from agentlings.tools import tool
from agentlings.tools.loader import load_tools_from_directory
from agentlings.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Registry integration
# --------------------------------------------------------------------------- #


class TestRegisterToolObject:
    def test_registers_name_description_and_schema(self) -> None:
        @tool
        def adder(a: int, b: int) -> int:
            """Add two integers."""
            return a + b

        registry = ToolRegistry()
        registry.register_tool_object(adder)

        schemas = registry.list_schemas()
        assert len(schemas) == 1
        s = schemas[0]
        assert s["name"] == "adder"
        assert s["description"] == "Add two integers."
        assert s["input_schema"]["type"] == "object"
        assert set(s["input_schema"]["properties"].keys()) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result_with_output(self) -> None:
        @tool
        def adder(a: int, b: int) -> int:
            """t"""
            return a + b

        registry = ToolRegistry()
        registry.register_tool_object(adder)

        result = await registry.execute("adder", {"a": 2, "b": 3})
        assert result.is_error is False
        assert result.output == "5"

    @pytest.mark.asyncio
    async def test_async_tool_executes_through_registry(self) -> None:
        @tool
        async def reverse(s: str) -> str:
            """t"""
            return s[::-1]

        registry = ToolRegistry()
        registry.register_tool_object(reverse)

        result = await registry.execute("reverse", {"s": "hello"})
        assert result.output == "olleh"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_validation_failure_surfaces_as_error_result(self) -> None:
        """Bad inputs must come back as is_error=True, not raise — the LLM
        retries with corrected args."""
        @tool
        def adder(a: int, b: int) -> int:
            """t"""
            return a + b

        registry = ToolRegistry()
        registry.register_tool_object(adder)

        result = await registry.execute("adder", {"a": "not-an-int", "b": 1})
        assert result.is_error is True
        assert "adder" in result.output

    @pytest.mark.asyncio
    async def test_runtime_exception_surfaces_as_error_result(self) -> None:
        @tool
        def boom(x: int) -> int:
            """t"""
            raise RuntimeError("nope")

        registry = ToolRegistry()
        registry.register_tool_object(boom)

        result = await registry.execute("boom", {"x": 1})
        assert result.is_error is True
        assert "boom" in result.output


# --------------------------------------------------------------------------- #
# Folder scan loader
# --------------------------------------------------------------------------- #


def _write(path: Path, body: str) -> None:
    path.write_text(textwrap.dedent(body).lstrip())


class TestFolderScan:
    def test_loads_tools_from_directory(self, tmp_path: Path) -> None:
        _write(tmp_path / "greet.py", """
            from agentlings.tools import tool

            @tool
            def greet(name: str) -> str:
                "Say hi."
                return f"hi {name}"
        """)

        registry = ToolRegistry()
        names = load_tools_from_directory(tmp_path, registry)

        assert names == ["greet"]
        assert "greet" in registry.tool_names()

    def test_loads_multiple_tools_per_file(self, tmp_path: Path) -> None:
        _write(tmp_path / "math_ops.py", """
            from agentlings.tools import tool

            @tool
            def add(a: int, b: int) -> int:
                "t"
                return a + b

            @tool
            def sub(a: int, b: int) -> int:
                "t"
                return a - b
        """)

        registry = ToolRegistry()
        names = load_tools_from_directory(tmp_path, registry)

        assert sorted(names) == ["add", "sub"]
        assert set(registry.tool_names()) == {"add", "sub"}

    def test_skips_files_starting_with_underscore(self, tmp_path: Path) -> None:
        _write(tmp_path / "_private.py", """
            from agentlings.tools import tool

            @tool
            def secret(x: int) -> int:
                "t"
                return x
        """)
        _write(tmp_path / "public.py", """
            from agentlings.tools import tool

            @tool
            def hello(x: int) -> int:
                "t"
                return x
        """)

        registry = ToolRegistry()
        names = load_tools_from_directory(tmp_path, registry)

        assert names == ["hello"]
        assert "secret" not in registry.tool_names()

    def test_ignores_non_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("# tools")
        (tmp_path / "config.yaml").write_text("foo: bar")
        _write(tmp_path / "real.py", """
            from agentlings.tools import tool

            @tool
            def alive(x: int) -> int:
                "t"
                return x
        """)

        registry = ToolRegistry()
        names = load_tools_from_directory(tmp_path, registry)

        assert names == ["alive"]

    def test_does_not_recurse_into_subdirs(self, tmp_path: Path) -> None:
        sub = tmp_path / "nested"
        sub.mkdir()
        _write(sub / "buried.py", """
            from agentlings.tools import tool

            @tool
            def buried(x: int) -> int:
                "t"
                return x
        """)

        registry = ToolRegistry()
        names = load_tools_from_directory(tmp_path, registry)
        assert names == []
        assert "buried" not in registry.tool_names()

    def test_broken_file_is_logged_and_skipped(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        _write(tmp_path / "broken.py", "this is not valid python {")
        _write(tmp_path / "good.py", """
            from agentlings.tools import tool

            @tool
            def ok(x: int) -> int:
                "t"
                return x
        """)

        registry = ToolRegistry()
        with caplog.at_level(logging.ERROR, logger="agentlings.tools.loader"):
            names = load_tools_from_directory(tmp_path, registry)

        assert names == ["ok"], "good tool must still register despite broken neighbor"
        assert any("broken.py" in r.message for r in caplog.records), \
            "broken file must be logged"

    def test_missing_directory_logs_and_returns_empty(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        missing = tmp_path / "does-not-exist"
        registry = ToolRegistry()
        with caplog.at_level(logging.INFO, logger="agentlings.tools.loader"):
            names = load_tools_from_directory(missing, registry)

        assert names == []
        assert any("does not exist" in r.message for r in caplog.records)

    def test_path_pointing_at_a_file_is_rejected(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        f = tmp_path / "not-a-dir.py"
        f.write_text("# nothing")
        registry = ToolRegistry()
        with caplog.at_level(logging.WARNING, logger="agentlings.tools.loader"):
            names = load_tools_from_directory(f, registry)

        assert names == []
        assert any("not a directory" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_loaded_tool_is_executable_through_registry(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: a user file → registry → execute returns the result."""
        _write(tmp_path / "shouty.py", """
            from agentlings.tools import tool

            @tool
            def shout(message: str) -> str:
                "t"
                return message.upper()
        """)

        registry = ToolRegistry()
        load_tools_from_directory(tmp_path, registry)

        result = await registry.execute("shout", {"message": "hello"})
        assert result.is_error is False
        assert result.output == "HELLO"

    def test_module_isolation_does_not_pollute_sys_path(
        self, tmp_path: Path
    ) -> None:
        """The user-tools dir must not be injected into sys.path — otherwise a
        user could shadow installed packages by naming a file ``json.py``."""
        import sys

        before = list(sys.path)
        _write(tmp_path / "trivial.py", """
            from agentlings.tools import tool

            @tool
            def t(x: int) -> int:
                "t"
                return x
        """)

        registry = ToolRegistry()
        load_tools_from_directory(tmp_path, registry)

        # sys.path entries must not include the user tools dir.
        assert str(tmp_path) not in sys.path
        # And nothing else should have been added either.
        assert sys.path == before
