from __future__ import annotations

import pytest

from agentlings.tools.builtins import build_builtin_registry
from agentlings.tools.registry import ToolRegistry, ToolResult


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


class TestRegister:
    def test_register_adds_tool(self, registry: ToolRegistry) -> None:
        registry.register(
            name="echo",
            description="Echo input",
            input_schema={"type": "object", "properties": {}},
            execute_fn=lambda: ToolResult(output="ok"),
        )
        schemas = registry.list_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "echo"

    def test_list_schemas_format(self, registry: ToolRegistry) -> None:
        registry.register(
            name="test",
            description="A test tool",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
            execute_fn=lambda x="": ToolResult(output=x),
        )
        schema = registry.list_schemas()[0]
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema

    def test_tool_names(self, registry: ToolRegistry) -> None:
        registry.register("a", "desc", {}, lambda: ToolResult(output=""))
        registry.register("b", "desc", {}, lambda: ToolResult(output=""))
        assert set(registry.tool_names()) == {"a", "b"}


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self, registry: ToolRegistry) -> None:
        registry.register(
            name="greet",
            description="Greet",
            input_schema={},
            execute_fn=lambda name="world": ToolResult(output=f"hello {name}"),
        )
        result = await registry.execute("greet", {"name": "test"})
        assert result.output == "hello test"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, registry: ToolRegistry) -> None:
        result = await registry.execute("nonexistent", {})
        assert result.is_error is True
        assert "Unknown tool" in result.output

    @pytest.mark.asyncio
    async def test_execute_error_tool(self, registry: ToolRegistry) -> None:
        def fail() -> ToolResult:
            return ToolResult(output="something went wrong", is_error=True)

        registry.register("fail", "Fails", {}, fail)
        result = await registry.execute("fail", {})
        assert result.is_error is True


class TestRegisterTools:
    def test_no_tools_when_empty(self, registry: ToolRegistry) -> None:
        registry.register_tools([])
        assert registry.tool_names() == []

    def test_register_bash_group(self, registry: ToolRegistry) -> None:
        registry.register_tools(["bash"])
        assert "bash" in registry.tool_names()

    def test_register_filesystem_group(self, registry: ToolRegistry) -> None:
        registry.register_tools(["filesystem"])
        names = registry.tool_names()
        assert "read_file" in names
        assert "write_file" in names
        assert "edit_file" in names
        assert "list_directory" in names
        assert "search_files" in names

    def test_register_individual_tool(self, registry: ToolRegistry) -> None:
        registry.register_tools(["read_file"])
        assert registry.tool_names() == ["read_file"]

    def test_register_multiple_groups(self, registry: ToolRegistry) -> None:
        registry.register_tools(["bash", "filesystem"])
        names = registry.tool_names()
        assert "bash" in names
        assert "read_file" in names

    def test_unknown_tool_raises(self, registry: ToolRegistry) -> None:
        with pytest.raises(ValueError, match="Unknown tools"):
            registry.register_tools(["nonexistent"])


class TestBashTool:
    @pytest.fixture(autouse=True)
    def setup(self, registry: ToolRegistry) -> None:
        registry.register_tools(["bash"])
        self.registry = registry

    @pytest.mark.asyncio
    async def test_echo(self) -> None:
        result = await self.registry.execute("bash", {"command": "echo hello"})
        assert result.output == "hello"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_nonzero_exit(self) -> None:
        result = await self.registry.execute("bash", {"command": "exit 1"})
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        result = await self.registry.execute(
            "bash", {"command": "sleep 10", "timeout": 1}
        )
        assert result.is_error is True
        assert "timed out" in result.output


class TestFilesystemTools:
    @pytest.fixture(autouse=True)
    def setup(self, registry: ToolRegistry) -> None:
        registry.register_tools(["filesystem"])
        self.registry = registry

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("line 1\nline 2\nline 3")
        result = await self.registry.execute("read_file", {"path": str(f)})
        assert "line 1" in result.output
        assert "line 2" in result.output
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_read_file_with_offset_limit(self, tmp_path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("\n".join(f"line {i}" for i in range(100)))
        result = await self.registry.execute(
            "read_file", {"path": str(f), "offset": 10, "limit": 5}
        )
        assert "line 10" in result.output
        assert "line 14" in result.output
        assert "line 15" not in result.output

    @pytest.mark.asyncio
    async def test_read_file_missing(self) -> None:
        result = await self.registry.execute("read_file", {"path": "/nonexistent"})
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path) -> None:
        f = tmp_path / "out.txt"
        result = await self.registry.execute(
            "write_file", {"path": str(f), "content": "hello"}
        )
        assert result.is_error is False
        assert f.read_text() == "hello"

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        result = await self.registry.execute(
            "edit_file",
            {"path": str(f), "old_text": "world", "new_text": "there"},
        )
        assert result.is_error is False
        assert f.read_text() == "hello there"

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self, tmp_path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("hello world")
        result = await self.registry.execute(
            "edit_file",
            {"path": str(f), "old_text": "missing", "new_text": "x"},
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_edit_file_multiple_matches_fails(self, tmp_path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("aaa aaa")
        result = await self.registry.execute(
            "edit_file",
            {"path": str(f), "old_text": "aaa", "new_text": "bbb"},
        )
        assert result.is_error is True
        assert "2 times" in result.output

    @pytest.mark.asyncio
    async def test_edit_file_replace_all(self, tmp_path) -> None:
        f = tmp_path / "edit.txt"
        f.write_text("aaa aaa")
        result = await self.registry.execute(
            "edit_file",
            {
                "path": str(f),
                "old_text": "aaa",
                "new_text": "bbb",
                "replace_all": True,
            },
        )
        assert result.is_error is False
        assert f.read_text() == "bbb bbb"

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path) -> None:
        (tmp_path / "file.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = await self.registry.execute(
            "list_directory", {"path": str(tmp_path)}
        )
        assert "[DIR] subdir" in result.output
        assert "[FILE] file.txt" in result.output

    @pytest.mark.asyncio
    async def test_search_files(self, tmp_path) -> None:
        (tmp_path / "a.py").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").touch()
        result = await self.registry.execute(
            "search_files", {"path": str(tmp_path), "pattern": "*.py"}
        )
        assert "a.py" in result.output
        assert "c.py" in result.output
        assert "b.txt" not in result.output


class TestBashTimeout:
    def test_custom_timeout_via_register_tools(self) -> None:
        registry = ToolRegistry()
        registry.register_tools(["bash"], bash_timeout=120)
        assert "bash" in registry.tool_names()

    @pytest.mark.asyncio
    async def test_custom_timeout_applies(self) -> None:
        registry = ToolRegistry()
        registry.register_tools(["bash"], bash_timeout=1)
        result = await registry.execute("bash", {"command": "sleep 5"})
        assert result.is_error is True
        assert "timed out after 1 seconds" in result.output

    @pytest.mark.asyncio
    async def test_per_call_timeout_overrides_default(self) -> None:
        registry = ToolRegistry()
        registry.register_tools(["bash"], bash_timeout=120)
        result = await registry.execute("bash", {"command": "sleep 5", "timeout": 1})
        assert result.is_error is True
        assert "timed out after 1 seconds" in result.output

    def test_build_builtin_registry_default(self) -> None:
        reg = build_builtin_registry()
        assert "bash" in reg
        assert "30" in reg["bash"]["input_schema"]["properties"]["timeout"]["description"]

    def test_build_builtin_registry_custom(self) -> None:
        reg = build_builtin_registry(bash_timeout=90)
        assert "90" in reg["bash"]["input_schema"]["properties"]["timeout"]["description"]
