from __future__ import annotations

import pytest

from agentlings.tools import ToolRegistry, ToolResult


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


class TestBuiltins:
    def test_register_builtins(self, registry: ToolRegistry) -> None:
        registry.register_builtins()
        names = registry.tool_names()
        assert "shell" in names
        assert "read_file" in names
        assert "write_file" in names

    @pytest.mark.asyncio
    async def test_shell_echo(self, registry: ToolRegistry) -> None:
        registry.register_builtins()
        result = await registry.execute("shell", {"command": "echo hello"})
        assert result.output == "hello"
        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_shell_nonzero_exit(self, registry: ToolRegistry) -> None:
        registry.register_builtins()
        result = await registry.execute("shell", {"command": "exit 1"})
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_shell_timeout(self, registry: ToolRegistry) -> None:
        registry.register_builtins()
        result = await registry.execute(
            "shell", {"command": "sleep 10", "timeout": 1}
        )
        assert result.is_error is True
        assert "timed out" in result.output

    @pytest.mark.asyncio
    async def test_read_file(self, registry: ToolRegistry, tmp_path) -> None:
        registry.register_builtins()
        f = tmp_path / "test.txt"
        f.write_text("file content")
        result = await registry.execute("read_file", {"path": str(f)})
        assert result.output == "file content"

    @pytest.mark.asyncio
    async def test_read_file_missing(self, registry: ToolRegistry) -> None:
        registry.register_builtins()
        result = await registry.execute("read_file", {"path": "/nonexistent/file.txt"})
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_write_file(self, registry: ToolRegistry, tmp_path) -> None:
        registry.register_builtins()
        f = tmp_path / "out.txt"
        result = await registry.execute(
            "write_file", {"path": str(f), "content": "written"}
        )
        assert result.is_error is False
        assert f.read_text() == "written"
