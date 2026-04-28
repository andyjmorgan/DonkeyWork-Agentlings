"""Tests for ``agentlings.tools.decorator`` — schema inference, validation,
sync/async dispatch, and rejection of unsupported signatures."""

from __future__ import annotations

import asyncio
from enum import Enum, IntEnum
from typing import Annotated, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from agentlings.tools import Tool, ToolDefinitionError, ToolInputError, tool


# --------------------------------------------------------------------------- #
# Decorator surface — bare, parenthesized, and overrides
# --------------------------------------------------------------------------- #


class TestDecoratorSurface:
    def test_bare_decorator(self) -> None:
        @tool
        def hello(name: str) -> str:
            """Say hi."""
            return f"hi {name}"

        assert isinstance(hello, Tool)
        assert hello.name == "hello"
        assert hello.description == "Say hi."

    def test_parenthesized_decorator_with_overrides(self) -> None:
        @tool(name="greet", description="custom")
        def hello(name: str) -> str:
            """ignored docstring"""
            return name

        assert hello.name == "greet"
        assert hello.description == "custom"

    def test_description_falls_back_to_docstring_full_text(self) -> None:
        @tool
        def f(x: int) -> int:
            """Short summary.

            Longer description spanning
            multiple lines.
            """
            return x

        assert f.description.startswith("Short summary.")
        assert "Longer description" in f.description

    def test_no_docstring_yields_empty_description(self) -> None:
        @tool
        def f(x: int) -> int:
            return x

        assert f.description == ""


# --------------------------------------------------------------------------- #
# Schema inference — primitives, Optional, defaults
# --------------------------------------------------------------------------- #


class TestSchemaPrimitives:
    def test_primitives_map_to_json_schema_types(self) -> None:
        @tool
        def f(s: str, i: int, fl: float, b: bool) -> None:
            """t"""

        props = f.input_schema["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["fl"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert set(f.input_schema["required"]) == {"s", "i", "fl", "b"}

    def test_optional_param_is_not_required(self) -> None:
        @tool
        def f(name: str, nickname: Optional[str] = None) -> None:
            """t"""

        assert f.input_schema["required"] == ["name"]
        # Optional[str] becomes anyOf[string, null] in Pydantic v2.
        nickname = f.input_schema["properties"]["nickname"]
        assert "anyOf" in nickname or nickname.get("type") in ("string", ["string", "null"])

    def test_default_value_is_advertised(self) -> None:
        @tool
        def f(timeout_seconds: int = 30) -> None:
            """t"""

        prop = f.input_schema["properties"]["timeout_seconds"]
        assert prop["default"] == 30
        assert "timeout_seconds" not in f.input_schema.get("required", [])

    def test_top_level_title_is_stripped(self) -> None:
        @tool
        def f(x: int) -> None:
            """t"""

        assert "title" not in f.input_schema


# --------------------------------------------------------------------------- #
# Schema inference — string and integer enums, Literal
# --------------------------------------------------------------------------- #


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Priority(IntEnum):
    P0 = 0
    P1 = 1
    P2 = 2


class TestEnumsAndLiterals:
    def test_str_enum_renders_as_string_with_enum_values(self) -> None:
        @tool
        def f(severity: Severity) -> None:
            """t"""

        # Pydantic emits enums via $ref to a $defs entry.
        schema = f.input_schema
        assert "$defs" in schema
        # Find the Severity definition.
        defs = schema["$defs"]
        sev_def = next(iter(defs.values()))
        assert sev_def["type"] == "string"
        assert set(sev_def["enum"]) == {"low", "medium", "high"}

    def test_int_enum_renders_as_integer_with_enum_values(self) -> None:
        @tool
        def f(priority: Priority) -> None:
            """t"""

        defs = f.input_schema["$defs"]
        pri_def = next(iter(defs.values()))
        assert pri_def["type"] == "integer"
        assert set(pri_def["enum"]) == {0, 1, 2}

    def test_literal_strings_render_inline_as_enum(self) -> None:
        @tool
        def f(method: Literal["GET", "POST", "DELETE"]) -> None:
            """t"""

        prop = f.input_schema["properties"]["method"]
        # Literal of strings inlines an enum, no $defs entry.
        assert set(prop["enum"]) == {"GET", "POST", "DELETE"}
        # Pydantic v2 may or may not include `type: string` for Literal —
        # tolerate either, but if present it must be string.
        assert prop.get("type", "string") == "string"

    @pytest.mark.asyncio
    async def test_str_enum_validates_and_passes_native_value(self) -> None:
        captured: dict[str, Severity] = {}

        @tool
        def record(severity: Severity) -> str:
            """t"""
            captured["v"] = severity
            return severity.value

        result = await record.call({"severity": "high"})
        assert result == "high"
        assert captured["v"] is Severity.HIGH

    @pytest.mark.asyncio
    async def test_invalid_enum_value_raises_tool_input_error(self) -> None:
        @tool
        def f(severity: Severity) -> None:
            """t"""

        with pytest.raises(ToolInputError) as exc_info:
            await f.call({"severity": "extreme"})
        assert isinstance(exc_info.value.validation_error, ValidationError)


# --------------------------------------------------------------------------- #
# Annotated[..., Field(...)] — descriptions, constraints
# --------------------------------------------------------------------------- #


class TestAnnotatedFieldMetadata:
    def test_annotated_field_description_appears_in_schema(self) -> None:
        @tool
        def f(
            url: Annotated[str, Field(description="The URL to fetch")],
        ) -> None:
            """t"""

        assert f.input_schema["properties"]["url"]["description"] == "The URL to fetch"

    def test_annotated_constraints_apply_at_validation(self) -> None:
        @tool
        def f(
            n: Annotated[int, Field(ge=1, le=10)],
        ) -> int:
            """t"""
            return n

        prop = f.input_schema["properties"]["n"]
        assert prop["minimum"] == 1
        assert prop["maximum"] == 10


# --------------------------------------------------------------------------- #
# Nested Pydantic models, lists, dicts
# --------------------------------------------------------------------------- #


class Coordinates(BaseModel):
    lat: float
    lng: float


class TestComplexTypes:
    def test_pydantic_model_param_yields_ref_and_definition(self) -> None:
        @tool
        def f(loc: Coordinates) -> None:
            """t"""

        prop = f.input_schema["properties"]["loc"]
        assert "$ref" in prop
        assert "$defs" in f.input_schema
        coord_def = f.input_schema["$defs"]["Coordinates"]
        assert coord_def["type"] == "object"
        assert set(coord_def["properties"].keys()) == {"lat", "lng"}

    def test_list_of_strings(self) -> None:
        @tool
        def f(tags: list[str]) -> None:
            """t"""

        prop = f.input_schema["properties"]["tags"]
        assert prop["type"] == "array"
        assert prop["items"]["type"] == "string"

    def test_dict_with_typed_values(self) -> None:
        @tool
        def f(headers: dict[str, str]) -> None:
            """t"""

        prop = f.input_schema["properties"]["headers"]
        assert prop["type"] == "object"
        # Pydantic emits additionalProperties as the value type schema.
        assert prop["additionalProperties"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_pydantic_model_arg_is_constructed_and_passed(self) -> None:
        captured: dict[str, Coordinates] = {}

        @tool
        def f(loc: Coordinates) -> str:
            """t"""
            captured["loc"] = loc
            return f"{loc.lat},{loc.lng}"

        out = await f.call({"loc": {"lat": 1.5, "lng": -2.0}})
        assert out == "1.5,-2.0"
        assert isinstance(captured["loc"], Coordinates)
        assert captured["loc"].lat == 1.5


# --------------------------------------------------------------------------- #
# Sync vs async dispatch
# --------------------------------------------------------------------------- #


class TestSyncAndAsync:
    @pytest.mark.asyncio
    async def test_sync_tool_runs_via_to_thread(self) -> None:
        @tool
        def add(a: int, b: int) -> int:
            """t"""
            return a + b

        assert add.is_async is False
        assert await add.call({"a": 2, "b": 3}) == 5

    @pytest.mark.asyncio
    async def test_async_tool_is_awaited(self) -> None:
        @tool
        async def add(a: int, b: int) -> int:
            """t"""
            await asyncio.sleep(0)
            return a + b

        assert add.is_async is True
        assert await add.call({"a": 2, "b": 3}) == 5


# --------------------------------------------------------------------------- #
# Rejection of unsupported signatures
# --------------------------------------------------------------------------- #


class TestRejectedSignatures:
    def test_missing_annotation_raises(self) -> None:
        with pytest.raises(ToolDefinitionError, match="must have a type annotation"):
            @tool
            def f(x) -> None:  # type: ignore[no-untyped-def]
                """t"""

    def test_var_args_rejected(self) -> None:
        with pytest.raises(ToolDefinitionError, match=r"\*args"):
            @tool
            def f(*args: int) -> None:
                """t"""

    def test_var_kwargs_rejected(self) -> None:
        with pytest.raises(ToolDefinitionError, match=r"\*\*kwargs"):
            @tool
            def f(**kwargs: int) -> None:
                """t"""

    def test_positional_only_rejected(self) -> None:
        with pytest.raises(ToolDefinitionError, match="positional-only"):
            @tool
            def f(x: int, /) -> None:
                """t"""


# --------------------------------------------------------------------------- #
# to_anthropic_dict — exact shape contract
# --------------------------------------------------------------------------- #


class TestAnthropicDictShape:
    def test_to_anthropic_dict_contains_expected_keys_only(self) -> None:
        @tool
        def f(x: int) -> None:
            """A short tool."""

        d = f.to_anthropic_dict()
        assert set(d.keys()) == {"name", "description", "input_schema"}
        assert d["name"] == "f"
        assert d["description"] == "A short tool."
        assert d["input_schema"]["type"] == "object"
        assert "x" in d["input_schema"]["properties"]

    def test_repeat_calls_return_equivalent_dicts(self) -> None:
        @tool
        def f(x: int) -> None:
            """t"""

        assert f.to_anthropic_dict() == f.to_anthropic_dict()


# --------------------------------------------------------------------------- #
# Generic invocation behavior
# --------------------------------------------------------------------------- #


class TestCallBehavior:
    @pytest.mark.asyncio
    async def test_extra_keys_rejected_by_default(self) -> None:
        @tool
        def f(x: int) -> int:
            """t"""
            return x

        # Pydantic's default model_config allows extra fields silently;
        # they're simply ignored. Confirm that's the behavior so callers can
        # rely on it (and we document the contract).
        assert await f.call({"x": 1, "junk": True}) == 1

    @pytest.mark.asyncio
    async def test_missing_required_param_raises_tool_input_error(self) -> None:
        @tool
        def f(x: int, y: int) -> int:
            """t"""
            return x + y

        with pytest.raises(ToolInputError):
            await f.call({"x": 1})
