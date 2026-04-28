"""``@tool`` decorator: turn a typed Python function into a self-describing tool.

A tool is a self-contained unit. The decorator infers everything the LLM needs
from the function itself:

- ``name``         from ``func.__name__`` (or override).
- ``description``  from the docstring (or override).
- ``input_schema`` derived from the parameter type annotations using Pydantic.

Per-parameter descriptions and constraints attach via
``Annotated[T, Field(description="...", ge=..., le=...)]`` — the idiomatic
Python way and the single source of truth (no separate docstring parser).

Sync and async functions are both supported; the resulting ``Tool`` exposes a
unified ``async call(args)`` regardless.

Output: ``tool.to_anthropic_dict()`` returns ``{name, description,
input_schema}`` — exactly the shape the agent registry already passes to
``messages.create(tools=...)``.
"""

from __future__ import annotations

import asyncio
import inspect
from inspect import Parameter
from typing import Any, Awaitable, Callable, Generic, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ValidationError, create_model

R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])


class ToolDefinitionError(TypeError):
    """Raised when a function cannot be turned into a tool.

    Common causes: missing type annotations, ``*args``/``**kwargs``, or
    positional-only parameters.
    """


class ToolInputError(ValueError):
    """Raised when ``Tool.call`` receives arguments that fail schema validation.

    Carries the underlying ``pydantic.ValidationError`` for callers that want
    structured error reporting; ``str(e)`` is a human-readable summary.
    """

    def __init__(self, message: str, *, validation_error: ValidationError) -> None:
        super().__init__(message)
        self.validation_error = validation_error


class Tool(Generic[R]):
    """A typed, self-describing callable the agent can expose to the LLM.

    Attributes:
        func: The underlying user function.
        name: The identifier the LLM uses to invoke the tool.
        description: Human-readable description for the LLM.
        input_schema: JSON Schema describing the tool's input parameters.
        is_async: ``True`` when ``func`` is a coroutine function.
    """

    func: Callable[..., Any]
    name: str
    description: str
    input_schema: dict[str, Any]
    is_async: bool

    def __init__(
        self,
        func: Callable[..., R] | Callable[..., Awaitable[R]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.func = func
        self.is_async = asyncio.iscoroutinefunction(func)
        self.name = name or func.__name__

        if description is not None:
            self.description = description
        elif func.__doc__:
            self.description = inspect.cleandoc(func.__doc__)
        else:
            self.description = ""

        self._input_model = _build_input_model(func)
        self.input_schema = _clean_schema(
            self._input_model.model_json_schema(),
        )

    async def call(self, args: dict[str, Any]) -> R:
        """Validate ``args`` against the schema and invoke the tool.

        Sync functions run on a worker thread so an async caller never blocks.
        """
        try:
            validated = self._input_model.model_validate(args)
        except ValidationError as e:
            raise ToolInputError(
                f"Invalid arguments for tool {self.name!r}: {e.error_count()} "
                f"validation error(s).",
                validation_error=e,
            ) from e

        kwargs = {k: getattr(validated, k) for k in self._input_model.model_fields}
        if self.is_async:
            return await self.func(**kwargs)  # type: ignore[no-any-return]
        return await asyncio.to_thread(self.func, **kwargs)

    def to_anthropic_dict(self) -> dict[str, Any]:
        """Render the tool in the shape ``messages.create(tools=...)`` expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "async" if self.is_async else "sync"
        return f"<Tool {self.name!r} ({kind})>"


@overload
def tool(func: F) -> Tool[Any]: ...


@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[F], Tool[Any]]: ...


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool[Any] | Callable[[Callable[..., Any]], Tool[Any]]:
    """Wrap a function as a ``Tool``.

    Usable bare or with overrides::

        @tool
        async def fetch(url: str) -> str:
            ...

        @tool(name="custom", description="…")
        def other(...): ...
    """

    def _make(fn: Callable[..., Any]) -> Tool[Any]:
        return Tool(fn, name=name, description=description)

    if func is not None:
        return _make(func)
    return _make


def _build_input_model(func: Callable[..., Any]) -> type[BaseModel]:
    """Build a Pydantic model representing the function's keyword arguments.

    Uses ``get_type_hints(..., include_extras=True)`` so ``Annotated`` metadata
    (``pydantic.Field(...)``) is preserved and feeds ``model_json_schema``.
    """
    sig = inspect.signature(func)
    try:
        type_hints = get_type_hints(func, include_extras=True)
    except NameError as e:
        raise ToolDefinitionError(
            f"Tool {func.__name__!r}: could not resolve type hints ({e}). "
            f"Use ``from __future__ import annotations`` and ensure all referenced "
            f"types are importable from the function's module."
        ) from e

    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind == Parameter.VAR_POSITIONAL:
            raise ToolDefinitionError(
                f"Tool {func.__name__!r}: ``*{param_name}`` is not supported."
            )
        if param.kind == Parameter.VAR_KEYWORD:
            raise ToolDefinitionError(
                f"Tool {func.__name__!r}: ``**{param_name}`` is not supported."
            )
        if param.kind == Parameter.POSITIONAL_ONLY:
            raise ToolDefinitionError(
                f"Tool {func.__name__!r}: positional-only parameter "
                f"{param_name!r} is not supported."
            )
        if param_name not in type_hints:
            raise ToolDefinitionError(
                f"Tool {func.__name__!r}: parameter {param_name!r} must have a "
                f"type annotation."
            )

        annotation = type_hints[param_name]
        default: Any = param.default if param.default is not Parameter.empty else ...
        fields[param_name] = (annotation, default)

    return create_model(f"{func.__name__}__InputSchema", **fields)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Drop Pydantic's noisy top-level ``title`` field; keep everything else.

    The top-level ``title`` is a Pydantic auto-generated identifier (e.g.
    ``foo__InputSchema``) that clutters the LLM-facing schema. Nested ``title``
    fields are left alone — Pydantic generates them for enum classes and
    sub-models where they may carry meaning.
    """
    schema.pop("title", None)
    return schema
