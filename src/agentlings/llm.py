from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str | None = None


class BaseLLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse: ...

    @abstractmethod
    async def stream(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]: ...


class AnthropicLLMClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        import anthropic

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.messages.create(
            **kwargs,
            betas=["interleaved-thinking-2025-05-14"],
        )
        return LLMResponse(
            content=[block.model_dump() for block in response.content],
            stop_reason=response.stop_reason,
        )

    async def stream(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("Streaming not yet implemented")
        yield  # pragma: no cover


class MockLLMClient(BaseLLMClient):
    def __init__(
        self,
        tool_names: list[str] | None = None,
        compaction_threshold: int = 10,
    ) -> None:
        self._tool_names = tool_names or []
        self._compaction_threshold = compaction_threshold
        self._call_count = 0

    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        self._call_count += 1
        last_message = messages[-1] if messages else {}
        last_text = _extract_text(last_message)

        if last_message.get("role") == "tool":
            return LLMResponse(
                content=[{"type": "text", "text": f"Tool result received: {last_text}"}],
                stop_reason="end_turn",
            )

        for tool_name in self._tool_names:
            if tool_name in last_text:
                return LLMResponse(
                    content=[{
                        "type": "tool_use",
                        "id": f"toolu_{uuid4().hex[:12]}",
                        "name": tool_name,
                        "input": _build_mock_tool_input(tool_name, last_text),
                    }],
                    stop_reason="tool_use",
                )

        return LLMResponse(
            content=[{"type": "text", "text": f"Mock response to: {last_text}"}],
            stop_reason="end_turn",
        )

    async def stream(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        response = await self.complete(system, messages, tools)
        for block in response.content:
            yield block


def create_llm_client(
    backend: str,
    api_key: str = "",
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    tool_names: list[str] | None = None,
) -> BaseLLMClient:
    if backend == "mock":
        logger.info("using mock LLM backend")
        return MockLLMClient(tool_names=tool_names)

    logger.info("using Anthropic LLM backend (model=%s)", model)
    return AnthropicLLMClient(api_key=api_key, model=model, max_tokens=max_tokens)


def _extract_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and block.get("type") == "tool_result":
                parts.append(str(block.get("content", "")))
        return " ".join(parts)
    return str(content)


def _build_mock_tool_input(tool_name: str, text: str) -> dict[str, Any]:
    if tool_name == "shell":
        return {"command": "echo 'mock command'"}
    if tool_name == "read_file":
        return {"path": "/tmp/mock_file.txt"}
    if tool_name == "write_file":
        return {"path": "/tmp/mock_file.txt", "content": "mock content"}
    return {"input": text}
