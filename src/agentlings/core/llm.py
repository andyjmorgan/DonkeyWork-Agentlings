"""LLM client abstraction with Anthropic and mock backends."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

logger = logging.getLogger(__name__)

BATCH_MAX_REQUESTS = 100_000


@dataclass
class LLMResponse:
    """Container for an LLM completion response.

    Attributes:
        content: List of content blocks (text, tool_use, etc.) from the model.
        stop_reason: Why the model stopped generating (e.g. ``"end_turn"``, ``"tool_use"``).
    """

    content: list[dict[str, Any]]
    stop_reason: str | None = None


@dataclass
class BatchRequest:
    """A single request within a batch submission.

    Attributes:
        custom_id: Caller-provided ID for correlating results.
        system: System prompt blocks.
        messages: Conversation messages.
        max_tokens: Maximum response tokens.
        output_schema: JSON Schema for structured output (used with ``output_config``).
    """

    custom_id: str
    system: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    max_tokens: int = 4096
    output_schema: dict[str, Any] | None = None


@dataclass
class BatchItemResult:
    """Result for a single request within a completed batch.

    Attributes:
        custom_id: The caller-provided ID from the request.
        content: Response content blocks (or empty on failure).
        status: Whether this individual request succeeded or failed.
        error: Error message if the request failed.
    """

    custom_id: str
    content: list[dict[str, Any]] = field(default_factory=list)
    status: Literal["succeeded", "failed"] = "succeeded"
    error: str | None = None


@dataclass
class BatchStatus:
    """Status of a batch job.

    Attributes:
        batch_id: The batch identifier.
        processing_status: Current batch state.
        succeeded: Count of succeeded requests.
        failed: Count of failed requests.
        total: Total number of requests.
    """

    batch_id: str
    processing_status: str
    succeeded: int = 0
    failed: int = 0
    total: int = 0


class BaseLLMClient(ABC):
    """Abstract interface for LLM completion backends."""

    @abstractmethod
    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send a completion request and return the full response.

        Args:
            system: System prompt blocks.
            messages: Conversation message history.
            tools: Tool schemas available to the model.
            output_schema: JSON Schema for structured output. When provided,
                the model is constrained to return valid JSON matching this schema.

        Returns:
            The model's response content and stop reason.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream completion response blocks incrementally.

        Args:
            system: System prompt blocks.
            messages: Conversation message history.
            tools: Tool schemas available to the model.

        Yields:
            Individual content blocks as they arrive.
        """
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: The text to tokenize.

        Returns:
            The token count.
        """
        ...

    @abstractmethod
    async def batch_create(self, requests: list[BatchRequest], model: str | None = None) -> list[str]:
        """Submit batch requests, chunking at the API limit.

        Args:
            requests: The batch requests to submit.
            model: Model override (uses client default if ``None``).

        Returns:
            List of batch IDs (one per chunk).
        """
        ...

    @abstractmethod
    async def batch_status(self, batch_id: str) -> BatchStatus:
        """Poll the status of a batch job.

        Args:
            batch_id: The batch identifier.

        Returns:
            Current batch status.
        """
        ...

    @abstractmethod
    async def batch_results(self, batch_id: str) -> list[BatchItemResult]:
        """Retrieve results from a completed batch.

        Args:
            batch_id: The batch identifier.

        Returns:
            List of per-request results.
        """
        ...


class AnthropicLLMClient(BaseLLMClient):
    """LLM client backed by the Anthropic Messages API."""

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
        output_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        if output_schema:
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": output_schema,
                }
            }

        response = await self._client.messages.create(**kwargs)
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

    async def count_tokens(self, text: str) -> int:
        result = await self._client.messages.count_tokens(
            model=self._model,
            messages=[{"role": "user", "content": text}],
        )
        return result.input_tokens

    async def batch_create(self, requests: list[BatchRequest], model: str | None = None) -> list[str]:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        use_model = model or self._model
        batch_ids: list[str] = []

        for i in range(0, len(requests), BATCH_MAX_REQUESTS):
            chunk = requests[i : i + BATCH_MAX_REQUESTS]
            api_requests = []
            for req in chunk:
                params: dict[str, Any] = {
                    "model": use_model,
                    "max_tokens": req.max_tokens,
                    "system": req.system,
                    "messages": req.messages,
                }
                if req.output_schema:
                    params["output_config"] = {
                        "format": {
                            "type": "json_schema",
                            "schema": req.output_schema,
                        }
                    }
                api_requests.append(
                    Request(
                        custom_id=req.custom_id,
                        params=MessageCreateParamsNonStreaming(**params),
                    )
                )

            result = await self._client.messages.batches.create(requests=api_requests)
            batch_ids.append(result.id)
            logger.info("batch submitted: %s (%d requests)", result.id, len(chunk))

        return batch_ids

    async def batch_status(self, batch_id: str) -> BatchStatus:
        result = await self._client.messages.batches.retrieve(batch_id)
        counts = result.request_counts
        return BatchStatus(
            batch_id=result.id,
            processing_status=result.processing_status,
            succeeded=counts.succeeded,
            failed=counts.errored + counts.expired,
            total=counts.succeeded + counts.errored + counts.expired + counts.processing,
        )

    async def batch_results(self, batch_id: str) -> list[BatchItemResult]:
        items: list[BatchItemResult] = []
        async for entry in await self._client.messages.batches.results(batch_id):
            if entry.result.type == "succeeded":
                content = [block.model_dump() for block in entry.result.message.content]
                items.append(BatchItemResult(
                    custom_id=entry.custom_id,
                    content=content,
                    status="succeeded",
                ))
            else:
                error_msg = str(entry.result.error) if hasattr(entry.result, "error") else "unknown error"
                items.append(BatchItemResult(
                    custom_id=entry.custom_id,
                    status="failed",
                    error=error_msg,
                ))
        return items


class MockLLMClient(BaseLLMClient):
    """Deterministic mock LLM client for testing without API calls.

    Returns canned responses based on pattern matching:
    tool name in input triggers tool_use, tool results get text replies,
    everything else gets an echo response.
    """

    def __init__(
        self,
        tool_names: list[str] | None = None,
        compaction_threshold: int = 10,
    ) -> None:
        self._tool_names = tool_names or []
        self._compaction_threshold = compaction_threshold
        self._call_count = 0
        self._batch_store: dict[str, list[BatchRequest]] = {}

    async def complete(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        output_schema: dict[str, Any] | None = None,
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

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    async def batch_create(self, requests: list[BatchRequest], model: str | None = None) -> list[str]:
        batch_id = f"mock_batch_{uuid4().hex[:8]}"
        self._batch_store[batch_id] = requests
        return [batch_id]

    async def batch_status(self, batch_id: str) -> BatchStatus:
        requests = self._batch_store.get(batch_id, [])
        return BatchStatus(
            batch_id=batch_id,
            processing_status="ended",
            succeeded=len(requests),
            failed=0,
            total=len(requests),
        )

    async def batch_results(self, batch_id: str) -> list[BatchItemResult]:
        import json
        requests = self._batch_store.get(batch_id, [])
        results: list[BatchItemResult] = []
        for req in requests:
            text = _extract_text(req.messages[-1]) if req.messages else ""
            mock_output = {
                "summary": f"Summary of conversation: {text[:100]}",
                "memory_candidates": [],
            }
            results.append(BatchItemResult(
                custom_id=req.custom_id,
                content=[{"type": "text", "text": json.dumps(mock_output)}],
                status="succeeded",
            ))
        return results


def create_llm_client(
    backend: str,
    api_key: str = "",
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 4096,
    tool_names: list[str] | None = None,
) -> BaseLLMClient:
    """Factory that returns the appropriate LLM client for the given backend.

    Args:
        backend: Either ``"anthropic"`` for the real API or ``"mock"`` for testing.
        api_key: Anthropic API key (required for ``"anthropic"`` backend).
        model: Model identifier to use for completions.
        max_tokens: Maximum tokens in the model response.
        tool_names: Tool names the mock backend should recognize.

    Returns:
        A configured LLM client instance.
    """
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
    if tool_name == "bash":
        return {"command": "echo 'mock command'"}
    if tool_name == "read_file":
        return {"path": "/tmp/mock_file.txt"}
    if tool_name == "write_file":
        return {"path": "/tmp/mock_file.txt", "content": "mock content"}
    if tool_name == "edit_file":
        return {"path": "/tmp/mock_file.txt", "old_text": "old", "new_text": "new"}
    if tool_name == "list_directory":
        return {"path": "/tmp"}
    if tool_name == "search_files":
        return {"path": "/tmp", "pattern": "*.txt"}
    return {"input": text}
