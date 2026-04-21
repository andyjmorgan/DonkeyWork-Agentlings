from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from a2a.client.client_factory import ClientFactory
from a2a.client.middleware import ClientCallContext, ClientCallInterceptor
from a2a.types import AgentCard, Message, Part, Role, TextPart
from a2a.utils import get_message_text


@dataclass
class A2AResponse:
    context_id: str | None
    text: str
    raw: Any


@dataclass
class A2AError:
    code: int
    message: str
    raw: Any


class _APIKeyInterceptor(ClientCallInterceptor):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def intercept(
        self,
        method_name: str,
        request_payload: dict[str, Any],
        http_kwargs: dict[str, Any],
        agent_card: AgentCard | None,
        context: ClientCallContext | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        headers = http_kwargs.get("headers", {})
        headers["X-API-Key"] = self._api_key
        http_kwargs["headers"] = headers
        return request_payload, http_kwargs


def _user_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        messageId=str(uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=text))],
        contextId=context_id,
    )


class A2ATestClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url
        self._api_key = api_key

    async def send(
        self, text: str, context_id: str | None = None
    ) -> A2AResponse | A2AError:
        client = await ClientFactory.connect(
            agent=self._base_url,
            interceptors=[_APIKeyInterceptor(self._api_key)],
        )

        msg = _user_message(text, context_id)
        context_id_out = None
        response_text = ""
        task_id_out = None
        task_state = None

        async for event in client.send_message(msg):
            if isinstance(event, Message):
                response_text = get_message_text(event)
                context_id_out = event.context_id
                # Message events carry the task_id when the executor sets it
                # (fast path returns a Message for a completed task).
                if getattr(event, "task_id", None):
                    task_id_out = event.task_id
                    task_state = "completed"
            elif isinstance(event, tuple):
                task, update = event
                if task:
                    if task.context_id:
                        context_id_out = task.context_id
                    task_id_out = task.id
                    task_state = task.status.state.value if task.status else None
                    # If the task is complete, pull the final message out of history.
                    if task.status and task.status.state.value == "completed":
                        for m in task.history or []:
                            if m.role.value == "agent":
                                response_text = get_message_text(m)

        close_fn = getattr(client, "close", None)
        if close_fn is not None:
            result = close_fn()
            if asyncio.iscoroutine(result):
                await result
        return A2AResponse(
            context_id=context_id_out,
            text=response_text,
            raw={"task_id": task_id_out, "task_state": task_state},
        )

    async def get_task(self, task_id: str) -> dict[str, Any]:
        """Call A2A's ``tasks/get`` to fetch current task state."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        return await self.send_raw(payload)

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Call A2A's ``tasks/cancel`` to cancel a task."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tasks/cancel",
            "params": {"id": task_id},
        }
        return await self.send_raw(payload)

    async def send_raw(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{self._base_url}/a2a",
                json=payload,
                headers={
                    "X-API-Key": self._api_key,
                    "Content-Type": "application/json",
                },
            )
            return resp.json()

    async def send_no_auth(self, text: str) -> int:
        payload = {
            "jsonrpc": "2.0",
            "id": "1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": text}],
                }
            },
        }
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{self._base_url}/a2a",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            return resp.status_code
