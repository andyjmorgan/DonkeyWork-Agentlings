from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import (
    ClientCallContext,
    ClientCallInterceptor,
    ClientConfig,
    create_client,
)
from a2a.client.interceptors import AfterArgs, BeforeArgs
from a2a.helpers.proto_helpers import get_message_text
from a2a.types import (
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageRequest,
    TaskState,
)


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
    """Injects the ``X-API-Key`` header on every outgoing HTTP call.

    The 1.0 SDK feeds ``ClientCallContext.service_parameters`` (a plain
    ``dict[str, str]``) into the httpx request's ``headers`` kwarg, so we
    populate that before each call.
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def before(self, args: BeforeArgs) -> None:
        if args.context is None:
            args.context = ClientCallContext()
        params = args.context.service_parameters
        if params is None:
            params = {}
        params["X-API-Key"] = self._api_key
        args.context.service_parameters = params

    async def after(self, args: AfterArgs) -> None:
        return None


def _user_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        message_id=str(uuid4()),
        role=Role.ROLE_USER,
        parts=[Part(text=text)],
        context_id=context_id or "",
    )


_TASK_STATE_NAME = {
    TaskState.TASK_STATE_WORKING: "working",
    TaskState.TASK_STATE_COMPLETED: "completed",
    TaskState.TASK_STATE_FAILED: "failed",
    TaskState.TASK_STATE_CANCELED: "canceled",
    TaskState.TASK_STATE_SUBMITTED: "submitted",
    TaskState.TASK_STATE_INPUT_REQUIRED: "input-required",
    TaskState.TASK_STATE_REJECTED: "rejected",
    TaskState.TASK_STATE_AUTH_REQUIRED: "auth-required",
    TaskState.TASK_STATE_UNSPECIFIED: None,
}


class A2ATestClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        request_timeout: float | None = None,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._request_timeout = request_timeout

    async def send(
        self, text: str, context_id: str | None = None
    ) -> A2AResponse | A2AError:
        config_kwargs: dict[str, Any] = {"streaming": False}
        if self._request_timeout is not None:
            config_kwargs["httpx_client"] = httpx.AsyncClient(
                timeout=self._request_timeout,
            )
        client = await create_client(
            agent=self._base_url,
            client_config=ClientConfig(**config_kwargs),
            interceptors=[_APIKeyInterceptor(self._api_key)],
        )

        msg = _user_message(text, context_id)
        context_id_out = context_id
        response_text = ""
        task_id_out = None
        task_state = None

        request = SendMessageRequest(message=msg)
        async for event in client.send_message(request):
            # In 1.0 the client yields ``StreamResponse`` whose ``payload``
            # oneof is either ``task`` or ``message``.
            which = event.WhichOneof("payload") if event is not None else None
            if which == "message":
                m = event.message
                response_text = get_message_text(m)
                if m.context_id:
                    context_id_out = m.context_id
                if m.task_id:
                    task_id_out = m.task_id
                    task_state = "completed"
            elif which == "task":
                task = event.task
                if task.context_id:
                    context_id_out = task.context_id
                task_id_out = task.id
                task_state = _TASK_STATE_NAME.get(task.status.state)
                if task.status.state == TaskState.TASK_STATE_COMPLETED:
                    for history_msg in task.history:
                        if history_msg.role == Role.ROLE_AGENT:
                            response_text = get_message_text(history_msg)

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
        """Call A2A's ``tasks/get`` to fetch current task state (0.3 wire format)."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "tasks/get",
            "params": {"id": task_id},
        }
        return await self.send_raw(payload)

    async def cancel_task(self, task_id: str) -> dict[str, Any]:
        """Call A2A's ``tasks/cancel`` to cancel a task (0.3 wire format)."""
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
