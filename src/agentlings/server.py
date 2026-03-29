"""Starlette application wiring A2A and MCP endpoints on a single HTTP server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agentlings.a2a_handler import AgentlingExecutor
from agentlings.agent_card import generate_agent_card
from agentlings.config import AgentConfig
from agentlings.llm import create_llm_client
from agentlings.log import setup_logging
from agentlings.loop import MessageLoop
from agentlings.mcp_handler import create_mcp_server
from agentlings.store import JournalStore
from agentlings.tools import ToolRegistry

logger = logging.getLogger(__name__)

_PUBLIC_PATHS = {
    "/.well-known/agent.json",
    "/.well-known/agent-card.json",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces ``X-API-Key`` header authentication on non-public paths."""

    def __init__(self, app: Any, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Check the API key header and reject unauthorized requests.

        Args:
            request: The incoming HTTP request.
            call_next: Callable to pass the request to the next middleware/route.

        Returns:
            The downstream response, or a 401 JSON response if unauthorized.
        """
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        key = request.headers.get("x-api-key", "")
        if key != self._api_key:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


def _create_app(config: AgentConfig | None = None) -> Starlette:
    if config is None:
        config = AgentConfig()

    setup_logging(config.agent_log_level)

    store = JournalStore(config.agent_data_dir)
    tools = ToolRegistry()
    tools.register_tools(config.enabled_tools)

    llm = create_llm_client(
        backend=config.agent_llm_backend,
        api_key=config.anthropic_api_key,
        model=config.agent_model,
        max_tokens=config.agent_max_tokens,
        tool_names=tools.tool_names(),
    )

    loop = MessageLoop(config=config, store=store, llm=llm, tools=tools)
    agent_card = generate_agent_card(config)

    executor = AgentlingExecutor(loop)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    mcp_server = create_mcp_server(loop=loop, agent_card=agent_card)
    transport: StreamableHTTPServerTransport | None = None

    async def mcp_endpoint(request: Request) -> Response:
        assert transport is not None
        await transport.handle_request(
            request.scope, request.receive, request._send  # type: ignore[attr-defined]
        )
        return Response()

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        nonlocal transport
        transport = StreamableHTTPServerTransport(mcp_session_id=None)

        async with transport.connect() as (read_stream, write_stream):
            task = asyncio.create_task(
                mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
            )
            logger.info(
                "agentling '%s' started on %s:%s",
                config.agent_name,
                config.agent_host,
                config.agent_port,
            )
            try:
                yield
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    routes = [
        Route("/mcp", mcp_endpoint, methods=["GET", "POST", "DELETE"]),
    ]

    middleware = [
        Middleware(APIKeyMiddleware, api_key=config.agent_api_key),
    ]

    app = Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=lifespan,
    )

    a2a_app.add_routes_to_app(app, rpc_url="/a2a")

    return app


def run(config: AgentConfig | None = None) -> None:
    """Build the application and start the uvicorn server.

    Args:
        config: Optional agent configuration; defaults are loaded from the environment.
    """
    if config is None:
        config = AgentConfig()

    app = _create_app(config)
    uvicorn.run(
        app,
        host=config.agent_host,
        port=config.agent_port,
        log_level=config.agent_log_level.lower(),
    )
