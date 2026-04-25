"""Starlette application wiring A2A and MCP protocol endpoints on a single HTTP server."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import uvicorn
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from mcp.server.streamable_http import StreamableHTTPServerTransport
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agentlings.config import AgentConfig
from agentlings.core.llm import create_llm_client
from agentlings.core.loop import MessageLoop
from agentlings.core.memory_store import MemoryFileStore
from agentlings.core.scheduler import run_scheduler
from agentlings.core.sleep import SleepCycle
from agentlings.core.store import JournalStore
from agentlings.core.telemetry import init_telemetry
from agentlings.log import setup_logging
from agentlings.protocol.a2a import AgentlingExecutor
from agentlings.protocol.a2a_task_store import EngineTaskStore
from agentlings.protocol.agent_card import generate_agent_card
from agentlings.protocol.mcp import create_mcp_server
from agentlings.tools.memory import init_memory_tool
from agentlings.tools.registry import ToolRegistry

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
            call_next: Callable to pass the request to the next middleware or route.

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

    if config.telemetry_config:
        init_telemetry(config.telemetry_config)

    store = JournalStore(config.agent_data_dir)

    memory_store: MemoryFileStore | None = None
    if "memory" in config.enabled_tools or (
        config.definition.memory is not None
    ):
        memory_store = MemoryFileStore(config.agent_data_dir)
        init_memory_tool(memory_store)

    tools = ToolRegistry()
    tools.register_tools(config.enabled_tools, bash_timeout=config.definition.bash_timeout)

    if not tools.tool_names():
        logger.warning(
            "no tools enabled — add tools to agent.yaml or set AGENT_CONFIG "
            "(run 'agentling --list-tools' to see available options)"
        )

    llm = create_llm_client(
        backend=config.agent_llm_backend,
        api_key=config.anthropic_api_key,
        model=config.agent_model,
        max_tokens=config.agent_max_tokens,
        tool_names=tools.tool_names(),
        base_url=config.anthropic_base_url,
    )

    loop = MessageLoop(
        config=config, store=store, llm=llm, tools=tools,
        memory_store=memory_store,
    )
    # Startup crash recovery: reconcile orphaned task sub-journals and partial
    # merge-backs before accepting new traffic.
    try:
        loop.engine.recover_on_startup()
    except Exception:  # noqa: BLE001 — recovery is best-effort
        logger.exception("crash recovery pass failed")

    agent_card = generate_agent_card(config)

    executor = AgentlingExecutor(loop, config)
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=EngineTaskStore(loop.engine),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    mcp_server = create_mcp_server(loop=loop, agent_card=agent_card, config=config)
    transport: StreamableHTTPServerTransport | None = None

    async def mcp_endpoint(request: Request) -> Response:
        assert transport is not None
        await transport.handle_request(
            request.scope, request.receive, request._send  # type: ignore[attr-defined]
        )
        return Response()

    sleep_cycle: SleepCycle | None = None
    if config.sleep_config and config.sleep_config.enabled and memory_store:
        sleep_cycle = SleepCycle(
            config=config, llm=llm, memory_store=memory_store, store=store,
        )
    elif config.sleep_config and not config.sleep_config.enabled:
        logger.info("sleep cycle disabled by config (sleep.enabled=false)")

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        nonlocal transport
        transport = StreamableHTTPServerTransport(mcp_session_id=None)

        background_tasks: list[asyncio.Task[None]] = []

        async with transport.connect() as (read_stream, write_stream):
            mcp_task = asyncio.create_task(
                mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )
            )
            background_tasks.append(mcp_task)

            if sleep_cycle and config.sleep_config:
                scheduler_task = asyncio.create_task(
                    run_scheduler(
                        expression=config.sleep_config.schedule,
                        callback=sleep_cycle.run,
                    )
                )
                background_tasks.append(scheduler_task)
                logger.info(
                    "sleep scheduler started: %s", config.sleep_config.schedule,
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
                # Drain live task workers before background services so
                # merge-backs in flight complete against a valid store.
                try:
                    await loop.engine.shutdown()
                except Exception:  # noqa: BLE001 — shutdown is best-effort
                    logger.exception("task engine shutdown raised")

                for t in background_tasks:
                    t.cancel()
                    try:
                        await t
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
