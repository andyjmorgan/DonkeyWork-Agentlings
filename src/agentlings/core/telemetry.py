"""OpenTelemetry setup, helpers, and shared instrument cache.

Public surface used across the codebase:

- ``init_telemetry(config)`` — wire OTLP exporters from ``TelemetryConfig``.
  No-op when ``config.enabled`` is ``False`` or the OTel SDK is missing.
- ``otel_span(name, attributes)`` — context-managed span; safe when telemetry
  is disabled (returns a no-op span).
- ``get_tracer()`` / ``get_meter()`` — public accessors. Both return real
  OTel objects when initialized, no-op shims otherwise.
- ``record_llm_usage(usage, *, model, agent_name, path)`` — emit the full
  token telemetry surface (input, output, cache_creation, cache_read,
  total, hit-ratio) as histograms + monotonic counters with consistent
  labels. Returns the totals so callers can stamp them on parent spans.
- ``record_tool_duration(name, duration_seconds, is_error)`` — per-tool
  latency histogram.
- ``record_journal_append(target, byte_count, duration_seconds)`` and
  ``record_journal_replay(target, entry_count, duration_seconds)`` —
  journal I/O instrumentation.
- ``capture_context()`` / ``attach_context(ctx)`` — capture the current
  OTel context at one ``await`` point and re-attach it at another (used
  to make the task worker span a descendant of the inbound request span).

All entry points are safe when telemetry is disabled — they emit through
no-op shims so the only cost is a function call.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator, Iterator

from agentlings.config import TelemetryConfig

logger = logging.getLogger(__name__)

_tracer: Any = None
_meter: Any = None
_initialized = False
_instruments: dict[str, Any] = {}


def init_telemetry(config: TelemetryConfig) -> None:
    """Configure OpenTelemetry tracer and meter providers.

    No-op if telemetry is disabled or already initialized.

    Args:
        config: Telemetry configuration from the agent YAML.
    """
    global _tracer, _meter, _initialized

    if _initialized or not config.enabled:
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": config.service_name})

        headers = config.headers or {}

        if config.protocol == "grpc":
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter as GrpcMetricExporter,
            )
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as GrpcSpanExporter,
            )
            span_exporter = GrpcSpanExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
                headers=headers or None,
            )
            metric_exporter = GrpcMetricExporter(
                endpoint=config.endpoint,
                insecure=config.insecure,
                headers=headers or None,
            )
        else:
            span_exporter = OTLPSpanExporter(
                endpoint=f"{config.endpoint}/v1/traces",
                headers=headers or None,
            )
            metric_exporter = OTLPMetricExporter(
                endpoint=f"{config.endpoint}/v1/metrics",
                headers=headers or None,
            )

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)

        metric_reader = PeriodicExportingMetricReader(metric_exporter)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)

        _tracer = trace.get_tracer("agentling")
        _meter = metrics.get_meter("agentling")
        _initialized = True
        _instruments.clear()

        logger.info(
            "telemetry initialized: endpoint=%s, protocol=%s",
            config.endpoint, config.protocol,
        )
    except ImportError:
        logger.warning(
            "opentelemetry packages not installed, telemetry disabled — "
            "install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
        )
    except Exception:
        logger.exception("failed to initialize telemetry")


def get_tracer() -> Any:
    """Return the configured tracer, or a no-op proxy if telemetry is not initialized."""
    if _tracer is not None:
        return _tracer

    try:
        from opentelemetry import trace
        return trace.get_tracer("agentling")
    except ImportError:
        return _NoOpTracer()


def get_meter() -> Any:
    """Return the configured meter, or a no-op proxy if telemetry is not initialized."""
    if _meter is not None:
        return _meter

    try:
        from opentelemetry import metrics
        return metrics.get_meter("agentling")
    except ImportError:
        return _NoOpMeter()


@contextmanager
def otel_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[Any, None, None]:
    """Context manager that creates a traced span.

    Args:
        name: Span name (e.g. ``"agentling.completion"``).
        attributes: Span attributes to set on creation.

    Yields:
        The span object (or a no-op if telemetry is disabled).
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        yield span


sleep_span = otel_span


# --------------------------------------------------------------------------- #
# Context propagation across asyncio.create_task boundaries
# --------------------------------------------------------------------------- #


def capture_context() -> Any:
    """Capture the current OTel context.

    Returned object is opaque; pass it to ``attach_context`` from the
    target async task to make spans created there descendants of the
    captured span.

    Returns ``None`` (cheaply) when the OTel API is not importable, which
    keeps no-op deployments lean.
    """
    try:
        from opentelemetry import context as otel_context
        return otel_context.get_current()
    except ImportError:
        return None


@contextmanager
def attach_context(ctx: Any) -> Iterator[None]:
    """Re-attach a previously captured OTel context for the duration of the block.

    Used inside ``asyncio.create_task`` callbacks where the parent context
    would otherwise be lost. Pairs with ``capture_context``.
    """
    if ctx is None:
        yield
        return
    try:
        from opentelemetry import context as otel_context
    except ImportError:
        yield
        return
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


# --------------------------------------------------------------------------- #
# Shared instrument cache + recording helpers
# --------------------------------------------------------------------------- #


def _get_instrument(key: str, factory: Any) -> Any:
    """Return a cached metric instrument, creating it on first access."""
    inst = _instruments.get(key)
    if inst is not None:
        return inst
    inst = factory()
    _instruments[key] = inst
    return inst


def _llm_token_instruments() -> dict[str, Any]:
    """Lazily build the LLM token histograms + counters."""
    m = get_meter()

    def _build() -> dict[str, Any]:
        return {
            # Histograms — useful for percentiles and per-call distribution.
            "input_h": m.create_histogram(
                "agentling.llm.input_tokens",
                description="Fresh prompt tokens charged on an LLM call.",
            ),
            "output_h": m.create_histogram(
                "agentling.llm.output_tokens",
                description="Generated output tokens on an LLM call.",
            ),
            "cache_creation_h": m.create_histogram(
                "agentling.llm.cache_creation_input_tokens",
                description="Tokens written to the prompt cache.",
            ),
            "cache_read_h": m.create_histogram(
                "agentling.llm.cache_read_input_tokens",
                description="Tokens served from the prompt cache.",
            ),
            "total_h": m.create_histogram(
                "agentling.llm.total_tokens",
                description="Sum of input + output + cache tokens.",
            ),
            "hit_ratio_h": m.create_histogram(
                "agentling.llm.cache_hit_ratio",
                description="cache_read / (input + cache_read) per call.",
            ),
            # Monotonic counters — better for rate() queries.
            "input_c": m.create_counter(
                "agentling.llm.input_tokens_total",
                description="Cumulative fresh prompt tokens.",
            ),
            "output_c": m.create_counter(
                "agentling.llm.output_tokens_total",
                description="Cumulative output tokens.",
            ),
            "cache_creation_c": m.create_counter(
                "agentling.llm.cache_creation_input_tokens_total",
                description="Cumulative cache-write tokens.",
            ),
            "cache_read_c": m.create_counter(
                "agentling.llm.cache_read_input_tokens_total",
                description="Cumulative cache-read tokens.",
            ),
            "calls_c": m.create_counter(
                "agentling.llm.calls_total",
                description="Cumulative LLM call count.",
            ),
        }

    return _get_instrument("llm.tokens", _build)


def record_llm_usage(
    usage: dict[str, Any] | None,
    *,
    model: str,
    agent_name: str = "",
    path: str = "live",
) -> dict[str, int]:
    """Emit the full LLM token telemetry surface for a single call.

    Args:
        usage: Anthropic-style ``usage`` dict carrying ``input_tokens``,
            ``output_tokens``, ``cache_creation_input_tokens``,
            ``cache_read_input_tokens``. ``None`` records a single call
            event with zero tokens (useful for non-Anthropic backends).
        model: Model identifier — labels every metric for slicing.
        agent_name: Agent name from config (optional label).
        path: ``"live"`` for ``messages.create`` calls, ``"batch"`` for
            results coming out of the batch API. Lets dashboards filter
            interactive vs. nightly token spend.

    Returns:
        Dict with the four token counts: ``input``, ``output``,
        ``cache_creation``, ``cache_read``. Callers roll these up onto
        parent spans for at-a-glance trace inspection.
    """
    inst = _llm_token_instruments()
    attrs = {"llm.model": model, "llm.path": path}
    if agent_name:
        attrs["agent.name"] = agent_name

    u = usage or {}
    input_tokens = int(u.get("input_tokens", 0) or 0)
    output_tokens = int(u.get("output_tokens", 0) or 0)
    cache_creation = int(u.get("cache_creation_input_tokens", 0) or 0)
    cache_read = int(u.get("cache_read_input_tokens", 0) or 0)
    total = input_tokens + output_tokens + cache_creation + cache_read

    inst["calls_c"].add(1, attrs)

    if input_tokens:
        inst["input_h"].record(input_tokens, attrs)
        inst["input_c"].add(input_tokens, attrs)
    if output_tokens:
        inst["output_h"].record(output_tokens, attrs)
        inst["output_c"].add(output_tokens, attrs)
    if cache_creation:
        inst["cache_creation_h"].record(cache_creation, attrs)
        inst["cache_creation_c"].add(cache_creation, attrs)
    if cache_read:
        inst["cache_read_h"].record(cache_read, attrs)
        inst["cache_read_c"].add(cache_read, attrs)
    if total:
        inst["total_h"].record(total, attrs)
    # Cache hit ratio — only meaningful when there's *some* prompt input
    # (fresh + cached). Reporting 0 when there's no input would skew the
    # distribution.
    denom = input_tokens + cache_read
    if denom:
        inst["hit_ratio_h"].record(cache_read / denom, attrs)

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cache_creation": cache_creation,
        "cache_read": cache_read,
    }


def _tool_instruments() -> dict[str, Any]:
    m = get_meter()

    def _build() -> dict[str, Any]:
        return {
            "duration": m.create_histogram(
                "agentling.tool.duration_seconds",
                description="Tool execution wall-clock duration.",
            ),
        }

    return _get_instrument("tool", _build)


def record_tool_duration(name: str, duration_seconds: float, is_error: bool) -> None:
    """Record a tool-execution latency sample with name + error labels."""
    inst = _tool_instruments()
    inst["duration"].record(
        duration_seconds,
        {"tool.name": name, "tool.is_error": str(is_error).lower()},
    )


def _journal_instruments() -> dict[str, Any]:
    m = get_meter()

    def _build() -> dict[str, Any]:
        return {
            "append_seconds": m.create_histogram(
                "agentling.journal.append_seconds",
                description="JSONL append wall-clock duration.",
            ),
            "replay_seconds": m.create_histogram(
                "agentling.journal.replay_seconds",
                description="Journal replay wall-clock duration.",
            ),
            "entries_replayed": m.create_counter(
                "agentling.journal.entries_replayed_total",
                description="Cumulative count of journal entries replayed.",
            ),
            "bytes_appended": m.create_counter(
                "agentling.journal.bytes_appended_total",
                description="Cumulative bytes appended to JSONL journals.",
            ),
        }

    return _get_instrument("journal", _build)


def record_journal_append(target: str, byte_count: int, duration_seconds: float) -> None:
    """Record an append latency sample and bump the bytes-written counter.

    Args:
        target: ``"parent"`` or ``"sub"`` — which journal layer was hit.
        byte_count: Bytes written in this append.
        duration_seconds: Wall-clock duration of the append.
    """
    inst = _journal_instruments()
    attrs = {"journal.target": target}
    inst["append_seconds"].record(duration_seconds, attrs)
    if byte_count:
        inst["bytes_appended"].add(byte_count, attrs)


def record_journal_replay(target: str, entry_count: int, duration_seconds: float) -> None:
    """Record a replay latency sample and bump the entries-replayed counter."""
    inst = _journal_instruments()
    attrs = {"journal.target": target}
    inst["replay_seconds"].record(duration_seconds, attrs)
    if entry_count:
        inst["entries_replayed"].add(entry_count, attrs)


# --------------------------------------------------------------------------- #
# No-op fallbacks
# --------------------------------------------------------------------------- #


class _NoOpSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        pass

    def __enter__(self) -> _NoOpSpan:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class _NoOpTracer:
    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpMeter:
    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()

    def create_counter(self, name: str, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> _NoOpInstrument:
        return _NoOpInstrument()


class _NoOpInstrument:
    def record(self, value: Any, attributes: dict[str, Any] | None = None) -> None:
        pass

    def add(self, value: Any, attributes: dict[str, Any] | None = None) -> None:
        pass
