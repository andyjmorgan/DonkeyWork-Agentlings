"""OpenTelemetry setup and instrumentation for the agent lifecycle."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from agentlings.config import TelemetryConfig

logger = logging.getLogger(__name__)

_tracer: Any = None
_meter: Any = None
_initialized = False


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
        name: Span name (e.g. ``"agentling.loop.process_message"``).
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
