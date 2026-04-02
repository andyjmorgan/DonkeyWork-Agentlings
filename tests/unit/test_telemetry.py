"""Tests for telemetry initialization and no-op behavior."""

from __future__ import annotations

from agentlings.config import TelemetryConfig
from agentlings.core.telemetry import _NoOpMeter, _NoOpTracer, get_meter, get_tracer, otel_span, sleep_span


class TestNoOp:
    def test_noop_tracer_creates_span(self) -> None:
        tracer = _NoOpTracer()
        span = tracer.start_as_current_span("test")
        with span:
            span.set_attribute("key", "value")

    def test_noop_meter_creates_instruments(self) -> None:
        meter = _NoOpMeter()
        histogram = meter.create_histogram("test")
        histogram.record(1.0)
        counter = meter.create_counter("test")
        counter.add(1)

    def test_otel_span_works_without_init(self) -> None:
        with otel_span("test.span", {"key": "value"}) as span:
            span.set_attribute("extra", "attr")

    def test_sleep_span_is_otel_span(self) -> None:
        assert sleep_span is otel_span

    def test_nested_spans(self) -> None:
        with otel_span("parent") as parent:
            parent.set_attribute("level", "parent")
            with otel_span("child", {"level": "child"}) as child:
                child.set_attribute("extra", "value")


class TestGetTracer:
    def test_returns_tracer(self) -> None:
        tracer = get_tracer()
        assert tracer is not None


class TestGetMeter:
    def test_returns_meter(self) -> None:
        meter = get_meter()
        assert meter is not None
