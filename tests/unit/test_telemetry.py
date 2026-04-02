"""Tests for telemetry initialization and no-op behavior."""

from __future__ import annotations

from agentlings.config import TelemetryConfig
from agentlings.core.telemetry import _NoOpMeter, _NoOpTracer, get_meter, get_tracer, sleep_span


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

    def test_sleep_span_works_without_init(self) -> None:
        with sleep_span("test.span", {"key": "value"}) as span:
            span.set_attribute("extra", "attr")


class TestGetTracer:
    def test_returns_tracer(self) -> None:
        tracer = get_tracer()
        assert tracer is not None


class TestGetMeter:
    def test_returns_meter(self) -> None:
        meter = get_meter()
        assert meter is not None
