"""Tests for telemetry initialization, no-op behavior, and emitted instrumentation.

The "wired" tests stand up a full in-memory OTel pipeline (trace +
metric exporters) and run real LLM completions and engine spawns
through it, asserting the spans + metrics observers see what we expect.
"""

from __future__ import annotations

from typing import Iterator

import pytest

import agentlings.core.telemetry as tele
from agentlings.config import TelemetryConfig
from agentlings.core.telemetry import (
    _NoOpMeter,
    _NoOpTracer,
    attach_context,
    capture_context,
    get_meter,
    get_tracer,
    otel_span,
    record_journal_append,
    record_journal_replay,
    record_llm_usage,
    record_tool_duration,
    sleep_span,
)


# --------------------------------------------------------------------------- #
# No-op safety net (telemetry disabled, no OTel global provider configured)
# --------------------------------------------------------------------------- #


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

    def test_record_helpers_safe_when_disabled(self) -> None:
        # All helpers must be safe to call when no provider is configured.
        record_llm_usage(
            {"input_tokens": 10, "output_tokens": 5},
            model="mock", agent_name="test",
        )
        record_tool_duration("bash", 0.123, False)
        record_journal_append("parent", 256, 0.001)
        record_journal_replay("parent", 17, 0.002)

    def test_capture_context_safe_when_disabled(self) -> None:
        ctx = capture_context()
        with attach_context(ctx):
            with otel_span("inside") as s:
                s.set_attribute("ok", True)


class TestGetTracer:
    def test_returns_tracer(self) -> None:
        tracer = get_tracer()
        assert tracer is not None


class TestGetMeter:
    def test_returns_meter(self) -> None:
        meter = get_meter()
        assert meter is not None


# --------------------------------------------------------------------------- #
# Wired-up OTel: real span + metric exporters in-memory
# --------------------------------------------------------------------------- #


@pytest.fixture
def otel_pipeline() -> Iterator[dict]:
    """Wire in-memory tracer + meter providers into the telemetry module.

    Pulls the tracer and meter directly off provider instances rather than
    via ``trace.set_tracer_provider`` / ``metrics.set_meter_provider`` —
    OTel forbids overriding an already-set global provider, and a prior
    test in the same process likely fixed the global to a default proxy.
    Bypassing the global lets the fixture work in any test order.

    Yields a dict carrying ``spans()`` and ``metrics()`` accessors. Cleans
    up the telemetry module's state so other tests stay isolated.
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[metric_reader])

    tele._tracer = tracer_provider.get_tracer("agentling")
    tele._meter = meter_provider.get_meter("agentling")
    tele._initialized = True
    tele._instruments.clear()

    # The task module's _TaskMetrics is built once at import via lru_cache,
    # so it captured a no-op counter before the fixture set the meter.
    # Refresh it so async-path counters land in the in-memory reader.
    import agentlings.core.task as task_module
    task_module._build_metrics.cache_clear()
    task_module._METRICS = task_module._build_metrics()

    def all_metrics() -> list:
        data = metric_reader.get_metrics_data()
        out: list = []
        if data is None:
            return out
        for rm in data.resource_metrics:
            for sm in rm.scope_metrics:
                for metric in sm.metrics:
                    out.append(metric)
        return out

    yield {
        "spans": span_exporter.get_finished_spans,
        "metrics": all_metrics,
    }

    tele._tracer = None
    tele._meter = None
    tele._initialized = False
    tele._instruments.clear()


def _metric_names(metrics_list: list) -> set[str]:
    return {m.name for m in metrics_list}


class TestLLMTokenTelemetry:
    """Token histograms + counters emit via the mock backend."""

    @pytest.mark.asyncio
    async def test_mock_complete_emits_token_metrics(self, otel_pipeline) -> None:
        from agentlings.core.llm import MockLLMClient

        client = MockLLMClient()
        response = await client.complete(
            system=[{"type": "text", "text": "sys"}],
            messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            tools=[],
        )

        assert response.usage["input_tokens"] >= 1
        assert response.usage["output_tokens"] >= 1
        assert response.model == MockLLMClient.MOCK_MODEL

        names = _metric_names(otel_pipeline["metrics"]())
        assert "agentling.llm.input_tokens" in names
        assert "agentling.llm.output_tokens" in names
        assert "agentling.llm.total_tokens" in names
        assert "agentling.llm.input_tokens_total" in names
        assert "agentling.llm.output_tokens_total" in names
        assert "agentling.llm.calls_total" in names

    def test_record_llm_usage_emits_cache_metrics(self, otel_pipeline) -> None:
        record_llm_usage(
            {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 20,
                "cache_read_input_tokens": 80,
            },
            model="claude-sonnet-4-6",
            agent_name="agent",
        )
        names = _metric_names(otel_pipeline["metrics"]())
        assert "agentling.llm.cache_creation_input_tokens" in names
        assert "agentling.llm.cache_read_input_tokens" in names
        assert "agentling.llm.cache_hit_ratio" in names


class TestCompletionSpanTree:
    """Completion + tool spans emit and stack as parent → child."""

    @pytest.mark.asyncio
    async def test_completion_span_records_token_totals(self, otel_pipeline) -> None:
        from agentlings.core.completion import run_completion
        from agentlings.core.llm import MockLLMClient
        from agentlings.tools.registry import ToolRegistry

        client = MockLLMClient()
        registry = ToolRegistry()

        result = await run_completion(
            llm=client,
            system=[],
            messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            tools=registry,
        )

        assert result.token_usage["input"] >= 1
        assert result.token_usage["output"] >= 1

        spans = otel_pipeline["spans"]()
        names = {s.name for s in spans}
        assert "agentling.completion" in names
        assert "agentling.completion.llm_call" in names
        assert "agentling.llm.complete" in names

        cycle_span = next(s for s in spans if s.name == "agentling.completion")
        assert cycle_span.attributes["completion.input_tokens"] >= 1
        assert cycle_span.attributes["completion.output_tokens"] >= 1


class TestEngineSpawnContextPropagation:
    """The worker span hangs off the caller's span via captured context."""

    @pytest.mark.asyncio
    async def test_worker_span_descends_from_caller_span(
        self, otel_pipeline, test_config, tmp_data_dir
    ) -> None:
        from agentlings.core.llm import MockLLMClient
        from agentlings.core.store import JournalStore
        from agentlings.core.task import TaskEngine
        from agentlings.tools.registry import ToolRegistry

        store = JournalStore(tmp_data_dir)
        llm = MockLLMClient()
        engine = TaskEngine(
            config=test_config, store=store, llm=llm, tools=ToolRegistry(),
        )

        with otel_span("test.outer") as outer:
            outer.set_attribute("test", True)
            await engine.spawn(message="hi", await_seconds=5.0)

        spans = otel_pipeline["spans"]()
        spans_by_name = {s.name: s for s in spans}

        outer = spans_by_name["test.outer"]
        spawn_span = spans_by_name["agentling.engine.spawn"]
        worker_span = spans_by_name["agentling.task.worker"]

        # spawn is a direct child of the outer caller span.
        assert spawn_span.parent.span_id == outer.context.span_id
        # The worker span lives inside the same trace because we re-attached
        # the captured context across the asyncio.create_task boundary.
        assert worker_span.context.trace_id == outer.context.trace_id

    @pytest.mark.asyncio
    async def test_worker_span_carries_token_totals(
        self, otel_pipeline, test_config, tmp_data_dir
    ) -> None:
        from agentlings.core.llm import MockLLMClient
        from agentlings.core.store import JournalStore
        from agentlings.core.task import TaskEngine
        from agentlings.tools.registry import ToolRegistry

        store = JournalStore(tmp_data_dir)
        llm = MockLLMClient()
        engine = TaskEngine(
            config=test_config, store=store, llm=llm, tools=ToolRegistry(),
        )

        await engine.spawn(message="ping", await_seconds=5.0)

        spans = otel_pipeline["spans"]()
        worker = next(s for s in spans if s.name == "agentling.task.worker")
        assert worker.attributes["task.input_tokens"] >= 1
        assert worker.attributes["task.output_tokens"] >= 1
        assert worker.attributes["task.terminal"] == "completed"


class TestAwaitWindowTelemetry:
    """Spawn/poll surface their await-window outcome via span attributes + metric."""

    @pytest.mark.asyncio
    async def test_spawn_completed_within_window_records_completed(
        self, otel_pipeline, test_config, tmp_data_dir
    ) -> None:
        from agentlings.core.llm import MockLLMClient
        from agentlings.core.store import JournalStore
        from agentlings.core.task import TaskEngine
        from agentlings.tools.registry import ToolRegistry

        engine = TaskEngine(
            config=test_config,
            store=JournalStore(tmp_data_dir),
            llm=MockLLMClient(),
            tools=ToolRegistry(),
        )
        await engine.spawn(message="hi", await_seconds=5.0)

        spans = otel_pipeline["spans"]()
        spawn = next(s for s in spans if s.name == "agentling.engine.spawn")
        assert spawn.attributes["task.await_outcome"] == "completed"

        names = _metric_names(otel_pipeline["metrics"]())
        assert "agentling.tasks.await_window_timeouts_total" not in names

    @pytest.mark.asyncio
    async def test_spawn_timeout_records_timed_out_and_increments_counter(
        self, otel_pipeline, test_config, tmp_data_dir
    ) -> None:
        from agentlings.core.store import JournalStore
        from agentlings.core.task import TaskEngine
        from agentlings.tools.registry import ToolRegistry
        from tests.unit.test_task import ControllableLLM

        llm = ControllableLLM()
        engine = TaskEngine(
            config=test_config,
            store=JournalStore(tmp_data_dir),
            llm=llm,
            tools=ToolRegistry(),
        )

        state = await engine.spawn(message="slow", await_seconds=0.05)

        spans = otel_pipeline["spans"]()
        spawn = next(s for s in spans if s.name == "agentling.engine.spawn")
        assert spawn.attributes["task.await_outcome"] == "timed_out"

        names = _metric_names(otel_pipeline["metrics"]())
        assert "agentling.tasks.await_window_timeouts_total" in names

        from agentlings.core.llm import LLMResponse

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "ok"}], stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            import asyncio as _a
            await _a.wait_for(rec.completion_event.wait(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_poll_records_wait_capping(
        self, otel_pipeline, test_config, tmp_data_dir
    ) -> None:
        from agentlings.core.llm import LLMResponse
        from agentlings.core.store import JournalStore
        from agentlings.core.task import TaskEngine
        from agentlings.tools.registry import ToolRegistry
        from tests.unit.test_task import ControllableLLM

        llm = ControllableLLM()
        engine = TaskEngine(
            config=test_config,
            store=JournalStore(tmp_data_dir),
            llm=llm,
            tools=ToolRegistry(),
        )
        state = await engine.spawn(message="slow", await_seconds=0.05)

        await engine.poll(state.task_id, wait_seconds=10.0, cap_seconds=0.05)

        spans = otel_pipeline["spans"]()
        poll = next(s for s in spans if s.name == "agentling.engine.poll")
        assert poll.attributes["task.await_outcome"] == "timed_out"
        assert poll.attributes["task.wait_capped_seconds"] == pytest.approx(0.05)

        llm.push(LLMResponse(
            content=[{"type": "text", "text": "ok"}], stop_reason="end_turn",
        ))
        rec = engine.registry.get(state.task_id)
        if rec is not None:
            import asyncio as _a
            await _a.wait_for(rec.completion_event.wait(), timeout=2.0)


class TestJournalMetrics:
    """JournalStore append/replay paths emit duration + bytes metrics."""

    def test_append_and_replay_record_metrics(self, otel_pipeline, tmp_data_dir) -> None:
        from agentlings.core.models import MessageEntry
        from agentlings.core.store import JournalStore

        store = JournalStore(tmp_data_dir)
        store.create("ctx-1")
        store.append("ctx-1", MessageEntry(
            ctx="ctx-1",
            role="user",
            content=[{"type": "text", "text": "hello"}],
        ))
        store.replay("ctx-1")

        names = _metric_names(otel_pipeline["metrics"]())
        assert "agentling.journal.append_seconds" in names
        assert "agentling.journal.replay_seconds" in names
        assert "agentling.journal.bytes_appended_total" in names
