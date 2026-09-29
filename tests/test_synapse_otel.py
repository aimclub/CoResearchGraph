"""Regression tests for Synapse bridge OpenTelemetry setup."""

import asyncio
import builtins
import importlib
import logging
import sys
import threading
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

from packaging.requirements import Requirement


def _declared_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    return {Requirement(value).name for value in project["dependencies"]}


def _install_fake_otlp_exporter(monkeypatch, exporter_type):
    parent = importlib.import_module("opentelemetry")
    parent_name = "opentelemetry"
    for child_name in ("exporter", "otlp", "proto", "http", "trace_exporter"):
        module_name = f"{parent_name}.{child_name}"
        module = ModuleType(module_name)
        if child_name != "trace_exporter":
            module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
        monkeypatch.setattr(parent, child_name, module, raising=False)
        parent = module
        parent_name = module_name
    parent.OTLPSpanExporter = exporter_type


def _capture_provider(monkeypatch, synapse):
    providers = []
    monkeypatch.setattr(synapse._ot_trace, "set_tracer_provider", providers.append)
    monkeypatch.setattr(
        synapse._ot_trace,
        "get_tracer",
        lambda name: providers[-1].get_tracer(name),
    )
    return providers


def test_networkx_is_a_declared_runtime_dependency():
    assert "networkx" in _declared_dependencies()


def test_configured_runtime_can_import_otlp_http_exporter():
    module = importlib.import_module(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    )
    assert module.OTLPSpanExporter is not None


def test_setup_otel_uses_batch_processor_with_bounded_timeout(monkeypatch):
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

    from CoScientist.checkpoints import synapse

    created = []

    class RecordingExporter:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            return None

    _install_fake_otlp_exporter(monkeypatch, RecordingExporter)
    providers = _capture_provider(monkeypatch, synapse)
    monkeypatch.setattr(
        synapse,
        "_synapse_cfg",
        lambda: SimpleNamespace(
            enabled=True, otlp_endpoint="http://collector/v1/traces"
        ),
    )
    monkeypatch.setattr(synapse, "_TRACER", None)
    monkeypatch.setattr(synapse, "_OTEL_READY", False)

    synapse.setup_otel()

    processors = providers[0]._active_span_processor._span_processors
    assert any(isinstance(processor, BatchSpanProcessor) for processor in processors)
    assert any(
        processor.__class__.__name__ == "RunIdSpanProcessor" for processor in processors
    )
    assert created == [{"endpoint": "http://collector/v1/traces", "timeout": 3.0}]
    providers[0].shutdown()


def test_blocking_failed_exporter_does_not_block_async_callbacks(monkeypatch):
    from CoScientist.checkpoints import synapse

    heartbeat_seen = threading.Event()
    release_exporter = threading.Event()
    heartbeat_preceded_release = []
    exported_spans = []

    class BlockingFailureExporter:
        def __init__(self, **kwargs):
            pass

        def export(self, spans):
            exported_spans.extend(spans)
            release_exporter.wait(timeout=2.0)
            raise RuntimeError("collector unavailable")

        def shutdown(self):
            release_exporter.set()

    _install_fake_otlp_exporter(monkeypatch, BlockingFailureExporter)
    providers = _capture_provider(monkeypatch, synapse)
    monkeypatch.setattr(
        synapse,
        "_synapse_cfg",
        lambda: SimpleNamespace(
            enabled=True, otlp_endpoint="http://collector/v1/traces"
        ),
    )
    monkeypatch.setattr(synapse, "_TRACER", None)
    monkeypatch.setattr(synapse, "_OTEL_READY", False)
    synapse.clear_runs()
    synapse.register_run(
        "ctx-live",
        "run-live",
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    )
    plugin = synapse.SynapseTracePlugin()
    invocation = SimpleNamespace(
        invocation_id="invocation-live",
        session=SimpleNamespace(id="ctx-live", app_name="science"),
        agent=SimpleNamespace(name="HypothesesAgent"),
    )

    def release_after_heartbeat():
        heartbeat_preceded_release.append(heartbeat_seen.wait(timeout=0.5))
        release_exporter.set()

    observer = threading.Thread(target=release_after_heartbeat)
    observer.start()

    async def exercise_callbacks():
        await plugin.before_run_callback(invocation_context=invocation)
        science_callback = asyncio.create_task(
            plugin.after_run_callback(invocation_context=invocation)
        )
        await asyncio.sleep(0)

        async def heartbeat():
            heartbeat_seen.set()

        heartbeat_task = asyncio.create_task(heartbeat())
        await asyncio.gather(science_callback, heartbeat_task)

    asyncio.run(exercise_callbacks())
    observer.join(timeout=1.0)
    providers[0].force_flush(timeout_millis=1000)

    assert not observer.is_alive()
    assert heartbeat_preceded_release == [True]
    assert len(exported_spans) == 1
    span = exported_spans[0]
    assert format(span.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(span.parent.span_id, "016x") == "b7ad6b7169203331"
    assert span.attributes["run_id"] == "run-live"
    providers[0].shutdown()


def test_setup_failure_is_logged_and_can_be_retried(monkeypatch, caplog):
    from opentelemetry.sdk.trace.export import SpanExportResult

    from CoScientist.checkpoints import synapse

    attempts = []

    class FlakyExporter:
        def __init__(self, **kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise RuntimeError("setup unavailable")

        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            return None

    _install_fake_otlp_exporter(monkeypatch, FlakyExporter)
    providers = _capture_provider(monkeypatch, synapse)
    monkeypatch.setattr(
        synapse,
        "_synapse_cfg",
        lambda: SimpleNamespace(
            enabled=True, otlp_endpoint="http://collector/v1/traces"
        ),
    )
    monkeypatch.setattr(synapse, "_TRACER", None)
    monkeypatch.setattr(synapse, "_OTEL_READY", False)

    with caplog.at_level(logging.WARNING, logger=synapse.__name__):
        synapse.setup_otel()
        synapse.setup_otel()
        synapse.setup_otel()

    assert "OTel setup failed" in caplog.text
    assert len(attempts) == 2
    assert synapse._OTEL_READY is True
    assert synapse._TRACER is not None
    providers[0].shutdown()


def test_unconfigured_setup_does_not_import_exporter(monkeypatch):
    from CoScientist.checkpoints import synapse

    original_import = builtins.__import__

    def reject_exporter_import(name, *args, **kwargs):
        if name.startswith("opentelemetry.exporter"):
            raise AssertionError("exporter import is not allowed while unconfigured")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_exporter_import)
    monkeypatch.setattr(
        synapse,
        "_synapse_cfg",
        lambda: SimpleNamespace(enabled=False, otlp_endpoint=None),
    )
    monkeypatch.setattr(synapse, "_TRACER", None)
    monkeypatch.setattr(synapse, "_OTEL_READY", False)

    synapse.setup_otel()
    synapse.setup_otel()

    assert synapse._OTEL_READY is True
    assert synapse._TRACER is not None
