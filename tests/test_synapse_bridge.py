"""Unit tests for the Synapse v1 adapter bridge (CoScientist/checkpoints/synapse.py)."""
import asyncio
import tempfile
from types import SimpleNamespace


# ── Task 1: SynapseSettings ──────────────────────────────────────────────────

def test_synapse_settings_default_off(monkeypatch):
    monkeypatch.delenv("SYNAPSE__ENABLED", raising=False)
    from CoScientist.config.settings import Settings
    s = Settings()
    assert s.synapse.enabled is False
    assert s.synapse.callback_url is None
    assert s.synapse.bundle_base_url is None
    assert s.synapse.otlp_endpoint is None


def test_synapse_settings_from_env(monkeypatch):
    monkeypatch.setenv("SYNAPSE__ENABLED", "true")
    monkeypatch.setenv("SYNAPSE__CALLBACK_URL", "http://localhost:9999")
    from CoScientist.config.settings import Settings
    s = Settings()
    assert s.synapse.enabled is True
    assert s.synapse.callback_url == "http://localhost:9999"


# ── Task 2: run registry + snapshot_ref ──────────────────────────────────────

def test_run_registry_roundtrip():
    from CoScientist.checkpoints import synapse
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1a2b", "00-abc-def-01")
    assert synapse.run_id_for("ctx-1") == "run-1a2b"
    assert synapse.traceparent_for("ctx-1") == "00-abc-def-01"
    assert synapse.run_id_for("unknown") is None


def test_snapshot_ref_build(monkeypatch):
    from CoScientist.checkpoints import synapse
    monkeypatch.setattr(synapse, "_bundle_base_url", lambda: "http://host:8100")
    assert synapse.snapshot_ref_for("ckpt_X") == "http://host:8100/api/checkpoints/ckpt_X/bundle"


def test_snapshot_ref_none_when_unconfigured(monkeypatch):
    from CoScientist.checkpoints import synapse
    monkeypatch.setattr(synapse, "_bundle_base_url", lambda: None)
    assert synapse.snapshot_ref_for("ckpt_X") is None


# ── Task 3: platform run_id + snapshot_ref on the manifest ────────────────────

class _FakeSession:
    def __init__(self, sid):
        self.app_name = "orchestrator"
        self.user_id = "u"
        self.id = sid
        self.events = []
        self.state = {}


def test_capture_uses_platform_run_id(monkeypatch):
    from CoScientist.checkpoints import synapse, capture
    from CoScientist.checkpoints.store import LocalZipStore
    synapse.clear_runs()
    synapse.register_run("ctx-9", "run-PLATFORM", None)
    monkeypatch.setattr(synapse, "_bundle_base_url", lambda: "http://host:8100")
    store = LocalZipStore(tempfile.mkdtemp())
    m = asyncio.run(capture.capture_checkpoint(
        session=_FakeSession("ctx-9"), label="T1_after_literature_review", store=store))
    assert m is not None
    assert m.run_id == "run-PLATFORM"
    assert m.snapshot_ref == f"http://host:8100/api/checkpoints/{m.checkpoint_id}/bundle"


def test_capture_falls_back_to_run_key(monkeypatch):
    from CoScientist.checkpoints import synapse, capture
    from CoScientist.checkpoints.store import LocalZipStore
    synapse.clear_runs()  # nothing registered
    store = LocalZipStore(tempfile.mkdtemp())
    m = asyncio.run(capture.capture_checkpoint(
        session=_FakeSession("ctx-none"), label="T5_invocation_end", store=store))
    assert m.run_id == "orchestrator__ctx-none"


def test_capture_uses_propagated_run_id_for_remote_child(monkeypatch):
    from CoScientist.checkpoints import capture, synapse, trace_context
    from CoScientist.checkpoints.store import LocalZipStore

    synapse.clear_runs()
    synapse.register_run(
        "root-context",
        "run-PLATFORM-CHILD",
        "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
    )
    monkeypatch.setattr(trace_context, "_proof_secret", lambda: b"demo-secret")
    with trace_context.scope_for_request("root-context", {}):
        headers = trace_context.outbound_carrier()
    with trace_context.scope_for_request("unregistered-child", headers):
        manifest = asyncio.run(
            capture.capture_checkpoint(
                session=_FakeSession("unregistered-child"),
                label="T1_after_literature_review",
                store=LocalZipStore(tempfile.mkdtemp()),
            )
        )
    assert manifest is not None
    assert manifest.run_id == "run-PLATFORM-CHILD"


def test_capture_does_not_trust_unsigned_remote_run_id(monkeypatch):
    from CoScientist.checkpoints import capture, synapse, trace_context
    from CoScientist.checkpoints.store import LocalZipStore

    synapse.clear_runs()
    monkeypatch.setattr(trace_context, "_proof_secret", lambda: b"demo-secret")
    headers = {
        "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
        "baggage": "run_id=known-platform-run",
    }
    with trace_context.scope_for_request("unregistered-child", headers):
        manifest = asyncio.run(
            capture.capture_checkpoint(
                session=_FakeSession("unregistered-child"),
                label="T1_after_literature_review",
                store=LocalZipStore(tempfile.mkdtemp()),
            )
        )
    assert manifest is not None
    assert manifest.run_id == "orchestrator__unregistered-child"


# ── Task 4: outbound snapshot-ready callback ─────────────────────────────────

def test_notify_posts_point(monkeypatch):
    from CoScientist.checkpoints import synapse
    from CoScientist.checkpoints.model import CheckpointManifest, SessionRef
    captured = {}

    def enqueue(url, body):
        captured["url"] = url
        captured["body"] = body
        class R:
            status_code = 200
        return R()

    monkeypatch.setattr(synapse, "_synapse_cfg",
                        lambda: SimpleNamespace(enabled=True, callback_url="http://plat:9000"))
    monkeypatch.setattr(synapse, "enqueue_snapshot", enqueue)
    m = CheckpointManifest(
        checkpoint_id="ckpt_X", label="T1_after_literature_review", run_id="run-1",
        created_at="t", session=SessionRef(app_name="a", user_id="u", session_id="s"),
        snapshot_ref="http://host/api/checkpoints/ckpt_X/bundle")
    synapse.notify_snapshot_saved(m)
    assert captured["url"] == "http://plat:9000/points"
    assert captured["body"]["point_id"] == "ckpt_X"
    assert captured["body"]["run_id"] == "run-1"
    assert captured["body"]["label"] == "T1_after_literature_review"
    assert captured["body"]["snapshot_ref"] == "http://host/api/checkpoints/ckpt_X/bundle"


def test_notify_noop_when_disabled(monkeypatch):
    from CoScientist.checkpoints import synapse
    from CoScientist.checkpoints.model import CheckpointManifest, SessionRef
    monkeypatch.setattr(synapse, "_synapse_cfg",
                        lambda: SimpleNamespace(enabled=False, callback_url=None))
    called = {"n": 0}
    monkeypatch.setattr(synapse, "enqueue_snapshot",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    m = CheckpointManifest(
        checkpoint_id="c", label="L", run_id="r", created_at="t",
        session=SessionRef(app_name="a", user_id="u", session_id="s"))
    synapse.notify_snapshot_saved(m)
    assert called["n"] == 0


# ── Task 5: POST /api/checkpoints/runs ───────────────────────────────────────

def test_runs_endpoint_registers(monkeypatch):
    import CoScientist.config
    from pydantic import SecretStr
    token = "test-only-platform-admin-credential"
    monkeypatch.setattr(CoScientist.config, "get_settings", lambda: SimpleNamespace(
        checkpoints=SimpleNamespace(api_token=SecretStr(token))))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from CoScientist.checkpoints.api import make_checkpoint_router
    from CoScientist.checkpoints import synapse
    from CoScientist.checkpoints.store import LocalZipStore
    synapse.clear_runs()
    app = FastAPI()
    app.include_router(make_checkpoint_router(
        session_service=object(), app_name="orchestrator",
        store=LocalZipStore(tempfile.mkdtemp())))
    c = TestClient(app)
    r = c.post("/api/checkpoints/runs", headers={"Authorization": f"Bearer {token}"},
               json={"context_id": "ctx-77", "run_id": "run-77", "traceparent": "00-t-s-01"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert synapse.run_id_for("ctx-77") == "run-77"
    assert synapse.traceparent_for("ctx-77") == "00-t-s-01"


# ── Task 6: minimal OTel traceparent stitching ───────────────────────────────

def test_otel_span_parents_on_traceparent():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from CoScientist.checkpoints import synapse

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    synapse._TRACER = provider.get_tracer("test")   # test-local tracer

    synapse.clear_runs()
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    synapse.register_run("ctx-tr", "run-tr", tp)
    h = synapse._start_run_span("ctx-tr", "ScriptedOrchestrator", "run-tr")
    synapse._end_run_span(h)

    spans = exporter.get_finished_spans()
    assert any(s.name == "invoke_agent" for s in spans)
    inv = next(s for s in spans if s.name == "invoke_agent")
    assert format(inv.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert format(inv.parent.span_id, "016x") == "b7ad6b7169203331"
    assert inv.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert inv.attributes["gen_ai.agent.name"] == "ScriptedOrchestrator"
    assert inv.attributes["run_id"] == "run-tr"
