# Synapse v1 Adapter Layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CoScientist's existing checkpoint subsystem speak the Synapse v1 core contract — platform-issued run_id, a "snapshot ready" callback with a stable snapshot_ref, platform-driven restore, minimal OTel traceparent stitching, and a single adapter endpoint — verified end-to-end against a local mock platform.

**Architecture:** A thin bridge module `checkpoints/synapse.py` holds a process-local run registry (context_id → run_id/traceparent), builds the snapshot_ref, and fires the outbound callback; the existing capture/restore code is reused unchanged except for reading run_id from the registry. A `scripts/mock_synapse.py` stand-in issues run_ids, receives callbacks, and drives restore. Everything is OFF by default (`SYNAPSE__ENABLED=false`).

**Tech Stack:** Python 3.12.11, FastAPI (already the A2A app), google-adk 2.3.0 plugins, opentelemetry-sdk (already in venv), httpx (already a dep), pytest.

## Global Constraints

- Python **3.12.11**; run everything with the repo venv: `/Users/kiriill/Documents/Python/CoScientist/.venv/bin/python`.
- Work in the worktree `.../scratchpad/cp2` on branch `feat/synapse-v1-adapter`; set `PYTHONPATH` to that worktree root when running.
- **OFF by default:** with `SYNAPSE__ENABLED=false` (the default) behaviour is byte-identical to today. Every new code path is gated on it.
- **Best-effort, never break the run:** every bridge call is wrapped in try/except and logs on failure — a snapshot/callback/trace failure must never fail the science run (same doctrine as `capture.py`).
- **Do not break** `tests/e2e_checkpoint_a2a.py` (bare restore); it must stay green.
- Tests use the scripted deterministic agent pattern from `tests/e2e_checkpoint_a2a.py` — **no LLM, no VPN, no network** beyond localhost.
- Reuse, don't rewrite: restore (`checkpoints/restore.py`) and the store (`checkpoints/store.py`) are untouched.

---

### Task 1: SynapseSettings config

**Files:**
- Modify: `CoScientist/config/settings.py:243-274`
- Test: `tests/test_synapse_bridge.py`

**Interfaces:**
- Produces: `settings.synapse.enabled: bool`, `settings.synapse.callback_url: str | None`, `settings.synapse.bundle_base_url: str | None`, `settings.synapse.otlp_endpoint: str | None`. Env keys: `SYNAPSE__ENABLED`, `SYNAPSE__CALLBACK_URL`, `SYNAPSE__BUNDLE_BASE_URL`, `SYNAPSE__OTLP_ENDPOINT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synapse_bridge.py
import os

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k synapse_settings -v`
Expected: FAIL — `Settings` has no attribute `synapse`.

- [ ] **Step 3: Write minimal implementation**

Add after `CheckpointSettings` (settings.py:251), before `# MAIN SETTINGS`:

```python
# =========================
# SYNAPSE v1 ADAPTER
# =========================
class SynapseSettings(BaseModel):
    """Synapse platform v1 contract bridge (checkpoints/synapse.py).

    OFF by default: enabling makes checkpoints report to the platform
    (callback_url), stamps platform-issued run_ids, builds snapshot_refs from
    bundle_base_url, and exports OTel spans to otlp_endpoint. Override via
    SYNAPSE__ENABLED / SYNAPSE__CALLBACK_URL / SYNAPSE__BUNDLE_BASE_URL /
    SYNAPSE__OTLP_ENDPOINT.
    """
    enabled: bool = False
    callback_url: Optional[str] = None      # platform base URL for "snapshot ready"
    bundle_base_url: Optional[str] = None    # adapter base URL used to build snapshot_ref
    otlp_endpoint: Optional[str] = None      # OTLP HTTP collector for trace export
```

Add to the `Settings` class body after line 274 (`checkpoints: ...`):

```python
    synapse: SynapseSettings = SynapseSettings()
```

(`Optional` is already imported in settings.py.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k synapse_settings -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add CoScientist/config/settings.py tests/test_synapse_bridge.py
git commit -m "feat(synapse): add SynapseSettings (off by default)"
```

---

### Task 2: Bridge core — run registry + snapshot_ref

**Files:**
- Create: `CoScientist/checkpoints/synapse.py`
- Test: `tests/test_synapse_bridge.py`

**Interfaces:**
- Consumes: `settings.synapse.bundle_base_url` (Task 1).
- Produces:
  - `register_run(context_id: str, run_id: str, traceparent: str | None = None) -> None`
  - `run_id_for(context_id: str) -> str | None`
  - `traceparent_for(context_id: str) -> str | None`
  - `snapshot_ref_for(checkpoint_id: str) -> str | None`
  - `clear_runs() -> None`  (test helper)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synapse_bridge.py  (append)
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k "registry or snapshot_ref" -v`
Expected: FAIL — `No module named 'CoScientist.checkpoints.synapse'`.

- [ ] **Step 3: Write minimal implementation**

```python
# CoScientist/checkpoints/synapse.py
"""Synapse v1 contract bridge for the checkpoint subsystem.

Off unless SYNAPSE__ENABLED. Holds a process-local registry mapping an A2A
contextId (== ADK session id) to the platform-issued run_id + traceparent, so
capture stamps the platform's run_id instead of the self-generated one. Also
builds the snapshot_ref and fires the outbound "snapshot ready" callback.

Every function is best-effort: a bridge failure never breaks the science run.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_RUN_CONTEXT: Dict[str, Dict[str, Optional[str]]] = {}   # context_id -> {run_id, traceparent}


def _bundle_base_url() -> Optional[str]:
    from CoScientist.config import get_settings
    return get_settings().synapse.bundle_base_url


def register_run(context_id: str, run_id: str, traceparent: Optional[str] = None) -> None:
    with _LOCK:
        _RUN_CONTEXT[context_id] = {"run_id": run_id, "traceparent": traceparent}


def run_id_for(context_id: str) -> Optional[str]:
    with _LOCK:
        entry = _RUN_CONTEXT.get(context_id)
        return entry["run_id"] if entry else None


def traceparent_for(context_id: str) -> Optional[str]:
    with _LOCK:
        entry = _RUN_CONTEXT.get(context_id)
        return entry["traceparent"] if entry else None


def clear_runs() -> None:
    with _LOCK:
        _RUN_CONTEXT.clear()


def snapshot_ref_for(checkpoint_id: str) -> Optional[str]:
    base = _bundle_base_url()
    if not base:
        return None
    return f"{base.rstrip('/')}/api/checkpoints/{checkpoint_id}/bundle"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k "registry or snapshot_ref" -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add CoScientist/checkpoints/synapse.py tests/test_synapse_bridge.py
git commit -m "feat(synapse): run registry + snapshot_ref builder"
```

---

### Task 3: Platform run_id + snapshot_ref in the manifest

**Files:**
- Modify: `CoScientist/checkpoints/model.py:37-55`
- Modify: `CoScientist/checkpoints/capture.py:250-269`
- Test: `tests/test_synapse_bridge.py`

**Interfaces:**
- Consumes: `synapse.run_id_for` / `synapse.snapshot_ref_for` (Task 2).
- Produces: `CheckpointManifest.snapshot_ref: Optional[str]`; `capture_checkpoint(...)` stamps platform run_id when registered.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synapse_bridge.py  (append)
import asyncio, tempfile
from types import SimpleNamespace

class _FakeSession:
    def __init__(self, sid): self.app_name="orchestrator"; self.user_id="u"; self.id=sid; self.events=[]; self.state={}

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k capture -v`
Expected: FAIL — run_id is `orchestrator__ctx-9` (fallback) and `snapshot_ref` attribute missing.

- [ ] **Step 3: Write minimal implementation**

In `model.py`, add to `CheckpointManifest` (after `warnings`, line 55):

```python
    snapshot_ref: Optional[str] = None   # Synapse v1: platform-held reference to the bundle
```

In `capture.py`, replace line 250 (`rid = run_key(session)`) with:

```python
        from CoScientist.checkpoints import synapse
        rid = synapse.run_id_for(session.id) or run_key(session)
```

In `capture.py`, after the `manifest = CheckpointManifest(...)` block (after line 269), before `parts = {...}`:

```python
        manifest.snapshot_ref = synapse.snapshot_ref_for(manifest.checkpoint_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k capture -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add CoScientist/checkpoints/model.py CoScientist/checkpoints/capture.py tests/test_synapse_bridge.py
git commit -m "feat(synapse): stamp platform run_id + snapshot_ref on the manifest"
```

---

### Task 4: Outbound "snapshot ready" callback

**Files:**
- Modify: `CoScientist/checkpoints/synapse.py`
- Modify: `CoScientist/checkpoints/capture.py:277` (after `store.save`)
- Test: `tests/test_synapse_bridge.py`

**Interfaces:**
- Consumes: `settings.synapse.enabled`, `settings.synapse.callback_url`; `CheckpointManifest` (Task 3).
- Produces: `notify_snapshot_saved(manifest: CheckpointManifest) -> None` — best-effort `POST {callback_url}/points`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synapse_bridge.py  (append)
def test_notify_posts_point(monkeypatch):
    from CoScientist.checkpoints import synapse
    from CoScientist.checkpoints.model import CheckpointManifest, SessionRef
    captured = {}
    def fake_post(url, json, timeout):
        captured["url"] = url; captured["body"] = json
        class R: status_code = 200
        return R()
    monkeypatch.setattr(synapse, "_synapse_cfg", lambda: SimpleNamespace(enabled=True, callback_url="http://plat:9000"))
    monkeypatch.setattr(synapse.httpx, "post", fake_post)
    m = CheckpointManifest(checkpoint_id="ckpt_X", label="T1_after_literature_review",
        run_id="run-1", created_at="t", session=SessionRef(app_name="a", user_id="u", session_id="s"),
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
    monkeypatch.setattr(synapse, "_synapse_cfg", lambda: SimpleNamespace(enabled=False, callback_url=None))
    called = {"n": 0}
    monkeypatch.setattr(synapse.httpx, "post", lambda *a, **k: called.__setitem__("n", called["n"]+1))
    m = CheckpointManifest(checkpoint_id="c", label="L", run_id="r", created_at="t",
        session=SessionRef(app_name="a", user_id="u", session_id="s"))
    synapse.notify_snapshot_saved(m)
    assert called["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k notify -v`
Expected: FAIL — `synapse` has no `httpx`/`notify_snapshot_saved`/`_synapse_cfg`.

- [ ] **Step 3: Write minimal implementation**

In `synapse.py`, add `import httpx` at the top and:

```python
def _synapse_cfg():
    from CoScientist.config import get_settings
    return get_settings().synapse


def notify_snapshot_saved(manifest) -> None:
    """Tell the platform a snapshot is ready (best-effort). v1 §5: one call,
    the platform records the point itself — no separate 'snapshot created' event."""
    cfg = _synapse_cfg()
    if not cfg.enabled or not cfg.callback_url:
        return
    body = {
        "point_id": manifest.checkpoint_id,
        "run_id": manifest.run_id,
        "time": manifest.created_at,
        "label": manifest.label,
        "snapshot_ref": manifest.snapshot_ref,
    }
    try:
        httpx.post(f"{cfg.callback_url.rstrip('/')}/points", json=body, timeout=5.0)
    except Exception as exc:  # noqa: BLE001 — never break the run
        logger.warning("synapse: snapshot-ready callback failed: %s", exc)
```

In `capture.py`, change the save line (line 277) from `return store.save(manifest, parts)` to:

```python
        saved = store.save(manifest, parts)
        synapse.notify_snapshot_saved(saved)
        return saved
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k notify -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add CoScientist/checkpoints/synapse.py CoScientist/checkpoints/capture.py tests/test_synapse_bridge.py
git commit -m "feat(synapse): outbound snapshot-ready callback"
```

---

### Task 5: `POST /api/runs` — platform registers a run

**Files:**
- Modify: `CoScientist/checkpoints/api.py:60-75` (add route inside `make_checkpoint_router`)
- Test: `tests/test_synapse_bridge.py`

**Interfaces:**
- Consumes: `synapse.register_run` (Task 2).
- Produces: route `POST /api/checkpoints/runs` with body `{context_id, run_id, traceparent?}` → registers the run; returns `{"ok": True}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synapse_bridge.py  (append)
def test_runs_endpoint_registers(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from CoScientist.checkpoints.api import make_checkpoint_router
    from CoScientist.checkpoints import synapse
    from CoScientist.checkpoints.store import LocalZipStore
    synapse.clear_runs()
    app = FastAPI()
    app.include_router(make_checkpoint_router(
        session_service=object(), app_name="orchestrator", store=LocalZipStore(tempfile.mkdtemp())))
    c = TestClient(app)
    r = c.post("/api/checkpoints/runs",
               json={"context_id": "ctx-77", "run_id": "run-77", "traceparent": "00-t-s-01"})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert synapse.run_id_for("ctx-77") == "run-77"
    assert synapse.traceparent_for("ctx-77") == "00-t-s-01"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k runs_endpoint -v`
Expected: FAIL — 404 (route not found).

- [ ] **Step 3: Write minimal implementation**

In `api.py`, add a request model near `RestoreRequest` (line 41):

```python
class RunRegisterRequest(BaseModel):
    context_id: str
    run_id: str
    traceparent: Optional[str] = None
```

Inside `make_checkpoint_router`, alongside the other routes (after the `list_checkpoints` route, ~line 66):

```python
    @router.post("/runs")
    async def register_run(body: RunRegisterRequest):
        from CoScientist.checkpoints import synapse
        synapse.register_run(body.context_id, body.run_id, body.traceparent)
        return {"ok": True}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k runs_endpoint -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CoScientist/checkpoints/api.py tests/test_synapse_bridge.py
git commit -m "feat(synapse): POST /api/checkpoints/runs registers platform run_id + traceparent"
```

---

### Task 6: OTel minimal — traceparent stitching + OTLP export

**Files:**
- Modify: `CoScientist/checkpoints/synapse.py`
- Modify: `CoScientist/a2a/server.py:128-134` (register the trace plugin when synapse enabled)
- Test: `tests/test_synapse_bridge.py`

**Interfaces:**
- Consumes: `traceparent_for` (Task 2), `settings.synapse` (Task 1).
- Produces:
  - `setup_otel() -> None` — idempotent; installs a TracerProvider with OTLP exporter (or no-op if unconfigured).
  - `SynapseTracePlugin` (google-adk `BasePlugin`): `before_run_callback` starts an `invoke_agent` span parented on the incoming traceparent; `after_run_callback` ends it.
  - `_start_run_span(context_id, agent_name, run_id)` / `_end_run_span(handle)` — used by the plugin and directly testable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synapse_bridge.py  (append)
def test_otel_span_parents_on_traceparent():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from CoScientist.checkpoints import synapse

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)      # test-local provider
    synapse._TRACER = provider.get_tracer("test")

    synapse.clear_runs()
    tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    synapse.register_run("ctx-tr", "run-tr", tp)
    h = synapse._start_run_span("ctx-tr", "ScriptedOrchestrator", "run-tr")
    synapse._end_run_span(h)

    spans = exporter.get_finished_spans()
    assert any(s.name == "invoke_agent" for s in spans)
    inv = next(s for s in spans if s.name == "invoke_agent")
    assert format(inv.context.trace_id, "032x") == "0af7651916cd43dd8448eb211c80319c"
    assert inv.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert inv.attributes["gen_ai.agent.name"] == "ScriptedOrchestrator"
    assert inv.attributes["run_id"] == "run-tr"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k otel -v`
Expected: FAIL — `synapse` has no `_start_run_span` / `_TRACER`.

- [ ] **Step 3: Write minimal implementation**

Append to `synapse.py`:

```python
from opentelemetry import trace as _ot_trace
from opentelemetry.trace import (
    SpanContext, TraceFlags, NonRecordingSpan, set_span_in_context,
)

_TRACER = None
_OTEL_READY = False


def setup_otel() -> None:
    """Install an OTLP exporter once (best-effort). No-op if unconfigured."""
    global _TRACER, _OTEL_READY
    if _OTEL_READY:
        return
    _OTEL_READY = True
    cfg = _synapse_cfg()
    try:
        if cfg.otlp_endpoint:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            provider = TracerProvider()
            provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint)))
            _ot_trace.set_tracer_provider(provider)
        _TRACER = _ot_trace.get_tracer("coscientist.synapse")
    except Exception as exc:  # noqa: BLE001
        logger.warning("synapse: OTel setup failed: %s", exc)


def _parent_ctx(traceparent: Optional[str]):
    """W3C traceparent -> an OTel context whose current span is the remote parent."""
    if not traceparent:
        return None
    try:
        _v, trace_id, span_id, flags = traceparent.split("-")
        sc = SpanContext(
            trace_id=int(trace_id, 16), span_id=int(span_id, 16),
            is_remote=True, trace_flags=TraceFlags(int(flags, 16)),
        )
        return set_span_in_context(NonRecordingSpan(sc))
    except Exception:  # noqa: BLE001
        return None


def _start_run_span(context_id: str, agent_name: str, run_id: str):
    if _TRACER is None:
        return None
    ctx = _parent_ctx(traceparent_for(context_id))
    span = _TRACER.start_span(
        "invoke_agent", context=ctx,
        attributes={"gen_ai.operation.name": "invoke_agent",
                    "gen_ai.agent.name": agent_name, "run_id": run_id},
    )
    return span


def _end_run_span(handle) -> None:
    if handle is not None:
        try:
            handle.end()
        except Exception:  # noqa: BLE001
            pass


from google.adk.plugins.base_plugin import BasePlugin as _BasePlugin


class SynapseTracePlugin(_BasePlugin):
    """Hangs one invoke_agent span per run under the platform's traceparent."""

    def __init__(self) -> None:
        super().__init__(name="synapse_trace")
        setup_otel()
        self._spans = {}

    async def before_run_callback(self, *, invocation_context):
        s = invocation_context.session
        rid = run_id_for(s.id) or f"{s.app_name}__{s.id}"
        self._spans[invocation_context.invocation_id] = _start_run_span(
            s.id, invocation_context.agent.name, rid)
        return None

    async def after_run_callback(self, *, invocation_context):
        _end_run_span(self._spans.pop(invocation_context.invocation_id, None))
        return None
```

In `a2a/server.py`, after the checkpoint-plugin block (line 134), add:

```python
    if get_settings().synapse.enabled:
        from CoScientist.checkpoints.synapse import SynapseTracePlugin
        plugins.insert(0, SynapseTracePlugin())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -k otel -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CoScientist/checkpoints/synapse.py CoScientist/a2a/server.py tests/test_synapse_bridge.py
git commit -m "feat(synapse): minimal OTel — invoke_agent span stitched under platform traceparent"
```

---

### Task 7: Mock Synapse platform

**Files:**
- Create: `scripts/mock_synapse.py`
- Test: covered by the e2e in Task 8 (this is a harness, not production code).

**Interfaces:**
- Produces a runnable FastAPI app + helpers:
  - `POST /points` — records incoming "snapshot ready" callbacks into an in-memory list.
  - `GET /points` — returns the recorded points (for assertions).
  - `issue_run() -> tuple[str, str]` — returns `(run_id, traceparent)` with a fresh W3C traceparent.
  - CLI `python scripts/mock_synapse.py --port 9100` runs the receiver.

- [ ] **Step 1: Write the mock (no separate unit test; exercised in Task 8)**

```python
# scripts/mock_synapse.py
"""Local stand-in for the Synapse platform — exercises the v1 adapter contract.

Receives "snapshot ready" callbacks (POST /points), and provides helpers to
issue a platform run_id + W3C traceparent. Real Synapse replaces this later.
"""
from __future__ import annotations

import argparse
import secrets

from fastapi import FastAPI, Request

_POINTS: list[dict] = []

app = FastAPI()


@app.post("/points")
async def receive_point(request: Request):
    _POINTS.append(await request.json())
    return {"ok": True}


@app.get("/points")
async def list_points():
    return {"points": _POINTS}


def issue_run() -> tuple[str, str]:
    run_id = f"run-{secrets.token_hex(3)}"
    trace_id = secrets.token_hex(16)
    span_id = secrets.token_hex(8)
    traceparent = f"00-{trace_id}-{span_id}-01"
    return run_id, traceparent


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9100)
    args = p.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
```

- [ ] **Step 2: Sanity-run the mock imports**

Run: `.venv/bin/python -c "import scripts.mock_synapse as m; print(m.issue_run())"`
Expected: prints a `('run-xxxxxx', '00-...-...-01')` tuple.

- [ ] **Step 3: Commit**

```bash
git add scripts/mock_synapse.py
git commit -m "feat(synapse): local mock platform (callback receiver + run issuer)"
```

---

### Task 8: End-to-end through the mock (the integration proof)

**Files:**
- Create: `tests/e2e_synapse_v1.py`
- Test: itself (runnable e2e, patterned on `tests/e2e_checkpoint_a2a.py`).

**Interfaces:**
- Consumes: everything above; the scripted-agent + subprocess-server harness from `tests/e2e_checkpoint_a2a.py`.

**Scenario:** mock issues run_id+traceparent → `POST /api/checkpoints/runs` on the adapter → A2A `message/send` (scripted agent, contextId = the registered ctx) → adapter auto-saves checkpoints stamped with the platform run_id, POSTs "snapshot ready" to the mock → assert the mock received a point whose `run_id` == the issued one and whose `snapshot_ref` resolves via `GET …/bundle` → `POST …/{point_id}/restore` (platform-driven) → new contextId; assert restored state.

- [ ] **Step 1: Write the e2e**

Copy the server/scripted-agent + client helpers from `tests/e2e_checkpoint_a2a.py` (scripted `ScriptedOrchestrator`, `make_a2a_app`, `a2a_send`, `wait_ready`, `spawn_server`, `kill`, `_http`). Set the adapter server env to enable the bridge:

```python
def spawn_server(ckpt_dir: str, mock_port: int) -> subprocess.Popen:
    env = {**os.environ,
           "CHECKPOINTS__ENABLED": "1", "CHECKPOINTS__DIR": ckpt_dir,
           "SYNAPSE__ENABLED": "1",
           "SYNAPSE__CALLBACK_URL": f"http://127.0.0.1:{mock_port}",
           "SYNAPSE__BUNDLE_BASE_URL": f"http://127.0.0.1:{PORT}",
           "A2A_DISABLE_OPIK": "1", "LOG_AGENT_EVENTS": "0"}
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve"],
                            cwd=REPO_ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
```

Orchestration body:

```python
def main() -> None:
    import uvicorn, threading
    from scripts.mock_synapse import app as mock_app, _POINTS, issue_run
    mock_port = 9100
    threading.Thread(target=lambda: uvicorn.run(mock_app, host="127.0.0.1",
                     port=mock_port, log_level="warning"), daemon=True).start()
    time.sleep(1.5)

    ckpt_dir = tempfile.mkdtemp(prefix="synapse_e2e_")
    server = spawn_server(ckpt_dir, mock_port); wait_ready(server)
    run_id, traceparent = issue_run()
    ctx = "syn-ctx-1"

    _http("POST", f"{BASE}/api/checkpoints/runs",
          {"context_id": ctx, "run_id": run_id, "traceparent": traceparent})
    a2a_send("start the literature phase for GSK3B", context_id=ctx)

    points = _http("GET", f"http://127.0.0.1:{mock_port}/points")["points"]
    ok("platform received a snapshot-ready callback", len(points) >= 1)
    ok("point carries the PLATFORM run_id", all(p["run_id"] == run_id for p in points),
       f"{[p['run_id'] for p in points]} vs {run_id}")
    ref = points[0]["snapshot_ref"]
    ok("snapshot_ref points at the adapter bundle URL",
       ref and ref.startswith(f"{BASE}/api/checkpoints/") and ref.endswith("/bundle"))

    listing = _http("GET", f"{BASE}/api/checkpoints?run_id={run_id}")["checkpoints"]
    ok("adapter lists only this run's points (single endpoint)",
       listing and all(c["run_id"] == run_id for c in listing))

    pid = points[0]["point_id"]
    restored = _http("POST", f"{BASE}/api/checkpoints/{pid}/restore", {})
    ok("platform-driven restore returns a new contextId", bool(restored.get("context_id")))
    kill(server)
    print("\n[e2e] ALL SYNAPSE CHECKS PASSED")
```

Add the `--serve` / `main()` dispatch and an `ok(name, cond, detail="")` helper identical to `e2e_checkpoint_a2a.py`.

- [ ] **Step 2: Run it**

Run: `cd <cp2 worktree> && PYTHONPATH=. A2A_DISABLE_OPIK=1 .venv/bin/python tests/e2e_synapse_v1.py`
Expected: `[e2e] ALL SYNAPSE CHECKS PASSED` with every `[PASS]`.

- [ ] **Step 3: Confirm the bare-restore e2e still passes**

Run: `PYTHONPATH=. A2A_DISABLE_OPIK=1 .venv/bin/python tests/e2e_checkpoint_a2a.py`
Expected: `[e2e] ALL CHECKS PASSED` (no regression).

- [ ] **Step 4: Run the unit suite**

Run: `.venv/bin/python -m pytest tests/test_synapse_bridge.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e_synapse_v1.py
git commit -m "test(synapse): e2e — platform run_id, snapshot-ready callback, single-endpoint list, platform-driven restore"
```

---

## Self-Review

**1. Spec coverage:**
- Single adapter endpoint → Task 8 asserts list filtered by the platform run_id (served by one agent). ✅
- (1) platform run_id → Tasks 2,3,5. ✅
- (2) snapshot-ready callback → Task 4. ✅
- (3) stable snapshot_ref → Tasks 2,3 (`…/bundle` URL). ✅
- (4) platform-initiated restore → Task 8 (mock drives `POST …/restore`; reuses existing endpoint). ✅
- (5) OTel minimal (traceparent + 3-name/attrs, OTLP export) → Task 6. ✅
- Mock platform → Task 7. ✅
- Off-by-default / no regression → gating in every task; Task 8 Step 3 guards `e2e_checkpoint_a2a.py`. ✅

**2. Placeholder scan:** none — every code step has concrete code.

**3. Type consistency:** `register_run/run_id_for/traceparent_for/snapshot_ref_for/notify_snapshot_saved/_start_run_span/_end_run_span/setup_otel` are named identically across Tasks 2/3/4/5/6/8; `RunRegisterRequest` fields (context_id, run_id, traceparent) match the mock's `issue_run` + the e2e POST body; callback body keys (point_id, run_id, time, label, snapshot_ref) match Task 4 impl and Task 8 assertions.

## Notes / deferred (from spec non-goals)
- `replay` (strict/fresh), a dedicated `/fork` endpoint, and full GenAI attribute coverage are **out of scope** (v1 doesn't require them).
- Adjacent gap (not this plan): A2A-mode `input-required` reaching the external endpoint (HITL handler not swapped on A2A servers).
