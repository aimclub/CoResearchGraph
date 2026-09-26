"""E2E: the Synapse v1 adapter contract, end-to-end against the local mock.

Scenario (run: ``python tests/e2e_synapse_v1.py``):

1. start the mock Synapse platform in-process (callback receiver);
2. start an A2A adapter server (subprocess) wrapping a SCRIPTED agent — no LLM,
   deterministic — with CHECKPOINTS__ENABLED + SYNAPSE__ENABLED;
3. the mock issues a platform run_id + traceparent and registers them for a
   contextId via ``POST /api/checkpoints/runs``;
4. send message #1 over A2A with that contextId → the agent commits
   ``search_results`` (T1 fires); the adapter stamps the PLATFORM run_id on the
   point and POSTs "snapshot ready" to the mock;
5. assert: the mock received a point whose run_id == the issued run_id and whose
   snapshot_ref resolves via ``GET …/bundle``; the adapter lists only this run's
   points; a platform-driven ``POST …/restore`` returns a fresh contextId.

The scripted agent lives in this file; ``--serve`` runs the server role.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PORT = int(os.getenv("E2E_SYNAPSE_PORT", "8138"))
MOCK_PORT = int(os.getenv("E2E_SYNAPSE_MOCK_PORT", "9138"))
BASE = f"http://127.0.0.1:{PORT}"
APP_NAME = "orchestrator"
TOKEN = "test-only-platform-admin-credential"


# ─────────────────────────── server role ────────────────────────────────────

def serve() -> None:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.events.event import Event
    from google.adk.events.event_actions import EventActions
    from google.genai import types

    class ScriptedOrchestrator(BaseAgent):
        async def _run_async_impl(self, ctx: InvocationContext):
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=types.Content(role="model",
                                      parts=[types.Part(text="literature review done")]),
                actions=EventActions(state_delta={"search_results": "GSK3B: 42 candidate molecules"}),
            )
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=types.Content(role="model", parts=[types.Part(text="phase 1 complete")]),
            )

    import uvicorn
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    from CoScientist.a2a.server import make_a2a_app

    card = AgentCard(
        name="ScriptedOrchestrator", description="e2e synapse test agent",
        url=f"{BASE}/", version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"], defaultOutputModes=["text/plain"],
        skills=[AgentSkill(id="e2e", name="e2e", description="scripted run", tags=["test"])],
    )
    app = make_a2a_app(ScriptedOrchestrator(name="ScriptedOrchestrator"), card, APP_NAME)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


# ─────────────────────────── client helpers ─────────────────────────────────

def _http(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    import urllib.request
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {TOKEN}"}
                                             if url.startswith(f"{BASE}/api/checkpoints") else {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def a2a_send(text: str, context_id: str) -> dict:
    message = {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
               "contextId": context_id, "parts": [{"kind": "text", "text": text}]}
    payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex,
               "method": "message/send", "params": {"message": message}}
    out = _http("POST", f"{BASE}/", payload, timeout=120.0)
    if "error" in out:
        raise AssertionError(f"A2A error: {out['error']}")
    return out["result"]


def wait_ready(url: str, proc: subprocess.Popen | None = None, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f"server died on startup (exit {proc.returncode})")
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=3.0)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.0)
    raise AssertionError(f"not ready in {timeout}s: {last}")


def spawn_server(ckpt_dir: str) -> subprocess.Popen:
    env = {**os.environ,
           "CHECKPOINTS__ENABLED": "1", "CHECKPOINTS__API_TOKEN": TOKEN, "CHECKPOINTS__DIR": ckpt_dir,
           "SYNAPSE__ENABLED": "1",
           "SYNAPSE__CALLBACK_URL": f"http://127.0.0.1:{MOCK_PORT}",
           "SYNAPSE__BUNDLE_BASE_URL": BASE,
           "A2A_DISABLE_OPIK": "1", "LOG_AGENT_EVENTS": "0"}
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve"],
                            cwd=REPO_ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def kill(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill(); proc.wait(timeout=15)


# ─────────────────────────── orchestration ──────────────────────────────────

def main() -> None:
    import uvicorn
    from scripts.mock_synapse import app as mock_app, issue_run

    threading.Thread(
        target=lambda: uvicorn.run(mock_app, host="127.0.0.1", port=MOCK_PORT,
                                   log_level="warning"),
        daemon=True).start()
    wait_ready(f"http://127.0.0.1:{MOCK_PORT}/points")

    checks: list[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        checks.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        print(checks[-1])
        if not cond:
            raise AssertionError(name)

    ckpt_dir = tempfile.mkdtemp(prefix="synapse_e2e_")
    print(f"[e2e] checkpoint dir: {ckpt_dir}")
    server = spawn_server(ckpt_dir)
    try:
        wait_ready(f"{BASE}/.well-known/agent-card.json", server)
        run_id, traceparent = issue_run()
        ctx = "syn-ctx-1"
        print(f"[e2e] platform issued run_id={run_id}")

        _http("POST", f"{BASE}/api/checkpoints/runs",
              {"context_id": ctx, "run_id": run_id, "traceparent": traceparent})
        a2a_send("start the literature phase for GSK3B", context_id=ctx)

        deadline = time.monotonic() + 10
        while True:
            points = _http("GET", f"http://127.0.0.1:{MOCK_PORT}/points")["points"]
            if any(p["label"] == "T1_after_literature_review" for p in points):
                break
            assert time.monotonic() < deadline, "snapshot callback timed out"
            time.sleep(0.05)
        ok("platform received a snapshot-ready callback", len(points) >= 1,
           f"{len(points)} points")
        ok("point carries the PLATFORM run_id",
           all(p["run_id"] == run_id for p in points),
           f"{[p['run_id'] for p in points]} vs {run_id}")
        ref = points[0]["snapshot_ref"]
        ok("snapshot_ref points at the adapter bundle URL",
           bool(ref) and ref.startswith(f"{BASE}/api/checkpoints/") and ref.endswith("/bundle"),
           str(ref))
        # the ref actually resolves to a downloadable bundle
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(
                ref, headers={"Authorization": f"Bearer {TOKEN}"}), timeout=10) as r:
            ok("snapshot_ref resolves (bundle downloadable)", r.status == 200)

        listing = _http("GET", f"{BASE}/api/checkpoints?run_id={run_id}")["checkpoints"]
        ok("adapter lists only this run's points (single endpoint)",
           bool(listing) and all(c["run_id"] == run_id for c in listing),
           f"{[c['run_id'] for c in listing]}")

        pid = next(p["point_id"] for p in points
                   if p["label"] == "T1_after_literature_review")
        restored = _http("POST", f"{BASE}/api/checkpoints/{pid}/restore", {})
        ok("platform-driven restore returns a new contextId",
           bool(restored.get("context_id")), str(restored.get("context_id")))
    finally:
        kill(server)

    print("\n[e2e] ALL SYNAPSE CHECKS PASSED")
    for line in checks:
        print("  " + line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="run the A2A server role")
    args = parser.parse_args()
    if args.serve:
        serve()
    else:
        main()
