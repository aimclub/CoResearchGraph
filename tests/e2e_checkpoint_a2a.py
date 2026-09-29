"""E2E: checkpoint save/restore of a run served over A2A.

Scenario (run: ``python tests/e2e_checkpoint_a2a.py``):

1. start an A2A server (subprocess) wrapping a SCRIPTED agent — no LLM calls,
   fully deterministic — through the real ``make_a2a_app`` factory with
   ``CHECKPOINTS__ENABLED=1``;
2. send message #1 over A2A: the agent commits ``search_results`` via a
   state_delta (→ T1 checkpoint fires) and then writes ``deployed_mcps`` by
   DIRECT mutation, exactly like ``WebToolsDeployerAgent`` does (this key
   exists in no state_delta — the classic checkpoint-killer);
3. verify checkpoint bundles appeared via ``GET /api/checkpoints``;
4. KILL the server (the InMemory session dies with it);
5. start a FRESH server process, restore the T5 checkpoint via
   ``POST /api/checkpoints/{id}/restore`` → new contextId;
6. send "continue" over plain A2A with that contextId — the agent must see the
   pre-crash state: ``search_results`` (delta-committed), ``deployed_mcps``
   (direct mutation → proves the authoritative state blob works) and the full
   event history;
7. differential check: restoring the T1 checkpoint instead must yield a state
   WITHOUT ``deployed_mcps`` (T1 was taken before the mutation) — proves
   checkpoints capture the point in time, not just the final state.

The scripted agent lives in this file; ``--serve`` runs the server role.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # script-by-path puts tests/ on sys.path, not the repo root
    sys.path.insert(0, str(REPO_ROOT))
PORT = int(os.getenv("E2E_CHECKPOINT_PORT", "8123"))
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
        """Deterministic stand-in for the orchestrator: phase 1 produces a
        'literature' result (delta) + a direct-mutation key; the 'continue'
        turn echoes what it sees in session state/history."""

        async def _run_async_impl(self, ctx: InvocationContext):
            text = ""
            if ctx.user_content and ctx.user_content.parts:
                text = "".join(p.text or "" for p in ctx.user_content.parts)

            if "continue" in text.lower():
                state = ctx.session.state
                report = (
                    f"CONTINUED|search_results={state.get('search_results')}"
                    f"|deployed_mcps={state.get('deployed_mcps')}"
                    f"|events={len(ctx.session.events)}"
                )
                yield Event(
                    author=self.name, invocation_id=ctx.invocation_id,
                    content=types.Content(role="model", parts=[types.Part(text=report)]),
                )
                return

            # module boundary: output lands as a state_delta (like output_key)
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=types.Content(role="model", parts=[types.Part(text="literature review done")]),
                actions=EventActions(state_delta={"search_results": "GSK3B: 42 candidate molecules"}),
            )
            # DIRECT mutation, bypassing the delta protocol — the pattern of
            # custom_agents.py:34 that only the authoritative state blob keeps
            ctx.session.state["deployed_mcps"] = ["http://fake-mcp:1"]
            yield Event(
                author=self.name, invocation_id=ctx.invocation_id,
                content=types.Content(role="model", parts=[types.Part(text="phase 1 complete")]),
            )

    import uvicorn
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    from CoScientist.a2a.server import make_a2a_app

    card = AgentCard(
        name="ScriptedOrchestrator",
        description="e2e checkpoint test agent",
        url=f"{BASE}/",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[AgentSkill(id="e2e", name="e2e", description="scripted run", tags=["test"])],
    )
    app = make_a2a_app(ScriptedOrchestrator(name="ScriptedOrchestrator"), card, APP_NAME)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


# ─────────────────────────── client helpers ─────────────────────────────────

def _http(method: str, url: str, body: dict | None = None, timeout: float = 60.0) -> dict:
    import urllib.request

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {TOKEN}"}
                                             if url.startswith(f"{BASE}/api/checkpoints") else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def a2a_send(text: str, context_id: str | None) -> dict:
    """Plain A2A JSON-RPC message/send; returns the JSON-RPC result."""
    message: dict = {
        "kind": "message",
        "role": "user",
        "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": text}],
    }
    if context_id:
        message["contextId"] = context_id
    payload = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": message},
    }
    out = _http("POST", f"{BASE}/", payload, timeout=120.0)
    if "error" in out:
        raise AssertionError(f"A2A error: {out['error']}")
    return out["result"]


def result_text(result: dict) -> str:
    """Collect all text parts from a Task (artifacts + status) or Message."""
    chunks: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("kind") == "text" and obj.get("text"):
                chunks.append(obj["text"])
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(result)
    return "\n".join(chunks)


def wait_ready(proc: subprocess.Popen, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server died on startup (exit {proc.returncode})")
        try:
            _http("GET", f"{BASE}/.well-known/agent-card.json", timeout=3.0)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0)
    raise AssertionError(f"server not ready in {timeout}s: {last_err}")


def spawn_server(ckpt_dir: str) -> subprocess.Popen:
    env = {
        **os.environ,
        "CHECKPOINTS__ENABLED": "1", "CHECKPOINTS__API_TOKEN": TOKEN,
        "CHECKPOINTS__DIR": ckpt_dir,
        "A2A_DISABLE_OPIK": "1",
        "LOG_AGENT_EVENTS": "0",
    }
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--serve"],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )


def kill(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


# ─────────────────────────── orchestration ──────────────────────────────────

def main() -> None:
    ckpt_dir = tempfile.mkdtemp(prefix="coscientist_e2e_ckpt_")
    print(f"[e2e] checkpoint dir: {ckpt_dir}")
    checks: list[str] = []

    def ok(name: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        checks.append(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        print(checks[-1])
        if not cond:
            raise AssertionError(name)

    server = spawn_server(ckpt_dir)
    try:
        wait_ready(server)
        print("[e2e] server #1 ready; sending phase-1 message")
        r1 = a2a_send("start the literature phase for GSK3B", context_id="e2e-ctx-1")
        ok("phase-1 run completed over A2A", "phase 1 complete" in result_text(r1))

        listing = _http("GET", f"{BASE}/api/checkpoints")["checkpoints"]
        labels = [c["label"] for c in listing]
        ok("T1 checkpoint auto-saved (state_delta trigger)",
           "T1_after_literature_review" in labels, f"labels={labels}")
        ok("T5 checkpoint auto-saved (turn boundary)",
           "T5_invocation_end" in labels, f"labels={labels}")
        t5 = next(c for c in listing if c["label"] == "T5_invocation_end")
        t1 = next(c for c in listing if c["label"] == "T1_after_literature_review")

        print("[e2e] killing server #1 (session state dies with the process)")
        kill(server)
    except BaseException:
        kill(server)
        raise

    server = spawn_server(ckpt_dir)
    try:
        wait_ready(server)
        print("[e2e] server #2 ready (fresh InMemory sessions)")

        listing2 = _http("GET", f"{BASE}/api/checkpoints")["checkpoints"]
        ok("bundles survive the process restart",
           {c["checkpoint_id"] for c in listing2} >= {t5["checkpoint_id"], t1["checkpoint_id"]})

        restored = _http("POST", f"{BASE}/api/checkpoints/{t5['checkpoint_id']}/restore", {})
        ctx5 = restored["context_id"]
        print(f"[e2e] T5 restored -> contextId={ctx5} "
              f"({restored['event_count']} events, warnings={restored['warnings']})")

        r2 = a2a_send("continue", context_id=ctx5)
        text2 = result_text(r2)
        ok("run continues over plain A2A message/send after restore", "CONTINUED" in text2, text2)
        ok("delta-committed state survived (search_results)",
           "search_results=GSK3B: 42 candidate molecules" in text2, text2)
        ok("DIRECT-mutation state survived via authoritative blob (deployed_mcps)",
           "deployed_mcps=['http://fake-mcp:1']" in text2, text2)

        restored1 = _http("POST", f"{BASE}/api/checkpoints/{t1['checkpoint_id']}/restore", {})
        r3 = a2a_send("continue", context_id=restored1["context_id"])
        text3 = result_text(r3)
        ok("T1 restore is point-in-time: deployed_mcps NOT yet set at T1",
           "deployed_mcps=None" in text3, text3)
        ok("T1 restore still carries its own module result",
           "search_results=GSK3B: 42 candidate molecules" in text3, text3)
    finally:
        kill(server)

    print("\n[e2e] ALL CHECKS PASSED")
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
