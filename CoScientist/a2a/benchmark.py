"""Smoke-test / performance client for the CoScientist A2A agents.

Two modes:

* --stream (default): uses the A2A `message/stream` SSE endpoint and prints
  EVERY intermediate event as it happens — status transitions, agent thoughts,
  tool calls/responses, sub-agent steps and artifacts. Use this to see what is
  happening inside the system.

* --no-stream: single `message/send` call, prints only the final answer plus a
  latency summary (good for rough perf numbers over -n repetitions).

Examples (run from /app):
    # Watch everything happen live:
    python -m CoScientist.a2a.benchmark --agent hypotheses \
        --text "Generate a hypothesis about why sleep aids memory."

    # Orchestrator (sub-agents must also be running):
    python -m CoScientist.a2a.benchmark --agent orchestrator \
        --text "Research recent advances in battery chemistry."

    # Latency over 5 runs, no event trace:
    python -m CoScientist.a2a.benchmark --agent research \
        --text "What is CRISPR?" -n 5 --no-stream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid

import httpx

from CoScientist.a2a.config import AGENT_PORTS

A2A_HOST = "127.0.0.1"

# Print a "still working" line if no stream event arrives for this many seconds.
_HEARTBEAT_SECS = 15.0

# ── ANSI colors ──────────────────────────────────────────────────────────────
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"


def _endpoint(agent: str) -> str:
    if agent not in AGENT_PORTS:
        raise SystemExit(
            f"Unknown agent '{agent}'. Choose from: {', '.join(AGENT_PORTS)}"
        )
    return f"http://{A2A_HOST}:{AGENT_PORTS[agent]}/"


def _build_message_params(text: str) -> dict:
    return {
        "message": {
            "messageId": str(uuid.uuid4()),
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
        }
    }


def _rpc(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params}


# ── Event rendering ──────────────────────────────────────────────────────────


def _render_parts(parts: list[dict], indent: str = "    ") -> None:
    """Print message/artifact parts: thoughts, text, tool calls, tool results."""
    for part in parts:
        kind = part.get("kind")
        meta = part.get("metadata") or {}

        if kind == "text":
            text = part.get("text", "")
            if meta.get("adk_thought"):
                print(f"{indent}{DIM}{MAGENTA}💭 thought:{RESET} {DIM}{text}{RESET}")
            else:
                print(f"{indent}{GREEN}🗎 text:{RESET} {text}")

        elif kind == "data":
            # ADK encodes function calls/responses as data parts, tagged via
            # metadata "adk_type" (function_call | function_response).
            data = part.get("data", {})
            adk_type = meta.get("adk_type")
            blob = json.dumps(data, ensure_ascii=False)
            if len(blob) > 600:
                blob = blob[:600] + " …"
            if adk_type == "function_call":
                label = f"🔧 tool call: {BOLD}{data.get('name')}{RESET}"
            elif adk_type == "function_response":
                label = f"📥 tool result: {BOLD}{data.get('name')}{RESET}"
            else:
                label = "🔧 data"
            print(f"{indent}{YELLOW}{label}{RESET} {DIM}{blob}{RESET}")

        elif kind == "file":
            f = part.get("file", {})
            print(f"{indent}{YELLOW}📎 file:{RESET} {f.get('name', '(unnamed)')}")

        else:
            print(f"{indent}{DIM}({kind}) {json.dumps(part, ensure_ascii=False)[:300]}{RESET}")


def _handle_stream_event(result: dict, t0: float) -> None:
    """Render one streamed A2A event (Task / Message / status / artifact update)."""
    ts = f"{DIM}[{time.perf_counter() - t0:6.2f}s]{RESET}"
    kind = result.get("kind")

    if kind == "task":
        state = result.get("status", {}).get("state", "?")
        print(f"{ts} {CYAN}▶ task created{RESET} (state={state})")

    elif kind == "status-update":
        status = result.get("status", {})
        state = status.get("state", "?")
        final = result.get("final", False)
        marker = "✔" if state == "completed" else ("✖" if state == "failed" else "…")
        color = GREEN if state == "completed" else (RED if state == "failed" else CYAN)
        print(f"{ts} {color}{marker} status: {state}{RESET}{' (final)' if final else ''}")
        msg = status.get("message")
        if msg and msg.get("parts"):
            _render_parts(msg["parts"])

    elif kind == "artifact-update":
        artifact = result.get("artifact", {})
        name = artifact.get("name") or artifact.get("artifactId", "")[:8]
        print(f"{ts} {GREEN}📦 artifact:{RESET} {name}")
        _render_parts(artifact.get("parts", []))

    elif kind == "message":
        role = result.get("role", "?")
        author = (result.get("metadata") or {}).get("adk_agent") or role
        print(f"{ts} {BOLD}💬 message{RESET} ({author})")
        _render_parts(result.get("parts", []))

    else:
        print(f"{ts} {DIM}event: {json.dumps(result, ensure_ascii=False)[:300]}{RESET}")


async def _stream_call(client: httpx.AsyncClient, url: str, text: str, timeout: float) -> float:
    """Open an SSE stream and render every event. Returns total elapsed seconds."""
    payload = _rpc("message/stream", _build_message_params(text))
    t0 = time.perf_counter()
    n_events = 0
    async with client.stream(
        "POST", url, json=payload,
        headers={"Accept": "text/event-stream"}, timeout=timeout,
    ) as resp:
        resp.raise_for_status()
        # A background task reads SSE lines into a queue; the main loop pulls
        # with a short timeout so it can emit a heartbeat when the stream goes
        # quiet — over A2A a delegated sub-agent's internal steps don't reach this
        # stream, so a long CoderAgent/Research call would otherwise look frozen.
        # (We can't wrap aiter_lines().__anext__() in wait_for directly: a timeout
        # cancels the pending read and corrupts the httpx line iterator.)
        # Silence is measured since the last RENDERED event, so SSE keep-alive
        # pings (~every 15s) don't reset the timer.
        queue: asyncio.Queue = asyncio.Queue()

        async def _reader() -> None:
            try:
                async for ln in resp.aiter_lines():
                    await queue.put(("line", ln))
            except Exception as exc:  # noqa: BLE001
                await queue.put(("err", exc))
            finally:
                await queue.put(("end", None))

        reader = asyncio.create_task(_reader())
        last_event = time.perf_counter()
        last_beat = last_event
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=4.0)
                except asyncio.TimeoutError:
                    kind, payload = "tick", None
                now = time.perf_counter()
                if now - last_event >= _HEARTBEAT_SECS and now - last_beat >= _HEARTBEAT_SECS:
                    print(
                        f"{DIM}… still working ({now - t0:.0f}s; a delegated agent is busy — "
                        f"watch the run_all console for its live steps){RESET}",
                        flush=True,
                    )
                    last_beat = now
                if kind in ("tick",):
                    continue
                if kind == "end":
                    break
                if kind == "err":
                    print(f"{RED}✖ stream error: {payload}{RESET}")
                    break
                line = payload.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                try:
                    envelope = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in envelope:
                    print(f"{RED}✖ error: {json.dumps(envelope['error'])}{RESET}")
                    continue
                result = envelope.get("result")
                if result is None:
                    continue
                n_events += 1
                last_event = now
                _handle_stream_event(result, t0)
        finally:
            reader.cancel()
    elapsed = time.perf_counter() - t0
    print(f"\n{DIM}── {n_events} events in {elapsed:.2f}s ──{RESET}")
    return elapsed


# ── Non-streaming (latency) mode ─────────────────────────────────────────────


def _extract_answer(result: dict) -> str:
    texts: list[str] = []
    for artifact in result.get("artifacts", []):
        for part in artifact.get("parts", []):
            if part.get("kind") == "text" and not (part.get("metadata") or {}).get("adk_thought"):
                texts.append(part["text"])
    if texts:
        return "\n".join(texts)
    for msg in reversed(result.get("history", [])):
        if msg.get("role") == "agent":
            for part in msg.get("parts", []):
                if part.get("kind") == "text":
                    return part["text"]
    return "(no text answer found)"


async def _send_call(client: httpx.AsyncClient, url: str, text: str, timeout: float):
    started = time.perf_counter()
    resp = await client.post(url, json=_rpc("message/send", _build_message_params(text)), timeout=timeout)
    elapsed = time.perf_counter() - started
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        return elapsed, None, body["error"]
    result = body.get("result", {})
    return elapsed, _extract_answer(result), result.get("status", {}).get("state", "unknown")


# ── Main ─────────────────────────────────────────────────────────────────────


async def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="A2A smoke-test / perf client")
    parser.add_argument("--agent", default="hypotheses",
                        help=f"Target agent ({', '.join(AGENT_PORTS)})")
    parser.add_argument("--text", required=True, help="User message to send")
    parser.add_argument("-n", "--repeat", type=int, default=1, help="Repetitions")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout (s)")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Disable event streaming; just measure latency")
    parser.set_defaults(stream=True)
    args = parser.parse_args(argv)

    url = _endpoint(args.agent)
    mode = "stream (live events)" if args.stream else "send (latency only)"
    print(f"→ Target: {args.agent} @ {url}")
    print(f"→ Prompt: {args.text!r}")
    print(f"→ Mode: {mode} | Repetitions: {args.repeat}\n")

    latencies: list[float] = []
    async with httpx.AsyncClient() as client:
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"{BOLD}═══ run {i + 1}/{args.repeat} ═══{RESET}")
            try:
                if args.stream:
                    elapsed = await _stream_call(client, url, args.text, args.timeout)
                else:
                    elapsed, answer, state = await _send_call(client, url, args.text, args.timeout)
                    print(f"[{i + 1}/{args.repeat}] {elapsed:6.2f}s  state={state}")
                    print(f"    {answer if answer else 'ERROR: ' + str(state)}\n")
            except httpx.ConnectError:
                raise SystemExit(
                    f"✗ Could not connect to {url}. Is the server running?\n"
                    f"  Start it with: python -m CoScientist.a2a.serve {args.agent}"
                )
            latencies.append(elapsed)

    if len(latencies) > 1:
        print("── Latency summary ──")
        print(f"  count : {len(latencies)}")
        print(f"  min   : {min(latencies):.2f}s")
        print(f"  mean  : {statistics.mean(latencies):.2f}s")
        print(f"  max   : {max(latencies):.2f}s")
        print(f"  stdev : {statistics.stdev(latencies):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
