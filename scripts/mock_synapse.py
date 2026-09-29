"""Local stand-in for the Synapse platform — exercises the v1 adapter contract.

Receives "snapshot ready" callbacks (POST /points), and provides helpers to
issue a platform run_id + W3C traceparent. Real Synapse replaces this later.

Run standalone:  python scripts/mock_synapse.py --port 9100
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
    """Return (run_id, traceparent) with a fresh W3C traceparent."""
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
