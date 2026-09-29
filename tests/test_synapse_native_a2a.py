"""Regression coverage for native A2A/ADK trace correlation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
ROOT_SPAN_ID = "b7ad6b7169203331"
RUN_ID = "run-native"


def _run_native_probe() -> dict:
    probe = Path(__file__).with_name("_synapse_native_trace_probe.py")
    env = {
        **os.environ,
        "A2A_DISABLE_OPIK": "1",
        "LOG_AGENT_EVENTS": "0",
        "LLM__MAIN_MODEL": "test-model",
    }
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = next(
        line.removeprefix("TRACE_PROBE=")
        for line in completed.stdout.splitlines()
        if line.startswith("TRACE_PROBE=")
    )
    return json.loads(marker)


def test_registered_http_run_correlates_native_adk_spans():
    result = _run_native_probe()

    assert result["status"] == 200
    assert "error" not in result["body"]
    native = [
        span
        for span in result["spans"]
        if span["name"] in {"invocation", "invoke_agent TinyAgent", "probe ctx-native"}
    ]
    assert {span["name"] for span in native} == {
        "invocation",
        "invoke_agent TinyAgent",
        "probe ctx-native",
    }
    assert {span["trace_id"] for span in native} == {TRACE_ID}
    assert {span["attributes"].get("run_id") for span in native} == {RUN_ID}

    handler = next(
        span
        for span in result["spans"]
        if span["name"].endswith("JSONRPCHandler.on_message_send")
    )
    assert handler["parent_span_id"] == ROOT_SPAN_ID
    assert handler["trace_id"] == TRACE_ID
    assert handler["attributes"]["run_id"] == RUN_ID


def test_trace_scope_api_requires_cleanup_for_streams_and_concurrency():
    os.environ.setdefault("LLM__MAIN_MODEL", "test-model")
    from CoScientist.checkpoints import trace_context

    assert trace_context.active_run_id() is None
    assert trace_context.scope_for_request is not None
    assert trace_context.RunIdSpanProcessor is not None
