"""Regression proof for RemoteA2aAgent across a real HTTP socket boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TRACE_ID = "5af7651916cd43dd8448eb211c80319c"
RUN_ID = "run-distributed"


def test_remote_agent_keeps_canonical_trace_without_sharing_registry():
    probe = Path(__file__).with_name("_synapse_remote_trace_probe.py")
    completed = subprocess.run(
        [sys.executable, str(probe)],
        cwd=Path(__file__).resolve().parent.parent,
        env={
            **os.environ,
            "A2A_DISABLE_OPIK": "1",
            "LOG_AGENT_EVENTS": "0",
            "LLM__MAIN_MODEL": "test-model",
        },
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    marker = next(
        line.removeprefix("REMOTE_TRACE_PROBE=")
        for line in completed.stdout.splitlines()
        if line.startswith("REMOTE_TRACE_PROBE=")
    )
    result = json.loads(marker)

    assert result["event_count"] > 0
    assert result["science_state"] == {"hypothesis": {"target": "GSK3B"}}
    assert result["remote_contexts"]
    assert result["remote_registered"] == [None] * len(result["remote_contexts"])
    remote = [
        span for span in result["spans"] if span["name"].startswith("remote-probe ")
    ]
    assert len(remote) == 1
    assert remote[0]["trace_id"] == TRACE_ID
    assert remote[0]["attributes"]["run_id"] == RUN_ID
