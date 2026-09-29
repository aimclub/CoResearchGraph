"""Launch a SEPARATE CoScientist instance for the microfluidics case.

Builds the system from CoScientist/microfluidics/microfluidics.yaml (ТЗ agent +
planner + orchestrator + literature analysis only) and serves the usual web
UI on its own port, so it can run side by side with the default CoScientist.

Usage (from the repo root):
    python scripts/microfluidics/run_microfluidics_web.py            # HITL on: human reviews
                                                       # the ТЗ table, the
                                                       # queries and the plan
    python scripts/microfluidics/run_microfluidics_web.py --no-hitl  # headless (testing
                                                       # without a human)

Environment overrides:
    COSCIENTIST_WEB_PORT — port for this instance (default 8010)
    COSCIENTIST_WEB_HOST — bind address (default 127.0.0.1). A deploy host sets
                           this to serve the instance to the team, because no
                           reverse proxy is in front of it.
    HITL__ENABLED        — same switch as --no-hitl (false disables all HITL)
"""
import os
import sys
from pathlib import Path

# Must be set BEFORE any CoScientist import: the system profile and the HITL
# wiring are resolved once per process when the agent system is first built.
os.environ.setdefault("COSCIENTIST_CONFIG", "microfluidics")
os.environ.setdefault("COSCIENTIST_WEB_PORT", "8010")
if "--no-hitl" in sys.argv:
    os.environ["HITL__ENABLED"] = "false"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The same launcher as `python -m CoScientist web`: its server guarantees that
# Ctrl+C ends the process (forced after a grace period, or on a second Ctrl+C)
# even when a tool thread is blocked — a plain uvicorn.run() waits for it.
from CoScientist.cli import run_web  # noqa: E402


if __name__ == "__main__":
    # The default keeps a local run on the loopback address. Only a deploy host
    # sets COSCIENTIST_WEB_HOST, so this change does not move a developer run
    # onto the network.
    run_web(
        host=os.environ.get("COSCIENTIST_WEB_HOST", "127.0.0.1"),
        port=int(os.environ["COSCIENTIST_WEB_PORT"]),
    )
