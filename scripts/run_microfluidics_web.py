"""Launch a SEPARATE CoScientist instance for the microfluidics case.

Builds the system from CoScientist/agents/microfluidics.yaml (ТЗ agent +
planner + orchestrator + literature analysis only) and serves the usual web
UI on its own port, so it can run side by side with the default CoScientist.

Usage (from the repo root):
    python scripts/run_microfluidics_web.py            # HITL on: human reviews
                                                       # the ТЗ table, the
                                                       # queries and the plan
    python scripts/run_microfluidics_web.py --no-hitl  # headless (testing
                                                       # without a human)

Environment overrides:
    COSCIENTIST_WEB_PORT — port for this instance (default 8010)
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn  # noqa: E402


if __name__ == "__main__":
    uvicorn.run(
        "CoScientist.web.server:app",
        host="127.0.0.1",
        port=int(os.environ["COSCIENTIST_WEB_PORT"]),
        reload=False,
        log_level="info",
    )
