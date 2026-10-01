"""The web plan tracker lists the stages of a linear pipeline, with the
roadmap's tasks nested under the stage that executes them.

The tracker is plain browser JavaScript; the scenario lives in
tests/unit/js/plan_tracker_stages.js and runs under node with a stubbed DOM.
Skipped where node is not installed."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TRACKER = ROOT / "CoScientist" / "web" / "static" / "js" / "plan_tracker.js"
SCENARIO = Path(__file__).parent / "js" / "plan_tracker_stages.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_tracker_lists_the_stages_of_a_linear_pipeline():
    run = subprocess.run(
        ["node", str(SCENARIO), str(TRACKER)],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr or run.stdout
    assert run.stdout.strip() == "ok"
