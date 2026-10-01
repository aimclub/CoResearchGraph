"""The web status indicator counts the stages of a linear pipeline.

The reducer is plain browser JavaScript; the scenario lives in
tests/unit/js/status_indicator_stages.js and runs under node with a stubbed
DOM. Skipped where node is not installed."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
INDICATOR = ROOT / "CoScientist" / "web" / "static" / "status_indicator.js"
SCENARIO = Path(__file__).parent / "js" / "status_indicator_stages.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_indicator_leads_with_the_stage_of_a_linear_pipeline():
    run = subprocess.run(
        ["node", str(SCENARIO), str(INDICATOR)],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr or run.stdout
    assert run.stdout.strip() == "ok"
