#!/usr/bin/env python3
"""Run the real molecular-design tool and its tests without booting the app.

Usage: .venv/bin/python scripts/microfluidics/test_molecular_design_offline.py
Only parent package initializers are bypassed in this standalone process;
RDKit, domain models, fixed-target logic and the async tool are real imports.
No LLM, A2A, CFD or equipment is called. Outputs go to /tmp by default.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
import time
import types


def run_async(coroutine):
    """Drive real async code with bounded selector waits in the test sandbox.

    The sandbox can miss the thread-pool socket wakeup (even to_thread(lambda:
    42) reproduces it). A short timer drains completion callbacks; no tool or
    executor is mocked. Close the loop directly after work finishes.
    """
    loop = asyncio.new_event_loop()
    task = loop.create_task(coroutine)
    deadline = time.monotonic() + 60
    try:
        while not task.done():
            if time.monotonic() >= deadline:
                task.cancel()
                loop.run_until_complete(asyncio.sleep(0))
                raise TimeoutError("Offline molecular design exceeded 60 seconds")
            loop.run_until_complete(asyncio.sleep(0.01))
        return task.result()
    finally:
        loop.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/molecular_design_phenolic_results.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # __init__.py eagerly starts the full application, including remote clients.
    # Namespace shells affect only this process; actual domain modules are loaded
    # from the repository unchanged, with their real dependencies.
    for name, path in (("CoScientist", root / "CoScientist"),
                       ("CoScientist.hitl", root / "CoScientist/hitl")):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        sys.modules[name] = package

    print("Loading molecular-design module...", flush=True)
    from CoScientist.microfluidics.molecular_design import molecular_design
    import rdkit
    import pytest

    case = json.loads((root / "tests/unit/microfluidics/fixtures/molecular_design/phenolic_antioxidant.json").read_text())

    async def calculate():
        results = {"rdkit_version": rdkit.__version__, "case": case, "runs": {}}
        for label, generate, limit in (("screening", False, 30), ("generation", True, 30), ("default_limit", True, 10)):
            ctx = types.SimpleNamespace(state={key: case[key] for key in ("structured_tz", "literature_analysis")})
            request = {**case["requirements"], "generate": generate, "max_candidates": limit}
            started = time.perf_counter()
            print("Running", label, flush=True)
            result = await molecular_design(json.dumps(request, ensure_ascii=False), ctx)
            results["runs"][label] = {"seconds": time.perf_counter() - started, "request": request, "result": result}
            print(label, {k: result.get(k) for k in ("status", "eligible_count", "enumerated")},
                  "returned:", len(result["candidates"]), flush=True)
        return results

    results = run_async(calculate())
    exit_code = pytest.main([str(root / "tests/unit/microfluidics/test_molecular_design_phenolic.py"), "-q"])
    results["pytest_exit_code"] = int(exit_code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print("Results:", args.output)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
