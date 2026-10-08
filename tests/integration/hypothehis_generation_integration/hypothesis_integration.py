"""
Isolated integration test: HypothesisGenerator + MooseChemMCPTool.

Uses the hypothesis_subsystem pipeline (retrieve_validation_tools →
MooseChemMCPTool) as a black box via build_hypothesis_subsystem().

Verifies that the subsystem produces structured Hypothesis objects with
validation_tool_matching annotations.

Reads test_cases.json, runs all cases, saves structured output.

Usage:
    .venv\Scripts\python.exe tests/integration/hypothehis_generation_integration/hypothesis_integration.py [--cases test_cases.json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Opik (best-effort) ----
os.environ.setdefault("OPIK_API_KEY", "test")
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:9999")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

_MODEL = os.getenv("LLM__MAIN_MODEL", "")
_API_KEY = os.getenv("LLM__OPENAI_API_KEY", "")
_API_URL = os.getenv("LLM__MAIN_URL", "")

import litellm
litellm.api_key = _API_KEY
litellm.api_base = _API_URL

# ---- Subsystem imports ----
from CoScientist.hypothesis_subsystem import build_hypothesis_subsystem
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger


async def run_one_case(case: dict, case_index: int) -> dict:
    """Run a single case through the Generator+MCPTool subsystem."""
    case_id = case.get("id", f"case_{case_index}")
    question = case["research_question"]

    case_dir = OUTPUT_DIR / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(str(case_dir / "orchestrator.log"), encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    console_handler.setFormatter(fmt)
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    log = logging.getLogger("test")

    summary = {
        "case_id": case_id,
        "question": question,
        "model": _MODEL,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "duration_sec": 0.0,
        "hypotheses_count": 0,
        "with_tool_matching": 0,
        "without_tool_matching": 0,
        "hypotheses": [],
        "error": None,
    }

    try:
        t0 = time.monotonic()

        log.info("=" * 60)
        log.info(f"Case [{case_id}]: {question[:100]}...")
        log.info("=" * 60)

        # Build the subsystem (single black-box AgentTool)
        subsystem = build_hypothesis_subsystem(model=_MODEL)
        log.info(f"Subsystem built: agent={subsystem.agent.name}, model={_MODEL}")

        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        session_service = InMemorySessionService()
        await session_service.create_session(
            app_name="test", user_id="integration_test", session_id=case_id,
        )
        runner = Runner(
            agent=subsystem.agent,
            app_name="test",
            session_service=session_service,
        )

        log.info("Running hypothesis subsystem...")
        t_run = time.monotonic()

        async for event in runner.run_async(
            user_id="integration_test",
            session_id=case_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text=question)],
            ),
        ):
            pass  # Events are collected below

        log.info(f"Subsystem run done in {(time.monotonic() - t_run) * 1000:.0f}ms")

        # Read hypotheses from session state (written by _generate_via_moosechem)
        session = await session_service.get_session(
            app_name="test", user_id="integration_test", session_id=case_id,
        )
        state = getattr(session, "state", {}) or {}
        generated = state.get("generated_hypotheses")

        # Parse generated hypotheses
        raw_hyps = generated.get("hypotheses", []) if isinstance(generated, dict) else []
        from CoScientist.hypothesis_subsystem.models import Hypothesis, ValidationToolInfo

        for i, h in enumerate(raw_hyps, 1):
            try:
                hyp = Hypothesis(**h) if isinstance(h, dict) else h
            except Exception:
                continue

            tool_matches = hyp.validation_tool_matching or []
            tool_names = [t.name if hasattr(t, "name") else t.get("name", "?")
                         for t in tool_matches]

            log.info(f"H#{i} [{hyp.strategy_type}] {hyp.claim[:120]}...")
            log.info(f"  domain={hyp.domain}")
            log.info(f"  tools={hyp.tools}")
            log.info(f"  validation_tool_matching={tool_names}")

            hy_summary = {
                "claim": hyp.claim,
                "strategy": hyp.strategy_type,
                "domain": hyp.domain,
                "hypothesis_tools": hyp.tools,
                "validation_tool_matching": [
                    t.model_dump(mode="json") if hasattr(t, "model_dump")
                    else (t if isinstance(t, dict) else str(t))
                    for t in tool_matches
                ],
                "refutation": hyp.refutation_conditions[:300],
            }
            summary["hypotheses"].append(hy_summary)

            if tool_matches:
                summary["with_tool_matching"] += 1
            else:
                summary["without_tool_matching"] += 1

        summary["hypotheses_count"] = len(summary["hypotheses"])
        summary["duration_sec"] = round(time.monotonic() - t0, 1)

        if not summary["hypotheses"]:
            summary["status"] = "FAIL"
            summary["error"] = "No hypotheses produced"

    except Exception as exc:
        summary["status"] = "ERROR"
        summary["error"] = str(exc)
        log.error(f"Case [{case_id}] FAILED: {exc}")

    # ---- Save outputs ----
    with open(case_dir / "hypotheses.json", "w", encoding="utf-8") as f:
        json.dump(summary["hypotheses"], f, ensure_ascii=False, indent=2, default=str)

    with open(case_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    root.removeHandler(console_handler)
    root.removeHandler(file_handler)
    return summary


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(PROJECT_ROOT / "tests" / "test_cases.json"))
    args = parser.parse_args()

    cases_path = Path(args.cases)
    if not cases_path.exists():
        cases_path = Path(__file__).resolve().parent.parent / "test_cases.json"
    if not cases_path.exists():
        print(f"ERROR: test_cases.json not found at {cases_path}")
        cases = [{"id": "default", "research_question": "What molecular features predict EGFR kinase inhibition?", "max_hypotheses": 2}]
    else:
        with open(cases_path, encoding="utf-8") as f:
            cases = json.load(f)

    print(f"Model: {_MODEL}")
    print(f"Cases: {len(cases)}  Output: {OUTPUT_DIR}")
    print("=" * 70)

    all_summaries = []
    total_start = time.monotonic()

    for i, case in enumerate(cases, 1):
        case_id = case.get("id", f"case_{i}")
        print(f"\n[{i}/{len(cases)}] {case_id}: {case['research_question'][:80]}...")

        case_start = time.monotonic()
        summary = await run_one_case(case, i)
        dt = time.monotonic() - case_start

        icon = {"PASS": "OK", "FAIL": "FAIL", "ERROR": "ERR"}.get(summary["status"], "???")
        print(f"  [{icon}] {summary['status']} | {summary['hypotheses_count']} hyps | "
              f"{summary['with_tool_matching']} tool-matched, "
              f"{summary['without_tool_matching']} unmatched | {dt:.0f}s")
        if summary["error"]:
            print(f"  Error: {summary['error']}")

        all_summaries.append({
            "case_id": summary["case_id"], "status": summary["status"],
            "duration_sec": summary["duration_sec"],
            "hypotheses_count": summary["hypotheses_count"],
            "with_tool_matching": summary["with_tool_matching"],
            "without_tool_matching": summary["without_tool_matching"],
            "error": summary["error"],
        })

    run_summary = {
        "model": _MODEL,
        "total_duration_sec": round(time.monotonic() - total_start, 1),
        "cases_total": len(cases),
        "passed": sum(1 for s in all_summaries if s["status"] == "PASS"),
        "failed": sum(1 for s in all_summaries if s["status"] != "PASS"),
        "cases": all_summaries,
    }
    with open(OUTPUT_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nDONE. {run_summary['passed']}/{run_summary['cases_total']} passed "
          f"({run_summary['total_duration_sec']:.0f}s)")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
