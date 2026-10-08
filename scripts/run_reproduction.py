"""
Reproduction runner for CoScientist paper experiments.

Usage:
    uv run python scripts/run_reproduction.py [--paper tox-antitargets|heracleum-tox] [--out-dir logs/reproduction]

Captures:
  - Full streaming event log  → <out_dir>/<paper>_<timestamp>_events.log
  - Final response (markdown) → <out_dir>/<paper>_<timestamp>_result.md
  - Combined run log          → <out_dir>/<paper>_<timestamp>_run.log
"""

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Prompts from REPRODUCTION_QUESTIONS-*.md
# ---------------------------------------------------------------------------

PROMPTS = {
    "tox-antitargets": (
        "You are reproducing Nikitin et al. 2025 (Pharmaceutics 17, 1573). "
        "Call the tox-antitargets tools, then state each conclusion using only the returned numbers. "
        "Do not introduce any value or claim not present in the tool output. "
        "Where a tool returns a `finding` or `reproduced_statement`, treat it as the authoritative interpretation. "
        "Report any value that differs from the paper and by how much.\n\n"
        "Task: Using the tox-antitargets tools, reproduce all the findings of Nikitin et al. 2025 "
        "linking antitargets to rodent acute toxicity, and state each conclusion with the supporting numbers."
    ),
    "heracleum-tox": (
        "You are reproducing Rassabina & Fedorov 2025 (Plants 14, 3253) with open-source analogues of Syntelly. "
        "Call the heracleum-tox tools, then state each conclusion using only the returned numbers. "
        "Where a tool returns a `finding` or `reproduced_statement`, treat it as authoritative. "
        "Report any value that differs from the paper and say by how much — "
        "in particular, surface the documented DILI/cardiotoxicity divergence (C5) rather than hiding it.\n\n"
        "Task: Using the heracleum-tox tools, reproduce all the findings of Rassabina & Fedorov 2025 on the "
        "toxicological profile of Heracleum sosnowskyi metabolites, and state each conclusion with the "
        "supporting numbers."
    ),
}


# ---------------------------------------------------------------------------
# Tee: write to both a file and the original stream
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(paper: str, out_dir: Path, session_id: str, latex: str = "skip"):
    from dotenv import load_dotenv
    load_dotenv()

    from CoScientist.main import CoScientistManager
    from CoScientist.config import ReportConfig

    prompt = PROMPTS[paper]
    report_config = ReportConfig(latex=latex)

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    events_path = out_dir / f"{paper}_{ts}_events.log"
    result_path = out_dir / f"{paper}_{ts}_result.md"
    run_log_path = out_dir / f"{paper}_{ts}_run.log"

    events_file = open(events_path, "w", encoding="utf-8")

    # Logging to file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(run_log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    root_logger.addHandler(file_handler)

    # Tee stdout so verbose event prints also land in events.log
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, events_file)

    print(f"[run_reproduction] paper      : {paper}")
    print(f"[run_reproduction] session_id : {session_id}")
    print(f"[run_reproduction] started    : {ts}")
    print(f"[run_reproduction] events log : {events_path}")
    print(f"[run_reproduction] result     : {result_path}")
    print(f"[run_reproduction] run log    : {run_log_path}")
    print("-" * 72)
    print("PROMPT:")
    print(prompt)
    print("-" * 72)

    try:
        manager = CoScientistManager(
            app_name="coscientist_reproduction",
            user_id="reproducer",
            session_id=session_id,
        )

        result = await manager.run(prompt, verbose=True, report_config=report_config)
        final = result.markdown

        print("-" * 72)
        print("[run_reproduction] FINAL RESPONSE:")
        print(final)
        print("-" * 72)
        if result.report_dir:
            print(f"[run_reproduction] 📁 Report deliverable: {result.report_dir}")

        result_path.write_text(
            f"# Reproduction result — {paper}\n\n"
            f"**Paper:** {paper}  \n"
            f"**Run timestamp:** {ts}  \n"
            f"**Session ID:** {session_id}  \n"
            f"**Report folder:** {result.report_dir or '(none)'}  \n\n"
            "---\n\n"
            "## Prompt\n\n"
            f"{prompt}\n\n"
            "---\n\n"
            "## Response\n\n"
            f"{final}\n",
            encoding="utf-8",
        )

        print(f"[run_reproduction] ✅  Result saved to {result_path}")

    except Exception as exc:
        logging.exception("Reproduction run failed")
        print(f"[run_reproduction] ❌  Run failed: {exc}")
        raise

    finally:
        sys.stdout = original_stdout
        events_file.close()
        root_logger.removeHandler(file_handler)
        file_handler.close()

    return result_path, events_path, run_log_path


def main():
    parser = argparse.ArgumentParser(description="CoScientist paper reproduction runner")
    parser.add_argument(
        "--paper",
        choices=list(PROMPTS.keys()),
        default="tox-antitargets",
        help="Which paper to reproduce (default: tox-antitargets)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("logs/reproduction"),
        help="Directory for logs and results (default: logs/reproduction)",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--latex",
        choices=["skip", "standalone", "body", "tree"],
        default="skip",
        help="LaTeX output mode for the final report (default: skip).",
    )
    args = parser.parse_args()

    session_id = args.session_id or f"repro_{args.paper}_{uuid.uuid4().hex[:8]}"

    result, events, run_log = asyncio.run(
        run(args.paper, args.out_dir, session_id, latex=args.latex)
    )

    print(f"\nDone. Outputs:")
    print(f"  result  → {result}")
    print(f"  events  → {events}")
    print(f"  run log → {run_log}")


if __name__ == "__main__":
    main()
