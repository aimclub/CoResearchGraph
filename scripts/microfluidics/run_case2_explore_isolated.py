"""Case 2 (2-bromo-1-(3,5-di-tert-butyl-4-hydroxyphenyl)ethan-1-one), continued.

The structured ТЗ and 5 LIT-xx queries were already produced by a prior run
(evaluation/microfluidics/runs/custom_case_02_explore_only.json) — that run's
LiteratureOrchestrator then crashed (1.29M-token context) trying to process
all 5 ResearchAgent delegations in one continuous session. This script reuses
the already-good ТЗ/queries and re-runs ONLY the literature step, one
ISOLATED ResearchAgent call per LIT query (fresh session each time — no
context carries over between tasks), restricted to explore_scientific_database
only (paper_analysis + task_tracker), per the user's instruction.

Run from the repo root:
    python scripts/microfluidics/run_case2_explore_isolated.py
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_microfluidics_literature_batch import APP, USER, _drain  # noqa: E402
from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import resolve_config_path  # noqa: E402
from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402

SEED_PATH = Path("evaluation/microfluidics/runs/custom_case_02_explore_only.json")
OUT_PATH = Path("evaluation/microfluidics/runs/custom_case_02_explore_isolated_v2.json")
YAML_PATH = Path("CoScientist/microfluidics/microfluidics.yaml")
FULL_TOOLS_LINE = "    tools: [websearch, paper_analysis, papers_search, task_tracker]"
RESTRICTED_TOOLS_LINE = "    tools: [paper_analysis, task_tracker]"


def _restrict_research_agent_tools() -> None:
    text = YAML_PATH.read_text(encoding="utf-8")
    assert FULL_TOOLS_LINE in text, "ResearchAgent tools line not found as expected"
    YAML_PATH.write_text(text.replace(FULL_TOOLS_LINE, RESTRICTED_TOOLS_LINE), encoding="utf-8")


def _revert_yaml() -> None:
    # Revert INSIDE this same process, right after build_system() has already
    # parsed the restricted file into memory — no race with a separate shell
    # command (that raced and lost last time, silently leaving the full
    # toolset active for the whole run).
    subprocess.run(["git", "checkout", "--", str(YAML_PATH)], check=True)


async def main():
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    structured_tz = seed["state"]["structured_tz"]
    queries = seed["state"]["tz_literature_queries"]["queries"]

    _restrict_research_agent_tools()
    try:
        system = build_system(config_path=resolve_config_path("microfluidics"))
    finally:
        _revert_yaml()

    try:
        ra = system.agent("ResearchAgent")
        print(f"ResearchAgent.tools (sanity check): {ra.tools!r}", flush=True)
    except Exception as exc:
        print(f"(sanity check skipped: {exc})", flush=True)

    results = []

    for q in queries:
        session_service = InMemorySessionService()
        session_id = f"case2_{q['id']}"
        await session_service.create_session(
            app_name=APP, user_id=USER, session_id=session_id,
            state={"structured_tz": structured_tz},
        )
        message = (
            f"{q['task']}\n\n"
            f"Search query (query_en, use VERBATIM): {q['query_en']}\n"
            f"Extract: {q.get('extract')}"
        )
        tool_calls: list[str] = []
        print(f"[{q['id']}] running...", flush=True)
        runner = Runner(agent=system.agent("ResearchAgent"), app_name=APP, session_service=session_service)
        err = await _drain(runner, session_id, message, tool_calls)

        session = await session_service.get_session(app_name=APP, user_id=USER, session_id=session_id)
        state = dict(session.state)
        results.append({
            "id": q["id"],
            "task": q["task"],
            "query_en": q["query_en"],
            "extract": q.get("extract"),
            "tool_calls": tool_calls,
            "error": err,
            "search_results": state.get("search_results"),
            "literature_smiles": state.get("literature_smiles"),
            "literature_reactions": state.get("literature_reactions"),
        })
        print(f"[{q['id']}] done — error={bool(err)} "
              f"smiles={state.get('literature_smiles')} reactions={state.get('literature_reactions')}",
              flush=True)
        await asyncio.sleep(3)

    out = {
        "original_request": seed["original_request"],
        "structured_tz": structured_tz,
        "queries": queries,
        "per_task_results": results,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
