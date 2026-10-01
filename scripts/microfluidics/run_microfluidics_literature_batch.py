"""Live batch run of the microfluidics literature stage over the 14 свод requests.

Two tasks (--full-ids, default 1,5) get the FULL module A run (TZAgent ->
PlannerAgent -> LiteratureOrchestrator -> ResearchAgent, on the customer's
free-form request) followed directly by MolDesignAgent on the resulting
session state — so a real design_candidates comes out the other end.

The remaining tasks get a single DIRECT ResearchAgent call on that request's
LIT-02 query (маршруты синтеза) — no TZ/planner overhead, just the literature
search that feeds capture_literature_smiles / capture_literature_reactions.

Requires the real LLM (.env) and network (paper_analysis / papers_search MCP
reachable — VPN); HITL is disabled for the run.

Run from the repo root:
    python scripts/microfluidics/run_microfluidics_literature_batch.py
"""
import asyncio
import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path

# Running this file directly (not via `-m`) sets sys.path[0] to scripts/, not
# the repo root — add the root explicitly so `import CoScientist` works
# regardless of invocation method or current working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("HITL__ENABLED", "false")
# The internal paper_analysis corpus in this environment doesn't cover the
# lubricant-antioxidant domain (confirmed by direct probe — see commit msg),
# so ResearchAgent genuinely needs more than the default 2 OpenAlex attempts
# per task to find anything real. Must be set before CoScientist is imported —
# assembly/bindings.py reads it once, at build_system() time.
os.environ.setdefault("RESEARCH_MAX_SEARCHES", "6")

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(errors="replace")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import resolve_config_path  # noqa: E402

APP = "microfluidics_svod_batch"
USER = "svod_batch"
SVOD_PATH = Path("tz_documents/TZ_svod_20260710_141503.md")
OUT_DIR = Path("evaluation/microfluidics/runs")
FULL_IDS = {1, 5}
PAUSE_BETWEEN_TASKS_S = 8  # be polite to the shared lab MCP servers


def parse_svod(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    req_starts = [(m.start(), int(m.group(1))) for m in re.finditer(r"^## Запрос (\d+)\.", text, re.M)]
    req_starts.append((len(text), None))

    results = {}
    for i in range(len(req_starts) - 1):
        start, num = req_starts[i]
        end = req_starts[i + 1][0]
        chunk = text[start:end]

        m = re.search(
            r"### Исходный запрос заказчика в свободной форме\s*\n+>\s*(.+?)\n\n", chunk, re.S
        )
        original = m.group(1).strip().replace("\n> ", " ").replace("\n", " ") if m else None

        lit_rows = re.findall(r"^\|\s*(LIT-\d+)\s*\|(.+?)\|(.+?)\|(.+?)\|$", chunk, re.M)
        queries = [
            {"id": lit_id.strip(), "task_ru": task_ru.strip(),
             "query_en": query_en.strip(), "extract": extract.strip()}
            for lit_id, task_ru, query_en, extract in lit_rows
        ]
        results[num] = {"original_request": original, "queries": queries}
    return results



# capture_literature_smiles / capture_literature_reactions write to session
# state inside an after_tool callback — BEFORE the agent's final synthesis
# turn. So a late-run crash (e.g. accumulated context blowing past the
# model's window after several escalations) does not erase what was already
# extracted; only the final prose answer is lost. _drain therefore always
# lets the caller fetch state, error or not.
_STEER = (
    " ВАЖНО про источники для этого запроса: внутренняя научная база "
    "(explore_scientific_database) в этом окружении НЕ содержит статей по "
    "теме присадок/антиоксидантов для масел (она про металлоорганический "
    "катализ и экстракцию из оливкового масла) — один пробный вызов "
    "допустим, но если он явно скажет, что релевантной информации нет, "
    "сразу переходи к search_papers/download_papers_from_search (OpenAlex) "
    "как к основному источнику для ЭТОЙ темы, не трать больше попыток на "
    "explore_scientific_database. download_papers_from_search возвращает "
    "только метаданные + S3-ключи — это дёшево; проанализируй explore_my_papers "
    "не более 2-3 самых релевантных скачанных статей, не все сразу. "
    "tavily_search в этом окружении не настроен (нет ключа) — не используй его."
)


async def _drain(runner, session_id, message, tool_calls) -> str | None:
    """Runs one agent turn, returns an error string (or None on success)."""
    try:
        async for event in runner.run_async(
            user_id=USER, session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            if not (event.content and event.content.parts):
                continue
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc:
                    tool_calls.append(fc.name)
                    print(f"    -> {fc.name}()", flush=True)
    except Exception as exc:
        print(f"    !! turn failed: {exc}", flush=True)
        return str(exc)
    return None


async def run_full_task(system, session_service, num: int, req: dict) -> dict:
    session_id = f"req_{num:02d}_full"
    await session_service.create_session(app_name=APP, user_id=USER, session_id=session_id)
    tool_calls: list[str] = []
    errors = []

    print(f"[req {num}] module A (TZ -> planner -> literature)...", flush=True)
    runner_a = Runner(agent=system.agent("ModuleA_TZLiterature"), app_name=APP, session_service=session_service)
    err = await _drain(runner_a, session_id, req["original_request"] + _STEER, tool_calls)
    if err:
        errors.append({"step": "module_a", "error": err})

    print(f"[req {num}] MolDesignAgent...", flush=True)
    runner_design = Runner(agent=system.agent("MolDesignAgent"), app_name=APP, session_service=session_service)
    err = await _drain(runner_design, session_id, "Выполни свою стадию по данным из состояния сессии.", tool_calls)
    if err:
        errors.append({"step": "mol_design", "error": err})

    session = await session_service.get_session(app_name=APP, user_id=USER, session_id=session_id)
    return {"mode": "full", "tool_calls": tool_calls, "errors": errors, "state": dict(session.state)}


async def run_reactions_only_task(system, session_service, num: int, req: dict) -> dict:
    session_id = f"req_{num:02d}_reactions"
    await session_service.create_session(app_name=APP, user_id=USER, session_id=session_id)
    tool_calls: list[str] = []
    errors = []

    # LIT-03 ("проточный/микрофлюидный синтез") is the query type the internal
    # corpus actually has content for — the prior run (evaluation/microfluidics/
    # literature_results.html, 19.08.2026) shows it reranking up to 51% там, где
    # LIT-01/02 (specific antioxidant structures/routes) top out at ~18% on
    # off-topic hits (Zr/Ti catalysis, olive oil extraction). LIT-02 is still the
    # fallback since not every request's numbering lines up the same way.
    lit02 = (
        next((q for q in req["queries"] if q["id"] == "LIT-03"), None)
        or next((q for q in req["queries"] if q["id"] == "LIT-02"), None)
        or req["queries"][0]
    )
    message = (
        f"{lit02['task_ru']}\n\n"
        f"Search query (query_en, use VERBATIM): {lit02['query_en']}\n"
        f"Extract: {lit02['extract']}"
    )

    print(f"[req {num}] ResearchAgent direct ({lit02['id']})...", flush=True)
    runner = Runner(agent=system.agent("ResearchAgent"), app_name=APP, session_service=session_service)
    err = await _drain(runner, session_id, message + _STEER, tool_calls)
    if err:
        errors.append({"step": "research_agent", "error": err})

    session = await session_service.get_session(app_name=APP, user_id=USER, session_id=session_id)
    return {
        "mode": "reactions_only", "lit_query": lit02, "tool_calls": tool_calls,
        "errors": errors, "state": dict(session.state),
    }


async def main():
    svod = parse_svod(SVOD_PATH)
    system = build_system(config_path=resolve_config_path("microfluidics"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for num in sorted(svod):
        req = svod[num]
        session_service = InMemorySessionService()
        t0 = time.time()
        try:
            if num in FULL_IDS:
                result = await run_full_task(system, session_service, num, req)
            else:
                result = await run_reactions_only_task(system, session_service, num, req)
        except Exception as exc:
            print(f"[req {num}] FAILED: {exc}", flush=True)
            result = {"mode": "error", "error": str(exc)}

        result["original_request"] = req["original_request"]
        result["elapsed_s"] = round(time.time() - t0, 1)

        out_path = OUT_DIR / f"req_{num:02d}.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        state = result.get("state", {})
        n_smiles = len(state.get("literature_smiles") or [])
        n_rxn = len(state.get("literature_reactions") or [])
        print(f"[req {num}] done in {result['elapsed_s']}s — "
              f"smiles={n_smiles} reactions={n_rxn} -> {out_path}", flush=True)
        summary.append({"num": num, "mode": result.get("mode"), "smiles": n_smiles,
                         "reactions": n_rxn, "elapsed_s": result["elapsed_s"]})

        await asyncio.sleep(PAUSE_BETWEEN_TASKS_S)

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== SUMMARY ===")
    for s in summary:
        print(s)


if __name__ == "__main__":
    asyncio.run(main())
