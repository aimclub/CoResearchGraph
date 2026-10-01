"""Live Module A run (TZAgent -> PlannerAgent -> LiteratureOrchestrator ->
ResearchAgent) on a single ad-hoc customer request, NOT from the свод — for a
one-off case check. Stops after Module A — no MolDesignAgent, no design stub.

Run from the repo root:
    python scripts/microfluidics/run_single_custom_case.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_microfluidics_literature_batch import APP, USER, _drain  # noqa: E402
from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import resolve_config_path  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402

ORIGINAL_REQUEST = (
    "Требуется разработать технологию получения антиокислительной присадки для "
    "моторного масла, близкой по функции к промышленным фенольным антиоксидантам. "
    "Продукт должен быть совместим с углеводородной средой, обеспечивать "
    "антиокислительный эффект в составе масла и потенциально производиться в "
    "малотоннажном масштабе. Желательно использовать доступные реагенты, "
    "избегать особо дефицитных компонентов и получить лабораторный образец не "
    "менее 1 г с подтверждением структуры и чистоты. Ориентир по чистоте — не "
    "ниже 80 %. Предельная себестоимость пока не задана, но важно оценить "
    "стоимость сырья и сравнить варианты маршрутов. Предварительно это вещество "
    "2-бром-1-(3,5-ди-трет-бутил-4-гидроксифенил)этан-1-он."
)

OUT_PATH = Path("evaluation/microfluidics/runs/custom_case_02_explore_only.json")


async def main():
    system = build_system(config_path=resolve_config_path("microfluidics"))
    session_service = InMemorySessionService()
    session_id = "custom_case_02_explore_only"
    await session_service.create_session(app_name=APP, user_id=USER, session_id=session_id)

    tool_calls: list[str] = []
    from google.adk.runners import Runner
    runner_a = Runner(agent=system.agent("ModuleA_TZLiterature"), app_name=APP, session_service=session_service)
    err = await _drain(runner_a, session_id, ORIGINAL_REQUEST, tool_calls)

    session = await session_service.get_session(app_name=APP, user_id=USER, session_id=session_id)
    result = {
        "mode": "module_a_only",
        "original_request": ORIGINAL_REQUEST,
        "tool_calls": tool_calls,
        "errors": [{"step": "module_a", "error": err}] if err else [],
        "state": dict(session.state),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    state = result["state"]
    print(f"errors: {result['errors']}")
    print(f"tool_calls: {tool_calls}")
    print(f"literature_smiles: {state.get('literature_smiles')}")
    print(f"literature_reactions: {state.get('literature_reactions')}")
    print(f"structured_tz present: {bool(state.get('structured_tz'))}")
    print(f"tz_literature_queries present: {bool(state.get('tz_literature_queries'))}")
    print(f"search_results present: {bool(state.get('search_results'))}")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
