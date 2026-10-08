"""Live end-to-end test of the MICROFLUIDICS profile.

Runs the real reduced pipeline (TZAgent -> PlannerAgent -> OrchestratorAgent
-> ResearchAgent) on a microfluidics-case request and verifies the contract
the profile exists for:

  1. the ТЗ agent produces a structured ТЗ table from the free-form request;
  2. literature queries (LIT-xx) are derived FROM that ТЗ;
  3. the planner registers them as tasks assigned to ResearchAgent;
  4. the request that actually REACHES ResearchAgent carries the ТЗ-derived
     query (query_en terms), i.e. the ТЗ table drives the literature search.

Requires the real LLM (.env) and network; HITL is disabled for the run.

Run from the repo root:
    pytest tests/integration/test_microfluidics_e2e.py -q -s
"""
import asyncio
import contextlib
import os
import re
import sys

# Must be set BEFORE any CoScientist import: HITL handlers and tools are wired
# at import/build time from settings.
os.environ.setdefault("HITL__ENABLED", "false")

# The requests carry Cyrillic + chemistry glyphs (Ca²⁺); a cp1251 Windows
# console must not kill the run over a diagnostic print.
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

APP, USER, SESSION = "microfluidics_e2e", "test_user", "session_e2e"

QUERY = (
    "Нужен отечественный ПАВ для повышения нефтеотдачи (МУН): работа в "
    "минерализованной воде с ионами Ca2+/Mg2+ при 60–90 °C, критично низкое "
    "межфазное натяжение нефть/вода. Синтез должен быть реализуем на проточной "
    "микрофлюидной установке из доступного в РФ сырья. Нужен обзор литературы: "
    "перспективные классы ПАВ, их свойства и маршруты синтеза."
)

# Generic English words that prove nothing about the ТЗ -> research handoff.
_STOPWORDS = {
    "with", "from", "into", "using", "under", "based", "high", "води",
    "search", "find", "paper", "papers", "download", "literature", "review",
    "data", "study", "studies", "research", "analysis", "properties",
}


def _significant_tokens(text: str) -> set:
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text.lower())
        if t not in _STOPWORDS
    }


async def _run_pipeline():
    system = build_system(config_path=resolve_config_path("microfluidics"))

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP, user_id=USER, session_id=SESSION
    )
    runner = Runner(agent=system.root, app_name=APP, session_service=session_service)

    research_requests: list[dict] = []
    async for event in runner.run_async(
        user_id=USER,
        session_id=SESSION,
        new_message=types.Content(role="user", parts=[types.Part(text=QUERY)]),
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name == "ResearchAgent":
                args = dict(fc.args or {})
                research_requests.append(args)
                print(f"\n[e2e] -> ResearchAgent request #{len(research_requests)}: "
                      f"{str(args)[:600]}")

    session = await session_service.get_session(
        app_name=APP, user_id=USER, session_id=SESSION
    )
    return dict(session.state), research_requests


def test_tz_table_drives_the_literature_requests():
    state, research_requests = asyncio.run(_run_pipeline())

    # 1) Structured ТЗ was built from the request (document-shaped: blocks of
    # field rows, see Пример_уточненного_структурированного_ТЗ_для_агентов.md).
    tz = state.get("structured_tz")
    assert isinstance(tz, dict), f"structured_tz missing/not a dict: {type(tz)}"
    assert tz.get("original_request"), "ТЗ lost the original request"
    blocks = tz.get("blocks") or []
    assert blocks, f"ТЗ has no blocks: {tz}"
    customer_set = [
        f"{b.get('title')}/{f.get('name')}"
        for b in blocks for f in (b.get("fields") or [])
        if f.get("status") == "задано заказчиком"
    ]
    assert customer_set, f"no ТЗ field was recognised as customer-set: {tz}"
    print(f"\n[e2e] ТЗ: {len(blocks)} blocks; fields set by customer: "
          f"{customer_set[:10]}{'…' if len(customer_set) > 10 else ''}")

    # 1a) The rendered ТЗ document was produced alongside the JSON.
    doc = state.get("structured_tz_document") or ""
    assert "## Правила интерпретации полей" in doc, "ТЗ document not rendered"
    assert "## Проверка соответствия исходному опроснику" in doc

    # 2) Literature queries were derived from the ТЗ.
    lq = state.get("tz_literature_queries") or {}
    queries = lq.get("queries") or []
    assert queries, f"no literature queries derived from the ТЗ: {lq}"
    for q in queries:
        assert q.get("id") and q.get("query_en"), f"malformed query: {q}"
    print(f"[e2e] ТЗ produced {len(queries)} literature queries: "
          f"{[q['id'] for q in queries]}")

    # 3) The plan assigns the literature tasks to ResearchAgent.
    active = state.get("active_tasks")
    assert isinstance(active, list), f"active_tasks must be a session-local list: {active!r}"
    tasks = active
    assert tasks, f"planner registered no tasks: {active!r}"
    research_tasks = [t for t in tasks if t.get("assignee") == "ResearchAgent"]
    assert research_tasks, f"no plan task assigned to ResearchAgent: {tasks}"
    print(f"[e2e] plan: {len(research_tasks)} ResearchAgent task(s): "
          f"{[t.get('title') for t in research_tasks]}")

    # 4) The ТЗ-derived query actually reached ResearchAgent.
    assert research_requests, "the orchestrator never delegated to ResearchAgent"
    joined = " ".join(str(r) for r in research_requests).lower()
    matched = []
    for q in queries:
        tokens = _significant_tokens(
            " ".join([q.get("query_en", ""), " ".join(q.get("extract") or [])])
        )
        common = sorted(t for t in tokens if t in joined)
        if q["id"].lower() in joined or len(common) >= 2:
            matched.append((q["id"], common[:8]))
    assert matched, (
        "No ResearchAgent request matches any ТЗ-derived literature query.\n"
        f"Queries: {[q['query_en'] for q in queries]}\n"
        f"Requests: {[str(r)[:200] for r in research_requests]}"
    )
    print(f"[e2e] ТЗ -> ResearchAgent handoff confirmed for: {matched}")
