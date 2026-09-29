"""Run NirReportAgent alone against a finished session and watch it work.

The full pipeline is a long way to reach one agent. This wires up just
NirReportAgent — its own model, its three tools, nothing else — points it at a
session that already has a research graph on disk, and streams what it does:
every tool call, every result, the prose it writes and the link it ends with.

    MCP__NORMCONTROL_URL=http://127.0.0.1:8001/mcp \
    python scripts/nir_agent_demo.py --session session_892e756a051b495da63127b8f81d4b72

Everything else comes from .env: LLM__NIR_MODEL picks the model (GLM 5.3 via
OpenRouter here) and LLM__OPENAI_API_KEY carries the key litellm uses for it.

Two things this script does that the real run does through other machinery, and
which are the reason it can exist at all:

* It pins ``graph_scope_*`` in session state, which is how every graph reader
  finds the study. That is normally the manager's own scoping — the tools read
  it rather than the ADK session id precisely so a delegated agent still sees
  the parent's graph, and here it lets a fresh ADK session read a finished one.
* It writes ``nir_report_request`` itself. In a real run the ask_nir_report
  callback writes it after the operator answers; there is no operator here, so
  the requisites below stand in for the form.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Everything this prints is Russian, and a Windows console defaults to cp1251,
# which cannot encode it. Without this the script dies mid-run on its own
# output — after the model has already been paid for the turn.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable stream
        pass

from CoScientist.config import get_settings  # noqa: E402
from CoScientist.graph.session_scope import (  # noqa: E402
    GRAPH_SCOPE_SESSION_KEY,
    GRAPH_SCOPE_USER_KEY,
)
from CoScientist.reporting.nir import hitl_form  # noqa: E402

APP_NAME = "nir_agent_demo"

#: Stand-ins for the HITL form. Deliberately incomplete: the empty fields show
#: what a draft looks like when an organisation has not supplied its requisites,
#: which is the common case and the one worth seeing.
DEMO_REQUISITES = {
    "udc": "004.942:615.9",
    "registration_nioktr": "АААА-А26-126090300001-1",
    "registration_ikrbs": "",
    "report_type": "заключительный",
    "stage": "",
    "program_code": "",
    "approval_position": "Проректор по научной работе",
    "approval_name": "В.О. Никифоров",
    "approval_degree": "доктор технических наук",
    "approval_academic_title": "профессор",
    "approval_date": "",
    "supervisor_role": "Руководитель НИР",
    "supervisor_name": "И.И. Иванов",
    "supervisor_position": "руководитель проекта",
    "supervisor_degree": "кандидат технических наук",
    "supervisor_academic_title": "",
    "performers_raw": (
        "Исполнитель | инженер-исследователь | П.П. Петров | сбор данных, разделы 1-2\n"
        "Исполнитель | младший научный сотрудник | С.С. Сидоров | эксперименты, разделы 3-4"
    ),
    "page_count": None,
}

TASK = textwrap.dedent("""\
    Подготовь отчёт о научно-исследовательской работе по ГОСТ 7.32-2017 на
    материале завершённого исследования этой сессии.

    Начни с nir_report_outline, напиши текст по каждому разделу, проверь его
    через nir_report_draft и отправь nir_report_submit. В ответе верни ссылку
    на документ и предупреждения сервера.
""")


def find_scope(session_id: str) -> tuple[str, str]:
    """``(user_id, session_id)`` for a session, found by its graph directory."""
    root = Path(os.getenv("GRAPH_SNAPSHOT_DIR") or "graph_runs") / "sessions"
    for candidate in root.glob(f"*/{session_id}"):
        if (candidate / "research_active.json").is_file():
            return candidate.parent.name, session_id
    raise SystemExit(
        f"no research graph for {session_id} under {root}\n"
        f"available: " + ", ".join(sorted(p.name for p in root.glob('*/session_*'))[:5])
    )


def short(value, limit: int = 400) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + f" …(+{len(text) - limit})"


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()

    url = args.mcp_url or settings.mcp.normcontrol_url
    if not url:
        raise SystemExit("set MCP__NORMCONTROL_URL (or pass --mcp-url)")
    settings.mcp.normcontrol_url = url
    # The agent is gated on nir_ready, which needs this on. Flipped here rather
    # than demanded in .env so the demo leaves the deployment's own switch alone.
    settings.nir.enabled = True

    model = settings.llm.nir_model or settings.llm.main_model
    user_id, session_id = find_scope(args.session)
    print(f"session   {user_id}/{session_id}")
    print(f"model     {model}")
    print(f"mcp       {url}\n")

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from CoScientist.assembly import build_system

    system = build_system()
    agent = system.agent("NirReportAgent")

    service = InMemorySessionService()
    await service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            # Every graph reader resolves the study through these two keys.
            GRAPH_SCOPE_USER_KEY: user_id,
            GRAPH_SCOPE_SESSION_KEY: session_id,
            "report_config": {"latex": "skip", "reports_root": str(
                Path(os.getenv("REPORTS_ROOT") or "logs/reports"))},
            "report_language": "ru",
            hitl_form.STATE_REQUEST_KEY: {
                "enabled": True,
                "mode": settings.nir.mode,
                "requisites": DEMO_REQUISITES,
            },
        },
    )

    runner = Runner(agent=agent, app_name=APP_NAME, session_service=service)
    message = types.Content(role="user", parts=[types.Part(text=TASK)])

    final = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session_id, new_message=message
    ):
        for part in (getattr(event.content, "parts", None) or []):
            call = getattr(part, "function_call", None)
            if call:
                print(f"CALL {call.name}({short(dict(call.args or {}), 260)})")
            response = getattr(part, "function_response", None)
            if response:
                print(f"  -> {response.name}: {short(response.response, 500)}\n")
            text = getattr(part, "text", None)
            if text and event.is_final_response():
                final = text

    print("-" * 78)
    print(final.strip() or "(агент не вернул текста)")
    print("-" * 78)

    session = await service.get_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id
    )
    saved = (session.state or {}).get(hitl_form.STATE_RESULT_KEY)
    if saved:
        print("сохранено:")
        for key in ("local_path", "bucket", "s3_key", "download_link", "sha256", "expires_at"):
            if saved.get(key):
                print(f"  {key:14} {saved[key]}")
        local = saved.get("local_path")
        if local and Path(local).is_file():
            print(f"  {'размер':14} {Path(local).stat().st_size} байт")
        return 0

    print("документ не сохранён — смотри ответ агента выше")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--session", required=True, help="a session with a research graph on disk")
    parser.add_argument("--mcp-url", help="overrides MCP__NORMCONTROL_URL")
    raise SystemExit(asyncio.run(run(parser.parse_args())))
