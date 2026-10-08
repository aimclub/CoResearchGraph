"""Live tests of the HITL review + per-question interview on the ТЗ stage.

Runs ONLY the TZAgent composite (TZSpecAgent -> TZQueryGenAgent) with the real
LLM and verifies the two-phase operator review:

  1. the rendered ТЗ document is approved first;
  2. then the operator is interviewed ONE HITL WINDOW PER QUESTION about the
     remaining «не задано» fields; the scripted operator exercises every
     answer kind: a text answer, «не знаю» (skip), «на усмотрение агента»,
     and «завершить опрос» (stop);
  3. collected answers trigger ONE rewrite, the updated document is approved,
     the stage publishes the document and produces the literature queries.

Also verifies the headless mode: with no handler wired (HITL__ENABLED=false)
the stage passes through with zero human interaction.

NOTE: this module declares HITL__ENABLED=true at import, while
test_microfluidics_e2e.py declares it false — run the integration files in
SEPARATE pytest invocations (settings are resolved once per process).

Run from the repo root:
    pytest tests/integration/test_microfluidics_hitl.py -q -s
"""
import asyncio
import contextlib
import os
import sys

os.environ["HITL__ENABLED"] = "true"

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from CoScientist.assembly import build_system  # noqa: E402
from CoScientist.assembly.schema import resolve_config_path  # noqa: E402
from CoScientist.hitl.handler import AbstractHITLHandler  # noqa: E402
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse  # noqa: E402
from CoScientist.microfluidics.questionnaire import (  # noqa: E402
    OPT_AGENT,
    OPT_SKIP,
    OPT_STOP,
    QUESTION_OPTIONS,
)

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(errors="replace")

QUERY = (
    "Нужен отечественный ПАВ для повышения нефтеотдачи: минерализованная вода "
    "с Ca2+/Mg2+, 60–90 °C, низкое межфазное натяжение, синтез на проточной "
    "микрофлюидной установке."
)

TEXT_ANSWER = "лабораторная рецептура и опытная партия до 1 кг"


class ScriptedHuman(AbstractHITLHandler):
    """Deterministic operator: approves documents; in the interview answers
    Q1 with text, skips Q2, delegates Q3 and stops the interview at Q4."""

    def __init__(self):
        self.requests: list[HITLRequest] = []
        self.proposals: dict[str, list[str]] = {}
        self.question_cards: list[HITLRequest] = []

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        self.requests.append(request)
        if request.options:  # a question window of the interview
            self.question_cards.append(request)
            k = len(self.question_cards)
            if k == 1:
                return HITLResponse(
                    action=HITLAction.EDIT, approved=False, instructions=TEXT_ANSWER
                )
            if k == 2:
                return HITLResponse(
                    action=HITLAction.SELECT, approved=True, selected_option=OPT_SKIP
                )
            if k == 3:
                return HITLResponse(
                    action=HITLAction.SELECT, approved=True, selected_option=OPT_AGENT
                )
            return HITLResponse(
                action=HITLAction.SELECT, approved=True, selected_option=OPT_STOP
            )
        # Document / queries review — accept as-is.
        proposals = self.proposals.setdefault(request.agent_name, [])
        proposals.append(str(request.context.get("output", "")))
        return HITLResponse(action=HITLAction.APPROVE, approved=True)


async def _run_tz_stage(wire_handler):
    """Build the profile, let the caller (re)wire HITL, run ONLY TZAgent.

    Returns the final session state and every text chunk the stage emitted
    into the event stream (what the chat would display)."""
    system = build_system(config_path=resolve_config_path("microfluidics"))
    wire_handler(system)

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name="mf_hitl", user_id="u", session_id="s"
    )
    runner = Runner(
        agent=system.agent("TZAgent"), app_name="mf_hitl",
        session_service=session_service,
    )
    chat_texts: list[str] = []
    async for event in runner.run_async(
        user_id="u", session_id="s",
        new_message=types.Content(role="user", parts=[types.Part(text=QUERY)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if getattr(part, "text", None) and not getattr(part, "thought", False):
                    chat_texts.append(part.text)
    session = await session_service.get_session(
        app_name="mf_hitl", user_id="u", session_id="s"
    )
    return dict(session.state), chat_texts


def _assert_tz_artifacts(state):
    tz = state.get("structured_tz")
    assert isinstance(tz, dict) and tz.get("original_request"), f"bad ТЗ: {tz!r}"
    assert tz.get("blocks"), f"ТЗ has no blocks: {tz!r}"
    doc = state.get("structured_tz_document") or ""
    assert "## Правила интерпретации полей" in doc, "ТЗ document not rendered"
    queries = (state.get("tz_literature_queries") or {}).get("queries") or []
    assert queries and all(q.get("id") and q.get("query_en") for q in queries), (
        f"bad queries: {queries!r}"
    )
    return tz, queries


def _assert_document_published_to_chat(chat_texts):
    published = [
        t for t in chat_texts
        if "Структурированное ТЗ сформировано" in t
        and "## Правила интерпретации полей" in t
    ]
    assert published, "the ТЗ document was not published into the chat stream"
    assert "tz_documents" in published[0], "chat message lacks the saved file path"
    return published[0]


def test_interview_one_window_per_question():
    """Document approval first, then ONE HITL window per question; answers
    (text / skip / delegate / stop) are applied via a single rewrite."""
    human = ScriptedHuman()

    def wire(system):
        for name in ("TZSpecAgent", "TZQueryGenAgent"):
            handler = system.agent(name).hitl_handler
            assert handler is not None, f"{name}: HITL handler not wired at build"
            handler.set_delegate(human)

    state, chat_texts = asyncio.run(_run_tz_stage(wire))

    # The interview asked one window per question and stopped when told to:
    # text answer, skip, delegate, stop => exactly 4 windows.
    cards = human.question_cards
    assert len(cards) == 4, f"expected 4 question windows, got {len(cards)}"
    for i, card in enumerate(cards, 1):
        assert list(card.options) == list(QUESTION_OPTIONS), card.options
        assert f"вопрос {i} из" in card.message, card.message
        output = str(card.context.get("output", ""))
        assert "Как ответить" in output and card.message.split("«")[1].split("»")[0] in output

    # The document was reviewed twice: the draft and the post-interview rewrite.
    doc_proposals = human.proposals.get("TZSpecAgent", [])
    assert len(doc_proposals) == 2, (
        f"expected draft + rewritten document reviews, got {len(doc_proposals)}"
    )
    assert doc_proposals[0] != doc_proposals[1], "rewrite identical to the draft"
    for proposal in doc_proposals:
        assert "## Правила интерпретации полей" in proposal
        assert "## Уточняющие вопросы оператору" not in proposal, (
            "the old inline questionnaire dump is back in the document review"
        )
    assert human.proposals.get("TZQueryGenAgent"), "queries never went to review"

    tz, queries = _assert_tz_artifacts(state)
    _assert_document_published_to_chat(chat_texts)
    answered_blocks = [c.message.split("«")[1].split("»")[0] for c in cards]
    print(f"\n[hitl] interview: {len(cards)} windows over blocks {answered_blocks}")
    print(f"[hitl] Q1 answered with text; Q2 skipped; Q3 delegated; Q4 stopped the interview")
    print(f"[hitl] final ТЗ: {len(tz.get('blocks') or [])} blocks; {len(queries)} queries")
    for title in answered_blocks[:1]:
        block = next((b for b in tz["blocks"]
                      if str(b.get("title", "")).lower() == title.lower()), {})
        fields = [(f.get("name"), f.get("value"), f.get("status"))
                  for f in block.get("fields") or []]
        print(f"[hitl] ТЗ «{title}» after the text answer: {fields}")


def test_pass_through_without_handler():
    """No handler (what HITL__ENABLED=false wires) — the stage runs headless:
    no document review, no interview, artifacts still produced."""
    human = ScriptedHuman()

    def wire(system):
        for name in ("TZSpecAgent", "TZQueryGenAgent"):
            system.agent(name).hitl_handler = None

    state, chat_texts = asyncio.run(_run_tz_stage(wire))

    assert not human.requests, "no human should have been consulted"
    _, queries = _assert_tz_artifacts(state)
    _assert_document_published_to_chat(chat_texts)
    print(f"\n[hitl] pass-through: ТЗ + {len(queries)} queries + published document "
          "with zero human interaction")
