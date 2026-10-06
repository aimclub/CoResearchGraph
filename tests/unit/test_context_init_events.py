"""Context initialization through ADK's actual event/state persistence boundary.

The model is scripted; no network or semantic critic is involved. Tests read
the session and graph saved by the actual Runner, without preloading its output.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import Field

from CoScientist.context_init.agent import (
    FRAME_COMPLETED_STATE_KEY,
    FRAME_STATE_KEY,
    ContextInitSessionAgent,
)
from CoScientist.context_init.models import ResearchFrame
from CoScientist.requirements.models import StatementDraft
from CoScientist.graph.research.store import ResearchGraphStore


SOURCES = [
    pytest.param(
        "Synthesize a novel tyrosine hydroxylase activator with cellular specificity.",
        id="c2",
    ),
    pytest.param("Generate molecules which can bind to GSK-3beta protein.", id="s"),
    pytest.param("Identify additives that improve polymer thermal stability.", id="additives"),
]


def _frame(source: str, *, draft: bool = True, formulation: str | None = None) -> dict:
    frame = ResearchFrame.blank(source)
    if draft:
        frame.statement_draft = StatementDraft.model_validate({"parts": [{
            "id": "DL-1",
            "kind": "deliverable",
            "formulation": formulation or source,
            "source": "user_request",
            "quote": source,
        }]})
    return frame.model_dump(mode="json")


class _ScriptedFrameModel(BaseLlm):
    model: str = "scripted-context-init"
    frames: list[dict | LlmResponse]
    calls: list[list[dict]] = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self.calls.append([
            content.model_dump(mode="json") for content in llm_request.contents
        ])
        index = len(self.calls) - 1
        # Bound retries so an accidental infinite loop fails immediately.
        if index >= len(self.frames):
            raise AssertionError("ContextInit requested an unexpected extra model pass")
        if isinstance(self.frames[index], LlmResponse):
            yield self.frames[index]
            return
        yield LlmResponse(content=types.Content(
            role="model",
            parts=[types.Part(text=json.dumps(self.frames[index], ensure_ascii=False))],
        ))


@pytest.fixture
def run_frame(monkeypatch, tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path / "graph"))
    monkeypatch.setattr("CoScientist.context_init.agent.get_research_graph", lambda _ctx: store)
    monkeypatch.setenv("OPIK_TRACK_DISABLE", "true")

    def run(source, frame, *, initial_state=None, correction=None):
        frames = [frame] if correction is None else [frame, correction]
        model = _ScriptedFrameModel(frames=frames)
        before_model_states = []

        def record_before_model(callback_context, llm_request):
            before_model_states.append(callback_context.state.get(FRAME_STATE_KEY))

        agent = ContextInitSessionAgent(
            name="ContextInitAgent", model=model, output_key=FRAME_STATE_KEY,
            output_schema=ResearchFrame,
            instruction="Return a complete ResearchFrame JSON with statement_draft.",
            before_model_callback=record_before_model,
        )
        service = InMemorySessionService()
        runner = Runner(agent=agent, app_name="context-events", session_service=service)

        async def go():
            session = await service.create_session(
                app_name="context-events", user_id="operator", state=initial_state or {},
            )
            assert FRAME_STATE_KEY not in session.state
            events = [event async for event in runner.run_async(
                user_id="operator", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text=source)]),
            )]
            saved = await service.get_session(
                app_name="context-events", user_id="operator", session_id=session.id,
            )
            return saved, events

        saved, events = asyncio.run(go())
        assert len(model.calls) == len(frames)
        assert before_model_states[0] is None
        return saved, events, store

    return run


@pytest.mark.parametrize("source", SOURCES)
def test_final_frame_and_catalog_survive_adk_persistence(run_frame, source):
    saved, events, store = run_frame(source, _frame(source), initial_state={
        "research_frame_initialization_error": "previous failed attempt",
    })
    assert any(FRAME_STATE_KEY in event.actions.state_delta for event in events)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    assert saved.state[FRAME_STATE_KEY]["original_request"] == source
    statement = saved.state["normalized_statement"]
    assert statement["source_request"] == source
    assert statement["parts"][0]["provenance"]["quote"] == source
    assert saved.state["requirement_refs"][0]["id"] == "DL-1"
    persisted = store.full()
    assert any((node.get("attrs") or {}).get("stable_id") == "DL-1"
               for node in persisted["nodes"])
    assert saved.state["research_frame_initialization_error"] == ""


@pytest.mark.parametrize("source", SOURCES)
def test_missing_draft_is_saved_as_uncertain_not_initialized(run_frame, source):
    saved, _events, _store = run_frame(source, _frame(source, draft=False), correction=_frame(source, draft=False))
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is False
    assert saved.state["normalized_statement"]["uncertain"] is True
    assert saved.state["requirement_refs"] == []
    assert saved.state["research_frame_initialization_error"]


@pytest.mark.parametrize("failure_at", ["get_research_graph", "seed_frame", "commit_statement"])
def test_storage_failure_is_persisted_and_keeps_initialization_incomplete(run_frame, monkeypatch, failure_at):
    reason = "storage write denied"

    def fail(*_args, **_kwargs):
        raise RuntimeError(reason)

    module = "CoScientist.requirements.graph_io" if failure_at == "commit_statement" else "CoScientist.context_init.agent"
    monkeypatch.setattr(f"{module}.{failure_at}", fail)
    source = "Return the measurements archive."
    saved, _events, _store = run_frame(source, _frame(source))
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is False
    assert reason in saved.state["research_frame_initialization_error"]


def test_accepted_frame_field_becomes_a_persisted_obligation(run_frame):
    source = "Уточню результат в форме"
    frame = ResearchFrame.blank(source)
    field = next(f for f in frame.block("Основание и приёмка").fields if f.name == "deliverables")
    field.value, field.status = "Архив", "уточнено оператором"
    frame.statement_draft = StatementDraft.model_validate({"parts": [{
        "id": "DL-1", "kind": "deliverable", "formulation": "Архив",
        "source": "frame_field", "quote": "Архив",
        "field_id": "Основание и приёмка.deliverables",
    }]})
    saved, _events, _store = run_frame(source, frame.model_dump(mode="json"))
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    part = saved.state["requirement_refs"][0]
    assert part["obligation"] is True
    assert part["provenance"]["accepted"] is True
    assert part["provenance"]["quote"] == "Архив"


def test_criteria_and_separate_deliverables_survive_persistence(run_frame):
    source = "Нужен образец с выходом не ниже 80 %. Нужна методика одностадийного синтеза из массовых реагентов."
    frame = ResearchFrame.blank(source)
    frame.statement_draft = StatementDraft.model_validate({"parts": [
        {"id": "sample", "kind": "deliverable", "formulation": "Образец",
         "source": "user_request", "quote": "Нужен образец с выходом не ниже 80 %",
         "physical_sample": True, "criteria": [{
             "text": "Выход не ниже 80 %", "origin": "user",
             "quote": "не ниже 80 %", "threshold": "не ниже 80 %"}]},
        {"id": "method", "kind": "deliverable", "formulation": "Методика",
         "source": "user_request", "quote": "Нужна методика одностадийного синтеза из массовых реагентов",
         "protocol_only": True, "criteria": [{
             "text": "Одностадийный синтез из массовых реагентов", "origin": "user",
             "quote": "одностадийного синтеза из массовых реагентов"}]},
    ]})
    saved, _events, store = run_frame(source, frame.model_dump(mode="json"))
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    parts = saved.state["normalized_statement"]["parts"]
    assert [part["id"] for part in parts] == ["sample", "method"]
    assert parts[0]["physical_sample"] is True
    assert parts[0]["criteria"][0]["threshold"] == "не ниже 80 %"
    assert parts[1]["protocol_only"] is True
    assert parts[1]["criteria"][0]["quote"] == "одностадийного синтеза из массовых реагентов"
    persisted = next(node["attrs"]["normalized_statement"] for node in store.full()["nodes"]
                     if "normalized_statement" in node.get("attrs", {}))
    assert persisted["parts"] == parts


def _qualitative_frame(source):
    """A corrected semantic draft preserves broad outcomes without inventing numbers."""
    frame = _frame(source)
    part = frame["statement_draft"]["parts"][0]
    if source.startswith("Synthesize"):
        part["physical_sample"] = True
        part["criteria"] = [
            {"text": "The activator is novel", "origin": "user", "quote": "novel",
             "threshold": None},
            {"text": "The activator has cellular specificity", "origin": "user",
             "quote": "cellular specificity", "threshold": None},
        ]
    else:
        part["criteria"] = [
            {"text": "The molecules can bind to GSK-3beta protein", "origin": "user",
             "quote": "bind to GSK-3beta protein", "threshold": None},
        ]
    frame["statement_draft"]["uncertainty"] = ""
    frame["statement_draft"]["conditions"] = []
    return ResearchFrame.model_validate(frame).model_dump(mode="json")


@pytest.mark.parametrize("source", SOURCES[:2])
def test_known_qualitative_outcome_does_not_require_invented_parameters(source):
    from CoScientist.requirements.routing import normalize_statement

    frame = _qualitative_frame(source)
    statement = normalize_statement(source, frame["statement_draft"])
    assert statement.uncertain is False
    assert statement.uncertainty == ""
    assert not statement.rejected
    assert len(statement.obligations()) == 1
    assert statement.parts[0].kind == "deliverable"
    assert statement.parts[0].hypothesis_mode is None
    assert statement.parts[0].protocol_only is False
    assert statement.parts[0].criteria
    assert all(criterion.threshold is None for criterion in statement.parts[0].criteria)
    assert statement.volume.requested is None
    assert statement.volume.run_cap is None
    assert statement.volume.input_count is None
    assert statement.volume.limit_scope == ""
    assert statement.conditions == []


def test_an_unknown_requested_outcome_remains_uncertain(run_frame):
    source = "Help with my research."
    reason = "The requested research outcome has not been specified."
    frame = ResearchFrame.blank(source)
    frame.statement_draft = StatementDraft(parts=[], clarification={"question": reason, "quote": source})
    saved, _events, _store = run_frame(source, frame.model_dump(mode="json"))
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is False
    assert saved.state["normalized_statement"]["uncertainty"] == reason
    assert saved.state["requirement_refs"] == []
    assert saved.state["research_frame_initialization_error"] == reason


@pytest.mark.parametrize("source", SOURCES[:2])
def test_qualitative_catalog_can_initialize_with_open_frame_fields(run_frame, source):
    frame = _qualitative_frame(source)
    assert all(field["status"] == "не задано"
               for block in frame["blocks"] for field in block["fields"])
    saved, _events, _store = run_frame(source, frame)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    statement = saved.state["normalized_statement"]
    assert statement["volume"]["requested"] is None
    assert statement["conditions"] == []


@pytest.mark.parametrize("draft", [True, False])
def test_pipeline_runs_computation_only_after_successful_initialization(monkeypatch, tmp_path, draft):
    from CoScientist.assembly.assembler import ResearchPipeline
    from google.adk.agents import BaseAgent
    from google.adk.events import Event, EventActions

    class Computation(BaseAgent):
        async def _run_async_impl(self, ctx):
            yield Event(author=self.name, actions=EventActions(state_delta={"computation_started": True}))

    source = "Return the measurements archive."
    store = ResearchGraphStore(directory=str(tmp_path / "graph"))
    monkeypatch.setattr("CoScientist.context_init.agent.get_research_graph", lambda _ctx: store)
    model = _ScriptedFrameModel(frames=[_frame(source, draft=draft)] * (1 if draft else 2))
    initializer = ContextInitSessionAgent(name="ContextInitAgent", model=model,
        output_key=FRAME_STATE_KEY, output_schema=ResearchFrame, instruction="Return ResearchFrame JSON.")
    pipeline = ResearchPipeline(name="pipeline", sub_agents=[initializer, Computation(name="computation")])
    service = InMemorySessionService()
    runner = Runner(agent=pipeline, app_name="pipeline-test", session_service=service)

    async def go():
        session = await service.create_session(app_name="pipeline-test", user_id="operator", state={
            "research_frame_initialization_error": "old error",
        })
        events = [event async for event in runner.run_async(user_id="operator", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=source)]))]
        saved = await service.get_session(app_name="pipeline-test", user_id="operator", session_id=session.id)
        return saved, events

    saved, events = asyncio.run(go())
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is draft
    assert bool(saved.state.get("computation_started")) is draft
    if not draft:
        assert saved.state["research_frame_initialization_error"]
        assert any(event.content and any("initialization incomplete" in (part.text or "")
                   for part in event.content.parts) for event in events)


def test_planning_notes_are_saved_without_blocking_initialization(run_frame):
    source = "Find candidate molecules that bind to protein X."
    frame = _frame(source)
    notes = ["The number of candidates and the binding evaluation method must be selected in the plan."]
    frame["statement_draft"]["planning_notes"] = notes
    saved, _events, store = run_frame(source, frame)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    assert saved.state["normalized_statement"]["planning_notes"] == notes
    assert not saved.state["normalized_statement"]["uncertain"]
    assert any(n.get("attrs", {}).get("normalized_statement", {}).get("planning_notes") == notes
               for n in store.full()["nodes"])


def test_invalid_counts_are_repaired_once_without_losing_the_condition(run_frame):
    source = "Generate candidates. If more than 40 are requested, generate 12; keep supplied inputs intact."
    bad = _frame(source)
    bad["statement_draft"]["volume"] = {"requested": 40, "run_cap": 12, "input_count": 0}
    good = _frame(source)
    clause = "If more than 40 are requested, generate 12; keep supplied inputs intact."
    good["statement_draft"]["conditions"] = [{"text": clause, "quote": clause}]
    saved, events, _store = run_frame(source, bad, correction=good)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    statement = saved.state["normalized_statement"]
    assert statement["volume"]["requested"] is None
    assert statement["volume"]["input_count"] is None
    assert statement["conditions"] == [{"text": clause, "quote": clause}]
    feedback = [e for e in events if e.content and any("Validation errors:" in (p.text or "") for p in e.content.parts)]
    assert len(feedback) == 1
    assert feedback[0].author == "ContextInitAgent"
    assert "count lacks a source quote" in feedback[0].content.parts[0].text


def test_invalid_counts_still_block_when_the_single_repair_fails(run_frame):
    source = "Return candidate molecules."
    bad = _frame(source)
    bad["statement_draft"]["volume"] = {"requested": 100}
    saved, _events, _store = run_frame(source, bad, correction=bad)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is False
    assert saved.state["normalized_statement"]["rejected"]
    assert saved.state["research_frame_initialization_error"]


def test_legacy_uncertainty_requires_repair_instead_of_silent_acceptance(run_frame):
    source = "Return candidate molecules."
    old = _frame(source)
    old["statement_draft"].pop("clarification")
    old["statement_draft"]["uncertainty"] = "The requested number is unspecified."
    good = _frame(source)
    good["statement_draft"]["planning_notes"] = ["Choose and document the number of candidates."]
    saved, _events, _store = run_frame(source, old, correction=good)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    assert saved.state["normalized_statement"]["planning_notes"]


def test_contradictory_catalog_and_clarification_get_one_structural_repair(run_frame):
    source = "Find materials resistant to heat."
    bad = _frame(source)
    bad["statement_draft"]["clarification"] = {"question": "Which measurement method?", "quote": source}
    good = _frame(source)
    good["statement_draft"]["planning_notes"] = ["Choose a measurement method."]
    saved, _events, _store = run_frame(source, bad, correction=good)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    assert saved.state["normalized_statement"]["planning_notes"] == ["Choose a measurement method."]


def test_model_cannot_rewrite_the_source_to_authorize_a_count(run_frame):
    source = "Return candidate molecules."
    bad = _frame(source)
    bad["original_request"] = "Return 100 candidate molecules."
    bad["statement_draft"]["volume"] = {"requested": 100, "requested_quote": "Return 100 candidate molecules."}
    saved, _events, _store = run_frame(source, bad, correction=bad)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is False
    assert saved.state["normalized_statement"]["source_request"] == source
    assert saved.state["normalized_statement"]["rejected"]


@pytest.mark.parametrize("source,text,threshold", [
    ("Оцени стоимость для трёх соединений.", "Стоимость для три соединений", "три"),
    ("Find molecules that bind to X.", "Binding score below -7", "-7"),
])
def test_literal_user_criterion_survives_an_unsupported_threshold(run_frame, source, text, threshold):
    frame = _frame(source)
    frame["statement_draft"]["parts"][0]["criteria"] = [{
        "id": "C1", "text": text, "origin": "user", "quote": source, "threshold": threshold,
    }]
    saved, _events, _store = run_frame(source, frame)
    assert saved.state[FRAME_COMPLETED_STATE_KEY] is True
    criterion = saved.state["normalized_statement"]["parts"][0]["criteria"][0]
    assert criterion["text"] == source
    assert criterion["obligation"] is True
    assert criterion["threshold"] is None
    assert not saved.state["normalized_statement"]["rejected"]
