"""Context-initialization pre-stage: research frame, gap detection, graph seeding.

The 6-layer meta-model is already the research-graph schema; this stage FILLS the
framing entities each run. These tests cover the frame model, the structured-form
round-trip, the privileged graph seeding (with human-vs-agent provenance), and the
schema permissions for the new ContextInitAgent writer.
"""
from collections import Counter

from CoScientist.context_init.agent import (
    FRAME_COMPLETED_STATE_KEY,
    FRAME_STATE_KEY,
    ContextInitSessionAgent,
    apply_form_values,
    apply_review_response,
    coerce_frame,
    frame_is_initialized,
    frame_to_form,
)
from CoScientist.context_init.commit import frame_to_init_kwargs, seed_frame
from CoScientist.context_init.models import CANONICAL_FRAME_BLOCKS, ResearchFrame
from CoScientist.graph.research import schema
from CoScientist.graph.research.store import ResearchGraphStore
from CoScientist.hitl.models import (
    HITLAction,
    HITLDecisionSource,
    HITLResponse,
)


def _filled_frame() -> ResearchFrame:
    f = ResearchFrame.blank("Does drug X reduce tumor growth?")
    q = f.block("Вопрос исследования")
    for fld in q.fields:
        if fld.name == "formulation":
            fld.value, fld.status = "Does drug X reduce tumor growth?", "задано заказчиком"
        if fld.name == "domain":
            fld.value, fld.status = "oncology", "уточнено оператором"
    eth = f.block("Этика и регуляторика")
    eth.fields[0].value, eth.fields[0].status = "IRB approval", "уточнено оператором"
    res = f.block("Ресурсы и бюджеты")
    res.fields[0].value, res.fields[0].status = "100 / 100", "задано заказчиком"
    cc = f.block("Условия подтверждения")
    cc.fields[0].value, cc.fields[0].status = "p<0.05", "уточнено оператором"
    cm = f.block("Модель стоимости")
    cm.fields[0].value, cm.fields[0].status = "GPU-hours * rate", "уточнено оператором"
    return f


# ── frame model + gap detection ───────────────────────────────────────────────

def test_blank_frame_has_every_canonical_block_all_open():
    f = ResearchFrame.blank("q")
    assert [b.title for b in f.blocks] == list(CANONICAL_FRAME_BLOCKS)
    # every field is open in a blank frame
    assert all(not fld.is_set() for b in f.blocks for fld in b.fields)
    assert len(f.open_fields()) == sum(len(b.fields) for b in f.blocks)


def test_open_fields_shrink_as_fields_are_filled():
    f = _filled_frame()
    open_names = {name for _title, name in f.open_fields()}
    assert "formulation" not in open_names   # filled
    assert "trl" in open_names               # still open


def test_normalized_reimposes_structure_and_keeps_values():
    f = ResearchFrame.blank("q")
    # an LLM that dropped a block and added a junk one
    f.blocks = [b for b in f.blocks if b.title != "Инструменты"]
    f.blocks.append(type(f.blocks[0])(title="Мусор", fields=[]))
    q = f.block("Вопрос исследования")
    q.fields[0].value, q.fields[0].status = "kept", "задано заказчиком"
    n = f.normalized()
    assert [b.title for b in n.blocks] == list(CANONICAL_FRAME_BLOCKS)  # junk gone, block back
    assert n.block("Вопрос исследования").fields[0].value == "kept"     # value preserved


# ── structured form round-trip ────────────────────────────────────────────────

def test_frame_to_form_marks_open_fields():
    form = frame_to_form(_filled_frame())
    assert form["blocks"][0]["title"] == "Вопрос исследования"
    by_name = {fld["name"]: fld for fld in form["blocks"][0]["fields"]}
    assert by_name["formulation"]["open"] is False
    assert by_name["trl"]["open"] is True


def test_apply_form_values_sets_operator_status_and_is_soft():
    f = ResearchFrame.blank("q")
    out = apply_form_values(f, {"Вопрос исследования": {"domain": "physics"}})
    q = out.block("Вопрос исследования")
    domain = next(x for x in q.fields if x.name == "domain")
    assert (domain.value, domain.status) == ("physics", "уточнено оператором")
    # untouched fields stay open — the soft gate does not force every field
    assert next(x for x in q.fields if x.name == "trl").is_set() is False


def test_apply_form_values_none_is_noop():
    f = _filled_frame()
    assert apply_form_values(f, None).model_dump() == f.normalized().model_dump()


def test_auto_review_does_not_forge_operator_provenance():
    frame = ResearchFrame.blank("q")
    domain = next(
        field for field in frame.block("Вопрос исследования").fields
        if field.name == "domain"
    )
    domain.value = "physics"
    domain.status = "предложено агентом"
    response = HITLResponse(
        action=HITLAction.APPROVE,
        approved=True,
        form_values={"Вопрос исследования": {
            "domain": "physics",
            "trl": "Не задано",
        }},
        decision_source=HITLDecisionSource.MODE_AUTO,
    )

    reviewed = apply_review_response(frame, response)
    fields = {field.name: field for field in reviewed.block("Вопрос исследования").fields}
    assert fields["domain"].status == "предложено агентом"
    assert fields["trl"].status == "не задано"


def test_coerce_frame_accepts_dict_and_json():
    f = _filled_frame()
    assert isinstance(coerce_frame(f.model_dump()), ResearchFrame)
    assert isinstance(coerce_frame(f.model_dump_json()), ResearchFrame)


def test_frame_is_initialized_only_after_completion_marker():
    # A drafted frame alone may belong to an interrupted first turn, so it
    # cannot suppress the retry.  Only the post-seeding marker does.
    assert frame_is_initialized({"research_frame": _filled_frame().model_dump()}) is False
    assert frame_is_initialized({FRAME_COMPLETED_STATE_KEY: True}) is True


# ── privileged graph seeding ──────────────────────────────────────────────────

def test_seed_frame_writes_expected_nodes(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    result = seed_frame(store, _filled_frame())
    assert result["ok"] is True

    g = store.full_graph()
    types = Counter(d.get("type") for _n, d in g.nodes(data=True))
    assert types["ResearchQuestion"] == 1
    assert types["Constraint"] == 1
    assert types["Resource"] == 1
    assert types["ConfirmationCriteria"] == 1
    assert types["CostModel"] == 1


def test_seed_frame_provenance_and_edges(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    seed_frame(store, _filled_frame())
    g = store.full_graph()

    # operator/customer-set fields are attributed to the human, not the agent
    for _n, d in g.nodes(data=True):
        if d.get("type") in ("Constraint", "Resource", "CostModel"):
            assert d.get("source") == "human"

    # CostModel applies_to the root question; Constraint contextualizes it
    edge_types = {k for _u, _v, k in g.edges(keys=True)}
    assert "applies_to" in edge_types
    assert "contextualizes" in edge_types


def test_init_research_honors_per_node_source(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    out = store.init_research(
        source="ContextInitAgent",
        question="q",
        constraints=[{"subtype": "ethics", "content": "x", "source": "human"}],
        cost_models=[{"attrs": {"rule": "r"}}],  # no per-node source -> default
    )
    assert out["ok"] is True
    g = store.full_graph()
    sources = {d["type"]: d["source"] for _n, d in g.nodes(data=True)}
    assert sources["Constraint"] == "human"          # per-node override
    assert sources["CostModel"] == "ContextInitAgent"  # falls back to default


# ── schema permissions for the new writer ─────────────────────────────────────

def test_context_init_agent_may_write_its_frame():
    assert schema.validate_node_draft(
        "ContextInitAgent", "Constraint", "active",
        {"subtype": "ethics", "content": "x"}) == []
    assert schema.validate_node_draft(
        "ContextInitAgent", "CostModel", "created", {}) == []
    assert schema.validate_edge(
        "ContextInitAgent", "contextualizes", "Constraint", "ResearchQuestion") == []
    assert schema.validate_edge(
        "ContextInitAgent", "applies_to", "CostModel", "ResearchQuestion") == []


def test_context_init_agent_cannot_write_others_nodes():
    # A statement may record the hypothesis or deliverable it actually contains.
    # Evidence and a support edge stay with the agents that produce them.
    assert schema.validate_node_draft(
        "ContextInitAgent", "Hypothesis", "formulated", {"formulation": "x"}) == []
    assert schema.validate_node_draft(
        "ContextInitAgent", "Deliverable", "specified", {"formulation": "x"}) == []
    assert schema.validate_node_draft(
        "ContextInitAgent", "Evidence", "obtained", {"subtype": "literature"})
    assert schema.validate_edge(
        "ContextInitAgent", "supports", "Evidence", "Hypothesis")


def test_post_final_events_seeds_graph_and_emits_no_chat_message(tmp_path, monkeypatch):
    from types import SimpleNamespace
    store = ResearchGraphStore(directory=str(tmp_path))
    monkeypatch.setattr("CoScientist.context_init.agent.get_research_graph", lambda _ctx: store)

    frame = _filled_frame()
    from CoScientist.requirements.models import StatementDraft
    frame.statement_draft = StatementDraft.model_validate({"parts": [{
        "kind": "question", "formulation": frame.original_request,
        "source": "user_request", "quote": frame.original_request,
    }]})
    agent = ContextInitSessionAgent(name="ContextInitSessionAgent", output_key="research_frame")
    ctx = SimpleNamespace(
        session=SimpleNamespace(state={"research_frame": frame.model_dump()}),
        invocation_id="inv-42",
        branch="main",
    )

    events = list(agent._post_final_events(ctx, ""))
    assert len(events) == 1
    event = events[0]

    # Verify state delta is populated correctly
    assert event.actions.state_delta[FRAME_STATE_KEY]["original_request"] == frame.original_request
    assert event.actions.state_delta["normalized_statement"]["uncertain"] is False
    assert event.actions.state_delta[FRAME_COMPLETED_STATE_KEY] is True
    assert event.actions.state_delta["orchestrator_root_goal"] == frame.original_request

    # Verify graph is seeded
    g = store.full_graph()
    assert len(g.nodes) > 0

    # Verify no visible chat text is emitted
    assert event.content is None


def _post_events(frame, store, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("CoScientist.context_init.agent.get_research_graph", lambda _ctx: store)
    agent = ContextInitSessionAgent(name="ContextInitSessionAgent", output_key="research_frame")
    ctx = SimpleNamespace(session=SimpleNamespace(state={"research_frame": frame.model_dump()}),
                          invocation_id="test", branch="main")
    return list(agent._post_final_events(ctx, ""))[0].actions.state_delta


def test_missing_catalog_is_visible_and_does_not_mark_frame_initialized(tmp_path, monkeypatch):
    delta = _post_events(_filled_frame(), ResearchGraphStore(directory=str(tmp_path)), monkeypatch)
    assert delta["normalized_statement"]["uncertain"] is True
    assert delta["requirement_refs"] == []
    assert delta[FRAME_COMPLETED_STATE_KEY] is False


def test_failed_catalog_commit_does_not_mark_frame_initialized(tmp_path, monkeypatch):
    from CoScientist.requirements.models import StatementDraft
    frame = _filled_frame()
    frame.statement_draft = StatementDraft.model_validate({"parts": [{
        "kind": "question", "formulation": frame.original_request,
        "source": "user_request", "quote": frame.original_request,
    }]})
    monkeypatch.setattr("CoScientist.requirements.graph_io.commit_statement", lambda *_: {"ok": False})
    delta = _post_events(frame, ResearchGraphStore(directory=str(tmp_path)), monkeypatch)
    assert delta["normalized_statement"]["uncertain"] is False
    assert delta[FRAME_COMPLETED_STATE_KEY] is False


def test_context_init_prompt_and_schema_request_the_semantic_catalog():
    from CoScientist.agents.prompts.templates import context_init
    from CoScientist.requirements.prompt import STATEMENT_PROMPT
    prompt = context_init(None)
    assert STATEMENT_PROMPT in prompt
    assert '"statement_draft"' in prompt
    schema = ResearchFrame.model_json_schema()
    assert "StatementDraft" in schema["$defs"]
    assert "hypothesis_mode" in schema["$defs"]["StatementPartDraft"]["properties"]


def test_statement_repair_receives_the_rejected_draft_and_trusted_request():
    import json
    from types import SimpleNamespace
    from CoScientist.requirements.models import StatementDraft

    source = "Compare these two approaches."
    frame = ResearchFrame.blank(source)
    frame.statement_draft = StatementDraft.model_validate({"parts": [{
        "kind": "question", "formulation": source, "source": "user_request",
        "quote": source, "hypothesis_mode": "check",
    }], "volume": {"input_count": 2, "input_count_quote": "these two approaches"}})
    agent = ContextInitSessionAgent(name="ContextInitAgent", output_key="research_frame")
    state = {"research_frame": frame.model_dump()}
    ctx = SimpleNamespace(session=SimpleNamespace(state=state))
    feedback = agent._unfinished_feedback(ctx)
    assert "hypothesis_mode belongs only to a hypothesis" in feedback
    assert "count lacks a source quote containing its value" in feedback
    repair = json.loads(feedback.split("Draft to repair against its trusted source: ", 1)[1])
    assert repair["original_request"] == source
    assert repair["statement_draft"] == frame.statement_draft.model_dump()
    # The repair is model work; validation does not rewrite or accept the rejected draft.
    assert state["research_frame"] == frame.model_dump()
