"""Context-initialization pre-stage: research frame, gap detection, graph seeding.

The 6-layer meta-model is already the research-graph schema; this stage FILLS the
framing entities each run. These tests cover the frame model, the structured-form
round-trip, the privileged graph seeding (with human-vs-agent provenance), and the
schema permissions for the new ContextInitAgent writer.
"""
from collections import Counter

from CoScientist.context_init.agent import (
    apply_form_values,
    coerce_frame,
    frame_to_form,
)
from CoScientist.context_init.commit import frame_to_init_kwargs, seed_frame
from CoScientist.context_init.models import CANONICAL_FRAME_BLOCKS, ResearchFrame
from CoScientist.graph.research import schema
from CoScientist.graph.research.store import ResearchGraphStore


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


def test_coerce_frame_accepts_dict_and_json():
    f = _filled_frame()
    assert isinstance(coerce_frame(f.model_dump()), ResearchFrame)
    assert isinstance(coerce_frame(f.model_dump_json()), ResearchFrame)


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
    assert schema.validate_node_draft(
        "ContextInitAgent", "Hypothesis", "formulated", {})
    assert schema.validate_edge(
        "ContextInitAgent", "supports", "Evidence", "Hypothesis")
