"""Who took part in a node — recorded, not inferred, with the basis named.

The request behind this was explicit: do not create bugs where records get
confused between research-graph nodes. Reconnaissance found that "which agents
participated" was nowhere written down; three heuristics stood in for it, and
one of them is observably wrong — in a real run all seven ExperimentTasks were
attributed to the same plan step.

So participation is written at the moment it happens, and every row says on
what basis. A planned assignee is an intention; a commit, a status change, a
delegation, a tool call and a claimed work order are things the system watched
happen. The panel draws that difference, and these tests hold it.
"""
from __future__ import annotations

import pytest

from CoScientist.config import get_settings
from CoScientist.graph.research.store import ResearchGraphStore


@pytest.fixture
def store(tmp_path):
    return ResearchGraphStore(directory=str(tmp_path))


def _seeded(store):
    store.init_research(source="OrchestratorAgent",
                        question="Does compound X inhibit target Y?")
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h",
                         "attrs": {"formulation": "X inhibits Y at low dose"}}])
    return "H1"


def _rows(store, nid):
    return (store._g.nodes[nid].get("attrs") or {}).get("contributors") or []


# ── the bug this exists to prevent ──────────────────────────────────────────
def test_a_later_commit_does_not_erase_the_contributors(store):
    """The attrs merge is a shallow `{**stored, **incoming}`.

    Anything routed through it is replaced wholesale by the next writer that
    mentions the key — which for an append-only record means losing everyone
    but the last. That is why participation is not written through `commit`.
    """
    nid = _seeded(store)
    store.add_contributors([
        {"node_id": nid, "agent": "ResearchAgent", "basis": "provenance"},
        {"node_id": nid, "agent": "CoderAgent", "basis": "delegation"},
    ])
    # Three: the two just added, plus the agent that wrote the node.
    assert len(_rows(store, nid)) == 3

    store.commit(source="HypothesesAgent",
                 nodes=[{"id": nid, "attrs": {"rationale": "dose-response fits"}}])

    agents = {r["agent"] for r in _rows(store, nid)}
    assert agents == {"HypothesesAgent", "ResearchAgent", "CoderAgent"}, (
        "an ordinary edit kept every one of them")
    assert store._g.nodes[nid]["attrs"]["rationale"] == "dose-response fits"


def test_a_contributors_list_sent_by_an_agent_is_dropped(store):
    """An agent sees a node's attrs in its context slice and may echo them back.

    Dropped rather than refused: the rest of the commit is worth more than the
    key, and the caller is told what happened.
    """
    nid = _seeded(store)
    store.add_contributors([{"node_id": nid, "agent": "ResearchAgent",
                             "basis": "provenance"}])

    result = store.commit(source="HypothesesAgent", nodes=[{
        "id": nid,
        "attrs": {"rationale": "still fits",
                  "contributors": [{"agent": "ImaginaryAgent", "basis": "commit"}]}}])

    assert result.ok, result.errors
    assert any("contributors" in w for w in result.warnings), result.warnings
    agents = {r["agent"] for r in _rows(store, nid)}
    assert agents == {"HypothesesAgent", "ResearchAgent"}, (
        "the graph's own record stood and the invented agent never landed")
    assert store._g.nodes[nid]["attrs"]["rationale"] == "still fits"


def test_a_reserved_attribute_cannot_be_smuggled_in_on_a_new_node(store):
    """The create path is the other door into attrs."""
    _seeded(store)
    result = store.commit(source="ResearchAgent", nodes=[{
        "type": "Evidence", "ref": "e",
        "attrs": {"subtype": "literature", "content": "a finding",
                  "report_artifact_id": "not-yours.md",
                  "contributors": [{"agent": "ImaginaryAgent", "basis": "commit"}]}}])

    assert result.ok, result.errors
    attrs = store._g.nodes["E1"]["attrs"]
    assert "report_artifact_id" not in attrs
    assert {r["agent"] for r in attrs.get("contributors") or []} == {"ResearchAgent"}


# ── what the record says, and how it says it ────────────────────────────────
def test_writing_a_node_is_recorded_as_taking_part_in_it(store):
    nid = _seeded(store)
    rows = _rows(store, nid)
    assert [(r["agent"], r["basis"]) for r in rows] == [("HypothesesAgent", "commit")]


def test_moving_a_node_is_recorded_too(store, monkeypatch):
    monkeypatch.setattr(get_settings().web, "max_active_hypotheses", 1)
    nid = _seeded(store)
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": nid, "status": "under_verification"}])
    assert {(r["agent"], r["basis"]) for r in _rows(store, nid)} == {
        ("HypothesesAgent", "commit"), ("OrchestratorAgent", "status")}


def test_a_row_points_at_the_run_not_only_at_the_name(store):
    """An agent that worked on three nodes in one study is three runs.

    Without the run's id a row says only "ResearchAgent took part", and a reader
    following it lands on whatever that agent happened to do last — which for a
    long study is almost never the run that produced the node in front of them.
    """
    nid = _seeded(store)
    store.commit(source="ResearchAgent", exec_id="agent:ResearchAgent@turn-7",
                 nodes=[{"type": "Evidence", "ref": "e",
                         "attrs": {"subtype": "literature", "content": "a finding"}}])
    rows = _rows(store, "E1")
    assert [(r["agent"], r["basis"], r.get("exec_id")) for r in rows] == [
        ("ResearchAgent", "commit", "agent:ResearchAgent@turn-7")]


def test_the_same_participation_is_recorded_once(store):
    """A resolver may run again on the next commit; that is not a second act."""
    nid = _seeded(store)
    for _ in range(3):
        store.add_contributors([{"node_id": nid, "agent": "ResearchAgent",
                                 "basis": "delegation"}])
    assert sum(1 for r in _rows(store, nid) if r["basis"] == "delegation") == 1


def test_the_same_agent_on_two_bases_is_two_rows(store):
    """Delegating the work and making the call behind it are different acts."""
    nid = _seeded(store)
    store.add_contributors([
        {"node_id": nid, "agent": "ResearchAgent", "basis": "delegation"},
        {"node_id": nid, "agent": "ResearchAgent", "basis": "provenance"},
    ])
    assert len([r for r in _rows(store, nid) if r["agent"] == "ResearchAgent"]) == 2


def test_a_planned_assignee_is_not_an_observed_executor(store):
    """The whole point of storing a basis.

    `PlanStep.attrs.assignee` is who the plan named. Nothing observes that they
    did the work, and drawing them as an executor is exactly how a reader ends
    up believing the wrong agent did something.
    """
    _seeded(store)
    store.commit(source="plan-mirror", nodes=[{
        "type": "PlanStep", "ref": "ps",
        "attrs": {"title": "gather literature", "plan_task_id": "TASK-1",
                  "assignee": "ResearchAgent"}}])
    store.add_contributors([
        {"node_id": "PS1", "agent": "ResearchAgent", "basis": "assignee"},
        {"node_id": "PS1", "agent": "CoderAgent", "basis": "work_order"},
    ])
    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    shown = drawn["PS1"]["contributors"]
    assert [(c["agent"], c["observed"]) for c in shown] == [
        ("CoderAgent", True), ("ResearchAgent", False)], (
        "observed first, and the planned one marked as not observed")


def test_a_mirror_is_not_a_participant(store):
    """`plan-mirror` writes every PlanStep there is; crediting it says nothing."""
    _seeded(store)
    store.commit(source="plan-mirror", nodes=[{
        "type": "PlanStep", "ref": "ps",
        "attrs": {"title": "gather literature", "plan_task_id": "TASK-1"}}])
    assert _rows(store, "PS1") == []


def test_an_unknown_basis_is_not_recorded(store):
    """The vocabulary is closed: a row whose basis nobody can read is noise."""
    nid = _seeded(store)
    store.add_contributors([{"node_id": nid, "agent": "X", "basis": "vibes"},
                            {"node_id": "NOPE", "agent": "X", "basis": "commit"}])
    assert [r for r in _rows(store, nid) if r["agent"] == "X"] == []


# ── what must not leak ──────────────────────────────────────────────────────
def test_the_bookkeeping_does_not_reach_the_reader(store):
    nid = _seeded(store)
    store.add_contributors([{"node_id": nid, "agent": "ResearchAgent",
                             "basis": "delegation"}])
    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    assert "contributors" not in (drawn[nid]["input"] or {})
    assert "участ" not in drawn[nid]["label"].lower()


def test_a_context_slice_does_not_ship_the_bookkeeping(store):
    """A worker is told about its neighbours in 240 characters of attrs.

    Forty rows of participation would fill that window and hide what the node
    actually says — which is the only thing the worker needs from it.
    """
    nid = _seeded(store)
    store.add_contributors([{"node_id": nid, "agent": f"Agent{i}",
                             "basis": "delegation"} for i in range(8)])
    rendered = store.get_context_slice(nid)["rendered"]
    assert "contributors" not in rendered
    assert "X inhibits Y" in rendered


def test_bookkeeping_does_not_make_an_idle_study_look_worked_on(store):
    """`updated_at` is the card's end time and the study list sorts on it."""
    nid = _seeded(store)
    before = store._g.nodes[nid]["updated_at"]
    store.add_contributors([{"node_id": nid, "agent": "ResearchAgent",
                             "basis": "delegation"}])
    assert store._g.nodes[nid]["updated_at"] == before


def test_the_record_does_not_grow_without_bound(store):
    """A long run must not turn a node into a list of names."""
    nid = _seeded(store)
    store.add_contributors([{"node_id": nid, "agent": f"Agent{i}",
                             "basis": "delegation"} for i in range(60)])
    attrs = store._g.nodes[nid]["attrs"]
    assert len(attrs["contributors"]) == 40
    # 60 added on top of the author's own row; 40 kept, the rest counted.
    assert attrs["contributors_more"] == 21


def test_a_batch_writes_one_snapshot(store, monkeypatch):
    """`_save` rewrites the whole graph; once per batch, not once per row."""
    nid = _seeded(store)
    saves = []
    monkeypatch.setattr(store, "_save", lambda: saves.append(1))
    store.add_contributors([{"node_id": nid, "agent": f"Agent{i}",
                             "basis": "delegation"} for i in range(5)])
    assert len(saves) == 1


def test_recording_participation_survives_a_reload(store, tmp_path):
    nid = _seeded(store)
    store.add_contributors([{"node_id": nid, "agent": "ResearchAgent",
                             "basis": "provenance", "exec_id": "tool:call_7"}])
    reopened = ResearchGraphStore(directory=str(tmp_path))
    rows = (reopened._g.nodes[nid].get("attrs") or {}).get("contributors") or []
    assert [(r["agent"], r["basis"], r.get("exec_id")) for r in rows] == [
        ("HypothesesAgent", "commit", None),
        ("ResearchAgent", "provenance", "tool:call_7"),
    ]
