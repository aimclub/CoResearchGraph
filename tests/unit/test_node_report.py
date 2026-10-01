"""What one node of the research graph established, and who established it.

The design rests on two rules, and both are tested here rather than described:
the FACTS come from the record and the model only writes the joins, and an
agent's own account is cited rather than copied.
"""
from __future__ import annotations

import asyncio

import pytest

from CoScientist.graph import summary_store
from CoScientist.graph.research.store import ResearchGraphStore
from CoScientist.reporting import node_report


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    s = ResearchGraphStore(directory=str(tmp_path))
    s.init_research(source="OrchestratorAgent",
                    question="Are citrus furanocoumarins toxic?")
    s._scope = ("u", "s")
    return s


def _evidence(store, **attrs):
    base = {"subtype": "literature", "content": "toxic above 5 mg/kg",
            "source_ref": "PMC12610272; 10.3390/plants14213253"}
    base.update(attrs)
    store.commit(source="ResearchAgent", exec_id="agent:ResearchAgent@t1",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": base}])
    store.add_contributors([
        {"node_id": "E1", "agent": "OrchestratorAgent", "basis": "delegation",
         "exec_id": "agent:OrchestratorAgent@t1"},
        {"node_id": "E1", "agent": "PlannerAgent", "basis": "assignee"},
    ])
    return "E1"


def _facts(store, node_id="E1", summaries=None, lang="ru"):
    return node_report.build_facts(store.to_view(), node_id,
                                   summaries=summaries, lang=lang)


def _answers(monkeypatch, text="Связующий текст."):
    async def _complete(system, user):
        return text, "tiny-model"

    from CoScientist.graph import agent_summary
    monkeypatch.setattr(agent_summary, "_complete", _complete)


# ── the facts come from the record ──────────────────────────────────────────
def test_the_skeleton_states_the_record_with_no_model_at_all(store):
    node_id = _evidence(store)
    text = node_report.render_skeleton(_facts(store, node_id), "ru")

    assert "E1" in text
    assert "toxic above 5 mg/kg" in text
    # Who took part, and on what basis — the distinction the record exists for.
    assert "ResearchAgent" in text and "OrchestratorAgent" in text
    assert "PlannerAgent" in text
    assert "назначен планом, не подтверждено" in text
    # Both identifiers of a compound citation, as links.
    assert "https://doi.org/10.3390/plants14213253" in text
    assert "PMC12610272" in text


def test_a_planned_assignee_is_never_written_as_an_executor(store):
    _evidence(store)
    text = node_report.render_skeleton(_facts(store), "ru")
    planned = [line for line in text.splitlines() if "PlannerAgent" in line]
    assert planned and all("не подтверждено" in line for line in planned)


def test_the_files_of_a_node_are_named(store, tmp_path):
    from CoScientist.reporting import session_files as sf

    record = sf.put_bytes(("u", "s"), b"%PDF-1.7\nx" * 40, filename="furano.pdf",
                          media_type="application/pdf", source_kind="paper")
    _evidence(store, session_artifact_id=record["artifact_id"])
    text = node_report.render_skeleton(_facts(store), "ru")
    assert "furano.pdf" in text
    assert f"/artifacts/{record['artifact_id']}" in text


# ── the model writes only the joins ─────────────────────────────────────────
def test_a_model_that_fails_still_publishes_the_facts(store, monkeypatch):
    """There is no state in which this publishes an invention."""
    from CoScientist.graph import agent_summary

    async def _explode(system, user):
        raise RuntimeError("the model is down")

    monkeypatch.setattr(agent_summary, "_complete", _explode)
    node_id = _evidence(store)

    written = asyncio.run(node_report.write_report(
        store.to_view(), node_id, scope=("u", "s"), store=store, lang="ru"))

    assert "toxic above 5 mg/kg" in written["report"]
    assert written["artifact_id"], "and it is still published"


def test_the_narrative_is_added_to_the_facts_never_instead_of_them(
        store, monkeypatch):
    _answers(monkeypatch, "Узел устанавливает токсичность при УФ.")
    node_id = _evidence(store)

    written = asyncio.run(node_report.write_report(
        store.to_view(), node_id, scope=("u", "s"), store=store, lang="ru"))

    assert "toxic above 5 mg/kg" in written["report"], "the skeleton survives"
    assert "Узел устанавливает" in written["report"]
    assert written["report"].index("toxic above 5 mg/kg") < \
        written["report"].index("Узел устанавливает"), "facts first"


# ── an account is cited, not copied ─────────────────────────────────────────
def test_a_contributors_account_is_cited_with_a_link_not_copied(store, tmp_path):
    long_account = "СТРОКА. " * 200
    summary_store.put(("u", "s"), "agent:OrchestratorAgent@t1", lang="ru",
                      trace_stamp="abc", summary=long_account)
    _evidence(store)

    text = node_report.render_skeleton(
        _facts(store, summaries=summary_store.for_session(("u", "s"))), "ru")

    assert "agent:OrchestratorAgent@t1" in text, "the run is named, so it opens"
    assert len(text) < len(long_account), (
        "an excerpt — one document per thing, joined by the run id")


def test_a_contributor_with_no_account_yet_is_said_so(store):
    _evidence(store)
    text = node_report.render_skeleton(
        _facts(store, summaries=summary_store.for_session(("u", "s"))), "ru")
    assert "ещё не написан" in text


# ── the node points at its write-up ─────────────────────────────────────────
def test_the_node_is_stamped_and_the_body_reaches_the_panel(store, monkeypatch):
    _answers(monkeypatch)
    node_id = _evidence(store)

    written = asyncio.run(node_report.write_report(
        store.to_view(), node_id, scope=("u", "s"), store=store, lang="ru"))

    attrs = store._g.nodes[node_id]["attrs"]
    assert attrs["report_artifact_id"] == written["artifact_id"]
    assert attrs["report_stamp"] == written["stamp"]
    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    assert "toxic above 5 mg/kg" in drawn[node_id]["report"], (
        "the browser cannot fetch an artifact's text, so the server inlines it")


def test_the_body_never_goes_into_the_attributes(store, monkeypatch):
    """`_truncate_attrs` caps an ordinary attribute at 2 000 characters, and
    every card rides along on a 1.5-second poll."""
    _answers(monkeypatch)
    _evidence(store)
    asyncio.run(node_report.write_report(store.to_view(), "E1", scope=("u", "s"),
                                         store=store, lang="ru"))
    attrs = store._g.nodes["E1"]["attrs"]
    assert not any(len(str(v)) > 500 for v in attrs.values()), sorted(attrs)


def test_the_writer_may_write_nothing_but_its_own_fields(store):
    """Its rights are stated a (type, attribute) pair at a time."""
    _evidence(store)
    refused = store.commit(source=node_report.SOURCE, allow_reserved=True,
                           nodes=[{"id": "E1", "attrs": {"content": "rewritten"}}])
    assert not refused.ok
    assert store._g.nodes["E1"]["attrs"]["content"] == "toxic above 5 mg/kg"


# ── staleness ───────────────────────────────────────────────────────────────
def test_a_report_is_not_stale_the_moment_it_is_written(store, monkeypatch):
    """The easiest way to ship this broken.

    Writing the report puts `report_artifact_id` into attrs and moves
    `updated_at`, so a stamp over either would declare the report stale in the
    same breath as writing it.
    """
    _answers(monkeypatch)
    _evidence(store)
    before_write = store._g.nodes["E1"]["updated_at"]

    first = asyncio.run(node_report.write_report(
        store.to_view(), "E1", scope=("u", "s"), store=store, lang="ru"))

    # The hazard is real, not hypothetical: writing the report DID move the
    # node's clock and DID add attributes to it.
    assert store._g.nodes["E1"]["updated_at"] > before_write
    assert "report_artifact_id" in store._g.nodes["E1"]["attrs"]

    # And the stamp is over a named list that excludes both, so it did not move.
    again = node_report.stamp(_facts(store), "ru")
    assert again == first["stamp"]
    facts = _facts(store)
    assert "updated_at" not in facts and "t_end" not in facts, (
        "the node's clock must not be an input to its own staleness")
    assert not any(k.startswith("report_") for k in facts["fields"]), (
        "nor may the report's own attributes be")


def test_work_on_the_node_makes_its_report_stale(store, monkeypatch):
    _answers(monkeypatch)
    _evidence(store)
    before = node_report.stamp(_facts(store), "ru")
    store.add_contributors([{"node_id": "E1", "agent": "CoderAgent",
                             "basis": "work_order"}])
    assert node_report.stamp(_facts(store), "ru") != before


def test_correcting_the_nodes_own_text_makes_its_report_stale(store):
    """The bigger half of staleness, and the half left out at first.

    The stamp covered what HAPPENED to a node — its status, its history, who
    took part — and not what the node SAYS. So an Evidence whose text was
    corrected kept its stamp, and `write_report` answered with the old write-up,
    about the old text, marked current.
    """
    _evidence(store)
    before = node_report.stamp(_facts(store), "ru")
    store.commit(source="ResearchAgent",
                 nodes=[{"id": "E1", "attrs": {"content": "toxic above 50 mg/kg"}}])
    assert node_report.stamp(_facts(store), "ru") != before


def test_the_view_says_when_a_report_no_longer_describes_its_node(store,
                                                                  monkeypatch):
    """The panel cannot work this out for itself, and never could.

    The only stamp the browser ever held was the one the write RETURNED — the
    same value the node carries — so its staleness test compared a value with
    itself and could come out true only if publishing had failed.
    """
    _answers(monkeypatch)
    _evidence(store)
    asyncio.run(node_report.write_report(
        store.to_view(), "E1", scope=("u", "s"), store=store, lang="ru"))

    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    assert drawn["E1"]["report"], "written"
    assert drawn["E1"]["report_stale"] is False, "and current the moment it was"

    store.commit(source="ResearchAgent",
                 nodes=[{"id": "E1", "attrs": {"content": "toxic above 50 mg/kg"}}])
    drawn = {n["id"]: n for n in store.to_view()["nodes"]}
    assert drawn["E1"]["report_stale"] is True
    assert drawn["E1"]["report"], "and the old account is still shown, not hidden"


def test_each_language_is_its_own_report(store):
    _evidence(store)
    assert node_report.stamp(_facts(store), "ru") != \
        node_report.stamp(_facts(store, lang="en"), "en")


# ── which nodes get one ─────────────────────────────────────────────────────
def test_the_write_up_follows_the_sessions_language_not_the_browsers(
        tmp_path, monkeypatch):
    """A study is written in one language, and a node holds ONE write-up.

    The panel sends its own interface language. Two readers set differently
    would take turns overwriting each other's copy in that single slot — each
    paying for a model call to do it — and every other reader would meet
    whichever of them wrote last.
    """
    import uuid

    monkeypatch.setenv("GRAPH_SNAPSHOT_DIR", str(tmp_path / "graphs"))
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web"))
    monkeypatch.setenv("ARTIFACTS__MIRROR_TO_S3", "False")
    from starlette.testclient import TestClient

    from CoScientist.graph.research.store import get_research_graph
    from CoScientist.web.app import create_app

    asked = []

    async def _fake(view, node_id, *, scope, store, summaries=None, lang="ru",
                    force=False):
        asked.append(lang)
        return {"report": "…", "artifact_id": "", "stamp": "", "cached": False}

    monkeypatch.setattr(node_report, "write_report", _fake)

    with TestClient(create_app()) as client:
        user = client.post("/api/users", json={
            "nickname": f"t-{uuid.uuid4().hex[:8]}"}).json()["user"]
        session = client.post(f"/api/users/{user['id']}/sessions",
                              json={"title": "run"}).json()["session"]
        graph = get_research_graph(user_id=user["id"], session_id=session["id"])
        graph.init_research(source="OrchestratorAgent", question="Toxic?")
        graph.commit(source="ResearchAgent", nodes=[{
            "type": "Evidence", "ref": "e",
            "attrs": {"subtype": "literature", "content": "toxic above 5 mg/kg"}}])
        client.app.state.runtime.report_languages[
            (user["id"], session["id"])] = "en"

        r = client.post(
            f"/api/users/{user['id']}/sessions/{session['id']}/graph/node_report",
            json={"node_id": "E1", "lang": "ru"})
        assert r.status_code == 200, r.text

    assert asked == ["en"], "the session's choice, not the browser's"


def test_a_kind_that_has_no_report_is_refused(store):
    _evidence(store)
    view = store.to_view()
    assert node_report._reportable(view, "E1")
    root = next(n["id"] for n in view["nodes"] if n["kind"] == "researchquestion")
    assert not node_report._reportable(view, root)


def test_a_node_that_is_not_there_is_not_an_error(store):
    assert node_report.build_facts(store.to_view(), "E99") is None


def test_the_write_up_stays_out_of_the_chat_document_list(store, monkeypatch):
    """The main page is unchanged: a node's report belongs to the node."""
    from CoScientist.reporting import session_files as sf

    _answers(monkeypatch)
    _evidence(store)
    written = asyncio.run(node_report.write_report(
        store.to_view(), "E1", scope=("u", "s"), store=store, lang="ru"))

    record = sf.load_manifest("s", "u")[written["artifact_id"]]
    assert record["source_kind"] == "node_report"
    assert not record["source_kind"].startswith("doc:"), (
        "the chat list filters on that prefix")
