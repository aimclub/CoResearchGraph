"""graph_bridge: approved plan → VM/tested_by; result → Evidence/GeneratedData."""
from __future__ import annotations

from types import SimpleNamespace

from CoScientist.graph.research.store import ResearchGraphStore

from CoScientist.experiments.runtime.graph_bridge import (
    _XT_IDS_KEY as _XT_KEY,
)

from .helpers import NOW, _approved_state, _plan, _task


def _seeded_store(
    tmp_path,
    extra: list[dict] | None = None,
) -> ResearchGraphStore:
    """A research graph with a root question and Hypothesis H1 (+ optional extras).

    Extra hypotheses are committed one-by-one so the store's max_active
    selection does not auto-postpone them before the bridge runs.
    """
    store = ResearchGraphStore(directory=str(tmp_path))
    store.init_research(source="ContextInitAgent", question="Root question?")
    root = store.root_id()
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h1",
                "attrs": {"formulation": "Compound X inhibits target Y."}}],
        edges=[{"type": "motivates", "from": root, "to": "#h1"}],
        enforce_permissions=False,
    )
    for index, item in enumerate(extra or [], start=2):
        ref = f"h{index}"
        draft = {"type": "Hypothesis", "ref": ref,
                 "attrs": {"formulation": item.get("formulation") or f"Claim {index}."}}
        if item.get("status"):
            draft["status"] = item["status"]
        store.commit(
            source="HypothesesAgent",
            nodes=[draft],
            edges=[{"type": "motivates", "from": root, "to": f"#{ref}"}],
            enforce_permissions=False,
        )
    return store


def _node_status(store: ResearchGraphStore) -> dict[str, str]:
    return {n["id"]: n["status"] for n in store.full()["nodes"]}


def _nodes_by_type(store: ResearchGraphStore) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in store.full()["nodes"]:
        out.setdefault(node["type"], []).append(node["id"])
    return out


def _conditional_store(tmp_path) -> ResearchGraphStore:
    store = _seeded_store(
        tmp_path,
        extra=[{"formulation": "Fallback after H1 is refuted.", "status": "postponed"}],
    )
    linked = store.commit(
        source="HypothesesAgent", enforce_permissions=False,
        edges=[{"type": "conditional_successor", "from": "H1", "to": "H2",
                "attrs": {"required_status": "refuted"}}],
    )
    assert linked.ok, linked.errors
    return store


def test_publish_plan_creates_vm_and_tested_by(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("VerificationMethod", [])) == 1
    vm_id = by_type["VerificationMethod"][0]
    assert state["experiment_graph_vm_ids"]["EXP-1"] == vm_id
    edges = store.full()["edges"]
    assert any(e["type"] == "tested_by" and e["from"] == "H1" and e["to"] == vm_id
               for e in edges)
    # VerificationMethod node attrs must contain mcp_servers
    vm_node = next(n for n in store.full()["nodes"] if n["id"] == vm_id)
    assert "mcp_servers" in vm_node.get("attrs", {})
    mcp_servers = vm_node["attrs"]["mcp_servers"]
    assert len(mcp_servers) >= 1
    assert mcp_servers[0]["url"] == "http://127.0.0.1:8000/mcp"
    assert "estimate_property" in mcp_servers[0]["tools"]


def test_plan_publication_cannot_publish_a_locked_successor(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_plan_to_graph,
    )

    store = _conditional_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H2")))

    publish_plan_to_graph(store, state)
    publish_plan_detail_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert by_type.get("VerificationMethod") is None
    assert by_type.get("ExperimentTask") is None
    assert state.get("experiment_graph_vm_ids") is None
    assert state.get(_XT_KEY) is None


def test_automatic_evidence_is_not_attached_to_a_locked_successor(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _conditional_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)
    state["experiment_runtime"]["plan"]["tasks"][0]["design"]["also_tests"] = ["H2"]
    publish_result_to_graph(store, state, "EXP-1", {
        "result_id": "RES-chain",
        "status": "success",
        "summary": "Evidence for the active hypothesis only.",
        "artifacts": [],
    })

    evidence_id = _nodes_by_type(store)["Evidence"][0]
    linked = {edge["to"] for edge in store.full()["edges"]
              if edge["type"] == "relates_to" and edge["from"] == evidence_id}
    assert linked == {"H1"}
    assert _node_status(store)["H2"] == "postponed"


def test_publish_plan_is_idempotent(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)
    first_vm = dict(state["experiment_graph_vm_ids"])
    publish_plan_to_graph(store, state)

    assert _nodes_by_type(store).get("VerificationMethod") == [first_vm["EXP-1"]]
    assert state["experiment_graph_vm_ids"] == first_vm


def test_publish_result_writes_evidence_and_vm_status(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)
    vm_id = state["experiment_graph_vm_ids"]["EXP-1"]

    task_result = {
        "result_id": "RES-1",
        "status": "success",
        "summary": "EXP-1 produced a managed result.",
        "artifacts": [{"name": "metrics.json", "bucket": "b", "s3_key": "k/metrics.json"}],
    }
    publish_result_to_graph(store, state, "EXP-1", task_result)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("Evidence", [])) == 1
    assert len(by_type.get("GeneratedData", [])) == 1
    evidence_id = by_type["Evidence"][0]
    gd_id = by_type["GeneratedData"][0]
    edges = store.full()["edges"]
    assert any(e["type"] == "produces" and e["from"] == vm_id and e["to"] == evidence_id
               for e in edges)
    assert any(e["type"] == "derived_from" and e["from"] == gd_id and e["to"] == evidence_id
               for e in edges)
    assert any(e["type"] == "relates_to" and e["from"] == evidence_id and e["to"] == "H1"
               for e in edges)
    statuses = _node_status(store)
    assert statuses[vm_id] == "used"
    assert statuses["H1"] == "under_verification"


def test_publish_result_failure_is_received_but_cannot_judge_a_hypothesis(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)
    vm_id = state["experiment_graph_vm_ids"]["EXP-1"]

    publish_result_to_graph(store, state, "EXP-1", {
        "result_id": "RES-2", "status": "failure", "summary": "route failed",
        "error_message": "Docker host did not resolve",
    })

    by_type = _nodes_by_type(store)
    evidence_id = by_type["Evidence"][0]
    evidence = next(n for n in store.full()["nodes"] if n["id"] == evidence_id)
    assert evidence["status"] == "obtained"
    assert evidence["attrs"]["subtype"] == "meta"
    assert evidence["attrs"]["result_kind"] == "ошибка выполнения"
    assert "Docker host" in evidence["attrs"]["failure_reason"]
    edges = store.full()["edges"]
    assert any(e["type"] == "produces" and e["to"] == evidence_id for e in edges)
    assert not any(e["from"] == evidence_id and e["type"] in {
        "relates_to", "supports", "refutes", "refines",
    } for e in edges)
    vm_status = {n["id"]: n["status"] for n in store.full()["nodes"]}[vm_id]
    assert vm_status == "not_used"


def test_a_terminal_task_status_retries_one_refused_graph_write():
    from CoScientist.experiments.runtime.graph_bridge import _advance_task_card

    class _FlakyStore:
        def __init__(self):
            self.calls = 0

        def overview(self):
            return {"nodes": [{"id": "XT1", "type": "ExperimentTask",
                               "status": "running"}]}

        def commit(self, **_kwargs):
            self.calls += 1
            return SimpleNamespace(ok=self.calls == 2, errors=["temporary refusal"])

    graph = _FlakyStore()
    advanced = _advance_task_card(
        graph, {_XT_KEY: {"EXP-1": "XT1"}}, "EXP-1", "done", "success", {},
    )

    assert advanced is True
    assert graph.calls == 2


def test_publish_result_skips_when_no_vm(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_result_to_graph

    store = _seeded_store(tmp_path)
    state: dict = {}
    publish_result_to_graph(store, state, "EXP-1",
                            {"result_id": "RES-3", "status": "success", "summary": "x"})
    assert _nodes_by_type(store).get("Evidence") is None


def test_publish_plan_swallows_store_errors():
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    class _Boom:
        def overview(self):
            raise RuntimeError("graph down")

        def commit(self, *a, **k):
            raise RuntimeError("graph down")

    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    # Must not raise — best-effort contract.
    publish_plan_to_graph(_Boom(), state)
    state["experiment_graph_vm_ids"] = {"EXP-1": "VM1"}
    publish_result_to_graph(_Boom(), state, "EXP-1",
                            {"result_id": "RES-4", "status": "success", "summary": "x"})


def test_publish_result_links_also_tests(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )
    from .helpers import _design

    store = _seeded_store(
        tmp_path,
        extra=[{"formulation": "Secondary claim also tested by the same run."}],
    )
    design = _design("H1")
    design["also_tests"] = ["H2"]
    state = _approved_state(_plan(_task("EXP-1", design=design)))
    publish_plan_to_graph(store, state)

    publish_result_to_graph(store, state, "EXP-1", {
        "result_id": "RES-5",
        "status": "success",
        "summary": "Shared evidence.",
        "artifacts": [{"name": "out.csv", "bucket": "b", "s3_key": "k/out.csv"}],
    })

    evidence_id = _nodes_by_type(store)["Evidence"][0]
    edges = store.full()["edges"]
    linked = {e["to"] for e in edges
              if e["type"] == "relates_to" and e["from"] == evidence_id}
    assert linked == {"H1", "H2"}
    statuses = _node_status(store)
    assert statuses["H1"] == "under_verification"
    assert statuses["H2"] == "under_verification"


def test_publish_plan_postpones_uncovered_hypotheses(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(
        tmp_path,
        extra=[{"formulation": "No task can test this alternative."}],
    )
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)

    statuses = _node_status(store)
    assert statuses["H1"] == "formulated"
    assert statuses["H2"] == "postponed"
    h2 = next(n for n in store.full()["nodes"] if n["id"] == "H2")
    reasons = " ".join(str(row.get("reason") or "") for row in (h2.get("status_history") or []))
    assert "no_method_this_stage" in reasons


def test_publish_plan_revives_postponed_when_task_covers(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(
        tmp_path,
        extra=[{
            "formulation": "Parked alternative now assigned a task.",
            "status": "postponed",
        }],
    )
    assert _node_status(store)["H2"] == "postponed"
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H2")))
    publish_plan_to_graph(store, state)

    statuses = _node_status(store)
    assert statuses["H2"] == "formulated"
    assert statuses["H1"] == "postponed"


def test_publish_result_schedules_background_judgment(tmp_path, monkeypatch):
    from CoScientist.experiments.runtime import graph_bridge

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    graph_bridge.publish_plan_to_graph(store, state)
    called: list[object] = []
    monkeypatch.setattr(
        graph_bridge, "_schedule_hypothesis_judgments",
        lambda graph: called.append(graph) or 1,
    )
    graph_bridge.publish_result_to_graph(store, state, "EXP-1", {
        "result_id": "RES-6",
        "status": "success",
        "summary": "Ready to judge.",
        "artifacts": [{"name": "out.csv", "bucket": "b", "s3_key": "k/out.csv"}],
    })
    assert called == [store]


# ── the plan as a record, not as a name and a route ──────────────────────────
# A VerificationMethod used to carry the task id, the route and a blob of MCP
# servers. Everything else a human approved — the success criteria, the
# baselines, the analysis artifacts, what the task waits on, how long it takes —
# lived only in the plan JSON on the session state, so opening the method in the
# graph told a reader nothing about the method. And the tools the plan names,
# which are the feasibility layer's whole subject, were a string inside an
# attribute rather than nodes the method uses.


def test_the_method_carries_the_design_the_human_approved(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)

    attrs = next(n for n in store.full()["nodes"]
                 if n["type"] == "VerificationMethod")["attrs"]
    assert attrs["name"] == "Chemical computation EXP-1"
    assert attrs["cost"] == "≈1 min"
    assert "EXP-1-C1: The MCP execution completes." in attrs["success_criteria"]
    assert attrs["baselines"] == "no-tool control (method)"
    assert attrs["analysis_artifacts"] == "metrics_table.json [metrics_table]"
    # Nothing the plan left unset is invented: an empty warning list stays out
    # rather than becoming an attribute that reads as "checked, none".
    assert "limitations" not in attrs
    assert "optional" not in attrs


def test_a_task_that_waits_and_may_be_skipped_says_so(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    second = _task("EXP-2", hypothesis_ref="H1", depends_on=["EXP-1"], optional=True)
    second["warnings"] = ["the upstream artifact may be empty"]
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1"), second))

    publish_plan_to_graph(store, state)

    attrs = next(n for n in store.full()["nodes"]
                 if n["type"] == "VerificationMethod"
                 and n["attrs"]["task_id"] == "EXP-2")["attrs"]
    assert attrs["depends_on"] == "EXP-1"
    assert attrs["optional"] is True
    assert attrs["limitations"] == "the upstream artifact may be empty"


def test_the_tools_the_plan_names_become_tools_the_method_uses(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("Tool", [])) == 1
    tool_id = by_type["Tool"][0]
    tool = next(n for n in store.full()["nodes"] if n["id"] == tool_id)
    assert tool["attrs"]["name"] == "estimate_property"
    assert tool["attrs"]["location"] == "http://127.0.0.1:8000/mcp"
    assert tool["attrs"]["tool_type"] == "computational"
    assert tool["status"] == "available"
    vm_id = by_type["VerificationMethod"][0]
    assert any(e["type"] == "uses" and e["from"] == vm_id and e["to"] == tool_id
               for e in store.full()["edges"])
    assert state["experiment_graph_tool_ids"] == {"chem-ready:estimate_property": tool_id}


def test_two_tasks_on_the_same_tool_share_one_tool_node(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(
        _plan(_task("EXP-1", hypothesis_ref="H1"), _task("EXP-2", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type["Tool"]) == 1
    uses = [e for e in store.full()["edges"] if e["type"] == "uses"]
    assert len(uses) == 2
    assert {e["to"] for e in uses} == set(by_type["Tool"])


def test_republishing_the_plan_does_not_duplicate_its_tools(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)
    publish_plan_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type["Tool"]) == 1
    assert len(by_type["VerificationMethod"]) == 1
    assert len([e for e in store.full()["edges"] if e["type"] == "uses"]) == 1


def test_a_build_task_records_the_repo_as_a_tool_being_created(tmp_path):
    """There is no tool yet — the repo the pipeline will turn into one is the
    honest record, and ``being_created`` is the status that says so."""
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    from CoScientist.experiments.runtime import approve_plan, initialize_runtime

    store = _seeded_store(tmp_path)
    raw = _task("EXP-1", route="alembic_build", hypothesis_ref="H1")
    raw["repo_url"] = "https://github.com/example/solver"
    raw["post_build_route"] = "react_tools"
    raw["code_assessment"] = {
        "requirement": "reuse",
        "evidence": "Existing solver entrypoint covers the operation unchanged.",
        "entrypoints": ["solver.run"],
    }
    # Built by hand: the deterministic critic refuses alembic_build under the
    # default profile, and the route it refuses is the one under test here.
    state: dict = {}
    initialize_runtime(state, _plan(raw), critique={
        "schema_version": "plan-critique/0.1", "critique_id": "CRIT-build",
        "plan_id": "PLAN-acceptance", "verdict": "approve", "issues": [],
        "checked_at": NOW,
    })
    approve_plan(state)

    publish_plan_to_graph(store, state)

    tool = next(n for n in store.full()["nodes"] if n["type"] == "Tool")
    assert tool["status"] == "being_created"
    assert tool["attrs"]["name"] == "solver"
    assert tool["attrs"]["location"] == "https://github.com/example/solver"


def test_a_route_with_no_mcp_tools_adds_none(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import publish_plan_to_graph

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", route="coder", hypothesis_ref="H1")))

    publish_plan_to_graph(store, state)

    assert _nodes_by_type(store).get("Tool") is None
    assert not [e for e in store.full()["edges"] if e["type"] == "uses"]


def test_the_evidence_says_what_it_was_measured_on(tmp_path):
    """The research graph requires it of computational evidence, and a reader
    needs it: the summary is the claim, this is what backs it."""
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_to_graph,
        publish_result_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1", hypothesis_ref="H1")))
    publish_plan_to_graph(store, state)
    publish_result_to_graph(store, state, "EXP-1", {
        "result_id": "RES-9", "status": "success", "route_used": "fedot_mas",
        "summary": "The property was computed.",
        "artifacts": [{"name": "m.json", "bucket": "b", "s3_key": "k/m.json"}],
    })

    evidence = next(n for n in store.full()["nodes"] if n["type"] == "Evidence")
    measured_on = evidence["attrs"]["measured_on"]
    assert "dataset ready_mcp_inputs" in measured_on
    assert "route fedot_mas against chem-ready (http://127.0.0.1:8000/mcp)" in measured_on
    assert "artifact s3://b/k/m.json" in measured_on


def test_evidence_from_a_run_that_names_nothing_says_so(tmp_path):
    """Never invent a measurement target to satisfy a required field."""
    from CoScientist.experiments.runtime.graph_bridge import _measured_on

    assert _measured_on("EXP-7", None, {}, "") == (
        "experiment task EXP-7: the run record names no dataset, route or artifact"
    )


# ── the detailed plan as its own nodes ────────────────────────────────────────

def _with_outer_plan(store, task_id="TASK-3", assignee="ExperimentModuleAgent",
                     status="in_progress"):
    """A study whose OUTER plan already has the step the module elaborates."""
    store.commit(
        source="plan-mirror",
        nodes=[{"type": "PlanStep", "status": "in_progress", "attrs": {
            "title": "Run the computational experiments",
            "plan_task_id": task_id, "assignee": assignee}}],
    )
    return [{"id": task_id, "title": "Run the computational experiments",
             "assignee": assignee, "status": status}]


def test_the_detailed_plan_gets_its_own_nodes_under_the_general_step(tmp_path):
    """The outer planner writes five one-line steps; the module then designs a
    detailed task under the experimental ones, with its question, its dataset,
    its bar and its artifacts. Both are intentions at two grains, and the graph
    had a type for only the coarse one — so the design a human approved existed
    nowhere a reader could see it.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1"), _task("EXP-2",
                                                        depends_on=["EXP-1"])))
    state["_master_active_tasks"] = _with_outer_plan(store)

    publish_plan_detail_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("ExperimentTask", [])) == 2, by_type
    # Each one points at the general step it serves — and by `elaborates`, not
    # `realises`: a finer intention is not a record of the coarser one.
    step = by_type["PlanStep"][0]
    links = {(e["from"], e["type"], e["to"]) for e in store.full()["edges"]}
    for xt in by_type["ExperimentTask"]:
        assert (xt, "elaborates", step) in links, links
    # And it carries the design, not just a name and a route.
    attrs = {n["id"]: n["attrs"] for n in store.full()["nodes"]}
    first = attrs[by_type["ExperimentTask"][0]]
    assert first["experiment_task_id"] == "EXP-1"
    assert first["plan_task_id"] == "TASK-3"
    assert first["success_criteria"] and first["expected_artifacts"]
    assert first["hypothesis_refs"] == "H1"


def test_republishing_the_plan_amends_the_tasks_rather_than_doubling_them(tmp_path):
    """A plan is revised and re-approved, and the module publishes again. The
    tasks must be the same cards with new contents: `_stage_merge` consults the
    ACL table whatever the privilege flag says, so without a table entry this
    path was refused — and refused silently, because every failure here is
    swallowed by contract.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)
    first = _nodes_by_type(store)["ExperimentTask"]

    revised = _task("EXP-1")
    revised["name"] = "Chemical computation EXP-1, rerun on the wider set"
    state2 = _approved_state(_plan(revised))
    state2["_master_active_tasks"] = state["_master_active_tasks"]
    state2[_XT_KEY] = state[_XT_KEY]
    # How far it got is the runtime's record, not the plan's.
    state2.setdefault("experiment_runtime", {}).setdefault("tasks", {})[
        "EXP-1"] = {"status": "running"}
    publish_plan_detail_to_graph(store, state2)

    assert _nodes_by_type(store)["ExperimentTask"] == first, "a second card"
    node = next(n for n in store.full()["nodes"] if n["id"] == first[0])
    assert node["attrs"]["title"].endswith("wider set"), node["attrs"]["title"]
    assert node["status"] == "running"


def test_the_detailed_plan_is_still_drawn_when_there_is_no_outer_step(tmp_path):
    """A module run with no outer plan — or one whose steps went to somebody
    else — must still record its tasks. Inventing a PlanStep to hang them off
    would put a step in the graph that no planner ever wrote.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = []

    publish_plan_detail_to_graph(store, state)

    by_type = _nodes_by_type(store)
    assert len(by_type.get("ExperimentTask", [])) == 1, by_type
    assert "PlanStep" not in by_type
    assert not [e for e in store.full()["edges"] if e["type"] == "elaborates"]


def test_the_plan_column_can_tell_the_two_plans_apart(tmp_path):
    """Both plans are intentions, so both are drawn beside the bands rather
    than in them — but in two lanes, coarsest first. Stacked in one column a
    reader cannot tell the step the study planned from the task the module
    designed under it, which is the whole reason there are two types.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)

    lanes = {n["kind"]: (n.get("track"), n.get("lane"))
             for n in store.view_of(None)["nodes"]
             if n["kind"] in ("planstep", "experimenttask")}
    assert lanes["planstep"] == ("plan", 0)
    assert lanes["experimenttask"] == ("plan", 1)
    # The detailed card names the step in the step's own words, because "PS1"
    # tells a reader nothing.
    task = next(n for n in store.view_of(None)["nodes"]
                if n["kind"] == "experimenttask")
    assert task["elaborates"].startswith("Run the computational")


def test_a_finished_task_says_so_on_its_own_card(tmp_path):
    """The method advances when a result lands, but a reader looking at the
    plan column wants to know which of the plan's tasks are still outstanding
    without cross-referencing the methods. And a card that says a thing failed
    without saying why is the gap the graph reports as `unreasoned_failures`.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_plan_to_graph,
        publish_result_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1"), _task("EXP-2")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_to_graph(store, state)
    publish_plan_detail_to_graph(store, state)

    publish_result_to_graph(store, state, "EXP-1",
                            {"status": "success", "route_used": "fedot_mas"})
    publish_result_to_graph(store, state, "EXP-2",
                            {"status": "failed",
                             "error": "the MCP server refused the call"})

    cards = {n["id"]: n for n in store.full()["nodes"]
             if n["type"] == "ExperimentTask"}
    by_task = {(c["attrs"] or {}).get("experiment_task_id"): c
               for c in cards.values()}
    assert by_task["EXP-1"]["status"] == "done"
    assert by_task["EXP-2"]["status"] == "failed"
    assert "refused the call" in by_task["EXP-2"]["attrs"]["failure_reason"]
    step_id = _nodes_by_type(store)["PlanStep"][0]
    assert _node_status(store)[step_id] == "blocked"
    assert state["_master_active_tasks"][0]["status"] == "FAILED"

    # A retry may reopen only the block the failed experiment created. The
    # parent returns to live while the task runs and settles when it succeeds.
    publish_task_state_to_graph(store, _started(state, "EXP-2", "ready"), "EXP-2")
    publish_task_state_to_graph(store, _started(state, "EXP-2", "running"), "EXP-2")
    assert _node_status(store)[step_id] == "in_progress"
    assert state["_master_active_tasks"][0]["status"] == "IN_PROGRESS"
    publish_result_to_graph(store, state, "EXP-2", {
        "status": "success", "summary": "retry produced a real result",
    })
    assert _node_status(store)[step_id] == "done"
    assert state["_master_active_tasks"][0]["status"] == "DONE"


# ── what the graph is told while a task is actually running ───────────────────
# The graph used to learn about a task at exactly two moments: plan approval and
# `record_result`. `start_task` wrote nothing — and `start_task` is the only
# producer of the runtime's `running` — so a card read «запланирован» for the
# whole of a run that took minutes and then jumped straight to «выполнен». The
# operator's complaint that the graph never says which stage is going was, at
# bottom, this: nothing was writing it down.

def _started(state, task_id="EXP-1", status="running"):
    state.setdefault("experiment_runtime", {}).setdefault("tasks", {})[
        task_id] = {"status": status}
    return state


def test_a_task_that_starts_says_so_on_the_canvas(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)
    xt_id = state[_XT_KEY]["EXP-1"]
    assert _node_status(store)[xt_id] == "planned"

    publish_task_state_to_graph(store, _started(state), "EXP-1")
    assert _node_status(store)[xt_id] == "running"


def test_the_step_above_a_running_task_is_under_way_too(tmp_path):
    """The outer plan's own mirror runs only on an orchestrator tick, and by
    then the tracker has usually moved the step from «не начат» straight to
    «выполнен» — so a step being worked on for minutes was never once drawn as
    being worked on. The module knows the moment a task starts.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store, status="todo")
    store.commit(source="plan-mirror",
                 status_updates=[{"id": _nodes_by_type(store)["PlanStep"][0],
                                  "status": "todo"}])
    publish_plan_detail_to_graph(store, state)
    step_id = _nodes_by_type(store)["PlanStep"][0]
    assert _node_status(store)[step_id] == "todo"

    publish_task_state_to_graph(store, _started(state), "EXP-1")
    assert _node_status(store)[step_id] == "in_progress"


def test_a_step_already_finished_is_not_reopened_by_a_late_task(tmp_path):
    """`done → in_progress` is a legal move for a step and a wrong one to make
    from here: the module may only say that work has BEGUN, and a finished step
    is the outer plan's own record, not this module's to overwrite.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)
    step_id = _nodes_by_type(store)["PlanStep"][0]
    store.commit(source="plan-mirror",
                 status_updates=[{"id": step_id, "status": "done"}])

    publish_task_state_to_graph(store, _started(state), "EXP-1")
    assert _node_status(store)[step_id] == "done"
    # …and not at the price of a refused commit every run: a refusal is
    # counted as a gap on the reader's own banner.
    assert not [g for g in store.to_view()["gaps"]
                if g["code"] == "rejected_commits"]


def test_a_move_the_graph_would_refuse_is_not_attempted(tmp_path):
    """A commit is all-or-nothing and this one carries nothing else worth
    losing, so the legality is asked before the write rather than after the
    refusal. `running → planned` is not a move an experiment task has.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)
    xt_id = state[_XT_KEY]["EXP-1"]
    publish_task_state_to_graph(store, _started(state), "EXP-1")
    publish_task_state_to_graph(store, _started(state, status="ready"), "EXP-1")
    assert _node_status(store)[xt_id] == "running"


def test_the_control_tools_are_the_ones_that_report_progress(tmp_path):
    """Wiring, not behaviour: every tool that moves a task mirrors it. A tool
    added later that forgets to is a task the graph goes quiet about again.
    """
    import inspect

    from CoScientist.experiments.runtime import tools as control

    for name in ("start_task", "retry_task", "fallback_task", "skip_task"):
        body = inspect.getsource(getattr(control.ExperimentControlToolset, name))
        assert "_mirror_task_state_to_graph" in body, name


def test_a_step_blocked_by_someone_else_is_not_quietly_released(tmp_path):
    """A replan retires a step to `blocked`, and the retirement sweep only ever
    looks at steps that are still `todo` — so a step this module talked out of
    `blocked` would stay «выполняется» for the rest of the run, and flap between
    the two every time the outer mirror ticked. Whatever blocked it knows why;
    this module does not.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)
    step_id = _nodes_by_type(store)["PlanStep"][0]
    store.commit(source="plan-mirror",
                 status_updates=[{"id": step_id, "status": "blocked",
                                  "reason": "шаг убран при пересмотре плана"}])

    publish_task_state_to_graph(store, _started(state), "EXP-1")
    assert _node_status(store)[step_id] == "blocked"


def test_a_finished_task_stops_being_drawn_as_running_even_with_no_method(tmp_path):
    """`publish_plan_to_graph` can land its tasks and lose its methods — its
    own commit is all-or-nothing and the detail commit is not. The result
    recording used to give up at that point, which merely left the card out of
    date; now that `start_task` draws the card as running, giving up leaves a
    card pulsing «выполняется» on the canvas for the rest of the session.
    """
    from CoScientist.experiments.runtime.graph_bridge import (
        publish_plan_detail_to_graph,
        publish_result_to_graph,
        publish_task_state_to_graph,
    )

    store = _seeded_store(tmp_path)
    state = _approved_state(_plan(_task("EXP-1")))
    state["_master_active_tasks"] = _with_outer_plan(store)
    publish_plan_detail_to_graph(store, state)          # tasks, but no methods
    xt_id = state[_XT_KEY]["EXP-1"]
    publish_task_state_to_graph(store, _started(state), "EXP-1")
    assert _node_status(store)[xt_id] == "running"

    publish_result_to_graph(store, state, "EXP-1",
                            {"result_id": "RES-9", "status": "success",
                             "summary": "done without a method node"})
    assert _node_status(store)[xt_id] == "done"
    step_id = _nodes_by_type(store)["PlanStep"][0]
    assert _node_status(store)[step_id] == "done"
    assert state["_master_active_tasks"][0]["status"] == "DONE"
