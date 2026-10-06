"""Planner context, stash, repo candidates, discovered capabilities."""
from __future__ import annotations

from types import SimpleNamespace

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.runtime import (
    approve_plan,
    initialize_runtime,
)
from CoScientist.experiments.schemas import ExperimentPlan

from .helpers import (
    _plan,
    _task,
)

def test_discovered_capabilities_survive_filtered_tools_clear_and_revise():
    """Attempt clears must not erase critique inventory; revise keeps run id."""
    from CoScientist.experiments.context import (
        DISCOVERED_CAPABILITIES_KEY,
        RETRIEVED_CAPABILITIES_KEY,
        build_experiment_context,
        skip_executor_without_runtime,
        snapshot_experiment_discovered_capabilities,
        stash_experiment_retrieved_capabilities,
    )
    from CoScientist.experiments.critique import critique_plan

    tool = {
        "tool": "estimate_property",
        "server_id": "srv-chem",
        "description": "Compute a property.",
        "input_schema": {"type": "object"},
        "score": 0.9,
    }
    extra = {
        "tool": "calculate_docking",
        "server_id": "srv-dock",
        "description": "Dock a molecule.",
        "input_schema": {"type": "object"},
        "score": 0.4,
    }
    state: dict = {
        "accumulated_tools": [tool, extra],
        "filtered_tools": [tool],
        "experiment_source_request": "Estimate a chemical property with ready MCP tools.",
    }
    ctx = SimpleNamespace(state=state, user_content=None)
    stash_experiment_retrieved_capabilities(ctx)
    snapshot_experiment_discovered_capabilities(ctx)
    build_experiment_context(ctx)
    run_id = state["experiment_context"]["experiment_run_id"]
    assert state[RETRIEVED_CAPABILITIES_KEY][1]["tool"] == "calculate_docking"
    assert {c["tool"] for c in state["experiment_context"]["available_mcp_capabilities"]} == {
        "estimate_property",
        "calculate_docking",
    }
    assert {c["tool"] for c in state["experiment_context"]["critique_mcp_capabilities"]} >= {
        "estimate_property",
        "calculate_docking",
    }
    prompt_ctx = state["experiment_planner_context"]
    assert isinstance(prompt_ctx, str)
    assert "estimate_property" in prompt_ctx
    assert "calculate_docking" in prompt_ctx
    # Prompt projection must not triple-dump preferred/critique inventories.
    assert prompt_ctx.count("available_mcp_capabilities") == 1
    assert "critique_mcp_capabilities" not in prompt_ctx
    assert '"input_schema"' in prompt_ctx

    # Simulate post-attempt clear used by the runtime.
    state["filtered_tools"] = []
    state["experiment_active_envelope"] = None
    state["experiment_plan_critique"] = {
        "verdict": "revise",
        "issues": [{"severity": "blocker", "message": "fix"}],
    }

    build_experiment_context(ctx)
    caps = state["experiment_context"]["critique_mcp_capabilities"]
    assert {c["server_id"] for c in caps} >= {"srv-chem", "srv-dock"}
    assert state["experiment_context"]["experiment_run_id"] == run_id
    assert state[DISCOVERED_CAPABILITIES_KEY]

    plan = ExperimentPlan.model_validate(_plan(_task("EXP-1")))
    approved = critique_plan(
        plan,
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=caps,
    )
    assert approved.verdict == "approve"
    assert skip_executor_without_runtime(ctx) is not None
    state["experiment_plan_critique"] = approved.model_dump(mode="json")
    initialize_runtime(state, plan, critique=approved.model_dump(mode="json"))
    approve_plan(state)
    assert skip_executor_without_runtime(ctx) is None


def test_first_planner_entry_keeps_stashed_retrieval_without_prior_request():
    """ToolPreparer→Planner handoff: empty prev_request must not wipe RETRIEVED."""
    from CoScientist.experiments.context import (
        RETRIEVED_CAPABILITIES_KEY,
        build_experiment_context,
    )

    full = {
        "tool": "generate_case_mols",
        "server_id": "d36e3d994404e957",
        "description": "Generate case molecules.",
        "input_schema": {"type": "object", "properties": {"case": {}}},
        "score": 0.8,
    }
    kept = {
        "tool": "fetch_activity_data",
        "server_id": "bfd3f80438ba403b",
        "description": "Fetch activity.",
        "input_schema": {"type": "object"},
        "score": 0.9,
    }
    # After ToolRetriever+Reranker: full set stashed, accumulated cleared, keep-set in filtered.
    state: dict = {
        "accumulated_tools": [],
        "filtered_tools": [kept],
        RETRIEVED_CAPABILITIES_KEY: [full, kept],
        # No experiment_source_request yet — first planner entry.
    }
    user = SimpleNamespace(parts=[SimpleNamespace(text="Generate GSK-3beta inhibitors with high activit")])
    ctx = SimpleNamespace(state=state, user_content=user)
    build_experiment_context(ctx)
    tools = {c["tool"] for c in state["experiment_context"]["available_mcp_capabilities"]}
    assert tools == {"generate_case_mols", "fetch_activity_data"}
    assert state[RETRIEVED_CAPABILITIES_KEY][0]["tool"] == "generate_case_mols"
    assert "generate_case_mols" in state["experiment_planner_context"]


def test_stash_before_rerank_clear_preserves_full_retrieval():
    from CoScientist.experiments.context import (
        RETRIEVED_CAPABILITIES_KEY,
        stash_experiment_retrieved_capabilities,
    )

    tools = [
        {
            "tool": "generate_case_mols",
            "server_id": "d36e3d994404e957",
            "description": "gen",
            "input_schema": {},
        },
        {
            "tool": "calculate_docking",
            "server_id": "bfd3f80438ba403b",
            "description": "dock",
            "input_schema": {},
        },
    ]
    state: dict = {"accumulated_tools": tools}
    ctx = SimpleNamespace(state=state, user_content=None)
    stash_experiment_retrieved_capabilities(ctx)
    state["accumulated_tools"] = []  # rerank clear
    state["filtered_tools"] = [tools[0]]
    assert {c["tool"] for c in state[RETRIEVED_CAPABILITIES_KEY]} == {
        "generate_case_mols",
        "calculate_docking",
    }


def test_extract_repo_candidates_from_ask():
    from CoScientist.experiments.context import extract_repo_candidates

    refs = extract_repo_candidates(
        "Use https://github.com/whitead/synspace and also "
        "https://github.com/encode/httpx for nothing."
    )
    urls = [r["url"] for r in refs]
    assert "https://github.com/whitead/synspace" in urls
    assert "https://github.com/encode/httpx" in urls
    assert refs[0]["repo_name"] == "synspace"


def test_explicit_repo_url_survives_when_alembic_discovery_is_off():
    from CoScientist.experiments.context import resolve_repo_candidates

    out = resolve_repo_candidates(
        "Run https://github.com/whitead/synspace directly with Coder.",
        route_alembic=False,
    )
    assert [row["url"] for row in out] == ["https://github.com/whitead/synspace"]


def test_resolve_repo_candidates_skips_search_when_inventory_covers(monkeypatch):
    from CoScientist.experiments.context import resolve_repo_candidates

    called = {"n": 0}

    def _boom(*_a, **_k):
        called["n"] += 1
        raise AssertionError("search must not run when inventory covers")

    monkeypatch.setattr(
        "CoScientist.experiments.capabilities.repo_searcher.search_repos_sync",
        _boom,
    )
    caps = [{
        "server_id": "srv",
        "tool": "smiles2prop",
        "description": "molecular properties and SA score / synthesizability",
    }]
    out = resolve_repo_candidates(
        "Call smiles2prop to estimate synthetic accessibility SA score for SMILES",
        planner_caps=caps,
        route_alembic=True,
    )
    assert out == []
    assert called["n"] == 0


def test_resolve_repo_candidates_searches_when_inventory_empty(monkeypatch):
    from CoScientist.experiments.capabilities.repo_searcher import RepoCandidate, RepoSearchResult
    from CoScientist.experiments.context import resolve_repo_candidates

    def _fake_search(ask, **_kwargs):
        cand = RepoCandidate(
            url="https://github.com/whitead/synspace",
            owner="whitead",
            repo_name="synspace",
            source="github_search",
            fit_score=0.6,
            fit_reason="test",
            description="Synthesis generative model",
        )
        return RepoSearchResult(query=ask, search_queries=["q"], candidates=[cand])

    monkeypatch.setattr(
        "CoScientist.experiments.capabilities.repo_searcher.search_repos_sync",
        _fake_search,
    )
    out = resolve_repo_candidates(
        "Estimate synthetic accessibility SA score for SMILES",
        planner_caps=[],
        route_alembic=True,
    )
    assert any(c["url"] == "https://github.com/whitead/synspace" for c in out)
    assert out[0].get("source") == "github_search"


def test_planner_context_reads_numbered_frame_operations():
    from CoScientist.context_init.models import ResearchFrame
    from CoScientist.experiments.context.builder import build_experiment_context

    ask = (
        "Run the computational cycle. "
        "1. Review published methods for the endpoint. "
        "2. Curate a labeled dataset with inclusion criteria. "
        "3. Cluster the structures by similarity. "
        "4. Fit six predictive models for the endpoint. "
        "5. Quantify applicability domain for each prediction. "
        "6. Predict general toxicity for the densest cluster. "
        "7. Report. Conclusions and limitations."
    )
    frame = ResearchFrame.blank(ask)
    ctx = SimpleNamespace(
        user_content=SimpleNamespace(parts=[SimpleNamespace(text=ask)]),
        state={"research_frame": frame.model_dump(), "filtered_tools": []},
    )
    build_experiment_context(ctx)
    ops = ctx.state["experiment_context"]["operations"]
    ids = [row["operation_id"] for row in ops]
    assert ids == ["OP-2", "OP-3", "OP-4", "OP-5", "OP-6"]
    assert ctx.state["experiment_context"]["external_literature_operations"][0]["operation_id"] == "OP-1"
    assert "Fit six predictive models" in ops[2]["statement"]
    assert not any("Conclusions and limitations" in row["statement"] for row in ops)


# ── research_graph_snapshot (Task: graph-first snapshot) ─────────────────────

class _FakeSnapshotGraph:
    """Minimal research store stand-in exposing full()/overview()/is_empty()."""

    def __init__(self, nodes, *, edges=None, rendered="rendered overview"):
        self._nodes = nodes
        self._edges = edges or []
        self._rendered = rendered

    def is_empty(self):
        return not self._nodes

    def full(self):
        return {"nodes": self._nodes, "edges": self._edges}

    def overview(self):
        return {"rendered": self._rendered, "nodes": self._nodes}


def _snapshot_ctx(monkeypatch, store, *, enabled=True):
    import CoScientist.graph.research.store as research_store
    from CoScientist.config import get_settings
    from CoScientist.experiments.context import builder

    monkeypatch.setattr(get_settings().research_graph, "enabled", enabled, raising=False)
    monkeypatch.setattr(research_store, "get_research_graph", lambda ctx: store)
    # research_graph_snapshot requires an ADK session; fake a truthy one.
    monkeypatch.setattr(builder, "_invocation_session", lambda ctx: object())
    return SimpleNamespace(state={}, user_content=None)


def test_research_graph_snapshot_reads_typed_nodes(monkeypatch):
    from CoScientist.experiments.context.builder import research_graph_snapshot

    nodes = [
        {"id": "H1", "type": "Hypothesis", "status": "formulated",
         "attrs": {"formulation": "Compound X inhibits target Y."}},
        {"id": "H2", "type": "Hypothesis", "status": "refuted",
         "attrs": {"formulation": "Refuted idea should be skipped."}},
        {"id": "C1", "type": "Constraint", "status": "active",
         "attrs": {"subtype": "budget", "content": "Stay under 4 GPU-hours."}},
        {"id": "CC1", "type": "ConfirmationCriteria", "status": "not_met",
         "attrs": {"metric": "auc", "threshold": 0.8}},
        {"id": "EB1", "type": "EmpiricalBase", "status": "active",
         "attrs": {"base_type": "dataset", "source_ref": "s3://bucket/data.csv"}},
        {"id": "E1", "type": "Evidence", "status": "obtained",
         "attrs": {"subtype": "computational", "content": "Prior AUC 0.72.",
                   "source_ref": "s3://bucket/run1.json"}},
    ]
    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph(nodes))
    snap = research_graph_snapshot(ctx)

    assert [h["hypothesis_id"] for h in snap["hypothesis_refs"]] == ["H1"]
    assert snap["hypothesis_refs"][0]["statement"] == "Compound X inhibits target Y."
    assert snap["constraints"][0]["content"] == "Stay under 4 GPU-hours."
    assert snap["confirmation_criteria"][0]["threshold"] == 0.8
    assert snap["data_refs"][0]["source_ref"] == "s3://bucket/data.csv"
    assert snap["prior_evidence"][0]["node_id"] == "E1"
    assert snap["rendered"] == "rendered overview"


def test_generated_data_is_an_input_but_literature_prose_is_not_a_dataset(monkeypatch):
    from CoScientist.experiments.context.builder import research_graph_snapshot

    nodes = [
        {"id": "E1", "type": "Evidence", "status": "obtained",
         "attrs": {"content": "Paper mentions compound structures", "source_ref": "doi:paper"}},
        {"id": "GD1", "type": "GeneratedData", "status": "created",
         "attrs": {"path": "cos-artifact:dataset.csv", "schema": {"columns": ["name", "structure"]}}},
    ]
    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph(nodes))
    snapshot = research_graph_snapshot(ctx)
    assert len(snapshot["data_refs"]) == 1
    assert snapshot["data_refs"][0]["node_id"] == "GD1"
    assert snapshot["data_refs"][0]["validation_status"] == "unverified"
    assert {row["node_id"] for row in snapshot["prior_evidence"]} == {"E1", "GD1"}


def test_successor_context_reuses_predecessor_evidence_with_provenance(monkeypatch):
    from CoScientist.experiments.context.builder import research_graph_snapshot

    nodes = [
        {"id": "H1", "type": "Hypothesis", "status": "refuted",
         "attrs": {"formulation": "Primary claim."}, "source": "HypothesesAgent"},
        {"id": "H2", "type": "Hypothesis", "status": "formulated",
         "attrs": {"formulation": "Fallback claim."}, "source": "HypothesesAgent"},
        {"id": "E1", "type": "Evidence", "status": "validated",
         "attrs": {"subtype": "computational", "content": "Measured miss."},
         "source": "ExperimentModule"},
        {"id": "GD1", "type": "GeneratedData", "status": "available",
         "attrs": {"path": "s3://bucket/prior.csv"}, "source": "ExperimentModule"},
    ]
    edges = [
        {"type": "conditional_successor", "from": "H1", "to": "H2"},
        {"type": "refutes", "from": "E1", "to": "H1"},
        {"type": "derived_from", "from": "GD1", "to": "E1"},
    ]
    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph(nodes, edges=edges))

    chain = research_graph_snapshot(ctx)["hypothesis_chain_context"][0]

    assert chain["successor_id"] == "H2"
    assert chain["predecessor_status"] == "refuted"
    assert chain["predecessor_evidence"][0]["id"] == "E1"
    assert chain["predecessor_evidence"][0]["source"] == "ExperimentModule"
    assert chain["predecessor_artifacts"][0]["id"] == "GD1"
    assert chain["predecessor_artifacts"][0]["attrs"]["path"] == "s3://bucket/prior.csv"


def test_research_graph_snapshot_empty_when_disabled(monkeypatch):
    from CoScientist.experiments.context.builder import research_graph_snapshot

    nodes = [{"id": "H1", "type": "Hypothesis", "status": "formulated",
              "attrs": {"formulation": "Ignored while disabled."}}]
    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph(nodes), enabled=False)
    assert research_graph_snapshot(ctx) == {}


def test_research_graph_snapshot_empty_graph(monkeypatch):
    from CoScientist.experiments.context.builder import research_graph_snapshot

    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph([]))
    assert research_graph_snapshot(ctx) == {}


def test_research_graph_snapshot_swallows_store_errors(monkeypatch):
    from CoScientist.experiments.context.builder import research_graph_snapshot

    class _Boom:
        def is_empty(self):
            raise RuntimeError("graph unavailable")

    ctx = _snapshot_ctx(monkeypatch, _Boom())
    assert research_graph_snapshot(ctx) == {}


def test_build_experiment_context_prefers_graph_hypotheses(monkeypatch):
    from CoScientist.experiments.context.builder import build_experiment_context

    nodes = [
        {"id": "H1", "type": "Hypothesis", "status": "formulated",
         "attrs": {"formulation": "Graph hypothesis wins."}},
        {"id": "E1", "type": "Evidence", "status": "obtained",
         "attrs": {"subtype": "computational", "content": "Prior fact.",
                   "source_ref": "s3://b/e.json"}},
    ]
    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph(nodes))
    ctx.state["filtered_tools"] = []
    # A rich prose ask that would otherwise dominate hypothesis extraction.
    ctx.state["experiment_source_request"] = (
        "Hypothesis 1: Prose one.\nHypothesis 2: Prose two.\nHypothesis 3: Prose three."
    )
    build_experiment_context(ctx)
    context = ctx.state["experiment_context"]
    assert [h["hypothesis_id"] for h in context["hypothesis_refs"]] == ["H1"]
    assert context["hypothesis_refs"][0]["statement"] == "Graph hypothesis wins."
    assert context["prior_evidence"][0]["content"] == "Prior fact."
    prompt = ctx.state["experiment_planner_context"]
    assert '"hypothesis_refs"' in prompt
    assert '"hypotheses"' not in prompt
    assert "rendered overview" not in prompt


def test_build_experiment_context_falls_back_without_graph(monkeypatch):
    from CoScientist.experiments.context.builder import build_experiment_context

    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph([]))
    ask = "Hypothesis 1: Prose one works.\nHypothesis 2: Prose two works."
    ctx.state.update({"filtered_tools": [], "experiment_source_request": ask})
    ctx.user_content = SimpleNamespace(parts=[SimpleNamespace(text=ask)])
    build_experiment_context(ctx)
    refs = ctx.state["experiment_context"]["hypothesis_refs"]
    assert [r["hypothesis_id"] for r in refs] == ["H1", "H2"]
    assert refs[0]["statement"] == "Prose one works."


def test_agent_prose_is_not_split_into_character_hypotheses(monkeypatch):
    from CoScientist.experiments.context.builder import build_experiment_context

    ctx = _snapshot_ctx(monkeypatch, _FakeSnapshotGraph([]))
    ctx.state.update({"filtered_tools": [], "experiment_source_request": "Compare two methods.",
                      "hypotheses": "No hypotheses were requested."})
    build_experiment_context(ctx)
    assert ctx.state["experiment_context"]["hypothesis_refs"] == []
    ctx.state["hypotheses"] = [{"hypothesis_id": "H1", "statement": "Method A is faster."}]
    build_experiment_context(ctx)
    assert ctx.state["experiment_context"]["hypothesis_refs"] == [
        {"hypothesis_id": "H1", "statement": "Method A is faster."},
    ]


def test_the_planner_context_carries_the_effective_fedot_answer(monkeypatch):
    """experiment_context.route_fedot is the answer start_task gets - switch AND
    FedotAgent attached - where it used to be the bare switch. The planner's own
    JSON leaves it out: its prompt is built with or without the route."""
    from CoScientist.assembly.schema import load_config, resolve_config_path
    from CoScientist.config import get_settings
    from CoScientist.experiments.context import build_experiment_context
    from CoScientist.experiments.runtime import state_machine

    def build() -> dict:
        state: dict = {
            "filtered_tools": [],
            "experiment_source_request": "Estimate a chemical property with ready MCP tools.",
        }
        build_experiment_context(SimpleNamespace(state=state, user_content=None))
        return state

    on = build()
    assert on["experiment_context"]["route_fedot"] is True
    assert '"route_fedot"' not in on["experiment_planner_context"]

    monkeypatch.setattr(get_settings().experiments, "route_fedot", False)
    assert build()["experiment_context"]["route_fedot"] is False

    monkeypatch.setattr(get_settings().experiments, "route_fedot", True)
    detached = load_config(resolve_config_path("experiments"))
    detached.agents["ExperimentExecutorAgent"].subordinates.remove("FedotAgent")
    monkeypatch.setattr(state_machine, "_config_tree", lambda: detached)
    assert build()["experiment_context"]["route_fedot"] is False


def test_the_planner_context_carries_medical_tools_only_with_the_agent(monkeypatch):
    """An empty list would still name the medical route to the planner."""
    from CoScientist.config import get_settings
    from CoScientist.experiments.context import build_experiment_context

    def build() -> dict:
        state: dict = {
            "filtered_tools": [],
            "experiment_source_request": "Review the clinical literature on the compound.",
        }
        build_experiment_context(SimpleNamespace(state=state, user_content=None))
        return state

    on = build()
    assert on["experiment_context"]["route_medical"] is True
    assert on["experiment_context"]["available_medical_capabilities"]
    assert "get_pico" in on["experiment_planner_context"]
    assert "search_pubmed" not in on["experiment_planner_context"]

    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", False)
    off = build()
    assert off["experiment_context"]["route_medical"] is False
    assert off["experiment_context"]["available_medical_capabilities"] == []
    assert "medical" not in off["experiment_planner_context"]


def test_the_planner_context_asks_the_sessions_executor(monkeypatch):
    """Switch on, but this session's executor was built without MedicalAgent:
    the planner must not be offered a route start_task would refuse."""
    from CoScientist.experiments.context import build_experiment_context

    executor = SimpleNamespace(tools=[
        SimpleNamespace(agent=SimpleNamespace(name=name))
        for name in ("ExperimentAgent", "CoderAgent", "ResearchAgent")
    ])
    root = SimpleNamespace(find_agent=lambda name: executor if name == "ExperimentExecutorAgent" else None)
    planner = SimpleNamespace(root_agent=root)
    state: dict = {
        "filtered_tools": [],
        "experiment_source_request": "Review the clinical literature on the compound.",
    }
    ctx = SimpleNamespace(state=state, user_content=None,
                          _invocation_context=SimpleNamespace(agent=planner))
    build_experiment_context(ctx)
    assert state["experiment_context"]["route_medical"] is False
    assert state["experiment_context"]["route_fedot"] is False
    assert "medical" not in state["experiment_planner_context"]


def test_planning_notes_reach_planner_without_becoming_requirements():
    from CoScientist.experiments.context import build_experiment_context
    note = "Choose and document a ranking method after obtaining the measurements."
    state = {"experiment_source_request": "Compare the groups.",
             "normalized_statement": {"planning_notes": [note]}}
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    assert state["experiment_context"]["planning_notes"] == [note]
    assert note in state["experiment_planner_context"]
    assert not state["experiment_context"]["requirement_refs"]


def test_planner_receives_complete_tool_contracts():
    from CoScientist.experiments.context.builder import _normalize_capabilities, _cap_for_prompt

    schema = {
        "type": "object",
        "properties": {
            "case": {"type": "string", "enum": ["registered_case"],
                     "description": "Use a registered model, never a target protein name."},
            "count": {"type": "integer", "minimum": 1, "maximum": 30},
        },
        "required": ["case"],
        "additionalProperties": False,
    }
    output = {"type": "object", "properties": {"rows": {"type": "array", "items": schema}}}
    description = "Tool documentation. " * 30 + "Only pretrained cases are supported."
    capability = {"tool": "generate", "server_id": "srv", "description": description,
                  "input_schema": schema, "output_schema": output}
    projected = _cap_for_prompt(_normalize_capabilities([capability])[0])
    assert projected["input_schema"] == schema
    assert projected["output_schema"] == output
    assert projected["description"] == description


def test_planner_keeps_user_method_restrictions_separate_from_delegation():
    from CoScientist.experiments.context import build_experiment_context
    state = {
        'experiment_source_request': 'Use tool_a to estimate binding.',
        'normalized_statement': {'source_request': 'Estimate binding using an appropriate method.'},
    }
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    import json
    context = json.loads(state['experiment_planner_context'])
    assert context['source_request'] == 'Use tool_a to estimate binding.'
    assert context['user_request'] == 'Estimate binding using an appropriate method.'
