"""Unit tests for the Research Context Graph (CoScientist.graph.research).

Exercise the store/schema/queries directly with source= strings (no ADK
contexts needed) against a tmp_path store, plus a couple of assembly-level
checks that the tools are wired into the right agents' prompts.

Run from the repo root:  pytest tests/unit/test_research_graph.py -q
"""
import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.config import get_settings  # noqa: E402
from CoScientist.graph.research import queries, schema  # noqa: E402
from CoScientist.graph.research.store import ResearchGraphStore  # noqa: E402


@pytest.fixture
def store(tmp_path):
    return ResearchGraphStore(directory=str(tmp_path))


def _init(store):
    return store.init_research(
        source="OrchestratorAgent",
        question="Does compound X inhibit target Y?",
        constraints=[{"subtype": "ethics", "content": "no animal testing"}],
        tools=[{"name": "AutoDock", "tool_type": "computational"}],
        resources=[{"resource_type": "GPU-hours", "remaining": 100, "limit": 100}],
        empirical_bases=[{"base_type": "dataset", "volume": "12k"}],
    )


# ── schema invariants ─────────────────────────────────────────────────────────

def test_edge_pairs_reference_known_types():
    for edge, pairs in schema.EDGE_TYPES.items():
        for f, t in pairs:
            assert f in schema.NODE_TYPES, f"{edge}: unknown from-type {f}"
            assert t in schema.NODE_TYPES, f"{edge}: unknown to-type {t}"


def test_transitions_reference_declared_statuses():
    for typ, pairs in schema.STATUS_TRANSITIONS.items():
        declared = set(schema.NODE_TYPES[typ].statuses)
        for f, t in pairs:
            assert f in declared and t in declared, f"{typ}: {f}->{t} not declared"


def test_node_prefixes_unique():
    prefixes = [s.prefix for s in schema.NODE_TYPES.values()]
    assert len(prefixes) == len(set(prefixes))


def test_permissions_reference_known_types_edges_transitions():
    for agent, perm in schema.AGENT_PERMISSIONS.items():
        for t in perm.create | perm.update_attrs:
            assert t in schema.NODE_TYPES, f"{agent}: unknown create type {t}"
        for edge, f, t in perm.edges:
            assert (f, t) in schema.EDGE_TYPES[edge], f"{agent}: bad edge {edge} {f}->{t}"
        for typ, f, t in perm.transitions:
            assert (f, t) in schema.STATUS_TRANSITIONS[typ], f"{agent}: bad transition {typ} {f}->{t}"


def test_permission_agents_exist_in_system(request):
    """Every AGENT_PERMISSIONS key must be a real agent in system.yaml, EXCEPT
    the virtual write-sources: 'human' (writes via HITL) and 'ValidatorAgent'
    (writes via the fully-async background validator plugin, not a sub-agent)."""
    from CoScientist.assembly.schema import get_config
    agents = set(get_config().agents)
    virtual = {"human", "ValidatorAgent"}
    for name in schema.AGENT_PERMISSIONS:
        if name in virtual:
            continue
        assert name in agents, f"AGENT_PERMISSIONS has unknown agent {name!r}"


# ── init + happy-path commit ────────────────────────────────────────────────────

def test_init_creates_root_and_star(store):
    r = _init(store)
    assert r["ok"] and r["root_id"] == "Q1"
    types = {n["type"] for n in r["committed"]["nodes"]}
    assert {"ResearchQuestion", "Constraint", "Tool", "Resource", "EmpiricalBase"} <= types
    ov = store.overview()
    assert ov["root"] == "Q1"
    # context star edges exist
    full = store.full()
    edge_types = {e["type"] for e in full["edges"]}
    assert "contextualizes" in edge_types and "defines_scope" in edge_types


def test_commit_with_refs(store):
    _init(store)
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "X binds Y"}},
               {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "computational"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h"},
               {"type": "tested_by", "from": "#h", "to": "#vm"}],
    )
    assert r.ok, r.errors
    ids = {n["type"]: n["id"] for n in r.committed["nodes"]}
    assert ids["Hypothesis"] == "H1" and ids["VerificationMethod"] == "VM1"


# ── transactionality ────────────────────────────────────────────────────────────

def test_commit_is_atomic_on_any_error(store):
    _init(store)
    before = store.full()["nodes"]
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "ok"}},
               {"type": "Evidence", "attrs": {"subtype": "literature"}}],  # not allowed for Hypotheses
    )
    assert not r.ok
    assert any("may not create 'Evidence'" in e for e in r.errors)
    assert store.full()["nodes"] == before, "partial write on a failed commit"


# ── permission / transition / edge rejections ──────────────────────────────────

def test_orchestrator_cannot_create_context_star_mid_run(store):
    """The context-star types are seeded only via research_init. The orchestrator
    must NOT be able to invent them through a normal commit (selective-context:
    Tools come from Hypotheses/Coder, not the orchestrator mid-run)."""
    _init(store)
    for t in ("Tool", "Resource", "EmpiricalBase"):
        r = store.commit(source="OrchestratorAgent",
                         nodes=[{"type": t, "attrs": {"name": "x", "base_type": "d",
                                                      "resource_type": "gpu"}}])
        assert not r.ok, f"orchestrator should not create {t} mid-run"
        assert any("may not create" in e for e in r.errors)


def test_init_still_seeds_context_star_privileged(store):
    """research_init seeds Tool/Resource/EmpiricalBase/Constraint despite the
    orchestrator's narrowed general create-set (privileged path)."""
    r = _init(store)  # declares a Tool, Resource, EmpiricalBase, Constraint
    assert r["ok"], r.get("errors")
    types = {n["type"] for n in r["committed"]["nodes"]}
    assert {"Tool", "Resource", "EmpiricalBase", "Constraint"} <= types
    # but a structurally-invalid seed is still rejected even when privileged
    bad = store.init_research(source="OrchestratorAgent", question="Q?",
                              constraints=[{"content": "no subtype"}])  # Constraint needs subtype
    assert not bad["ok"] and any("subtype" in e for e in bad["errors"])


def test_orchestrator_can_wire_constraints(store):
    """regulates/constrains had no creator before; the orchestrator now wires
    seeded Constraints to the methods/hypotheses that appear later."""
    store.init_research(source="OrchestratorAgent", question="Q?",
                        constraints=[{"subtype": "ethics", "content": "no 3R breach"}])
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}},
                        {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "lab"}}],
                 edges=[{"type": "tested_by", "from": "#h", "to": "#vm"}])
    r = store.commit(source="OrchestratorAgent",
                     edges=[{"type": "constrains", "from": "C1", "to": "H1"},
                            {"type": "regulates", "from": "C1", "to": "VM1"}])
    assert r.ok, r.errors


def test_permission_rejection_lists_allowed_types(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Tool", "attrs": {"name": "foo"}}])
    assert not r.ok
    assert any("ResearchAgent" in e and "Tool" in e for e in r.errors)


def test_bad_initial_status_rejected(store):
    _init(store)
    r = store.commit(source="HypothesesAgent",
                     nodes=[{"type": "Hypothesis", "status": "confirmed",
                             "attrs": {"formulation": "x"}}])
    assert not r.ok
    assert any("not a valid initial status" in e for e in r.errors)


def test_missing_subtype_rejected(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "attrs": {"content": "no subtype"}}])
    assert not r.ok
    assert any("requires attrs.subtype" in e for e in r.errors)


def test_wrong_edge_pair_rejected(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    # supports must go Evidence->Hypothesis; here Hypothesis->? is wrong direction
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature"}}],
                     edges=[{"type": "supports", "from": "H1", "to": "#e"}])
    assert not r.ok
    assert any("must connect Evidence → Hypothesis" in e for e in r.errors)


def test_unknown_endpoint_lists_ids(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature"}}],
                     edges=[{"type": "relates_to", "from": "#e", "to": "H99"}])
    assert not r.ok
    assert any("no node 'H99'" in e for e in r.errors)


def test_illegal_transition_rejected(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    # refuted is not reachable from formulated in one step, and Hypotheses can't do it
    r = store.commit(source="OrchestratorAgent",
                     status_updates=[{"id": "H1", "status": "confirmed"}])
    assert not r.ok
    assert any("cannot go 'formulated' → 'confirmed'" in e for e in r.errors)


# ── RU aliases, enrichment, locking, persistence ────────────────────────────────

def test_russian_aliases_accepted(store):
    _init(store)
    r = store.commit(source="HypothesesAgent",
                     nodes=[{"type": "Гипотеза", "status": "сформулирована",
                             "attrs": {"formulation": "гипотеза"}}])
    assert r.ok, r.errors
    assert store.full()["nodes"][-1]["type"] == "Hypothesis"


def test_attrs_merge_enrichment(store):
    _init(store)
    r = store.commit(source="ResearchAgent",
                     nodes=[{"id": "EB1", "attrs": {"volume": "20k", "note": "cleaned"}}])
    assert r.ok, r.errors
    eb = next(n for n in store.full()["nodes"] if n["id"] == "EB1")
    assert eb["attrs"]["volume"] == "20k" and eb["attrs"]["note"] == "cleaned"


def test_hypothesis_branch_lock(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    assert store.commit(source="OrchestratorAgent",
                        status_updates=[{"id": "H1", "status": "under_verification"}]).ok
    r = store.commit(source="OrchestratorAgent",
                     status_updates=[{"id": "H1", "status": "under_verification"}])
    assert not r.ok
    assert any("already under verification" in e for e in r.errors)


def test_ref_and_id_matching_is_case_insensitive(store):
    """LLMs define ref 'E4' then cite '#e4', and may write 'h1' for node 'H1'.
    Both must resolve (to the canonical stored key) instead of erroring."""
    _init(store)
    # ref defined uppercase ("Ev"), cited lowercase ("#ev"); bare id "q1" cited
    # lowercase. The node created here is assigned id E1 (ref is just a handle).
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "Ev", "attrs": {"subtype": "literature"}}],
                     edges=[{"type": "relates_to", "from": "#ev", "to": "q1"}])
    assert r.ok, r.errors
    edges = store.full()["edges"]
    assert any(e["type"] == "relates_to" and e["from"] == "E1" and e["to"] == "Q1"
               for e in edges)
    # status update citing the node id in the wrong case ("e1" for "E1")
    assert store.commit(source="ResearchAgent",
                        status_updates=[{"id": "e1", "status": "validated"}]).ok
    # case-variant refs in one commit are a duplicate, not two nodes
    dup = store.commit(source="ResearchAgent",
                       nodes=[{"type": "Evidence", "ref": "x1", "attrs": {"subtype": "meta"}},
                              {"type": "Evidence", "ref": "X1", "attrs": {"subtype": "meta"}}])
    assert not dup.ok and any("duplicate ref" in e for e in dup.errors)


def test_duplicate_edge_is_idempotent(store):
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}],
                 edges=[{"type": "motivates", "from": "Q1", "to": "#h"}])
    r = store.commit(source="HypothesesAgent",
                     edges=[{"type": "motivates", "from": "Q1", "to": "H1"}])
    assert r.ok
    assert any("already exists" in w for w in r.warnings)
    motivates = [e for e in store.full()["edges"] if e["type"] == "motivates"]
    assert len(motivates) == 1


def test_persistence_roundtrip_and_archive(tmp_path):
    s = ResearchGraphStore(directory=str(tmp_path))
    _init(s)
    s.commit(source="HypothesesAgent",
             nodes=[{"type": "Hypothesis", "attrs": {"formulation": "x"}}])
    before = s.full()
    s2 = ResearchGraphStore(directory=str(tmp_path))  # reload from disk
    after = s2.full()
    assert len(after["nodes"]) == len(before["nodes"])
    assert len(after["edges"]) == len(before["edges"])
    assert s2.root_id() == "Q1"
    # re-init archives the old graph and starts fresh
    r = s2.init_research(source="OrchestratorAgent", question="New question?")
    assert r["ok"] and "archived" in r
    assert list(tmp_path.glob("research_Q1_*.json")), "old graph not archived"


def test_research_generation_ids_are_unique_within_same_second(store, monkeypatch):
    import CoScientist.graph.research.store as store_module

    original_datetime = store_module.datetime

    class FixedDatetime:
        @staticmethod
        def now():
            return original_datetime(2024, 1, 1, 12, 0, 0)

    monkeypatch.setattr(store_module, "datetime", FixedDatetime)
    first = store.init_research(
        source="OrchestratorAgent",
        question="First question?",
    )
    first_id = store.full()["research_id"]
    second = store.init_research(
        source="OrchestratorAgent",
        question="Second question?",
    )
    second_id = store.full()["research_id"]

    assert first["ok"] and second["ok"]
    assert first_id != second_id


def test_no_delete_api():
    for attr in ("delete_node", "delete_edge", "remove_node", "remove_edge"):
        assert not hasattr(ResearchGraphStore, attr)


def test_search_limiter_ignores_research_tools():
    """Regression: the web-search limiter matched "search" as a SUBSTRING, so
    "re-search" tools (research_commit, …) were counted as searches and blocked
    once the cap was hit — which stopped agents recording anything in the graph.
    It must match "search" as a name token instead."""
    from CoScientist.agents.callbacks.tool_callbacks import SearchLimiter

    class _Tool:
        def __init__(self, name): self.name = name

    class _Ctx:
        def __init__(self): self.state = {}

    lim, ctx = SearchLimiter(max_searches=2), _Ctx()
    research_tools = ["research_commit", "research_context_slice", "research_overview",
                      "research_provenance", "research_init", "research_triggers",
                      "research_set_focus"]
    for _ in range(5):
        for name in research_tools:
            assert lim.limit_searches(_Tool(name), {}, ctx) is None
    assert ctx.state.get("_search_limiter_count", 0) == 0  # none counted as a search

    # Real search tools are still capped.
    lim2, ctx2 = SearchLimiter(max_searches=2), _Ctx()
    assert lim2.limit_searches(_Tool("tavily_search"), {}, ctx2) is None
    assert lim2.limit_searches(_Tool("download_papers_from_search"), {}, ctx2) is None
    blocked = lim2.limit_searches(_Tool("search_papers"), {}, ctx2)
    assert blocked is not None and "limit" in blocked["result"].lower()


# ── triggers ────────────────────────────────────────────────────────────────────

def _build_verifiable(store):
    _init(store)
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "X binds Y"}},
               {"type": "VerificationMethod", "ref": "vm", "attrs": {"method_type": "computational"}},
               {"type": "ConfirmationCriteria", "ref": "cc", "attrs": {"threshold": "<-8"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h"},
               {"type": "tested_by", "from": "#h", "to": "#vm"},
               {"type": "formulated_for", "from": "#cc", "to": "#h"},
               {"type": "requires", "from": "#h", "to": "T1"},
               {"type": "uses", "from": "#vm", "to": "T1"}],
    )


def test_ready_trigger(store):
    _build_verifiable(store)
    ready = queries.ready_hypotheses(store)["items"]
    assert [i["hypothesis"] for i in ready] == ["H1"]


def test_ready_trigger_offers_one_hypothesis_at_a_time(store):
    """Several verifiable hypotheses ⇒ exactly ONE is actionable; the rest are
    reported as queued so the orchestrator does not verify them in parallel."""
    _build_verifiable(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h2",
                         "attrs": {"formulation": "alt", "priority": "high"}}],
                 edges=[{"type": "motivates", "from": "Q1", "to": "#h2"}])
    ready = queries.ready_hypotheses(store)
    assert [i["hypothesis"] for i in ready["items"]] == ["H2"]   # higher priority wins
    assert [i["hypothesis"] for i in ready["queued"]] == ["H1"]
    assert "QUEUED" in ready["rendered"]


def test_commit_keeps_one_hypothesis_active_and_postpones_the_rest(store, monkeypatch):
    """A batch of hypotheses proposed in one commit: the selected one stays
    `formulated`, the alternatives are stored as `postponed` backlog."""
    # Pinned rather than relying on the code default: HYPOTHESES__MAX_ACTIVE in
    # .env overrides it for local runs, and this test's math (1 stays active,
    # 2 postponed) only holds for max_active == 1.
    monkeypatch.setattr(get_settings().web, "max_active_hypotheses", 1)
    _init(store)
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "a", "attrs": {"formulation": "first"}},
               {"type": "Hypothesis", "ref": "b",
                "attrs": {"formulation": "the relevant one", "selected": "true"}},
               {"type": "Hypothesis", "ref": "c", "attrs": {"formulation": "third"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#b"}])
    assert r.ok, r.errors
    statuses = {n["id"]: n["status"] for n in store.full()["nodes"]
                if n["type"] == "Hypothesis"}
    assert statuses == {"H1": "postponed", "H2": "formulated", "H3": "postponed"}
    # the warning names the cap and how to steer it, so the agent can act on it
    assert any("may be verified at a time" in w and "attrs.selected" in w
               for w in r.warnings), r.warnings
    # only the selected one is offered for verification; the backlog stays quiet
    assert [i["hypothesis"] for i in queries.ready_hypotheses(store)["items"]] == ["H2"]
    assert not queries.postponed_hypotheses(store)["rendered"]


def test_postponed_backlog_surfaces_only_when_nothing_is_active(store, monkeypatch):
    # Same pin as above: this test's second hypothesis must auto-postpone at
    # creation, which only happens when max_active == 1.
    monkeypatch.setattr(get_settings().web, "max_active_hypotheses", 1)
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "a", "attrs": {"formulation": "one"}},
                        {"type": "Hypothesis", "ref": "b", "attrs": {"formulation": "two"}}])
    assert not queries.postponed_hypotheses(store)["rendered"]   # H1 is active
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "postponed"}])
    backlog = queries.postponed_hypotheses(store)
    assert [i["hypothesis"] for i in backlog["items"]] == ["H1", "H2"]
    assert "BACKLOG" in backlog["rendered"]


def test_blocked_trigger(store):
    _build_verifiable(store)
    # a hypothesis requiring a tool that is still being created is blocked, not ready
    store.commit(source="CoderAgent", nodes=[{"type": "Tool", "ref": "t2",
                 "status": "being_created", "attrs": {"name": "custom"}}])
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h2", "attrs": {"formulation": "needs custom"}}],
                 edges=[{"type": "requires", "from": "#h2", "to": "T2"}])
    blocked = [i["hypothesis"] for i in queries.blocked_hypotheses(store)["items"]]
    assert "H2" in blocked
    assert "H1" not in blocked


def test_closable_and_missing_criteria(store):
    _build_verifiable(store)
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "under_verification"}])
    store.commit(source="ExperimentAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "computational",
                         "content": "docking -9",
                         "measured_on": "AutoDock Vina 1.2.5, 5R84 receptor"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    # CC not met yet → awaiting, not closable
    res = queries.closable_hypotheses(store)
    assert not res["items"]
    assert [i["hypothesis"] for i in res["awaiting_criteria"]] == ["H1"]
    # meet CC → closable. Criteria transitions belong to the ValidatorAgent now,
    # not the orchestrator (verdict/criteria are the judge's job).
    assert not store.commit(source="OrchestratorAgent",
                            status_updates=[{"id": "CC1", "status": "met"}]).ok
    store.commit(source="ValidatorAgent", status_updates=[{"id": "CC1", "status": "met"}])
    res2 = queries.closable_hypotheses(store)
    assert [i["hypothesis"] for i in res2["items"]] == ["H1"]


def test_refuting_evidence_trigger(store):
    _build_verifiable(store)
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "under_verification"}])
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "contradicts"}}],
                 edges=[{"type": "refutes", "from": "#e", "to": "H1"}])
    items = queries.refuting_evidence(store)["items"]
    assert items and items[0]["hypothesis"] == "H1"


def test_resources_low_trigger(store):
    _init(store)
    store.commit(source="OrchestratorAgent",
                 nodes=[{"id": "R1", "attrs": {"remaining": 5, "limit": 100}}])
    low = [i["resource"] for i in queries.resources_low(store)["items"]]
    assert "R1" in low


def test_provenance_to_root(store):
    _build_verifiable(store)
    prov = store.get_provenance("VM1")
    assert prov["root"] == "Q1"
    ids = [n["id"] for n in prov["chain"]]
    assert ids[0] == "VM1" and ids[-1] == "Q1"


# ── deterministic maintainer + async background validator ───────────────────────

def test_maintainer_auto_advances_hypothesis_on_evidence(store):
    """Store invariant (deterministic, no LLM): attaching supporting/refuting
    evidence to a `formulated` hypothesis auto-moves it to `under_verification`,
    attributed to the graph-maintainer — so the graph looks live for free."""
    _init(store)
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "x"}}])
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                             "content": "backs it"}}],
                     edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    assert r.ok, r.errors
    h1 = next(n for n in store.full()["nodes"] if n["id"] == "H1")
    assert h1["status"] == "under_verification"
    assert any(u.get("auto") for u in r.committed["status_updates"])
    assert h1["status_history"][-1]["source"] == "graph-maintainer"


def test_background_validator_judges_hypothesis(store):
    """The async validator (with an injected fake LLM) turns an under_verification
    hypothesis with evidence into a verdict + Conclusion, committed as ValidatorAgent."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e", "attrs": {"subtype": "literature",
                         "content": "strong support"}}],
                 edges=[{"type": "supports", "from": "#e", "to": "H1"}])
    assert next(n for n in store.full()["nodes"] if n["id"] == "H1")["status"] == "under_verification"

    async def fake_complete(system, user):
        return ('{"verdict":"confirmed","criteria":{"CC1":"met"},'
                '"conclusion":"X binds Y","validity_bounds":"in vitro","reason":"E1 meets CC1"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake_complete))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    nodes = {n["id"]: n for n in store.full()["nodes"]}
    assert nodes["H1"]["status"] == "confirmed"
    assert nodes["CC1"]["status"] == "met"
    assert any(n["type"] == "Conclusion" and n["source"] == "ValidatorAgent"
               for n in nodes.values())


def test_focus_autolink_relates_evidence_to_hypothesis(store):
    """Option A: Evidence committed with a focus hypothesis but no explicit link
    is auto-attached (relates_to) to that hypothesis, and the maintainer then
    advances the hypothesis to under_verification."""
    _build_verifiable(store)  # H1 formulated
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e",
                             "attrs": {"subtype": "literature", "content": "a finding"}}],
                     autolink_focus="H1")
    assert r.ok, r.errors
    edges = store.full()["edges"]
    assert any(e["type"] == "relates_to" and e["from"] == "E1" and e["to"] == "H1"
               for e in edges)
    h1 = next(n for n in store.full()["nodes"] if n["id"] == "H1")
    assert h1["status"] == "under_verification"


def test_autolink_resolves_focus_on_method_or_tool_to_hypothesis(store):
    """Broadened auto-link: evidence recorded while focused on a hypothesis'
    VerificationMethod (or Tool) still attaches to the HYPOTHESIS — the
    orchestrator often focuses on the method it is verifying, not the hypothesis."""
    _build_verifiable(store)  # H1 -tested_by-> VM1 ; H1 -requires-> T1
    # focus on the METHOD
    r = store.commit(source="ResearchAgent",
                     nodes=[{"type": "Evidence", "ref": "e",
                             "attrs": {"subtype": "literature", "content": "via method"}}],
                     autolink_focus="VM1")
    assert r.ok, r.errors
    assert any(e["type"] == "relates_to" and e["from"] == "E1" and e["to"] == "H1"
               for e in store.full()["edges"])
    assert next(n for n in store.full()["nodes"] if n["id"] == "H1")["status"] == "under_verification"
    # focus on the TOOL resolves to the same hypothesis
    r2 = store.commit(source="ResearchAgent",
                      nodes=[{"type": "Evidence", "ref": "e2",
                              "attrs": {"subtype": "literature", "content": "via tool"}}],
                      autolink_focus="T1")
    assert r2.ok, r2.errors
    assert any(e["type"] == "relates_to" and e["from"] == "E2" and e["to"] == "H1"
               for e in store.full()["edges"])


def test_validator_assigns_polarity_to_autolinked_evidence(store):
    """The validator turns an auto-linked (polarity-unknown) relates_to evidence
    into a supports/refutes edge as part of its verdict."""
    import asyncio
    import CoScientist.graph.research.validator as V

    _build_verifiable(store)
    store.commit(source="ResearchAgent",
                 nodes=[{"type": "Evidence", "ref": "e",
                         "attrs": {"subtype": "literature", "content": "supports it"}}],
                 autolink_focus="H1")

    async def fake(system, user):
        return ('{"evidence":{"E1":"supports"},"verdict":"confirmed",'
                '"criteria":{"CC1":"met"},"conclusion":"c","validity_bounds":"b","reason":"r"}')

    orig = V.research_graph
    V.research_graph = store
    try:
        res = asyncio.run(V.judge_hypothesis("H1", complete=fake))
    finally:
        V.research_graph = orig

    assert res and res["ok"], res
    full = store.full()
    assert any(e["type"] == "supports" and e["from"] == "E1" and e["to"] == "H1"
               for e in full["edges"])
    nodes = {n["id"]: n for n in full["nodes"]}
    assert nodes["H1"]["status"] == "confirmed" and nodes["CC1"]["status"] == "met"
    assert any(n["type"] == "Conclusion" for n in full["nodes"])


def test_background_validator_retries_after_failed_judgment(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        def full(self):
            return {"research_id": "research-1"}

    graph = FakeGraph()
    item = {
        "hypothesis": "H1",
        "supporting": ["E1"],
        "refuting": [],
        "related": [],
    }
    outcomes = [None, {"ok": True}]
    calls = []

    async def fake_judge(hypothesis, *, graph, expected_research_id):
        calls.append((hypothesis, expected_research_id))
        return outcomes.pop(0)

    monkeypatch.setattr(V, "_enabled", lambda: True)
    monkeypatch.setattr(V, "get_research_graph", lambda context: graph)
    monkeypatch.setattr(
        V.queries,
        "unresolved_hypotheses",
        lambda selected_graph: {"items": [item]},
    )
    monkeypatch.setattr(V, "judge_hypothesis", fake_judge)
    plugin = V.BackgroundValidatorPlugin()
    tool = SimpleNamespace(name="research_commit")

    async def trigger():
        before = set(V._TASKS)
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        scheduled = set(V._TASKS) - before
        if scheduled:
            await asyncio.gather(*scheduled)
        return len(scheduled)

    async def scenario():
        # The duplicate callback while the first task is still in flight must
        # not schedule another LLM call.
        before = set(V._TASKS)
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        first_tasks = set(V._TASKS) - before
        assert len(first_tasks) == 1
        await asyncio.gather(*first_tasks)

        assert await trigger() == 1  # retry after the failed first result
        assert await trigger() == 0  # successful signature is now completed

    asyncio.run(scenario())
    assert calls == [("H1", "research-1"), ("H1", "research-1")]


def test_background_validator_dedup_tracks_related_evidence_and_research_id(
    monkeypatch,
):
    import asyncio
    from types import SimpleNamespace
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        research_id = "research-1"

        def full(self):
            return {"research_id": self.research_id}

    graph = FakeGraph()
    item = {
        "hypothesis": "H1",
        "supporting": ["E1"],
        "refuting": [],
        "related": [],
    }
    calls = []

    async def fake_judge(hypothesis, *, graph, expected_research_id):
        calls.append((hypothesis, expected_research_id, tuple(item["related"])))
        return {"ok": True}

    monkeypatch.setattr(V, "_enabled", lambda: True)
    monkeypatch.setattr(V, "get_research_graph", lambda context: graph)
    monkeypatch.setattr(
        V.queries,
        "unresolved_hypotheses",
        lambda selected_graph: {"items": [item]},
    )
    monkeypatch.setattr(V, "judge_hypothesis", fake_judge)
    plugin = V.BackgroundValidatorPlugin()
    tool = SimpleNamespace(name="research_commit")

    async def trigger():
        before = set(V._TASKS)
        await plugin.after_tool_callback(
            tool=tool,
            tool_args={},
            tool_context=object(),
            result={},
        )
        scheduled = set(V._TASKS) - before
        if scheduled:
            await asyncio.gather(*scheduled)

    async def scenario():
        await trigger()
        await trigger()
        item["related"] = ["E2"]
        await trigger()
        graph.research_id = "research-2"
        await trigger()

    asyncio.run(scenario())
    assert calls == [
        ("H1", "research-1", ()),
        ("H1", "research-1", ("E2",)),
        ("H1", "research-2", ("E2",)),
    ]


def test_validator_discards_result_if_research_changes_during_llm_call():
    import asyncio
    import CoScientist.graph.research.validator as V

    class FakeGraph:
        research_id = "research-1"
        committed = False

        def full(self):
            return {"research_id": self.research_id}

        def get_context_slice(self, hypothesis, depth):
            return {
                "nodes": [{
                    "id": "H1",
                    "type": "Hypothesis",
                    "status": "under_verification",
                    "attrs": {"formulation": "x"},
                }],
                "edges": [],
            }

        def commit(self, **kwargs):
            self.committed = True
            raise AssertionError("a stale verdict must not be committed")

    graph = FakeGraph()

    async def fake_complete(system, user):
        graph.research_id = "research-2"
        return '{"verdict":"postponed","reason":"insufficient evidence"}'

    result = asyncio.run(V.judge_hypothesis(
        "H1",
        complete=fake_complete,
        graph=graph,
        expected_research_id="research-1",
    ))

    assert result is None
    assert not graph.committed


def test_the_view_speaks_the_reader_s_language(tmp_path):
    """A scientist reads the graph, so it must not answer in storage format."""
    from CoScientist.graph.research.store import (
        ResearchGraphStore, _fields, _headline,
    )

    # A budget is a sentence, not the record it is stored as, and what the
    # sentence already says is not repeated underneath it.
    budget = {"resource_type": "GPU-hours", "remaining": "50", "limit": "50"}
    headline = _headline("Resource", budget)
    assert headline == "GPU-hours: 50 of 50 left"
    assert _fields(budget, headline, "Resource") == {}

    store = ResearchGraphStore(directory=str(tmp_path), active_file="v.json")
    store.commit(source="OrchestratorAgent", nodes=[
        {"type": "ResearchQuestion", "attrs": {"formulation": "Does it work?"}},
    ])
    store.commit(source="HypothesesAgent", nodes=[
        {"type": "Hypothesis", "attrs": {"formulation": "It works", "priority": "high"}},
    ])

    view = {n["kind"]: n for n in store.to_view()["nodes"]}

    question = view["researchquestion"]
    assert question["type_word"] == "Вопрос", "the reader of this graph reads Russian"
    assert question["label"] == "Does it work?", "the id belongs in the panel"
    assert not question["label"].startswith("Q1")

    hypothesis = view["hypothesis"]
    assert hypothesis["type_word"] == "Гипотеза"
    assert hypothesis["status_word"] == "предложена", "`formulated` is not a word"
    assert hypothesis["input"]["Priority"] == "high", "attrs need reader-facing names"


def test_a_tested_branch_is_not_recorded_as_untried(tmp_path):
    """The verdict for "we tested it and it did not settle" must be its own."""
    from CoScientist.graph.research import queries
    from CoScientist.graph.research.store import ResearchGraphStore

    store = ResearchGraphStore(directory=str(tmp_path), active_file="v.json")
    store.commit(source="OrchestratorAgent",
                 nodes=[{"type": "ResearchQuestion", "attrs": {"formulation": "Q"}}])
    store.commit(source="HypothesesAgent",
                 nodes=[{"type": "Hypothesis",
                         "attrs": {"formulation": "H", "selected": "true"}}])
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "under_verification",
                                  "reason": "start"}])

    # While a branch is open the study is open, whatever answer is being drafted.
    assert "STUDY NOT FINISHED" in queries.study_open(store)["rendered"]

    # Only the validator may reach a verdict, and it has one for this outcome.
    denied = store.commit(source="OrchestratorAgent",
                          status_updates=[{"id": "H1", "status": "inconclusive",
                                           "reason": "no numbers"}])
    assert not denied.ok

    verdict = store.commit(source="ValidatorAgent",
                           status_updates=[{"id": "H1", "status": "inconclusive",
                                            "reason": "no quantitative results"}])
    assert verdict.ok

    node = next(n for n in store.to_view()["nodes"] if n["id"] == "H1")
    assert node["status_word"] == "проверена — без ответа"
    assert node["status"] != "postponed", "tested is not the same as never tried"


def test_a_claim_cannot_outrun_the_bar_set_for_it(tmp_path):
    """The three ways one recorded run reported a result its graph denied."""
    from CoScientist.graph.research import queries
    from CoScientist.graph.research.store import ResearchGraphStore

    store = ResearchGraphStore(directory=str(tmp_path), active_file="v.json")
    store.commit(source="OrchestratorAgent",
                 nodes=[{"type": "ResearchQuestion", "attrs": {"formulation": "Q"}}])
    store.commit(source="HypothesesAgent", nodes=[
        {"type": "Hypothesis", "ref": "h",
         "attrs": {"formulation": "hierarchical beats flat", "selected": "true"}},
        {"type": "ConfirmationCriteria", "ref": "cc",
         "attrs": {"threshold": "p < 0.05 in at least 2 of 3 environments"}},
    ], edges=[{"type": "formulated_for", "from": "#cc", "to": "#h"}])
    store.commit(source="OrchestratorAgent",
                 status_updates=[{"id": "H1", "status": "under_verification"}])

    # 1. Work happening outside the record is visible while it is still fixable.
    assert "WORK NOT IN THE RECORD" in queries.verdict_without_evidence(store)["rendered"]

    # 2. Measured evidence must say what it was measured on. A run benchmarked a
    #    local stand-in for a repository that 404'd and reported it as the real
    #    comparison; the substitution appeared nowhere in the record.
    refused = store.commit(source="CoderAgent", nodes=[
        {"type": "Evidence", "attrs": {"subtype": "computational",
                                       "content": "hierarchical wins, p=0.0006"}}])
    assert not refused.ok
    assert "measured_on" in refused.errors[0]

    store.commit(source="CoderAgent", nodes=[
        {"type": "Evidence", "ref": "e",
         "attrs": {"subtype": "computational", "content": "hierarchical wins in 2 of 3",
                   "measured_on": "local reimplementation; upstream repo 404"}}],
        edges=[{"type": "supports", "from": "#e", "to": "H1"}])

    # 3. Confirmation cannot outrun the criteria written for the hypothesis.
    denied = store.commit(source="ValidatorAgent",
                          status_updates=[{"id": "H1", "status": "confirmed",
                                           "reason": "the benchmark says so"}])
    assert not denied.ok
    assert "CC1" in denied.errors[0]

    # Marking the criterion met in the SAME commit is the shape the schema asks
    # for, and the verdict may lead the list.
    allowed = store.commit(source="ValidatorAgent", status_updates=[
        {"id": "H1", "status": "confirmed", "reason": "criterion met"},
        {"id": "CC1", "status": "met"},
    ])
    assert allowed.ok, allowed.errors


def test_a_session_can_reach_every_study_it_holds(tmp_path):
    """research_init archives; only the live study used to be reachable."""
    from CoScientist.graph.research.store import ResearchGraphStore

    store = ResearchGraphStore(directory=str(tmp_path),
                               active_file="research_active.json")
    store.commit(source="OrchestratorAgent", nodes=[
        {"type": "ResearchQuestion", "attrs": {"formulation": "First question"}}])
    store.reset(archive=True)
    store.commit(source="OrchestratorAgent", nodes=[
        {"type": "ResearchQuestion", "attrs": {"formulation": "Second question"}}])

    studies = store.studies()
    assert [s["live"] for s in studies] == [True, False], "the live study comes first"
    assert studies[0]["label"] == "Second question"
    assert studies[1]["label"] == "First question", "a study is known by its question"

    # The default is the live study, and it lists the others beside it.
    live = store.view_of()
    assert live["study_id"] == "active"
    assert len(live["studies"]) == 2

    archived = store.view_of(studies[1]["study_id"])
    labels = [n["label"] for n in archived["nodes"] if not n.get("overlay")]
    assert labels == ["First question"]
    assert len(archived["studies"]) == 2, "the picker stays populated"
def test_tool_needs_adaptation_to_available_transition(store):
    _init(store)
    # T1 is available from _init; let's create a tool with needs_adaptation
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Tool", "status": "needs_adaptation", "attrs": {"name": "CVAE harness"}}],
    )
    assert r.ok, r.errors
    tool_id = r.committed["nodes"][0]["id"]

    # OrchestratorAgent can transition it to available
    r2 = store.commit(
        source="OrchestratorAgent",
        status_updates=[{"id": tool_id, "status": "available"}],
    )
    assert r2.ok, r2.errors
    tool_node = next(n for n in store.full()["nodes"] if n["id"] == tool_id)
    assert tool_node["status"] == "available"


def test_orchestrator_can_commit_evidence(store):
    _init(store)
    # HypothesesAgent creates hypothesis
    r = store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h", "attrs": {"formulation": "CVAE models molecule properties"}}],
    )
    assert r.ok, r.errors

    # OrchestratorAgent commits Evidence and links to H1
    r2 = store.commit(
        source="OrchestratorAgent",
        # `measured_on` is required of computational evidence: what the number
        # was actually measured on is the difference between a result and a claim.
        nodes=[{"type": "Evidence", "ref": "e", "attrs": {
            "subtype": "computational", "content": "95% validity",
            "measured_on": "ZINC-250k held-out split",
        }}],
        edges=[{"type": "supports", "from": "#e", "to": "H1"}],
    )
    assert r2.ok, r2.errors
    ev_node = next(n for n in store.full()["nodes"] if n["type"] == "Evidence")
    assert ev_node["status"] == "obtained"
    assert any(e["type"] == "supports" and e["to"] == "H1" for e in store.full()["edges"])



def test_the_view_links_tool_calls_without_drawing_them(store):
    """Provenance rides on the node; it is not a node of its own.

    One Evidence can be the product of a dozen calls. Drawn as nodes they
    outnumber the findings and bury what the reader came for, so the graph
    shows the research record and the card carries the links into the log.
    """
    _init(store)
    r = store.commit(
        source="OrchestratorAgent",
        nodes=[{"type": "Evidence", "ref": "e", "attrs": {
            "subtype": "computational", "content": "95% validity",
            "measured_on": "ZINC-250k held-out split",
            "_provenance": [
                {"tool": "execute_bash", "exec_id": "tool:1", "result": "ok"},
                {"tool": "tavily_search", "exec_id": "tool:2", "result": "12 papers"},
            ],
        }}],
    )
    assert r.ok, r.errors

    view = store.to_view()
    assert not [n for n in view["nodes"] if n["kind"] == "toolcall"]
    assert not [e for e in view["edges"] if e["type"] == "via"]

    evidence = next(n for n in view["nodes"] if n["kind"] == "evidence")
    assert [p["tool"] for p in evidence["provenance"]] == ["execute_bash", "tavily_search"]
    # The panel needs the call id to link into the execution log.
    assert [p["exec_id"] for p in evidence["provenance"]] == ["tool:1", "tool:2"]
    # And the raw bookkeeping key never reaches the reader's field list.
    assert "_provenance" not in (evidence["input"] or {})


def test_the_graph_page_sends_a_tool_call_to_its_own_tab():
    """Following a result back to the call must not cost you the graph.

    The link also cannot name a node any more: calls are folded into the agent
    that made them, so the page has to find the request holding the call.
    """
    from starlette.testclient import TestClient

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        page = client.get("/graph").text

    assert 'window.open("/graph?"' in page and '"_blank"' in page
    assert "openTurnHolding" in page, "a deep link must locate the request itself"
    # And the canvas keeps the scientific record, not the engineering around it.
    assert "RESEARCH_SHOWN" in page
    for gone in ('id="legend"', 'id="tiles"', 'id="pbar"'):
        assert gone not in page, gone


def test_the_graph_page_only_wires_controls_it_still_has():
    """A handler left behind on a removed control takes the whole page down.

    `document.getElementById("prov").onchange = …` survived the removal of the
    tool-call toggle. It throws on load, so everything after it — honouring
    `?view=`, building the network, the first poll — never ran, and every link
    into the page landed on the default view with nothing drawn.
    """
    import re

    from starlette.testclient import TestClient

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        page = client.get("/graph").text

    present = set(re.findall(r'id="([^"]+)"', page))
    wired = set(re.findall(r'getElementById\("([^"]+)"\)', page))
    assert not (wired - present), sorted(wired - present)


def test_a_headline_never_stands_in_for_the_thing_it_should_say(store):
    """The stock word won over content sitting under an unexpected key.

    Agents name the field whatever the prompt made natural. When none of the
    anticipated names matched, the card said "acceptance criteria" — which the
    type name already says — while the criteria themselves sat under details.
    """
    from CoScientist.graph.research.store import _headline

    invented = {"confirmation_criteria": "H1 подтверждается, если валидность ≥ 0.9"}
    assert _headline("ConfirmationCriteria", invented).startswith("H1 подтверждается")

    # Even a name nobody listed reaches the reader rather than a stock word.
    unheard_of = {"acceptance_rule_v2": "два независимых прогона сходятся"}
    said = _headline("ConfirmationCriteria", unheard_of)
    assert "два независимых прогона" in said

    # The same holds for the other types that had a stock word.
    assert _headline("Tool", {"purpose": "docking"}) != "tool"
    assert _headline("VerificationMethod", {"plan": "ретроспектива"}) != "method"
    assert _headline("Conclusion", {"verdict": "подтверждено"}) != "conclusion"


def test_what_the_headline_says_is_not_repeated_under_details(store):
    """One sentence, once: the panel used to print it twice."""
    from CoScientist.graph.research.store import _fields, _headline

    attrs = {"confirmation_criteria": "валидность ≥ 0.9", "reproducibility": "2 прогона"}
    headline = _headline("ConfirmationCriteria", attrs)
    assert "confirmation_criteria" not in _fields(attrs, headline, "ConfirmationCriteria")


def test_the_orchestrator_can_say_why_a_branch_was_left_untested(store):
    """The triggers tell it to; the store used to refuse.

    `study_open` ends with "commit attrs.not_tested_reason saying why the
    verdict already obtained makes testing it unnecessary" — and the
    orchestrator, the only agent that reads triggers, could not write that
    field on a Hypothesis. So the study stayed open, and the run learned to
    answer that closing it was somebody else's job and not critical anyway.
    """
    _init(store)
    r = store.commit(source="HypothesesAgent", nodes=[
        {"type": "Hypothesis", "attrs": {"formulation": "H one"}},
        {"type": "Hypothesis", "attrs": {"formulation": "H two"}},
    ])
    assert r.ok, r.errors
    backlog = r.committed["nodes"][1]["id"]

    r2 = store.commit(
        source="OrchestratorAgent",
        nodes=[{"id": backlog, "attrs": {
            "not_tested_reason": "H1 подтверждена; проверять эту ветку незачем"}}],
    )
    assert r2.ok, r2.errors
    node = next(n for n in store.full()["nodes"] if n["id"] == backlog)
    assert node["attrs"]["not_tested_reason"].startswith("H1 подтверждена")


def test_that_grant_does_not_open_the_rest_of_the_hypothesis(store):
    """One field, not the node: the formulation stays with the agent that owns it."""
    _init(store)
    r = store.commit(source="HypothesesAgent", nodes=[
        {"type": "Hypothesis", "attrs": {"formulation": "H one"}}])
    hid = r.committed["nodes"][0]["id"]

    denied = store.commit(
        source="OrchestratorAgent",
        nodes=[{"id": hid, "attrs": {"formulation": "rewritten"}}])
    assert not denied.ok
    assert "not_tested_reason" in " ".join(denied.errors), denied.errors

    # Nor smuggled in beside a field it may write.
    mixed = store.commit(
        source="OrchestratorAgent",
        nodes=[{"id": hid, "attrs": {"not_tested_reason": "ok", "priority": "high"}}])
    assert not mixed.ok


def test_the_prompt_tells_the_agent_about_the_field_it_may_write():
    """Prompt and enforcement come from one table, so neither can drift."""
    from CoScientist.graph.research import schema

    summary = schema.permitted_summary("OrchestratorAgent")
    assert any("not_tested_reason" in line for line in summary["update_attrs"]), summary
    validator = schema.permitted_summary("ValidatorAgent")
    assert any("inconclusive_reason" in line for line in validator["update_attrs"])
