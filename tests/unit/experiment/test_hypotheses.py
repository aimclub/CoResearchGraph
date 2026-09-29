"""Hypothesis seed, commit, enforce, bootstrap."""
from __future__ import annotations

import json
from types import SimpleNamespace


from .helpers import (
    _FakeInitGraph,
    _patch_research_graph,
)

def test_normalize_commit_coerces_string_nodes_arg():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import normalize_em_hypothesis_commit

    nodes_json = json.dumps(
        [
            {
                "type": "Hypothesis",
                "ref": "h1",
                "attrs": {"formulation": "Compound X inhibits target Y."},
            }
        ]
    )
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="research_commit",
                        args={"nodes": nodes_json},
                    )
                )
            ],
        )
    )
    out = normalize_em_hypothesis_commit(SimpleNamespace(state={}), response)
    assert out is not None
    fc = out.content.parts[0].function_call
    assert fc.name == "research_commit"
    # The JSON string must be repaired into a real list before dispatch.
    assert isinstance(fc.args["nodes"], list)
    assert fc.args["nodes"][0]["attrs"]["formulation"] == "Compound X inhibits target Y."


def test_normalize_commit_drops_unparseable_string_list_arg():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import normalize_em_hypothesis_commit

    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="research_commit",
                        args={
                            "nodes": [
                                {
                                    "type": "Hypothesis",
                                    "ref": "h1",
                                    "attrs": {"formulation": "Valid hypothesis statement here."},
                                }
                            ],
                            "status_updates": "not-json-at-all",
                        },
                    )
                )
            ],
        )
    )
    out = normalize_em_hypothesis_commit(SimpleNamespace(state={}), response)
    assert out is not None
    fc = out.content.parts[0].function_call
    # Unparseable list-typed arg is dropped, never shipped as a bare string.
    assert not isinstance(fc.args.get("status_updates"), str)


def test_seed_hypotheses_instructs_small_commits():
    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import seed_hypotheses_from_em_request

    state = {"experiment_source_request": "Design BTK and KRAS inhibitors."}
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="noise")])])
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=None), req)
    text = req.contents[0].parts[0].text
    assert "Hypothesis nodes ONLY" in text
    # One commit, not "at most three per call, make more calls if you need
    # more": every extra call used to get its own full allowance of active
    # hypotheses, which is how a run configured for one verified several.
    assert "ONE research_commit" in text
    assert "Extra calls do not raise the ceiling" in text


def test_the_seed_does_not_decide_how_many_hypotheses_there_are():
    """The count comes from one place — the operator's setting, via the agent's
    own instruction. This user turn used to override it with «one hypothesis per
    operation slot, do not skip a slot», and a six-step request became six
    hypotheses whatever the setting said."""
    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import seed_hypotheses_from_em_request

    state = {
        "experiment_source_request": "Profile the metabolites.",
        "experiment_operations": [
            {"operation_id": f"OP-{i}", "statement": f"step {i}"} for i in range(1, 7)
        ],
    }
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="noise")])])
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=None), req)
    text = req.contents[0].parts[0].text

    assert "one hypothesis per slot" not in text
    assert "H1 matches OP-1" not in text
    assert "one distinct hypothesis per distinct operation" not in text
    # The operations still reach the model — as the scope, which is what they are.
    assert "OP-6: step 6" in text
    assert "SCOPE" in text and "NOT a list of hypotheses" in text
    assert "stated in your instructions" in text


def test_seed_hypotheses_does_not_instruct_creating_research_question():
    """HypothesesAgent has no research_init tool and no ResearchQuestion in its
    ACL (schema.AGENT_PERMISSIONS) — the seed prompt must not tell it to try."""
    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import seed_hypotheses_from_em_request

    state = {"experiment_source_request": "Design BTK and KRAS inhibitors."}
    req = LlmRequest(contents=[types.Content(role="user", parts=[types.Part(text="noise")])])
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=None), req)
    text = req.contents[0].parts[0].text
    assert "Create a ResearchQuestion root" not in text
    assert "do NOT try to create the ResearchQuestion yourself" in text


def test_enforce_hypothesis_research_commit_forces_from_prose(monkeypatch):
    """Thought/prose-only exit → rewrite into a Hypothesis research_commit."""
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    state: dict = {}
    prose = (
        "Hypothesis 1: ATP-competitive binders fit the GSK-3β pocket better.\n"
        "Hypothesis 2: Selective hinge motifs reduce off-target kinase hits.\n"
    )
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=prose)])
    )
    out = enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    )
    assert out is not None
    fc = out.content.parts[0].function_call
    assert fc.name == "research_commit"
    assert len(fc.args["nodes"]) == 2
    assert all(n["type"] == "Hypothesis" for n in fc.args["nodes"])
    assert state["_em_hypotheses_commit_forced"] is True
    assert len(state["_em_hypotheses_from_fc"]) == 2


def test_enforce_hypothesis_research_commit_forces_from_numbered_thinking(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    thinking = (
        "Now I need to generate hypotheses:\n"
        "1. **ATP-competitive binding hypothesis**: Molecules with high structural "
        "similarity to known ATP-competitive GSK-3β inhibitors will show superior "
        "inhibitory activity due to optimal fitting in the ATP-binding pocket.\n"
        "2. **Selectivity kinase hypothesis**: Molecules designed with selective "
        "hinge-binding motifs targeting the unique Leu132 residue in GSK-3β will "
        "achieve higher selectivity and reduced off-target effects.\n"
    )
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=thinking, thought=True)],
        )
    )
    out = enforce_hypothesis_research_commit(
        SimpleNamespace(state={}, user_content=None),
        resp,
    )
    assert out is not None
    nodes = out.content.parts[0].function_call.args["nodes"]
    assert len(nodes) == 2
    assert "ATP-binding pocket" in nodes[0]["attrs"]["formulation"]


def test_enforce_hypothesis_research_commit_skips_when_tool_call_present(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_function_call(name="research_overview", args={})],
        )
    )
    out = enforce_hypothesis_research_commit(
        SimpleNamespace(state={}, user_content=None), resp
    )
    assert out is None


def test_enforce_hypothesis_research_commit_skips_when_already_have_refs(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    state = {
        "_em_hypotheses_from_fc": [
            {"hypothesis_id": "H1", "statement": "Already stashed hypothesis."}
        ]
    }
    prose = "Hypothesis 1: Should not force again because stash exists.\n"
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=prose)])
    )
    assert enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    ) is None


def test_enforce_hypothesis_research_commit_only_once(monkeypatch):
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import enforce_hypothesis_research_commit

    _patch_research_graph(monkeypatch, [])
    state: dict = {}
    prose = (
        "Hypothesis 1: First recoverable draft statement is long enough.\n"
        "Hypothesis 2: Second recoverable draft statement is long enough.\n"
    )
    resp = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=prose)])
    )
    first = enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    )
    assert first is not None
    # Clear stash to isolate the once-flag behavior.
    state["_em_hypotheses_from_fc"] = []
    second = enforce_hypothesis_research_commit(
        SimpleNamespace(state=state, user_content=None), resp
    )
    assert second is None


def test_extract_hypothesis_refs_from_neutral_multih_text():
    from CoScientist.experiments.context import extract_hypothesis_refs

    text = """
    Research ask.
    Hypotheses:
    • H1. Feature A anti-correlates with feature B under joint filters.
    • H2. Metric M decreases as complexity rises.
    H3: Protocol noise dominates sample-size gains.
    H4 - Ortholog scores are predictable from the primary target.
    H5. First cleaning pass captures most of the lift.
    """
    refs = extract_hypothesis_refs(text)
    assert [r["hypothesis_id"] for r in refs] == ["H1", "H2", "H3", "H4", "H5"]
    assert "anti-correlates" in refs[0]["statement"]
    assert "Ortholog" in refs[3]["statement"]

    merged = extract_hypothesis_refs(
        "No labels here.",
        legacy_hypotheses=[{"id": "H9", "statement": "Legacy only claim."}],
    )
    assert merged == [{"hypothesis_id": "H9", "statement": "Legacy only claim."}]


def test_extract_hypothesis_refs_from_system_hypothesis_n_prose():
    from CoScientist.experiments.context import extract_hypothesis_refs
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses
    from types import SimpleNamespace

    text = """
I have formulated one hypothesis for each.

**Hypothesis 1 (Parkinson's):**
*Statement:* Generative models can produce 10 novel BBB-permeable molecules.
*VerificationMethod:* Execute generate_case_mols with case=parkinson.

**Hypothesis 2 (Dyslipidemia):**
*Statement:* Case-conditioned generation yields molecules with bioavailability >= 80%.
*VerificationMethod:* generate_case_mols case=dyslipidemia.

Hypothesis 3 (KRAS):
Statement: Docking of candidate KRAS G12C inhibitors produces scored poses for ranking.
VerificationMethod: calculate_docking upload to S3.
"""
    refs = extract_hypothesis_refs(text)
    assert [r["hypothesis_id"] for r in refs] == ["H1", "H2", "H3"]
    assert "BBB-permeable" in refs[0]["statement"]
    assert "bioavailability" in refs[1]["statement"]
    assert "Docking" in refs[2]["statement"]

    state = {"hypotheses": text}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert [r["hypothesis_id"] for r in state["hypothesis_refs"]] == ["H1", "H2", "H3"]


def test_commit_experiment_hypotheses_normalizes_and_fallbacks():
    from types import SimpleNamespace

    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    state: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "KRAS binders exist."},
                {"hypothesis_id": "H2", "statement": "BTK modulators exist."},
            ]
        )
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert [r["hypothesis_id"] for r in state["hypothesis_refs"]] == ["H1", "H2"]
    assert state["hypotheses"] == state["experiment_hypotheses"] == state["hypothesis_refs"]

    empty: dict = {"hypotheses": "not-json", "experiment_source_request": "Make BBB antioxidant."}
    commit_experiment_hypotheses(SimpleNamespace(state=empty, user_content=None))
    assert empty["hypothesis_refs"][0]["hypothesis_id"] == "H1"
    assert "BBB antioxidant" in empty["hypothesis_refs"][0]["statement"]


def test_persist_and_seed_hypotheses_use_em_ask_not_tool_prep_noise():
    from types import SimpleNamespace

    from google.adk.models import LlmRequest
    from google.genai import types

    from CoScientist.experiments.hypotheses import (
        persist_experiment_em_request,
        seed_hypotheses_from_em_request,
    )

    ask = (
        "Complete as ONE stage:\n"
        "1. BTK non-covalent modulators for MS\n"
        "2. KRAS G12C inhibitors for lung cancer"
    )
    state: dict = {}
    user = types.Content(role="user", parts=[types.Part(text=ask)])
    persist_experiment_em_request(SimpleNamespace(state=state, user_content=user))
    assert state["experiment_source_request"] == ask

    # Noise must not overwrite a good ask.
    noise = types.Content(
        role="user",
        parts=[types.Part(text='{"mcp_scores":[{"index":0,"score":true}]}')],
    )
    persist_experiment_em_request(SimpleNamespace(state=state, user_content=noise))
    assert state["experiment_source_request"] == ask

    req = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part(text="[FullSetToolReranker] tools sufficient")],
            )
        ]
    )
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=noise), req)
    blob = req.contents[0].parts[0].text
    assert "BTK" in blob and "KRAS" in blob
    assert "FullSetToolReranker" not in blob
    assert "mcp_scores" not in blob
    assert state["_em_hypotheses_seeded"] is True

    # Second before_model must not wipe tool-turn history.
    req2 = LlmRequest(
        contents=[
            types.Content(role="user", parts=[types.Part(text="ASK:\nkept")]),
            types.Content(role="model", parts=[types.Part(text="overview done")]),
        ]
    )
    seed_hypotheses_from_em_request(SimpleNamespace(state=state, user_content=noise), req2)
    assert len(req2.contents) == 2
    assert req2.contents[1].parts[0].text == "overview done"


def test_commit_experiment_hypotheses_prefers_graph_style_nodes():
    from types import SimpleNamespace

    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    raw = json.dumps(
        {
            "nodes": [
                {
                    "type": "Hypothesis",
                    "ref": "h_pd",
                    "attrs": {"formulation": "PD dopamine modulators exist."},
                },
                {
                    "type": "Hypothesis",
                    "ref": "h_lipid",
                    "attrs": {"formulation": "Lipid clearance molecules exist."},
                },
                {"type": "VerificationMethod", "ref": "vm1", "attrs": {"method_type": "computational"}},
            ]
        }
    )
    state = {"hypotheses": raw}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert [r["hypothesis_id"] for r in state["hypothesis_refs"]] == ["H1", "H2"]
    assert "dopamine" in state["hypothesis_refs"][0]["statement"]
    assert "Lipid" in state["hypothesis_refs"][1]["statement"]


def test_commit_experiment_hypotheses_source_priority(monkeypatch):
    """Fixed source order — first non-empty wins: graph → FC → nodes/struct → text.

    The graph is authoritative regardless of how rich the prose channel is
    (no max-by-length heuristic). Empty sources are skipped; if all are empty,
    a separate test covers the H1 fallback.
    """
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    # Case 1: the research graph (1 node) beats a richer prose parse (3 refs).
    _patch_research_graph(monkeypatch, ["Graph-committed hypothesis."])
    prose = (
        "Hypothesis 1: Prose candidate one works.\n"
        "Hypothesis 2: Prose candidate two works.\n"
        "Hypothesis 3: Prose candidate three works.\n"
    )
    state: dict = {"hypotheses": prose}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    statements = [r["statement"] for r in state["hypothesis_refs"]]
    assert statements == ["Graph-committed hypothesis."]

    # Case 2: empty graph → the FC stash (1 ref) beats richer structured output.
    _patch_research_graph(monkeypatch, [])
    state2: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "Struct one."},
                {"hypothesis_id": "H2", "statement": "Struct two."},
                {"hypothesis_id": "H3", "statement": "Struct three."},
            ]
        ),
        "_em_hypotheses_from_fc": [
            {"hypothesis_id": "H1", "statement": "FC-committed hypothesis."}
        ],
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state2, user_content=None))
    statements2 = [r["statement"] for r in state2["hypothesis_refs"]]
    assert statements2 == ["FC-committed hypothesis."]

    # Case 3: empty graph and FC → structured output wins over prose.
    _patch_research_graph(monkeypatch, [])
    state3: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "Struct one."},
                {"hypothesis_id": "H2", "statement": "Struct two."},
            ]
        ),
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state3, user_content=None))
    statements3 = [r["statement"] for r in state3["hypothesis_refs"]]
    assert statements3 == ["Struct one.", "Struct two."]


def test_commit_experiment_hypotheses_prefers_graph_over_text(monkeypatch):
    """The graph is first in the fixed source order — it beats prose."""
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, ["Only graph hypothesis."])
    prose = "Hypothesis 1: Prose filler number 1 works."
    state: dict = {"hypotheses": prose}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert len(refs) == 1
    assert refs[0]["hypothesis_id"] == "H1"
    assert refs[0]["statement"] == "Only graph hypothesis."


def test_commit_experiment_hypotheses_prefers_fc_over_struct(monkeypatch):
    """FC (successful research_commit stash) precedes struct in the fixed order."""
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, [])
    state: dict = {
        "hypotheses": json.dumps(
            [
                {"hypothesis_id": "H1", "statement": "Struct alpha."},
                {"hypothesis_id": "H2", "statement": "Struct beta."},
            ]
        ),
        "_em_hypotheses_from_fc": [
            {"hypothesis_id": "H1", "statement": "FC alpha."},
            {"hypothesis_id": "H2", "statement": "FC beta."},
        ],
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    statements = [r["statement"] for r in state["hypothesis_refs"]]
    assert statements == ["FC alpha.", "FC beta."]


def test_commit_experiment_hypotheses_graph_beats_longer_text(monkeypatch):
    """Graph-first is absolute: 1 graph node beats 5 prose refs."""
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, ["Only graph hypothesis."])
    prose = "\n".join(
        f"Hypothesis {i}: Prose filler number {i} works." for i in range(1, 6)
    )
    state: dict = {"hypotheses": prose}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert len(refs) == 1
    assert refs[0]["hypothesis_id"] == "H1"
    assert refs[0]["statement"] == "Only graph hypothesis."


def test_commit_experiment_hypotheses_skips_postponed_graph_nodes(monkeypatch):
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(
        monkeypatch,
        ["Active KRAS hypothesis.", "Postponed leftover duplicate."],
        statuses=["formulated", "postponed"],
    )
    state: dict = {"hypotheses": ""}
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert [r["statement"] for r in refs] == ["Active KRAS hypothesis."]


def test_commit_experiment_hypotheses_all_sources_empty_falls_back_to_h1(monkeypatch):
    from CoScientist.experiments.hypotheses import commit_experiment_hypotheses

    _patch_research_graph(monkeypatch, [])
    state: dict = {
        "hypotheses": "no hypothesis markers here",
        "experiment_source_request": "Design a BBB-permeable antioxidant molecule.",
    }
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    refs = state["hypothesis_refs"]
    assert len(refs) == 1
    assert refs[0]["hypothesis_id"] == "H1"
    assert "BBB-permeable antioxidant" in refs[0]["statement"]


def test_bootstrap_research_question_if_empty_seeds_root(monkeypatch):
    """B: deterministic root bootstrap — no longer relies on OrchestratorAgent
    catching every empty-graph turn before delegating to HypothesesAgent."""
    import CoScientist.graph.research.store as research_store

    from CoScientist.experiments.hypotheses import bootstrap_research_question_if_empty

    graph = _FakeInitGraph(empty=True)
    monkeypatch.setattr(research_store, "get_research_graph", lambda ctx: graph)
    state = {"experiment_source_request": "Design a BBB-permeable antioxidant molecule."}
    bootstrap_research_question_if_empty(SimpleNamespace(state=state, user_content=None))
    assert len(graph.init_calls) == 1
    assert graph.init_calls[0]["question"] == "Design a BBB-permeable antioxidant molecule."


def test_bootstrap_research_question_if_empty_noop_when_graph_has_content(monkeypatch):
    """Must never archive/replace an already-seeded graph (e.g. Orchestrator
    already called research_init this turn)."""
    import CoScientist.graph.research.store as research_store

    from CoScientist.experiments.hypotheses import bootstrap_research_question_if_empty

    graph = _FakeInitGraph(empty=False)
    monkeypatch.setattr(research_store, "get_research_graph", lambda ctx: graph)
    state = {"experiment_source_request": "Design a BBB-permeable antioxidant molecule."}
    bootstrap_research_question_if_empty(SimpleNamespace(state=state, user_content=None))
    assert graph.init_calls == []


def test_bootstrap_research_question_if_empty_survives_store_error(monkeypatch):
    """Bootstrap must never break the run — a broken store is a no-op, not a crash."""
    import CoScientist.graph.research.store as research_store

    from CoScientist.experiments.hypotheses import bootstrap_research_question_if_empty

    def _boom(ctx):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(research_store, "get_research_graph", _boom)
    state = {"experiment_source_request": "Design a BBB-permeable antioxidant molecule."}
    bootstrap_research_question_if_empty(SimpleNamespace(state=state, user_content=None))  # no raise


def test_capture_hypotheses_after_research_commit_logs_failure(caplog):
    """D: a rejected research_commit (e.g. the ResearchQuestion-ACL error or the
    resulting malformed empty-commit retry) must leave an audit trail instead
    of silently falling through to the H1 fallback with no diagnosis."""
    from CoScientist.experiments.hypotheses import capture_hypotheses_after_research_commit

    state: dict = {}
    tool_response = {
        "ok": False,
        "message": "",
        "committed": {},
        "errors": ["empty commit — provide nodes, edges and/or status_updates"],
    }
    with caplog.at_level("WARNING", logger="CoScientist.experiments.hypotheses"):
        capture_hypotheses_after_research_commit(
            SimpleNamespace(name="research_commit"),
            {},
            SimpleNamespace(state=state),
            tool_response,
        )
    assert "EXPERIMENT_HYPOTHESES_COMMIT_FAILED" in caplog.text
    assert "empty commit" in caplog.text
    assert "_em_hypotheses_from_fc" not in state


def test_capture_hypotheses_after_research_commit_silent_on_non_hypothesis_ok_commit(caplog):
    """A successful commit with no Hypothesis nodes (e.g. VerificationMethod-only)
    is a normal case, not a failure — must not be logged as one."""
    from CoScientist.experiments.hypotheses import capture_hypotheses_after_research_commit

    state: dict = {}
    tool_response = {"ok": True, "committed": {"nodes": []}, "errors": []}
    with caplog.at_level("WARNING", logger="CoScientist.experiments.hypotheses"):
        capture_hypotheses_after_research_commit(
            SimpleNamespace(name="research_commit"),
            {"nodes": [{"type": "VerificationMethod", "ref": "vm0", "attrs": {}}]},
            SimpleNamespace(state=state),
            tool_response,
        )
    assert "EXPERIMENT_HYPOTHESES_COMMIT_FAILED" not in caplog.text
    assert state == {}

def test_normalize_em_hypothesis_commit_shrinks_and_stashes():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import (
        commit_experiment_hypotheses,
        normalize_em_hypothesis_commit,
    )

    fat_nodes = []
    for i in range(5):
        fat_nodes.append(
            {
                "type": "Hypothesis",
                "ref": f"h{i}",
                "attrs": {"formulation": f"Hypothesis about target {i} works.", "status": "formulated"},
            }
        )
        fat_nodes.append(
            {
                "type": "VerificationMethod",
                "ref": f"vm{i}",
                "attrs": {"description": "x" * 200, "protocol_steps": ["a", "b", "c"]},
            }
        )
    state: dict = {}
    resp = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="research_commit",
                    args={"nodes": fat_nodes, "edges": []},
                )
            ],
        )
    )
    out = normalize_em_hypothesis_commit(SimpleNamespace(state=state), resp)
    assert out is not None
    fc = out.content.parts[0].function_call
    assert fc.name == "research_commit"
    assert len(fc.args["nodes"]) == 3
    assert all(n["type"] == "Hypothesis" for n in fc.args["nodes"])
    assert len(state["_em_hypotheses_from_fc"]) == 3

    # even without output_key text, commit uses FC stash
    commit_experiment_hypotheses(SimpleNamespace(state=state, user_content=None))
    assert len(state["hypothesis_refs"]) == 3
    assert state["hypothesis_refs"][0]["hypothesis_id"] == "H1"



# ── the normaliser must not turn a change into a copy ───────────────────────

def _dispatched(args):
    """What `normalize_em_hypothesis_commit` would actually send."""
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.hypotheses import normalize_em_hypothesis_commit

    resp = LlmResponse(content=types.Content(
        role="model",
        parts=[types.Part.from_function_call(name="research_commit", args=args)]))
    out = normalize_em_hypothesis_commit(SimpleNamespace(state={}), resp)
    if out is None:                      # unchanged — the agent's own payload
        return args
    return dict(out.content.parts[0].function_call.args)


def test_a_draft_that_names_a_node_stays_an_update():
    """The defect behind the fourteen hypotheses.

    `{"id": "H3", "attrs": {…}}` is the documented way to change a node. The
    normaliser moved that id into `ref`, which the store reads as "create a
    node and call it that" — so ten attempts to write a formulation into H3
    produced H5…H14, each answered `ok: true`.
    """
    args = _dispatched({
        "nodes": [{"id": "H3", "attrs": {"formulation": "Гипотеза о порядке шагов " + "x" * 60}}],
    })
    draft = args["nodes"][0]
    assert draft["id"] == "H3"
    assert "ref" not in draft and "type" not in draft
    # And nothing invents a second «motivates» for a node the question already
    # points at.
    assert not args.get("edges")


def test_an_update_does_not_carry_a_display_status_onto_the_node():
    """`attrs.status` is what the card prints. When the normaliser rebuilds a
    draft it must not put «formulated» on a hypothesis since postponed — the
    card would then contradict the node. Shown on an overrun, because a
    payload that kept to the instruction is shipped exactly as written.
    """
    nodes = [{"id": "H3", "attrs": {"formulation": "y" * 80,
                                    "status": "formulated"}}]
    nodes += [{"type": "Hypothesis", "ref": "hh%d" % i,
               "attrs": {"formulation": "Гипотеза %d " % i + "q" * 60}}
              for i in range(5)]
    args = _dispatched({"nodes": nodes,
                        "status_updates": [{"id": "H3", "status": "postponed"}]})
    update = next(n for n in args["nodes"] if n.get("id") == "H3")
    assert "status" not in update["attrs"]
    # The creations keep theirs: there the attribute and the node agree.
    assert all(n["attrs"]["status"] == "formulated"
               for n in args["nodes"] if n.get("ref"))


def test_a_commit_that_kept_to_the_instruction_is_dispatched_as_written():
    """The seeded prompt asks for hypotheses only, at most three. A payload
    that obeys it used to be rewritten anyway — its criterion and its method
    were dropped, which is why that run kept re-sending them."""
    args = _dispatched({
        "nodes": [
            {"type": "Hypothesis", "ref": "h_new", "attrs": {"formulation": "z" * 80}},
            {"type": "ConfirmationCriteria", "ref": "cc_new",
             "attrs": {"threshold": "доля валидных ≥ 0.5"}},
            {"type": "VerificationMethod", "ref": "vm_new",
             "attrs": {"description": "докинг в 2FBW, Vina"}},
        ],
        "edges": [{"type": "formulated_for", "from": "#cc_new", "to": "#h_new"}],
    })
    kinds = [n.get("type") for n in args["nodes"]]
    assert kinds == ["Hypothesis", "ConfirmationCriteria", "VerificationMethod"]
    assert args["edges"][0]["type"] == "formulated_for"


def test_an_overrun_is_still_cut_down_and_keeps_the_ranking():
    """What the rewrite was built for: too many hypotheses in one call. The
    status changes ride along — they are an id and a word each, and dropping
    them lost the agent's own choice of which branch to verify."""
    args = _dispatched({
        "nodes": [{"type": "Hypothesis", "ref": "hh%d" % i,
                   "attrs": {"formulation": "Гипотеза %d " % i + "q" * 60}}
                  for i in range(5)],
        "status_updates": [{"id": "H1", "status": "postponed"}],
    })
    assert len(args["nodes"]) == 3
    assert args["status_updates"] == [{"id": "H1", "status": "postponed"}]
    assert len(args["edges"]) == 5
    assert [edge["type"] for edge in args["edges"]].count(
        "conditional_successor"
    ) == 2


def test_the_cut_falls_on_the_new_hypotheses_not_on_the_updates():
    """Five creations and one update: the update is cheap and is the one thing
    that cannot be re-derived from the text, so it survives the trim."""
    nodes = [{"id": "H1", "attrs": {"formulation": "Первая, уточнённая " + "w" * 60}}]
    nodes += [{"type": "Hypothesis", "ref": "hh%d" % i,
               "attrs": {"formulation": "Гипотеза %d " % i + "q" * 60}}
              for i in range(5)]
    args = _dispatched({"nodes": nodes})
    assert sum(1 for n in args["nodes"] if n.get("id") == "H1") == 1
    assert sum(1 for n in args["nodes"] if n.get("ref")) == 3


def test_an_overrun_keeps_the_type_on_a_draft_that_names_a_node():
    """The seeded prompt asks the model to number its own hypotheses ("H1
    matches OP-1"), so an id it writes may name nothing. Stripped of its type,
    such a draft reaches the store as a change to a node that does not exist
    and the whole commit is refused; with the type the store records it as a
    new node and says the id was not honoured."""
    nodes = [{"id": "H1", "type": "Hypothesis",
              "attrs": {"formulation": "Первая гипотеза " + "w" * 60}}]
    nodes += [{"type": "Hypothesis", "ref": "hh%d" % i,
               "attrs": {"formulation": "Гипотеза %d " % i + "q" * 60}}
              for i in range(5)]
    args = _dispatched({"nodes": nodes})
    update = next(n for n in args["nodes"] if n.get("id") == "H1")
    assert update["type"] == "Hypothesis"
    assert "ref" not in update


def test_the_key_spellings_this_module_tolerates_are_repaired_before_dispatch():
    """`node_type` and `attributes` are read everywhere in here because models
    write them, and while every commit was being rebuilt that tolerance was
    enough. Now a payload that kept to the instruction is dispatched as the
    agent wrote it, and the store has never heard of either spelling: one makes
    a node with no attrs, the other a draft with no type."""
    args = _dispatched({"nodes": [
        {"node_type": "Hypothesis", "ref": "h_new",
         "attributes": {"formulation": "Гипотеза о порядке шагов " + "z" * 60}}]})
    draft = args["nodes"][0]
    assert draft["type"] == "Hypothesis" and "node_type" not in draft
    assert draft["attrs"]["formulation"].startswith("Гипотеза о порядке шагов")
    assert "attributes" not in draft
