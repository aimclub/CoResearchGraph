"""A commit that names a node must reach that node, not make another one.

Every case here is taken from session_d9765b9e6de44530a3540da3aa0cd4c0, where
the operator asked for one more hypothesis and the graph answered with
fourteen, ten of them the same sentence. The store accepted all of them with
`ok: true`: a node draft could name an existing node in three different ways
and be read as a creation in all three.

Run from the repo root:  pytest tests/unit/test_graph_node_identity.py -q
"""
import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.graph.research.store import ResearchGraphStore  # noqa: E402

SAID = ("Обучая генеративный Transformer на упорядоченных траекториях, "
        "получаем бо́льшую долю сразу-валидных кандидатов.")
SRC = "HypothesesAgent"


@pytest.fixture
def store(tmp_path):
    s = ResearchGraphStore(directory=str(tmp_path))
    s.ensure_root("Можно ли получить структурно новые ингибиторы Q-сайта?")
    return s


def _hypothesis(store, said=SAID):
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_new", "attrs": {"formulation": said}}])
    assert r.ok, r.errors
    return r.committed["nodes"][0]["id"]


def _live(store, kind="Hypothesis"):
    return [n for n in store.full()["nodes"] if n["type"] == kind]


# ── an id is an identity ────────────────────────────────────────────────────

def test_an_id_beside_a_type_changes_the_node_it_names(store):
    """The form a model writes when it is being helpful.

    `{"id": "H1", "attrs": …}` was documented and worked; adding the type —
    which is true, and which every other draft in the same list carries — made
    the store create a second node and never look at the id again. That is the
    payload that put `{"selected": "true"}` on a brand-new empty hypothesis
    while H1 stayed unmarked.
    """
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"id": hid, "type": "Hypothesis", "attrs": {"priority": "high"}}])
    assert r.ok, r.errors
    live = _live(store)
    assert len(live) == 1, [n["id"] for n in live]
    assert live[0]["attrs"]["priority"] == "high"
    assert live[0]["attrs"]["formulation"] == SAID


def test_a_type_that_contradicts_the_stored_one_is_refused(store):
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"id": hid, "type": "Evidence", "attrs": {"content": "…"}}])
    assert not r.ok
    assert "is a Hypothesis, not a Evidence" in r.errors[0], r.errors


def test_a_status_on_an_update_is_called_out_rather_than_applied(store):
    """Status is a transition with a history, not a field to overwrite."""
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"id": hid, "type": "Hypothesis", "status": "postponed",
         "attrs": {"priority": "low"}}])
    assert r.ok, r.errors
    assert _live(store)[0]["status"] == "formulated"
    assert any("status_updates" in w for w in r.warnings), r.warnings


def test_an_id_that_names_nothing_still_records_the_node(store):
    """A model numbering its own draft must not lose the draft — but it has to
    hear that the number was not honoured, or it will point at it next turn."""
    r = store.commit(source=SRC, nodes=[
        {"id": "H7", "type": "Hypothesis", "ref": "h_new",
         "attrs": {"formulation": "Другая гипотеза про порядок шагов."}}])
    assert r.ok, r.errors
    assert len(_live(store)) == 1
    assert any("no node 'H7'" in w for w in r.warnings), r.warnings


# ── a ref is not an id ──────────────────────────────────────────────────────

def test_an_existing_id_used_as_a_ref_is_refused_by_name(store):
    """What the live run did ten times, each answered `ok: true`.

    A `ref` is a handle for a node created in THIS call. Writing an existing
    id there reads, to whoever wrote it, as "the node I mean"; the store read
    it as "a new node, call it that" and said nothing.
    """
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": hid,
         "attrs": {"formulation": "Переформулированная гипотеза о порядке."}}])
    assert not r.ok
    assert f"ref '{hid}' is the id of an existing node" in r.errors[0]
    # And it says what to write instead, so the retry is one edit away.
    assert f'{{"id": "{hid}"' in r.errors[0], r.errors[0]
    assert len(_live(store)) == 1


def test_a_ref_is_matched_against_ids_case_insensitively(store):
    """The live payload said `ref: "h1"`, which is how a model writes H1."""
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": hid.lower(),
         "attrs": {"formulation": "Ещё одна формулировка."}}])
    assert not r.ok and hid in r.errors[0]


def test_the_refs_the_system_writes_itself_can_never_be_ids():
    """The rule above is only safe because nothing inside the system hands the
    store a ref shaped like a node id.

    The first version of this test matched a fully literal token against whole
    source lines, and every ref the system writes is an f-string — the quoted
    text is `ps_{i}`, never `ps1` — so it inspected 25 lines, found 0 matches
    and asserted nothing at all. It would have passed on the code that caused
    the defect. This one RENDERS each template with a digit before judging it,
    and proves on synthetic lines that it can still see the old shapes.
    """
    import re

    from CoScientist.agents.callbacks import tool_callbacks
    from CoScientist.experiments import hypotheses
    from CoScientist.experiments.runtime import graph_bridge
    from CoScientist.graph.research import schema

    prefixes = sorted({spec.prefix for spec in schema.NODE_TYPES.values()})
    looks_like_an_id = re.compile(r"(?:PREFIXES)\d+$".replace(
        "PREFIXES", "|".join(prefixes)), re.I)
    # A ref as it is actually written: a literal or an f-string, in a dict
    # entry or assigned to a name ending in `ref`.
    site = re.compile(r'(?:"ref"\s*:\s*|\w*ref\w*\s*=\s*)f?"([^"]+)"')

    def offenders(text):
        out = []
        for template in site.findall(text):
            rendered = re.sub(r"\{[^}]*\}", "1", template)
            if looks_like_an_id.fullmatch(rendered):
                out.append((template, rendered))
        return out

    # It can see every shape this change had to rename — the assertion the
    # previous version could not make.
    assert offenders('ref = f"ps{i}"') == [("ps{i}", "ps1")]
    assert offenders('ref = f"xt{index}"') == [("xt{index}", "xt1")]
    assert offenders('nodes.append({"type": "GeneratedData", "ref": f"gd{i}"})')
    assert offenders('tool_ref = "e1"') == [("e1", "e1")]
    # And it lets the renamed ones through.
    assert offenders('ref = f"ps_{i}"') == []
    assert offenders('{"type": "Evidence", "ref": "e_0"}') == []
    # There are none left.
    for module in (tool_callbacks, hypotheses, graph_bridge):
        found = offenders(open(module.__file__, encoding="utf-8").read())
        assert not found, "%s: %s" % (module.__name__, found)


def test_the_examples_the_agents_are_shown_do_not_hand_them_an_id():
    """A prompt is not documentation — it is the payload the model copies.

    The worked example in the tool's own doc said `"ref"?: "e1"` and
    `use "#e1"`, and that text is rendered into the system prompt of every
    agent carrying the research-graph toolset. From the second Evidence node
    onwards, an agent following it loses its whole commit.
    """
    import re

    from CoScientist.assembly import bindings
    from CoScientist.graph.research import agent_tools, schema

    prefixes = "|".join(sorted({s.prefix for s in schema.NODE_TYPES.values()}))
    # A ref, by the two ways an example writes one: as the value of "ref", or
    # cited with a leading "#". An endpoint like "from": "E4" is NOT a ref —
    # it points at a node the graph already holds, which is what an edge is for.
    as_ref = re.compile(r'"ref"[?]?\s*:\s*"([^"]+)"')
    as_citation = re.compile(r'"#([A-Za-z_][\w-]*)"')
    an_id = re.compile(r"(?:PREFIXES)\d+$".replace("PREFIXES", prefixes), re.I)

    shown = []
    doc = bindings._RESEARCH_COMMIT_DOC
    shown += [str(u) for u in (getattr(doc, "usage", None) or ())]
    shown.append(str(getattr(doc, "summary", "") or ""))
    shown.append(agent_tools.ResearchGraphToolset.research_commit.__doc__ or "")
    for path in ("CoScientist/agents/prompts/templates.py",
                 "CoScientist/docs/research_graph.md"):
        shown.append(open(path, encoding="utf-8").read())

    for text in shown:
        for ref in as_ref.findall(text) + as_citation.findall(text):
            assert not an_id.fullmatch(ref), (
                "an agent is shown the ref %r, which is a node id — the store "
                "refuses that commit outright" % ref)


# ── the same sentence is one node ───────────────────────────────────────────

def test_the_same_formulation_reuses_the_node_instead_of_doubling_it(store):
    """The operator asked for one more hypothesis, not for the first one
    again. Re-sending what is already recorded has to be a no-op."""
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_again", "attrs": {"formulation": SAID}}])
    assert r.ok, r.errors
    assert len(_live(store)) == 1
    assert r.committed["nodes"][0]["id"] == hid
    assert r.committed["nodes"][0]["reused"] is True
    assert any("already says this" in w for w in r.warnings), r.warnings


def test_an_edge_drawn_to_a_reused_node_lands_on_it(store):
    """The whole commit has to survive the reuse — an agent that re-states a
    hypothesis while attaching its criterion must not lose the criterion."""
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_again", "attrs": {"formulation": SAID}},
        {"type": "ConfirmationCriteria", "ref": "cc_new",
         "attrs": {"threshold": "доля валидных ≥ 0.5"}},
    ], edges=[{"type": "formulated_for", "from": "#cc_new", "to": "#h_again"}])
    assert r.ok, r.errors
    assert len(_live(store)) == 1
    assert len(_live(store, "ConfirmationCriteria")) == 1
    assert [(e["from"], e["to"]) for e in r.committed["edges"]][0][1] == hid


def test_a_reused_node_takes_the_new_attributes_with_it(store):
    """Same sentence, more said about it: the additions belong on the node
    that already holds the sentence."""
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_again",
         "attrs": {"formulation": SAID, "rationale": "прямо отвечает на Q1"}}])
    assert r.ok, r.errors
    assert _live(store)[0]["attrs"]["rationale"] == "прямо отвечает на Q1"
    # …and it is named once, not once per bookkeeping step.
    assert [n["id"] for n in r.committed["nodes"]] == [hid]


def test_a_different_sentence_is_a_different_node(store):
    """The point is duplicates, not a ban on second hypotheses."""
    _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_alt",
         "attrs": {"formulation": "Выигрыш определяется составом корпуса, "
                                  "а не порядком шагов."}}])
    assert r.ok, r.errors
    assert len(_live(store)) == 2


def test_whitespace_and_case_do_not_make_a_second_node(store):
    hid = _hypothesis(store)
    restated = "  " + SAID.replace(" ", "  ").upper() + "\n"
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_again", "attrs": {"formulation": restated}}])
    assert r.ok, r.errors
    assert [n["id"] for n in _live(store)] == [hid]


def test_a_node_with_nothing_to_identify_it_is_simply_recorded(store):
    """A draft the store cannot judge must not be guessed about: two methods
    that say only «computational» are not one method."""
    for _ in range(2):
        r = store.commit(source=SRC, nodes=[
            {"type": "VerificationMethod", "ref": "vm_new",
             "attrs": {"method_type": "computational"}}])
        assert r.ok, r.errors
    assert len(_live(store, "VerificationMethod")) == 2


def test_identity_never_rests_on_a_word_from_a_fixed_vocabulary():
    """`_headline` falls back to whatever key the agent invented, which is
    right for a card and wrong for identity. Guard the difference."""
    from CoScientist.graph.research.store import _IDENTITY_ATTRS

    forbidden = {"method_type", "base_type", "subtype", "tool_type",
                 "resource_type", "status", "priority", "assignee", "route"}
    for ntype, keys in _IDENTITY_ATTRS.items():
        assert not (set(keys) & forbidden), f"{ntype}: {keys}"


def test_a_privileged_commit_may_change_what_it_may_create(store):
    """`init_research` and the experiment runtime write node types no role
    owns, through `enforce_permissions=False`. Reusing a node has to answer to
    the same flag: refusing the UPDATE while allowing the CREATE would make
    re-publishing one task result fail where publishing it had succeeded.
    """
    first = store.commit(
        source="ExperimentModule", enforce_permissions=False,
        nodes=[{"type": "ExperimentTask", "ref": "xt_0",
                "attrs": {"experiment_task_id": "EXP-1",
                          "title": "Кластеризация метаболитов"}}])
    assert first.ok, first.errors
    again = store.commit(
        source="ExperimentModule", enforce_permissions=False,
        nodes=[{"type": "ExperimentTask", "ref": "xt_0",
                "attrs": {"experiment_task_id": "EXP-1",
                          "title": "Кластеризация метаболитов",
                          "route": "mcp"}}])
    assert again.ok, again.errors
    tasks = _live(store, "ExperimentTask")
    assert len(tasks) == 1 and tasks[0]["attrs"]["route"] == "mcp"


def test_one_commit_saying_the_same_thing_twice_records_it_once(store):
    """The first commit of a run has no node to compare against — the copies
    are in the list itself. Both refs then have to mean the one node, or the
    edges drawn from them would split across a phantom."""
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_a", "attrs": {"formulation": SAID}},
        {"type": "Hypothesis", "ref": "h_b",
         "attrs": {"formulation": SAID, "priority": "high"}},
    ], edges=[{"type": "motivates", "from": "Q1", "to": "#h_b"}])
    assert r.ok, r.errors
    live = _live(store)
    assert len(live) == 1, [n["id"] for n in live]
    assert live[0]["attrs"]["priority"] == "high"
    assert r.committed["edges"][0]["to"] == live[0]["id"]
    assert any("recorded once" in w for w in r.warnings), r.warnings


# ── what identity is NOT ────────────────────────────────────────────────────

def test_two_hypotheses_may_hold_the_same_bar(store):
    """«p < 0.05» says nothing about WHICH claim it is the bar for. Keyed on
    the threshold alone, a criterion written for the second hypothesis was
    swallowed by the first one's, both `formulated_for` edges landed on one
    node, and meeting the bar for one claim met it for the other."""
    r1 = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_a", "attrs": {"formulation": SAID}},
        {"type": "ConfirmationCriteria", "ref": "cc_a", "attrs": {"threshold": "p < 0.05"}},
    ], edges=[{"type": "formulated_for", "from": "#cc_a", "to": "#h_a"}])
    r2 = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_b",
         "attrs": {"formulation": "Иная гипотеза о порядке шагов."}},
        {"type": "ConfirmationCriteria", "ref": "cc_b", "attrs": {"threshold": "p < 0.05"}},
    ], edges=[{"type": "formulated_for", "from": "#cc_b", "to": "#h_b"}])
    assert r1.ok and r2.ok, (r1.errors, r2.errors)
    assert len(_live(store, "ConfirmationCriteria")) == 2
    assert len({e["to"] for e in r1.committed["edges"] + r2.committed["edges"]}) == 2


def test_two_servers_may_offer_a_tool_of_the_same_name(store):
    """«search» is an ordinary MCP tool name. Keyed on the name alone the two
    became one card, and the survivor's `location` was the other server's."""
    r = store.commit(source="ExperimentModule", enforce_permissions=False, nodes=[
        {"type": "Tool", "ref": "tool_0",
         "attrs": {"name": "search", "location": "http://chem:8001/mcp"}},
        {"type": "Tool", "ref": "tool_1",
         "attrs": {"name": "search", "location": "http://pubmed:8002/mcp"}},
    ])
    assert r.ok, r.errors
    assert sorted((n["attrs"] or {}).get("location") for n in _live(store, "Tool")) == [
        "http://chem:8001/mcp", "http://pubmed:8002/mcp"]


def test_a_verdict_sentence_that_repeats_is_not_one_verdict(store):
    """The judge writes «Данных недостаточно для однозначного вывода.» for one
    hypothesis after another; keyed on the sentence, the second judgment
    overwrote the first one's card."""
    for _ in range(2):
        r = store.commit(source="OrchestratorAgent", enforce_permissions=False, nodes=[
            {"type": "Conclusion", "ref": "cl_new",
             "attrs": {"synthesis": "Данных недостаточно для однозначного вывода."}}])
        assert r.ok, r.errors
    assert len(_live(store, "Conclusion")) == 2


# ── the shapes that are not ambiguous at all ────────────────────────────────

def test_a_ref_naming_a_node_with_nothing_new_to_say_is_read_as_a_change(store):
    """The live payload: the agent marking which hypothesis it picked.

    `{"type": "Hypothesis", "ref": "h1", "attrs": {"selected": "true"}}` — a
    hypothesis with no formulation is not a new hypothesis under any reading,
    so refusing it would cost the criterion committed beside it. The edge drawn
    from that ref has to land on the node too.
    """
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": hid.lower(),
         "attrs": {"selected": "true", "rationale": "прямо отвечает на Q1"}},
        {"type": "ConfirmationCriteria", "ref": "cc1h",
         "attrs": {"threshold": "доля валидных ≥ 0.5"}},
    ], edges=[{"type": "formulated_for", "from": "#cc1h", "to": "#" + hid.lower()}])
    assert r.ok, r.errors
    assert len(_live(store)) == 1
    assert _live(store)[0]["attrs"]["selected"] == "true"
    assert len(_live(store, "ConfirmationCriteria")) == 1
    assert r.committed["edges"][0]["to"] == hid
    assert any("read as a change" in w for w in r.warnings), r.warnings


def test_a_ref_naming_a_node_while_saying_something_new_is_still_refused(store):
    """The ambiguous half stays refused: a draft that carries a formulation of
    its own may be a new hypothesis, and the store must not choose."""
    hid = _hypothesis(store)
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": hid,
         "attrs": {"formulation": "Совсем другая гипотеза."}}])
    assert not r.ok and "is the id of an existing node" in r.errors[0]


def test_a_formulation_longer_than_the_store_keeps_still_matches_itself(store):
    """`_truncate_attrs` caps a long attribute on the way in, so a raw draft
    and its own stored copy are different strings. Compared raw, a formulation
    over the cap could never match its twin and every resend made a node."""
    long_said = ("Гипотеза о порядке шагов эволюционного поиска. " * 90)[:3400]
    for _ in range(3):
        r = store.commit(source=SRC, nodes=[
            {"type": "Hypothesis", "ref": "h_new", "attrs": {"formulation": long_said}}])
        assert r.ok, r.errors
    assert len(_live(store)) == 1


def test_the_later_of_two_drafts_of_one_node_wins(store):
    """A model that says a thing twice in one breath is correcting itself, and
    the correction came second — the same way a live twin takes the new
    attributes. The echo names both refs, or a caller rebuilding a ref map from
    it loses whatever it was keeping under the second one."""
    r = store.commit(source=SRC, nodes=[
        {"type": "Hypothesis", "ref": "h_1",
         "attrs": {"formulation": SAID, "rationale": "черновая причина", "priority": "low"}},
        {"type": "Hypothesis", "ref": "h_2",
         "attrs": {"formulation": SAID, "rationale": "уточнённая причина", "priority": "high"}},
    ], edges=[{"type": "motivates", "from": "Q1", "to": "#h_2"}])
    assert r.ok, r.errors
    attrs = _live(store)[0]["attrs"]
    assert attrs["rationale"] == "уточнённая причина" and attrs["priority"] == "high"
    assert r.committed["nodes"][0]["refs"] == ["h_1", "h_2"]
    assert r.committed["edges"][0]["to"] == _live(store)[0]["id"]
