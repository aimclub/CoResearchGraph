"""Reaction-SMILES extraction from RAG/paper-search results, and the
ResearchAgent after_tool callback that captures them for SynthRouteAgent."""
from types import SimpleNamespace

from CoScientist.agents.callbacks.research_callbacks import capture_literature_reactions
from CoScientist.chemical_utils.smiles_extraction import extract_reactions, extract_smiles

_MANNICH_RXN = (
    "Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1.CN(C)C.C=O>>"
    "CN(C)Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"
)
_ANSWER_TEXT = (
    f"Based on papers: маршрут получения основания Манниха описан как {_MANNICH_RXN} "
    "с выходом 78%. Отдельно тот же фенол сам по себе: "
    "Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1. Также в тексте встречается сравнение "
    "5 > 3 стадий и Group II > Group I по цене."
)


def test_extract_reactions_finds_the_reaction():
    found = extract_reactions(_ANSWER_TEXT)
    assert len(found) == 1
    # canonicalized: reagent order may differ from the input, same chemistry.
    assert ">>" in found[0]
    assert "CN(C)Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1" in found[0]  # product survives canonicalization


def test_extract_reactions_ignores_bare_comparisons_and_prose():
    assert extract_reactions("5 > 3 стадий, Group II > Group I по цене.") == []
    assert extract_reactions("") == []
    assert extract_reactions("Просто текст без реакций.") == []


def test_extract_reactions_rejects_degenerate_fragments():
    # RDKit parses "CC>>" and ">>CC" fine but they have no product / no reactant.
    assert extract_reactions("Смесь CC>> ни во что не превращается.") == []
    assert extract_reactions("Продукт >>CC взялся ниоткуда.") == []


def test_molecules_and_reactions_are_mutually_exclusive():
    """A reaction SMILES never also shows up as a bare molecule, and the
    standalone molecule in the same text is still picked up by extract_smiles."""
    assert extract_smiles(_MANNICH_RXN) == []
    assert extract_smiles(_ANSWER_TEXT) == ["Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1"]


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(state={})


def test_capture_literature_reactions_stores_state_for_rag_tool():
    ctx = _ctx()
    tool_response = {"answer": _ANSWER_TEXT, "metadata": {}}

    capture_literature_reactions(_tool("explore_scientific_database"), {}, ctx, tool_response)

    assert len(ctx.state["literature_reactions"]) == 1
    assert ">>" in ctx.state["literature_reactions"][0]
    assert "Реакции" in ctx.state["literature_reactions_summary"]


def test_capture_literature_reactions_accumulates_and_dedupes_across_calls():
    ctx = _ctx()
    tool_response = {"answer": _ANSWER_TEXT}
    capture_literature_reactions(_tool("explore_scientific_database"), {}, ctx, tool_response)
    capture_literature_reactions(_tool("explore_my_papers"), {}, ctx, tool_response)  # same reaction again

    assert len(ctx.state["literature_reactions"]) == 1  # deduped, not doubled


def test_capture_literature_reactions_ignores_unrelated_tools():
    ctx = _ctx()
    capture_literature_reactions(_tool("tavily_search"), {}, ctx, {"answer": _ANSWER_TEXT})
    assert "literature_reactions" not in ctx.state


def test_capture_literature_reactions_noop_when_nothing_found():
    ctx = _ctx()
    capture_literature_reactions(
        _tool("search_papers"), {}, ctx, {"papers": [{"title": "Some paper"}]}
    )
    assert "literature_reactions" not in ctx.state
