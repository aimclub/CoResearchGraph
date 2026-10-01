"""SMILES extraction from RAG/paper-search results, and the ResearchAgent
after_tool callback that captures them into session state for MolDesignAgent."""
from types import SimpleNamespace

from CoScientist.agents.callbacks.research_callbacks import capture_literature_smiles
from CoScientist.chemical_utils.smiles_extraction import extract_smiles

_ANSWER_TEXT = (
    "Based on papers: a promising hybrid antioxidant is "
    "Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1, effective in automotive oils. "
    "Another reported analogue is CN(C)CCSc1ncccc1Cl. "
    "The CoScientist system and Instructions are not molecules, "
    "and file.txt or 2026 should not match either."
)


def test_extract_smiles_finds_real_molecules_only():
    found = extract_smiles(_ANSWER_TEXT)
    assert "Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1" in found
    assert "CN(C)CCSc1ncccc1Cl" in found
    assert len(found) == 2  # no false positives from prose/filenames/years


def test_extract_smiles_dedupes_and_preserves_order():
    text = f"{_ANSWER_TEXT} Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1 again."
    found = extract_smiles(text)
    assert found == ["Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1", "CN(C)CCSc1ncccc1Cl"]


def test_extract_smiles_empty_on_no_match():
    assert extract_smiles("Nothing chemical here, just words and 2026 dates.") == []
    assert extract_smiles("") == []


def _tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(state={})


def test_capture_literature_smiles_stores_state_for_rag_tool():
    ctx = _ctx()
    tool_response = {"answer": _ANSWER_TEXT, "metadata": {}}

    capture_literature_smiles(_tool("explore_scientific_database"), {}, ctx, tool_response)

    assert ctx.state["literature_smiles"] == [
        "Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1",
        "CN(C)CCSc1ncccc1Cl",
    ]
    assert "SMILES" in ctx.state["literature_smiles_summary"]
    assert "Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1" in ctx.state["literature_smiles_summary"]


def test_capture_literature_smiles_accumulates_across_calls():
    ctx = _ctx()
    capture_literature_smiles(
        _tool("explore_scientific_database"), {}, ctx, {"answer": "See CN(C)CCSc1ncccc1Cl."}
    )
    capture_literature_smiles(
        _tool("explore_my_papers"),
        {},
        ctx,
        {"answer": "Also Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1."},
    )

    assert ctx.state["literature_smiles"] == [
        "CN(C)CCSc1ncccc1Cl",
        "Cc1cccc(CC(C)NCC(=O)Nc2ccccc2)c1",
    ]


def test_capture_literature_smiles_ignores_unrelated_tools():
    ctx = _ctx()
    capture_literature_smiles(_tool("tavily_search"), {}, ctx, {"answer": _ANSWER_TEXT})
    assert "literature_smiles" not in ctx.state


def test_capture_literature_smiles_noop_when_nothing_found():
    ctx = _ctx()
    capture_literature_smiles(
        _tool("search_papers"), {}, ctx, {"papers": [{"title": "Some paper"}]}
    )
    assert "literature_smiles" not in ctx.state
