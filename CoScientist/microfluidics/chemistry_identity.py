"""Canonical identities for compounds whose names are part of this deployment."""
from __future__ import annotations

from typing import Any

from google.adk.agents.callback_context import CallbackContext

from CoScientist.microfluidics.models import LiteratureAnalysis

VANILLIN_SMILES = "COc1cc(C=O)ccc1O"
VANILLIC_ACID_SMILES = "COc1cc(C(=O)O)ccc1O"

_KNOWN = {
    "vanillin": VANILLIN_SMILES,
    "ванилин": VANILLIN_SMILES,
    "vanillic acid": VANILLIC_ACID_SMILES,
    "ванилиновая кислота": VANILLIC_ACID_SMILES,
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def canonical_smiles(value: str) -> str:
    if not value.strip():
        return ""
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(value)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else ""
    except ImportError:
        return value.strip()


def known_smiles(name: str) -> str:
    return _KNOWN.get(_norm(name), "")


def normalize_literature_identities(callback_context: CallbackContext) -> None:
    """Correct known name/SMILES mismatches before selection and structure search."""
    raw = callback_context.state.get("literature_analysis")
    if not raw:
        return None
    analysis = LiteratureAnalysis.model_validate(raw)

    expected = known_smiles(analysis.target_molecule.name)
    if expected:
        analysis.target_molecule.smiles = expected
    elif analysis.target_molecule.smiles:
        analysis.target_molecule.smiles = canonical_smiles(analysis.target_molecule.smiles)

    for analogue in analysis.analogues:
        expected = known_smiles(analogue.name)
        if expected:
            analogue.smiles = expected
        elif analogue.smiles:
            analogue.smiles = canonical_smiles(analogue.smiles) or analogue.smiles

    for index, route in enumerate(analysis.synthesis_routes, 1):
        if not route.route_id.strip():
            route.route_id = f"LIT-ROUTE-{index:02d}"
        expected = known_smiles(route.product)
        if expected:
            route.product_smiles = expected
        elif route.product_smiles:
            route.product_smiles = canonical_smiles(route.product_smiles) or route.product_smiles

    callback_context.state["literature_analysis"] = analysis.model_dump()
    return None


__all__ = [
    "VANILLIC_ACID_SMILES",
    "VANILLIN_SMILES",
    "canonical_smiles",
    "known_smiles",
    "normalize_literature_identities",
]
