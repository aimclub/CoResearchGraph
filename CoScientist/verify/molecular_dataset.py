"""Deterministic validator for a molecular dataset (evolution trajectories / pairs).

A dataset is REAL only if it actually contains chemistry: a column of RDKit-valid,
diverse SMILES. This rejects the fabrication patterns we keep seeing — toy integer
placeholders (``[1,2,3,4,5]`` / ``start_smiles=1``), a single repeated molecule,
or a synthetic fallback — regardless of what the agent claims.

Pure/deterministic: given the same file it always returns the same verdict, so it
can gate the pipeline without an LLM in the loop.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Column names that plausibly hold SMILES, in priority order.
_SMILES_HINTS = ("smiles", "canonical_smiles", "mol", "molecule", "child_smiles",
                 "start_smiles", "end_smiles", "parent_smiles", "structure")
# Column names that plausibly hold a fitness / property value.
_FITNESS_HINTS = ("fitness", "sa", "sa_score", "norm_sa_score", "score", "property",
                  "target", "qed", "value", "objective")


def _rdkit_valid(smiles: str) -> bool:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        if not isinstance(smiles, str) or not smiles.strip():
            return False
        # A bare integer / float string is never a molecule (toy-data guard).
        s = smiles.strip()
        try:
            float(s)
            return False
        except ValueError:
            pass
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False


def validate_molecular_dataset(
    path: str,
    min_valid: int = 30,
    min_unique: int = 15,
    smiles_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate that ``path`` is a CSV holding real, diverse molecules.

    Returns a dict with ``ok`` (bool), ``reasons`` (list of failure strings),
    and diagnostics (n_rows, n_valid_smiles, n_unique_valid, smiles_col,
    has_fitness, sample). ``ok`` is True only when a SMILES column has at least
    ``min_valid`` RDKit-valid rows and ``min_unique`` distinct valid molecules.
    """
    out: Dict[str, Any] = {
        "ok": False, "path": path, "reasons": [], "n_rows": 0,
        "n_valid_smiles": 0, "n_unique_valid": 0, "smiles_col": None,
        "has_fitness": False, "property_cols": [], "sample": [],
    }

    if not os.path.exists(path):
        out["reasons"].append(f"file not found: {path}")
        return out
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        out["reasons"].append(f"could not read CSV: {e}")
        return out

    out["n_rows"] = int(len(df))
    if len(df) == 0:
        out["reasons"].append("dataset is empty")
        return out

    cols = list(df.columns)
    # 1) pick the SMILES column: an explicit hint/override, else the column whose
    #    values are mostly RDKit-parseable.
    candidates: List[str] = []
    if smiles_col and smiles_col in cols:
        candidates = [smiles_col]
    else:
        candidates = [c for c in cols if c.lower() in _SMILES_HINTS] or list(cols)

    best_col, best_valid = None, -1
    sample_n = min(len(df), 200)
    for c in candidates:
        vals = df[c].dropna().astype(str).head(sample_n).tolist()
        if not vals:
            continue
        valid = sum(_rdkit_valid(v) for v in vals)
        if valid > best_valid:
            best_col, best_valid = c, valid

    if best_col is None or best_valid <= 0:
        out["reasons"].append(
            "no column contains valid SMILES — the 'molecules' are not real "
            "chemistry (e.g. integer/toy placeholders). Columns: " + ", ".join(map(str, cols)))
        return out

    out["smiles_col"] = best_col
    # 2) count over the WHOLE column (not just the sample).
    allvals = df[best_col].dropna().astype(str).tolist()
    valid_vals = [v for v in allvals if _rdkit_valid(v)]
    out["n_valid_smiles"] = len(valid_vals)
    out["n_unique_valid"] = len(set(valid_vals))
    out["sample"] = list(dict.fromkeys(valid_vals))[:10]

    # 3) fitness / property column present? Judge by CONTENT, not by an exact
    #    name match: a real dataset names its property anything (norm_sa, raw_sa,
    #    ic50, docking…), and rejecting it for the wrong spelling would block
    #    training on perfectly good data. A numeric column that is not an
    #    index/step counter is a property.
    _STRUCTURAL = {"trajectory_id", "traj_id", "trial_id", "id", "index",
                   "step", "generation", "gen", "iteration", "epoch"}
    numeric_props = []
    for c in cols:
        if str(c).lower() in _STRUCTURAL:
            continue
        try:
            series = pd.to_numeric(df[c], errors="coerce")
        except Exception:  # noqa: BLE001
            continue
        if series.notna().mean() > 0.8:  # mostly numeric -> a measured property
            numeric_props.append(str(c))
    named = [str(c) for c in cols
             if any(h in str(c).lower() for h in _FITNESS_HINTS)]
    out["property_cols"] = sorted(set(numeric_props) | set(named))
    out["has_fitness"] = bool(out["property_cols"])

    # 4) verdicts
    if out["n_valid_smiles"] < min_valid:
        out["reasons"].append(
            f"only {out['n_valid_smiles']} RDKit-valid molecules in column "
            f"'{best_col}' (need >= {min_valid}) — dataset too small or not real.")
    if out["n_unique_valid"] < min_unique:
        out["reasons"].append(
            f"only {out['n_unique_valid']} DISTINCT valid molecules (need >= "
            f"{min_unique}) — degenerate/low-diversity (mode collapse or a repeated stub).")
    if not out["has_fitness"]:
        out["reasons"].append(
            "no fitness/property column (fitness/sa/score/...) — an evolution "
            "trajectory dataset must carry the target property it was optimized for.")

    out["ok"] = not out["reasons"]
    return out
