"""Evidence-based candidate screening and bounded BRICS enumeration.

No fitted property predictor is available in this project. RDKit descriptors
are calculations on a structure, not measurements of surfactant performance.
Literature observations belong only to the original molecule; generated
structures never inherit them. Unknown criteria remain explicitly unknown.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from itertools import islice
from typing import Literal

from google.adk.tools import ToolContext
from pydantic import BaseModel, ConfigDict, Field, model_validator

from CoScientist.microfluidics.design import fixed_target_candidates
from CoScientist.microfluidics.models import DesignCandidate, DesignCandidates, LiteratureAnalysis


class Criterion(BaseModel):
    """Inclusive numeric bounds, compared only in identical units/conditions."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(min_length=1)
    minimum: float | None = None
    maximum: float | None = None
    unit: str = ""
    conditions: str = ""

    @model_validator(mode="after")
    def valid_bounds(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError("Specify minimum or maximum")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must not exceed maximum")
        return self


class DesignRequest(BaseModel):
    """Agent translates explicit ТЗ bounds; missing bounds must not be guessed."""

    model_config = ConfigDict(extra="forbid", strict=True)
    criteria: list[Criterion] = Field(default_factory=list, max_length=40)
    required_smarts: list[str] = Field(default_factory=list, max_length=10)
    forbidden_smarts: list[str] = Field(default_factory=list, max_length=10)
    generate: bool = True
    max_candidates: int = Field(default=10, ge=1, le=30)


def _norm(value: str) -> str:
    return " ".join(value.casefold().split())


def _descriptors(mol):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    values = {
        "MolWt": (Descriptors.MolWt(mol), "g/mol"),
        "TPSA": (rdMolDescriptors.CalcTPSA(mol), "angstrom^2"),
        "HBD": (rdMolDescriptors.CalcNumHBD(mol), ""),
        "HBA": (rdMolDescriptors.CalcNumHBA(mol), ""),
        "RotatableBonds": (rdMolDescriptors.CalcNumRotatableBonds(mol), ""),
        "FormalCharge": (Chem.GetFormalCharge(mol), ""),
    }
    # Neutral single-component molecules only: a salt's summed logP can be
    # particularly misleading, as can an ion's value without pH/speciation.
    if len(Chem.GetMolFrags(mol)) == 1 and all(a.GetFormalCharge() == 0 for a in mol.GetAtoms()):
        values["MolLogP"] = (Descriptors.MolLogP(mol), "")
    return {k: (float(v), u) for k, (v, u) in values.items() if math.isfinite(v)}


_SCALAR = re.compile(r"^\s*([+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")


def _check(criterion: Criterion, descriptors, properties, has_sources: bool):
    values = []
    if criterion.name in descriptors:
        value, unit = descriptors[criterion.name]
        if _norm(unit) == _norm(criterion.unit) and not criterion.conditions:
            values.append(value)
    elif has_sources:
        for prop in properties:
            match = _SCALAR.fullmatch(prop.value)
            if (_norm(prop.name) == _norm(criterion.name) and match
                    and _norm(match[2]) == _norm(criterion.unit)
                    and _norm(prop.conditions) == _norm(criterion.conditions)):
                value = float(match[1].replace(",", "."))
                if math.isfinite(value):
                    values.append(value)
    outcomes = [
        (criterion.minimum is None or v >= criterion.minimum)
        and (criterion.maximum is None or v <= criterion.maximum)
        for v in values
    ]
    # Disagreeing observations are not a pass or an automatic rejection.
    status = "unknown" if not outcomes or len(set(outcomes)) > 1 else ("pass" if outcomes[0] else "fail")
    return {**criterion.model_dump(), "status": status, "observed": values}


def design_molecules(request: DesignRequest, structured_tz, literature_analysis) -> dict:
    """Pure computation; session access and the thread boundary are in the tool."""
    fixed = fixed_target_candidates(structured_tz, literature_analysis)
    if fixed is not None:
        return {"status": "fixed_target", "stub": False, **fixed.model_dump()}

    from rdkit import Chem
    from rdkit.Chem import BRICS

    analysis = LiteratureAnalysis.model_validate(literature_analysis or {})
    required = [Chem.MolFromSmarts(p) for p in request.required_smarts]
    forbidden = [Chem.MolFromSmarts(p) for p in request.forbidden_smarts]
    if any(p is None or p.GetNumAtoms() == 0 for p in required + forbidden):
        raise ValueError("Invalid or empty SMARTS constraint")

    gaps = list(analysis.gaps)
    gaps.append("Полное соответствие ТЗ не установлено: проверены только переданные числовые и структурные ограничения.")
    gaps.append("ККМ, МПН, растворимость и устойчивость новых структур не прогнозируются: нужна модель или эксперимент.")
    pool = {}
    rejected = []
    for analogue in analysis.analogues[:100]:
        smiles = analogue.smiles.strip()
        # Bound expensive parsing/enumeration; mixtures and salts retain ALL
        # components instead of silently dropping the counterion.
        mol = Chem.MolFromSmiles(smiles) if smiles and len(smiles) <= 2000 else None
        if (mol is None or not mol.GetNumAtoms() or mol.GetNumHeavyAtoms() > 150
                or any(a.GetAtomicNum() == 0 or a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
            rejected.append({"name": analogue.name, "smiles": smiles, "reason": "Нет пригодной конкретной структуры SMILES"})
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in pool:
            old = pool[canonical]
            for prop in analogue.properties:
                if prop not in old["properties"] and analogue.sources:
                    old["properties"].append(prop)
            old["sources"] = sorted(set(old["sources"] + analogue.sources))
            continue
        pool[canonical] = {
            "mol": mol, "name": analogue.name, "compound_class": analogue.compound_class,
            "properties": list(analogue.properties) if analogue.sources else [],
            "sources": list(analogue.sources), "source": "литература", "parents": [],
            "route_ids": [],
        }
        if analogue.properties and not analogue.sources:
            gaps.append(f"{analogue.name}: свойства без ссылок исключены из числовой проверки.")

    if len(analysis.analogues) > 100:
        gaps.append("Обработаны первые 100 аналогов (лимит вычислений).")

    # A route-first literature search often reports a product only inside a
    # synthesis route.  It must be a candidate even when the prose summary did
    # not repeat it in ``analogues``.  We accept only an explicit, parseable
    # product SMILES; names are deliberately not resolved or guessed here.
    for route_index, route in enumerate(analysis.synthesis_routes, 1):
        route_id = route.route_id.strip() or f"LIT-ROUTE-{route_index:02d}"
        smiles = route.product_smiles.strip()
        if not smiles:
            candidate = Chem.MolFromSmiles(route.product.strip())
            smiles = route.product.strip() if candidate is not None else ""
        mol = Chem.MolFromSmiles(smiles) if smiles and len(smiles) <= 2000 else None
        if mol is None or not mol.GetNumAtoms() or mol.GetNumHeavyAtoms() > 150:
            if route.product.strip():
                gaps.append(
                    f"{route.product}: продукт литературного маршрута {route.product!r} "
                    "не включён в кандидаты без явного корректного SMILES."
                )
            continue
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        if canonical in pool:
            item = pool[canonical]
            item["route_ids"] = sorted(set(item["route_ids"] + [route_id]))
            item["sources"] = sorted(set(item["sources"] + route.sources))
            continue
        pool[canonical] = {
            "mol": mol,
            "name": route.product,
            "compound_class": "продукт литературного маршрута",
            "properties": [],
            "sources": list(route.sources),
            "source": "литература",
            "parents": [],
            "route_ids": [route_id],
        }

    enumerated = 0
    if request.generate and pool:
        # Deterministic, small fragment library and depth; no global RNG state.
        fragments = {}
        for smiles, item in list(pool.items())[:20]:
            mol = item["mol"]
            if len(Chem.GetMolFrags(mol)) != 1:
                continue  # recombining salts could lose counterions/stoichiometry
            for frag in sorted(BRICS.BRICSDecompose(mol, minFragmentSize=2)):
                if "*" in frag:
                    fragments.setdefault(frag, set()).add(smiles)
        selected = sorted(fragments)[:12]
        if selected:
            fragment_mols = [Chem.MolFromSmiles(f) for f in selected]
            for mol in islice(BRICS.BRICSBuild(fragment_mols, maxDepth=1, scrambleReagents=False), 100):
                enumerated += 1
                try:
                    Chem.SanitizeMol(mol)
                except ValueError:
                    continue
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                if smiles in pool or mol.GetNumHeavyAtoms() > 150:
                    continue
                pool[smiles] = {
                    "mol": mol, "name": f"BRICS-кандидат {enumerated}", "compound_class": "",
                    "properties": [], "sources": [], "source": "дизайн",
                    # This is the seed library, not a claimed synthetic route.
                    "parents": sorted(set().union(*(fragments[f] for f in selected))),
                    "route_ids": [],
                }
        gaps.append("BRICS-перебор ограничен 20 исходными структурами, 12 фрагментами, глубиной 1 и 100 продуктами; пространство не исчерпано.")

    candidates = []
    assessments = {}
    for smiles, item in pool.items():
        mol = item["mol"]
        if any(not mol.HasSubstructMatch(p, useChirality=True) for p in required) or any(
            mol.HasSubstructMatch(p, useChirality=True) for p in forbidden
        ):
            rejected.append({"name": item["name"], "smiles": smiles, "reason": "Структурные ограничения"})
            continue
        descriptors = _descriptors(mol)
        checks = [_check(c, descriptors, item["properties"], bool(item["sources"])) for c in request.criteria]
        assessments[smiles] = checks
        if any(c["status"] == "fail" for c in checks):
            rejected.append({"name": item["name"], "smiles": smiles, "reason": "Нарушены числовые ограничения", "criteria": checks})
            continue
        passed = sum(c["status"] == "pass" for c in checks)
        unknown = len(checks) - passed
        properties = [p.model_dump() for p in item["properties"]]
        properties += [
            {"name": k, "value": f"{v:.6g} {u}".strip(), "conditions": "Расчёт RDKit по SMILES; не эксперимент"}
            for k, (v, u) in descriptors.items()
        ]
        candidate = DesignCandidate(
            name=item["name"], smiles=smiles, compound_class=item["compound_class"],
            properties=properties, source=item["source"], stub=False,
            sources=item["sources"],
            derivation=("BRICS, библиотека исходных SMILES: " + "; ".join(item["parents"])) if item["parents"] else (
                "Продукт литературного маршрута" if item["route_ids"] else "Литературный аналог"
            ),
            route_ids=item["route_ids"],
            tz_fit=f"Проверено ограничений: {passed}/{len(checks)}; неизвестно: {unknown}. Полное соответствие ТЗ не подтверждено.",
            risks="Синтезируемость и целевые эксплуатационные свойства требуют проверки." + (
                " Новая структура-гипотеза; свойства исходных аналогов не перенесены." if item["source"] == "дизайн" else ""
            ),
        )
        # Rank evidence coverage, NOT an invented chemical fitness score.
        candidates.append(((-passed, unknown, not bool(item["route_ids"]), item["source"] != "литература", smiles), candidate))
    candidates.sort(key=lambda pair: pair[0])
    chosen = [c for _, c in candidates[:request.max_candidates]]
    if not chosen:
        gaps.append("Нет подходящих кандидатов: расширьте литературный поиск или уточните ограничения.")
    result = DesignCandidates(candidates=chosen, gaps=list(dict.fromkeys(gaps)))
    return {
        "status": "ok" if chosen else "no_candidates", "stub": False,
        **result.model_dump(), "criteria_checks": assessments, "rejected": rejected,
        "enumerated": enumerated, "eligible_count": len(candidates),
        "ranking": "Число проверенных ограничений; при равенстве — литературные аналоги, затем SMILES",
    }


async def molecular_design(requirements: str, tool_context: ToolContext) -> dict:
    """Screen literature analogues and optionally enumerate BRICS hypotheses.

    Args:
        requirements: JSON object with criteria (name, minimum/maximum, unit,
            conditions), required_smarts, forbidden_smarts, generate (bool),
            max_candidates (1..30). Use only explicit ТЗ requirements; {} is
            valid. Descriptor names: MolWt (g/mol), TPSA (angstrom^2), MolLogP,
            HBD, HBA, RotatableBonds, FormalCharge (latter five unitless).
            Other properties require exact literature name/unit/conditions.
        tool_context: Framework context. Reads structured_tz/literature_analysis.

    Returns:
        Actual candidates, criterion checks, rejected structures and data gaps.
        No external requests. Unknown properties are never invented.
    """
    try:
        request = DesignRequest.model_validate_json(requirements)
        result = await asyncio.to_thread(
            design_molecules, request, tool_context.state.get("structured_tz"),
            tool_context.state.get("literature_analysis"),
        )
    except (ValueError, ImportError) as exc:
        result = {"status": "error", "stub": False, "candidates": [], "gaps": [str(exc)]}
    tool_context.state["molecular_design_result"] = result
    return result
