"""Seed a confirmed ``ResearchFrame`` into the Research Context Graph.

Maps the frame's blocks onto ``store.init_research`` (the privileged seeding
path): the question block becomes the root ResearchQuestion's attributes; the
constraint blocks become the Constraint context star; budgets/tools/empirical
base become their nodes; the criteria and cost-model blocks become
ConfirmationCriteria and CostModel. Each node is attributed to ``human`` when
its value came from the operator or the user's request, else to
``ContextInitAgent`` — so the graph records the automation-vs-human split.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from CoScientist.context_init.models import FrameBlock, ResearchFrame

AGENT_SOURCE = "ContextInitAgent"
HUMAN_SOURCE = "human"
# Statuses that mean a human supplied the value (operator form or user request).
_HUMAN_STATUSES = ("уточнено оператором", "задано заказчиком")

# Frame field name -> EmpiricalBase.base_type.
_EB_TYPE = {"datasets": "dataset", "corpora": "corpus", "trusted_kb": "knowledge_base"}
_REMAINING_LIMIT = re.compile(r"^\s*([\d.]+)\s*/\s*([\d.]+)\s*$")


def _block_source(block: FrameBlock) -> str:
    """`human` when any set field of the block came from a human, else agent."""
    for f in block.set_fields():
        if f.status in _HUMAN_STATUSES:
            return HUMAN_SOURCE
    return AGENT_SOURCE


def _content(block: FrameBlock) -> str:
    """Join a block's set fields into one readable content string."""
    return "; ".join(f"{f.name}: {f.value}".strip() for f in block.set_fields())


def frame_to_init_kwargs(frame: ResearchFrame) -> Dict[str, Any]:
    """Translate a (normalized) frame into ``store.init_research`` keyword args."""
    frame = frame.normalized()
    question_attrs: Dict[str, str] = {}
    question_source: Optional[str] = None
    constraints: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []
    resources: List[Dict[str, Any]] = []
    empirical_bases: List[Dict[str, Any]] = []
    confirmation_criteria: List[Dict[str, Any]] = []
    cost_models: List[Dict[str, Any]] = []

    for block in frame.blocks:
        set_fields = block.set_fields()
        if not set_fields and block.kind != "question":
            continue
        src = _block_source(block)

        if block.kind == "question":
            for f in set_fields:
                if f.name == "formulation":
                    continue  # the root formulation is passed separately
                question_attrs[f.name] = f.value
                if f.status in _HUMAN_STATUSES:
                    question_source = HUMAN_SOURCE
            question_source = question_source or (
                HUMAN_SOURCE if src == HUMAN_SOURCE else question_source)

        elif block.kind == "constraint":
            constraints.append({"subtype": block.subtype, "content": _content(block),
                                "source": src})

        elif block.kind == "resources":
            for f in set_fields:
                attrs: Dict[str, Any] = {"resource_type": f.name}
                m = _REMAINING_LIMIT.match(f.value)
                if m:
                    attrs["remaining"], attrs["limit"] = m.group(1), m.group(2)
                else:
                    attrs["note"] = f.value
                resources.append({"attrs": attrs, "source": src})

        elif block.kind == "tools":
            for f in set_fields:
                tools.append({"attrs": {"name": f.value, "tool_type": f.name},
                              "status": "available", "source": src})

        elif block.kind == "empirical_base":
            for f in set_fields:
                empirical_bases.append({
                    "attrs": {"base_type": _EB_TYPE.get(f.name, f.name),
                              "source_ref": f.value},
                    "source": src})

        elif block.kind == "confirmation_criteria":
            confirmation_criteria.append({
                "attrs": {f.name: f.value for f in set_fields}, "source": src})

        elif block.kind == "cost_model":
            attrs = {f.name: f.value for f in set_fields}
            if attrs:
                # CostModel's documented attr is `rule`; fold the cost rule there.
                attrs.setdefault("rule", attrs.get("cost_rule", ""))
                cost_models.append({"attrs": attrs, "source": src})

    q_block = frame.block("Вопрос исследования")
    formulation = ""
    if q_block is not None:
        for f in q_block.fields:
            if f.name == "formulation" and f.is_set():
                formulation = f.value
                break
    question = formulation or frame.original_request

    return {
        "question": question,
        "attrs": question_attrs,
        "question_source": question_source,
        "constraints": constraints,
        "tools": tools,
        "resources": resources,
        "empirical_bases": empirical_bases,
        "confirmation_criteria": confirmation_criteria,
        "cost_models": cost_models,
    }


def _dump_frame(store, frame: ResearchFrame, result: Dict[str, Any]) -> None:
    """Persist the confirmed frame next to the graph, for durable per-field
    provenance. The graph node ``source`` is coarse (block-level), so the frame's
    per-field status is the audit record of who supplied each piece."""
    directory = getattr(store, "_dir", None)
    if directory is None:
        return
    try:
        import json
        from pathlib import Path

        Path(directory).mkdir(parents=True, exist_ok=True)
        payload = {"root_id": result.get("root_id"),
                   "frame": frame.normalized().model_dump()}
        (Path(directory) / "research_frame.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # a failed sidecar dump must never break seeding


def seed_frame(store, frame: ResearchFrame) -> Dict[str, Any]:
    """Seed the confirmed frame into ``store`` (a ResearchGraphStore)."""
    kwargs = frame_to_init_kwargs(frame)
    result = store.init_research(source=AGENT_SOURCE, **kwargs)
    if result.get("ok"):
        _dump_frame(store, frame, result)
    return result


__all__ = ["frame_to_init_kwargs", "seed_frame", "AGENT_SOURCE", "HUMAN_SOURCE"]
