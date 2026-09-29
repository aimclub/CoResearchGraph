"""One ExperimentPlan, shaped for a reader instead of for a model.

``render_experiment_plan`` (review.py) flattens a plan into Markdown for the
console and for the prompt that asks for a revision. The web UI used to be
handed that same blob and showed it in a ``<pre>``: a wall of pipe-separated
table rows a human has to parse by eye, exactly when they are being asked to
approve it.

``plan_to_view`` returns the same plan as data — one dict of typed fields with
nothing formatted into it — so three places can render it in their own terms
without re-deriving the plan:

* the web HITL card renders it as a design matrix plus one card per task;
* the execution (call) graph records it as the decision it was;
* the research graph reads the per-task slice back out of ``tasks``.

Nothing here is LLM-facing, so field names stay English and stable; the web UI
holds the translations (``static/js/i18n.js``, ``plan.*``). Placeholder design
values ("unspecified", "n/a", …) come back as ``None`` rather than as their
placeholder text: a reader must be able to see at a glance that the planner
left a slot empty.
"""
from __future__ import annotations

import re
from typing import Any

from CoScientist.experiments.schemas import (
    ExperimentPlan,
    ExperimentTask,
    is_design_placeholder,
)

#: Long free text is cut here. The full plan stays in ``context.output`` and in
#: the plan JSON on the runtime, so this bounds a card, never the record.
_TEXT_LIMIT = 1200


def _text(value: Any, limit: int = _TEXT_LIMIT) -> str | None:
    """Collapsed text, or ``None`` for an empty or placeholder value."""
    if is_design_placeholder(value):
        return None
    out = re.sub(r"\s+", " ", str(value)).strip()
    if not out:
        return None
    return out if len(out) <= limit else out[: limit - 1] + "…"


def _lines(values: Any, limit: int = 400) -> list[str]:
    out: list[str] = []
    for value in values or []:
        if text := _text(value, limit):
            out.append(text)
    return out


def _servers(task: ExperimentTask) -> list[dict[str, Any]]:
    return [
        {
            "name": server.name,
            "server_id": server.server_id,
            "url": server.url,
            "source": server.source,
            "health": server.health,
            "tools": [
                {
                    "name": tool.name,
                    "description": _text(tool.description, 240),
                    "required": bool(tool.required_for_task),
                }
                for tool in server.tools
            ],
        }
        for server in task.mcp_servers
    ]


def _inputs(task: ExperimentTask) -> list[dict[str, Any]]:
    rows = []
    for ref in task.input_data:
        # The durable reference, whole: a bare key names no bucket, and the
        # reviewer is being asked whether this is the right input.
        location = ref.url or ref.workspace_path or None
        if not location and ref.s3_key:
            location = f"s3://{ref.bucket}/{ref.s3_key}" if ref.bucket else ref.s3_key
        if not location and ref.source_task_id:
            location = f"{ref.source_task_id}/{ref.source_artifact_id or '*'}"
        rows.append({
            "data_id": ref.data_id,
            "kind": ref.kind,
            "description": _text(ref.description, 240),
            "location": _text(location, 240),
            "required": bool(ref.required),
            "prepare_instruction": _text(ref.prepare_instruction, 240),
        })
    return rows


def _criteria(task: ExperimentTask) -> list[dict[str, Any]]:
    rows = []
    for criterion in task.success_criteria:
        threshold = None
        if criterion.metric and criterion.operator is not None and criterion.target is not None:
            threshold = f"{criterion.metric} {criterion.operator} {criterion.target}"
        rows.append({
            "criterion_id": criterion.criterion_id,
            "kind": criterion.kind,
            "description": _text(criterion.description),
            "threshold": threshold,
            "verification": _text(criterion.verification, 400),
            "required": bool(criterion.required),
        })
    return rows


def _design(task: ExperimentTask) -> dict[str, Any]:
    design = task.design
    return {
        "hypothesis_ref": design.hypothesis_ref,
        "also_tests": list(design.also_tests),
        "operation_ref": design.operation_ref,
        "question": _text(design.experiment_question),
        "dataset": {
            "name": _text(design.dataset.name, 240),
            "ref": _text(design.dataset.ref, 240),
            "notes": _text(design.dataset.notes, 400),
        },
        "baselines": [
            {"name": b.name, "kind": b.kind, "ref": _text(b.ref, 240)}
            for b in design.baselines
        ],
        "metrics": [
            {
                "name": m.name,
                "direction": m.direction,
                "threshold": None if m.threshold is None else str(m.threshold),
                "test": _text(m.test, 120),
                "description": _text(m.description, 240),
            }
            for m in design.metrics
        ],
        "analysis_artifacts": [
            {
                "name": a.name,
                "role": a.role,
                "prepare_via": a.prepare_via,
                "path_or_tool": _text(a.path_or_tool, 240),
            }
            for a in design.analysis_artifacts
        ],
    }


def task_to_view(task: ExperimentTask) -> dict[str, Any]:
    """One plan task with every field a reviewer needs, nothing formatted."""
    return {
        "id": task.id,
        "name": _text(task.name, 240),
        "route": task.route.value,
        "optional": bool(task.optional),
        "depends_on": list(task.depends_on),
        "est_duration_min": task.est_duration_min,
        "description": _text(task.description),
        "rationale": (
            None if task.rationale == task.description else _text(task.rationale)
        ),
        "repo_url": task.repo_url,
        "post_build_route": task.post_build_route,
        "code_assessment": {
            "requirement": task.code_assessment.requirement.value,
            "evidence": _text(task.code_assessment.evidence),
            "entrypoints": _lines(task.code_assessment.entrypoints, 240),
        },
        "design": _design(task),
        "mcp_servers": _servers(task),
        "input_data": _inputs(task),
        "launch_params": {str(k): str(v) for k, v in (task.launch_params or {}).items()},
        "success_criteria": _criteria(task),
        "expected_artifacts": [
            {
                "name": a.name,
                "role": a.role,
                "media_type": a.media_type,
                "required": bool(a.required),
                "description": _text(a.description, 400),
            }
            for a in task.expected_artifacts
        ],
        "warnings": _lines(task.warnings),
    }


def _matrix_row(view: dict[str, Any]) -> dict[str, Any]:
    """The design-matrix row for a task: the columns that fit side by side."""
    design = view["design"]
    return {
        "task_id": view["id"],
        "hypothesis": design["hypothesis_ref"],
        "question": design["question"],
        "dataset": design["dataset"]["name"],
        "baselines": [b["name"] for b in design["baselines"]],
        "metrics": [
            f"{m['name']} ({m['direction']})" for m in design["metrics"]
        ],
        "tools": [
            f"{s['name']}:{tool['name']}"
            for s in view["mcp_servers"] for tool in s["tools"]
        ],
        "artifacts": [a["name"] for a in design["analysis_artifacts"]],
        "route": view["route"],
    }


def _critique_view(critique: Any) -> dict[str, Any] | None:
    """The deterministic critique as the reviewer's own verdict on the plan."""
    if not isinstance(critique, dict) or not critique.get("verdict"):
        return None
    issues = []
    for issue in critique.get("issues") or []:
        if not isinstance(issue, dict):
            continue
        issues.append({
            "issue_id": issue.get("issue_id"),
            "severity": issue.get("severity"),
            "category": issue.get("category"),
            "task_id": issue.get("task_id"),
            "message": _text(issue.get("message")),
            "suggestion": _text(issue.get("suggestion")),
        })
    return {"verdict": critique.get("verdict"), "issues": issues}


def plan_to_view(
    plan: ExperimentPlan,
    critique: Any = None,
    *,
    status: str = "proposed",
) -> dict[str, Any]:
    """The whole plan as one renderable record.

    ``status`` says where the plan stands for whoever reads the record later:
    ``proposed`` while the human is being asked, then ``approved`` /
    ``revision_requested`` / ``paused`` once they have answered.
    """
    tasks = [task_to_view(t) for t in plan.tasks]
    return {
        "kind": "experiment_plan",
        "schema_version": plan.schema_version,
        "status": status,
        "plan_id": plan.plan_id,
        "experiment_run_id": plan.experiment_run_id,
        "revision": plan.revision,
        "created_at": plan.created_at,
        "goal": _text(plan.goal),
        "hypothesis": _text(plan.hypothesis),
        "source_request": _text(plan.source_request),
        "methods": _lines(plan.methods, 240),
        "hypotheses": [
            {"id": h.hypothesis_id, "statement": _text(h.statement)}
            for h in plan.hypotheses
        ],
        "total_est_duration_min": plan.total_est_duration_min,
        "task_count": len(tasks),
        "routes": sorted({t["route"] for t in tasks}),
        "risks": _lines(plan.risks),
        "assumptions": _lines(plan.assumptions),
        "critique": _critique_view(critique),
        "matrix": [_matrix_row(t) for t in tasks],
        "tasks": tasks,
    }


def plan_headline(view: dict[str, Any]) -> str:
    """One line for a graph card: what the plan is, in the width a card has."""
    tasks = view.get("task_count") or 0
    return (
        f"plan rev {view.get('revision')} · "
        f"{tasks} task{'' if tasks == 1 else 's'} · "
        f"{view.get('total_est_duration_min')} min"
    )


__all__ = ["plan_headline", "plan_to_view", "task_to_view"]
