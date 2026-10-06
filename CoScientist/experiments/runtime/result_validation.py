"""Validate requirement outcomes before any result is committed.

Shared by the agent guard and the state machine, including direct callers.
The legacy unscoped output representation retains the same strict field types.
"""
from __future__ import annotations

from typing import Any


def requirement_result_error(
    state: Any, task_id: str, result: dict[str, Any], *, require_question_answers: bool = True,
) -> dict[str, Any] | None:
    """Validate types everywhere; explicit answers remain an agent input rule.

    Direct runtime callers also serve legacy plans whose answer is in summary.
    Their requirement completeness is assessed separately by coverage.
    """
    from pydantic import ValidationError
    from CoScientist.requirements.coverage import projection_from_state
    from CoScientist.requirements.models import OutcomeRecord

    outputs = result.get("outputs")
    if outputs is None:
        outputs = {}
    scoped = outputs.get("requirements", {}) if isinstance(outputs, dict) else None
    questions = [row for row in projection_from_state(state)
                 if row["kind"] == "question" and task_id in row["closing_tasks"]]
    repair_contract = {
        "envelope_keys": ["task_id", "attempt_id", "result"],
        "required_question_outcomes": [{"id": row["id"], "question": row["formulation"]}
                                       for row in questions],
        "requirement_outcome_schema": OutcomeRecord.model_json_schema(),
        "instructions": (
            "Keep all fields inside result. Put task criteria_checks at result level as a list "
            "of {criterion_id, passed, observed, evidence_artifact_ids, details}. "
            "Put each question outcome at result.outputs.requirements[EXACT requirement id]. "
            "Inside a requirement outcome, criteria_checks is only {criterion_id: true/false/null}. "
            "Record the observed answer and answer_grounded boolean, or an inconclusive verdict "
            "with limitation; do not invent an answer. Preserve raw outputs and artifact references."
        ),
    }
    try:
        if not isinstance(outputs, dict) or not isinstance(scoped, dict):
            raise TypeError("outputs and outputs.requirements must be objects")
        for part_id, payload in (scoped.items() if scoped else [("", outputs)]):
            OutcomeRecord.model_validate({**payload, "part_id": part_id})
    except (ValidationError, TypeError, AttributeError) as exc:
        return {
            "status": "refused", "error_code": "outcome_contract_invalid",
            "message": f"Requirement outcomes must satisfy OutcomeRecord: {exc}. "
            "produced is an integer count or null; describe the result in answer/summary. "
            "Correct outputs before recording. The attempt remains open.",
            "repair_contract": repair_contract,
        }
    if not require_question_answers or result.get("status") not in {"success", "partial"}:
        return None
    missing = []
    for row in questions:
        answer = scoped.get(row["id"], {}) if scoped else outputs
        grounded = answer.get("answer_grounded", answer.get("grounded"))
        if answer.get("verdict") == "inconclusive" and answer.get("limitation"):
            continue
        if not str(answer.get("answer") or "").strip() or not isinstance(grounded, bool):
            missing.append(row["id"])
    if missing:
        return {
            "status": "refused", "error_code": "question_outcome_missing",
            "message": f"Record an explicit outcome for closing questions {missing} in "
            "outputs.requirements: answer and answer_grounded (true or false), or "
            "verdict=inconclusive with limitation. Use only observed evidence; "
            "a summary/file alone does not record an answer. The attempt remains open.",
            "repair_contract": repair_contract,
        }
    return None
