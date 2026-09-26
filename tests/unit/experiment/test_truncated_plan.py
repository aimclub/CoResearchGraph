"""A planner answer cut by the output limit is named as such in the revision."""
from __future__ import annotations

from CoScientist.experiments.review import _truncated_plan_errors


def test_a_lone_task_object_is_reported_as_a_cut_off_plan():
    errors = _truncated_plan_errors({"id": "EXP-3", "name": "Grid", "optional": False})
    assert len(errors) == 1
    assert "cut off" in errors[0] and "EXP-3" in errors[0]


def test_a_plan_or_anything_else_passes_through():
    assert _truncated_plan_errors({"schema_version": "experiment-plan/1.0", "tasks": []}) == []
    assert _truncated_plan_errors({"id": "EXP-1", "tasks": []}) == []
    assert _truncated_plan_errors({"hypothesis_id": "H1"}) == []
    assert _truncated_plan_errors("text") == []


def test_a_stuttered_name_key_on_an_artifact_is_read_as_name():
    from CoScientist.experiments.schemas.models import ExpectedArtifact

    art = ExpectedArtifact.model_validate({
        "namename": "reference_values.json", "role": "data",
        "media_type": "application/json", "required": True, "description": "Reference",
    })
    assert art.name == "reference_values.json"
    art = ExpectedArtifact.model_validate({
        "namename_removed": True, "name": "x.csv", "role": "data", "description": "d",
    }) if False else ExpectedArtifact.model_validate({
        "namename": "x.csv", "namename_removed": True, "role": "data", "description": "d",
    })
    assert art.name == "x.csv"
