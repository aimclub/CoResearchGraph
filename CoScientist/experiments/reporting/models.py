"""Strict result/artifact contracts consumed by the Experiment Module runtime."""
from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from CoScientist.experiments.schemas.models import (
    ExecutionRoute,
    JsonObjectDict,
    StrictModel,
    _http_url_str,
    _require_utc,
    is_presigned_url,
)


class ScientificCheck(StrictModel):
    """Optional scientific claim status — separate from execution criteria_checks."""

    hypothesis_ref: str = Field(min_length=1)
    status: Literal["supported", "refuted", "inconclusive", "not_evaluated"]
    details: str = Field(min_length=1)


class ArtifactRef(StrictModel):
    artifact_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    role: Literal["data", "model", "plot", "report", "code", "log", "mcp_server"]
    name: str = Field(min_length=1)
    bucket: str | None = None
    s3_key: str | None = None
    workspace_path: str | None = None
    external_url: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    checksum_sha256: str | None = None
    producer_route: ExecutionRoute
    producer_tool: str | None = None
    derived_from: list[str] = Field(default_factory=list)
    created_at: datetime
    durability: Literal["managed", "workspace", "transient"]
    #: Our own copy in the session's artifact store, when one was taken.
    #:
    #: Deliberately NOT named `artifact_id` — that one is this runtime's own
    #: identity (``ART-<uuid>``) and means something else entirely. Conflating
    #: the two put ``cos-artifact:ART-…`` into a graph node, which is a link to
    #: a file that was never stored under that name.
    #:
    #: Not a canonical location either: it is a mirror OF whichever location
    #: below is canonical, so `validate_location` must not count it.
    session_artifact_id: str | None = None

    @field_validator("external_url")
    @classmethod
    def validate_external_url(cls, value: Any) -> str | None:
        return _http_url_str(value)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def validate_location(self) -> "ArtifactRef":
        managed = bool(self.bucket and self.s3_key)
        if bool(self.bucket) != bool(self.s3_key):
            raise ValueError("managed artifact requires both bucket and s3_key")
        if managed + bool(self.workspace_path) + bool(self.external_url) != 1:
            raise ValueError("artifact must have exactly one canonical location")
        if is_presigned_url(self.external_url):
            raise ValueError("ArtifactRef.external_url must not be a presigned S3 URL")
        if managed and self.durability != "managed":
            raise ValueError("bucket+s3_key artifact must have durability='managed'")
        return self


class CriterionCheck(StrictModel):
    criterion_id: str = Field(min_length=1)
    # Defaults to the historical behaviour for callers constructing a check
    # outside record_result.  The runtime replaces it with the plan's canonical
    # purpose before validation.
    purpose: Literal["execution", "assessment"] = "execution"
    passed: bool | None = None
    observed: Any | None = None
    evidence_artifact_ids: list[str] = Field(default_factory=list)
    details: str = Field(min_length=1)

    @field_validator("evidence_artifact_ids", mode="before")
    @classmethod
    def coerce_evidence_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(x) for x in value if x is not None and str(x).strip()]
        return [str(value)]

    @field_validator("details", mode="before")
    @classmethod
    def coerce_details(cls, value: Any) -> str:
        if value is None:
            return "n/a"
        text = str(value).strip()
        return text or "n/a"


class TaskResult(StrictModel):
    schema_version: Literal["task-result/0.1"]
    result_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_no: int = Field(ge=1)
    status: Literal["success", "partial", "failure", "skipped"]
    execution_status: Literal["completed", "completed_with_limits", "failed", "skipped"] = "failed"
    assessment_status: Literal["met", "not_met", "inconclusive", "not_evaluated"] = "not_evaluated"
    result_version: int = Field(default=1, ge=1)
    supersedes_result_id: str | None = None
    planned_route: ExecutionRoute
    route_used: ExecutionRoute
    started_at: datetime
    finished_at: datetime
    summary: str = Field(min_length=1)
    outputs: JsonObjectDict = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    criteria_checks: list[CriterionCheck]
    scientific_check: ScientificCheck | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def derive_separate_statuses(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        raw = dict(value)
        status = str(raw.get("status") or "")
        raw.setdefault("execution_status", {
            "success": "completed",
            "partial": "completed_with_limits",
            "failure": "failed",
            "skipped": "skipped",
        }.get(status, "failed"))
        if "assessment_status" not in raw:
            assessment = [
                item for item in (raw.get("criteria_checks") or [])
                if (
                    (isinstance(item, dict) and item.get("purpose") == "assessment")
                    or getattr(item, "purpose", None) == "assessment"
                )
            ]
            def _passed(item: Any) -> Any:
                return item.get("passed") if isinstance(item, dict) else getattr(item, "passed", None)
            if not assessment:
                raw["assessment_status"] = "not_evaluated"
            elif any(_passed(item) is False for item in assessment):
                raw["assessment_status"] = "not_met"
            elif all(_passed(item) is True for item in assessment):
                raw["assessment_status"] = "met"
            else:
                raw["assessment_status"] = "inconclusive"
        return raw

    @field_validator("started_at", "finished_at")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def validate_result(self) -> "TaskResult":
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")
        if self.status == "failure" and not (self.error_code and self.error_message):
            raise ValueError("failure requires error_code and error_message")
        if self.status == "success" and any(
            c.purpose == "execution" and c.passed is not True
            for c in self.criteria_checks
        ):
            raise ValueError("success requires all execution criteria checks to pass")
        if self.status != "failure" and self.retryable:
            raise ValueError("only technical failures may be retryable")
        for a in self.artifacts:
            if (a.plan_id, a.task_id, a.attempt_id) != (self.plan_id, self.task_id, self.attempt_id):
                raise ValueError("artifact identity must match the TaskResult attempt")
        return self


def artifact_name_from_location(value: dict[str, Any]) -> str:
    """Best-effort stable display name for a captured artifact."""
    for key in ("name", "s3_key", "workspace_path", "url", "external_url"):
        if raw := value.get(key):
            if name := PurePosixPath(str(raw).split("?", 1)[0]).name:
                return name
    return "artifact"
