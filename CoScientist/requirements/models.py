"""Contracts for a normalized research statement and its provenance.

Kinds are only ``question``, ``hypothesis`` and ``deliverable``. There is no
scope / claim_form / unknown_kind / artifact_kind axis.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, StrictBool, field_validator, model_validator

Kind = Literal["question", "hypothesis", "deliverable"]
Role = Literal["user_obligation", "supporting"]
SourceKind = Literal["user_request", "clarification", "frame_field", "agent_proposal"]
CriterionOrigin = Literal["user", "agent_method", "technical"]
HypothesisMode = Literal["check", "formulate"]
Fulfillment = Literal["fulfilled", "partial", "unfulfilled"]


class Provenance(BaseModel):
    """Where a part came from. A model may not mark a part accepted by writing a string."""

    source: SourceKind = "agent_proposal"
    quote: str = ""
    message_id: str = ""
    field_id: str = ""
    version: str = ""
    accepted: bool = False


class Criterion(BaseModel):
    """An evaluation condition attached to one part, not a fourth path."""

    id: str = ""
    text: str
    origin: CriterionOrigin = "agent_method"
    quote: str = ""
    threshold: Optional[str] = None
    obligation: bool = False
    technical: bool = False


class Volume(BaseModel):
    """Requested amount, the cap of this run, and what was actually produced.

    None means the number was not given. A conditional limit is not a fact
    that the requested amount exceeded a threshold.
    """

    requested: Optional[int] = None
    run_cap: Optional[int] = None
    produced: Optional[int] = None
    input_count: Optional[int] = None
    limit_scope: str = ""
    requested_quote: str = ""
    run_cap_quote: str = ""
    input_count_quote: str = ""


class RunCondition(BaseModel):
    """A condition stays a condition. It is not an obligation and not a fact."""

    text: str
    quote: str = ""


class StatementPartDraft(BaseModel):
    """Semantic decisions made by the parser, never inferred from keywords."""

    id: str = ""
    kind: Kind
    formulation: str
    source: SourceKind
    quote: str
    role: Role = "user_obligation"
    field_id: str = ""
    message_id: str = ""
    hypothesis_mode: Optional[HypothesisMode] = None
    physical_sample: StrictBool = False
    protocol_only: StrictBool = False
    criteria: list[Criterion] = Field(default_factory=list)


class RequestClarification(BaseModel):
    """A question needed to identify an ambiguous requested outcome."""

    question: str = Field(min_length=1, description="Ask which outcome the user wants, not how to compute a known outcome.")
    quote: str = Field(description="Verbatim ambiguous fragment of the user request.")


class StatementDraft(BaseModel):
    parts: list[StatementPartDraft]
    clarification: RequestClarification | None = Field(default=None,
        description="Null when the requested outcome is identifiable. A clarification response has parts=[]; only unresolved identity or scope of the ask blocks initialization.")
    planning_notes: list[str] = Field(default_factory=list,
        description="Nonblocking open method, count, ranking or threshold choices. Unknown scientific answers also do not require user clarification.")
    conditions: list[RunCondition] = Field(default_factory=list)
    volume: Volume = Field(default_factory=Volume)

    @model_validator(mode="before")
    @classmethod
    def preserve_legacy_uncertainty(cls, value):
        # Never silently turn an old blocked draft into an accepted one. The
        # missing source quote will request a bounded schema repair.
        if isinstance(value, dict) and value.get("uncertainty") and "clarification" not in value:
            return {**value, "clarification": {"question": value["uncertainty"], "quote": ""}}
        return value


class RequirementPart(BaseModel):
    id: str
    kind: Kind
    formulation: str
    role: Role = "supporting"
    provenance: Provenance = Field(default_factory=Provenance)
    criteria: list[Criterion] = Field(default_factory=list)
    obligation: bool = False
    statement_version: str = "1"
    hypothesis_mode: Optional[HypothesisMode] = None
    physical_sample: bool = False
    protocol_only: bool = False
    ready: bool = True
    retired: bool = False
    previous_formulation: str = ""
    previous_requested: Optional[int] = None


class NormalizedStatement(BaseModel):
    """The ask, kept apart from the verbatim source text."""

    source_request: str
    version: str = "1"
    parts: list[RequirementPart] = Field(default_factory=list)
    conditions: list[RunCondition] = Field(default_factory=list)
    volume: Volume = Field(default_factory=Volume)
    root_mode: Literal["question", "container", "uncertain"] = "uncertain"
    uncertain: bool = False
    uncertainty: str = ""
    planning_notes: list[str] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)

    def obligations(self) -> list[RequirementPart]:
        return [p for p in self.parts if p.obligation and not p.retired]

    def of_kind(self, kind: Kind) -> list[RequirementPart]:
        return [p for p in self.parts if p.kind == kind and not p.retired]


class Completion(BaseModel):
    fulfillment: Fulfillment
    report_ready: bool
    remaining: list[dict[str, Any]] = Field(default_factory=list)
    asserted: bool = True
    legacy: bool = False


class OutcomeRecord(BaseModel):
    """What a controlled executor or the graph actually produced for one part."""

    part_id: str
    summary: str = ""
    artifact_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    grounded: bool = False
    # An explicit answer assessment overrides generic evidence. None keeps
    # compatibility with executors and saved outcomes that only set grounded.
    answer_grounded: Optional[bool] = None
    verdict: str = ""
    executed: bool = False
    infrastructure_error: bool = False
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    produced: Optional[int] = None
    limitation: str = ""
    # Explicit item identities permit union across deliveries without double counting.
    produced_ids: list[str] | None = None
    property_verified: Optional[bool] = None
    criteria_checks: dict[str, Optional[bool]] = Field(default_factory=dict)

    @field_validator("produced_ids")
    @classmethod
    def nonempty_produced_ids(cls, ids: list[str] | None) -> list[str] | None:
        if ids is not None and any(not item.strip() for item in ids):
            raise ValueError("produced_ids must contain nonempty item identities")
        return ids
