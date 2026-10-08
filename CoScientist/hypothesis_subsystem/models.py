"""
Pydantic models for the Hypothesis Agent subsystem.

Defines the full structured output contract for hypothesis generation:
Hypothesis, Variable, Variables, Reference, AlternativeHypothesis,
HypothesisStatus, ScaleType, Provenance, HypothesisQuery, ToolResult,
CriticReview, and HypothesisList.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class ScaleType(str, Enum):
    """Measurement scale classification (Stevens' typology)."""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"


class HypothesisStatus(str, Enum):
    """Lifecycle status of a hypothesis."""
    PROPOSED = "proposed"
    ACTIVE = "active"
    REFUTED = "refuted"
    CONFIRMED = "confirmed"
    DEFERRED = "deferred"


class CriticVerdict(str, Enum):
    """Verdict from the Critic agent on a hypothesis."""
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


# ============================================================================
# Variable models
# ============================================================================

class Variable(BaseModel):
    """A single scientific variable with metadata."""
    name: str = Field(..., description="Variable name (e.g. 'Molecular Weight').")
    description: str = Field(
        ..., description="What this variable represents and how it's measured."
    )
    unit: Optional[str] = Field(
        None, description="Unit of measurement (e.g. 'Da', 'kcal/mol', 'nM')."
    )
    scale: ScaleType = Field(
        ..., description="Stevens' scale type (nominal/ordinal/interval/ratio)."
    )


class Variables(BaseModel):
    """Grouped independent, dependent, and covariate variables."""
    independent: List[Variable] = Field(
        default_factory=list,
        description="Independent variables — manipulated or predictor factors.",
    )
    dependent: List[Variable] = Field(
        default_factory=list,
        description="Dependent variables — measured outcomes or responses.",
    )
    covariates: List[Variable] = Field(
        default_factory=list,
        description="Covariates — controlled or confounding factors.",
    )


# ============================================================================
# Reference / Evidence model
# ============================================================================

class Reference(BaseModel):
    """A cited source supporting the hypothesis."""
    doi: Optional[str] = Field(None, description="DOI of the publication.")
    url: Optional[str] = Field(None, description="URL to the resource.")
    title: str = Field(..., description="Title of the paper, dataset, or experiment.")
    description: Optional[str] = Field(
        None, description="Brief note on how this reference supports the hypothesis."
    )


# ============================================================================
# Alternative hypothesis model
# ============================================================================

class AlternativeHypothesis(BaseModel):
    """A competing hypothesis with distinguishing observations."""
    claim: str = Field(..., description="The alternative claim.")
    distinguishing_observation: str = Field(
        ...,
        description="Observation that would allow distinguishing this alternative "
                    "from the primary hypothesis.",
    )


# ============================================================================
# Provenance models
# ============================================================================

class ProvenanceRecord(BaseModel):
    """A single entry in the provenance audit trail."""
    action: str = Field(
        ...,
        description="Action performed: 'created', 'revised', 'critiqued', 'status_changed'.",
    )
    agent: str = Field(
        ...,
        description="Who performed the action (agent name or 'human').",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of the action.",
    )
    detail: Optional[str] = Field(
        None, description="Additional context about the action."
    )


class Provenance(BaseModel):
    """Complete provenance tracking for a hypothesis."""
    creator: str = Field(
        ...,
        description="Original creator (e.g. 'HypothesisGenerator', 'MooseChem').",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of creation.",
    )
    history: List[ProvenanceRecord] = Field(
        default_factory=list,
        description="Ordered list of all actions performed on this hypothesis.",
    )


# ============================================================================
# Core Hypothesis model
# ============================================================================

class Hypothesis(BaseModel):
    """Full structured hypothesis as required by the CoScientist contract."""

    claim: str = Field(
        ...,
        description="Assertion of the form 'Compounds with feature X demonstrate "
                    "effect Y in domain Z'.",
    )
    variables: Variables = Field(
        default_factory=Variables,
        description="Independent, dependent, and covariate factors.",
    )
    domain: str = Field(
        ...,
        description="Applicability domain (class of objects and conditions, e.g. "
                    "'small organic molecules ≤500 Da at physiological pH 7.4').",
    )
    reasoning: str = Field(
        ...,
        description="Full logical chain: how data, literature, or prior experiments "
                    "lead to this hypothesis, including limitations and prior work issues.",
    )
    strategy_type: str = Field(
        ...,
        description="Source/strategy identifier (e.g. 'MooseChem', 'literature_mining').",
    )
    evidence_basis: List[Reference] = Field(
        default_factory=list,
        description="References to data, publications, datasets, or experiments "
                    "supporting the hypothesis.",
    )
    verification_plan: str = Field(
        ...,
        description="Detailed verification plan: data to collect, models/methods, "
                    "metrics, reproducible steps.",
    )
    tools: List[str] = Field(
        default_factory=list,
        description="Required tools (e.g. 'molecular dynamics', 'GNN', 'ChEMBL', "
                    "'Mann–Whitney U test').",
    )
    refutation_conditions: str = Field(
        ...,
        description="Popperian falsification criteria (e.g. 'MAE > 0.5', "
                    "'R² < 0.3 on hold-out', 'failure to reproduce under protocol').",
    )
    competing_with: List[AlternativeHypothesis] = Field(
        default_factory=list,
        description="Alternative hypotheses with distinguishing observations.",
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.PROPOSED,
        description="Current lifecycle status.",
    )
    provenance: Provenance = Field(
        ...,
        description="Origin information: creator, timestamp, generation context.",
    )
    validation_tool_matching: List[ValidationToolInfo] = Field(
        default_factory=list,
        description="Validation tools that could test this hypothesis, "
                    "matched during generation. Empty = no tool found yet "
                    "— the hypothesis may need future tool development.",
    )


# ============================================================================
# Validation tool models
# ============================================================================

class ValidationToolInfo(BaseModel):
    """Metadata about a single MCP validation tool available in the system."""
    name: str = Field(..., description="Tool name (e.g. 'docking_simulation').")
    description: str = Field(
        ..., description="What the tool does and what inputs it expects."
    )
    server_id: Optional[str] = Field(
        None, description="MCP server identifier (for future A2A routing)."
    )
    input_schema: Optional[Dict[str, Any]] = Field(
        None, description="JSON Schema of the tool's input parameters."
    )
    limitations: Optional[str] = Field(
        None, description="Known limitations: max molecule size, required SMILES, etc."
    )
    retrieval_score: Optional[float] = Field(
        None, description="RAG relevance score for this tool vs the research question."
    )


class ToolCatalog(BaseModel):
    """Discovered validation tools for a research question."""
    tools: List[ValidationToolInfo] = Field(
        default_factory=list,
        description="Available MCP validation tools with metadata."
    )
    retrieval_query: Optional[str] = Field(
        None, description="The query used to discover these tools."
    )
    source: str = Field(
        default="rag",
        description="Discovery source: 'rag' (MCP DB) or 'static_fallback'."
    )


# ============================================================================
# Query / Result models
# ============================================================================

class HypothesisQuery(BaseModel):
    """Input query for hypothesis generation tools."""
    research_question: str = Field(
        ..., description="The research question to generate hypotheses for."
    )
    background_survey: Optional[str] = Field(
        None, description="Optional background context or literature survey."
    )
    domain_constraints: Optional[str] = Field(
        None, description="Optional constraints on the applicability domain."
    )
    max_hypotheses: int = Field(
        default=5, description="Maximum number of hypotheses to generate.", ge=1, le=20
    )
    temperature: Optional[float] = Field(
        None, description="LLM temperature for generation.", ge=0.0, le=2.0
    )
    tool_catalog: Optional[ToolCatalog] = Field(
        None, description="Discovered validation tools available for hypothesis testing."
    )


class ToolResult(BaseModel):
    """Structured result from a hypothesis-generation tool invocation."""
    strategy_type: str = Field(
        ..., description="Strategy identifier (e.g. 'MooseChem')."
    )
    hypotheses: List[Hypothesis] = Field(
        default_factory=list, description="Generated hypotheses."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (corpus size, model used, duration, etc.).",
    )
    success: bool = Field(default=True, description="Whether the invocation succeeded.")
    error_message: Optional[str] = Field(
        None, description="Error message if success is False."
    )


class HypothesisList(BaseModel):
    """Wrapper for List[Hypothesis] for ADK output_schema compatibility."""
    hypotheses: List[Hypothesis] = Field(
        default_factory=list, description="List of hypotheses."
    )


# ============================================================================
# Critic models
# ============================================================================

class CriticReview(BaseModel):
    """Structured critique of a single hypothesis."""
    verdict: CriticVerdict = Field(
        ..., description="Approve, revise, or reject."
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Specific actionable suggestions for improvement.",
    )
    reasoning: Optional[str] = Field(
        None, description="Explanation for the verdict."
    )
    revised_hypothesis: Optional[Hypothesis] = Field(
        None,
        description="If verdict is 'revise', an improved version of the hypothesis "
                    "incorporating the suggestions.",
    )
    fields_to_revise: List[str] = Field(
        default_factory=list,
        description="If verdict is 'revise', which fields need changes "
                    "(e.g. 'reasoning', 'refutation_conditions', 'verification_plan').",
    )
