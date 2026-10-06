"""
HypothesisGenerator — ADK LlmAgent for hypothesis generation.

This agent owns the tool registry and orchestrates:
1. Validation tool discovery (retrieve_validation_tools)
2. Hypothesis generation via MooseChemMCPTool (generate_via_moosechem) — the MCP
   tool is ALWAYS the generation path, never a fallback.
3. Critic refinement loop (run_critic_loop)

All three tools are public module-level functions so their ``FunctionTool``
names are stable and the assembler can declare/register them in system.yaml +
bindings.py — which keeps guard_unknown_tools' whitelist populated.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from opik import track

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.models import (
    Hypothesis,
    HypothesisList,
    HypothesisQuery,
    Provenance,
    ToolCatalog,
    ToolResult,
)
from CoScientist.hypothesis_subsystem.prompts import GENERATOR_INSTRUCTION
from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry

logger = logging.getLogger(__name__)


# ============================================================================
# Function tool: retrieve_validation_tools
# ============================================================================

@track(name="retrieve_validation_tools")
async def retrieve_validation_tools(
    research_question: str,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """Discover MCP validation tools relevant to the research question.

    Queries the FedotMAS RAG database for tools that can test/validate
    hypotheses for this question. Falls back to a static list (chemical-mcp-server)
    when the RAG DB is unreachable. Stores the result in state['tool_catalog']
    for downstream use by generate_via_moosechem.

    Args:
        research_question: The research question to find validation tools for.

    Returns:
        Dict with 'tool_catalog' (serialized ToolCatalog) and 'source'.
    """
    registry: Optional[HypothesisToolRegistry] = None
    if tool_context:
        registry = tool_context.state.get("hypothesis_registry")

    if registry is None:
        from CoScientist.hypothesis_subsystem.tool_registry import HypothesisToolRegistry
        registry = HypothesisToolRegistry()

    catalog = await registry.discover_validation_tools(research_question)

    if tool_context:
        tool_context.state["tool_catalog"] = catalog

    return {
        "tool_catalog": [t.model_dump(mode="json") for t in catalog.tools],
        "source": catalog.source,
        "tool_count": len(catalog.tools),
        "message": (
            f"Discovered {len(catalog.tools)} validation tools "
            f"(source: {catalog.source}). Use these to prioritize "
            f"testable hypotheses in generate_via_moosechem."
        ),
    }


# ============================================================================
# Function tool: generate_via_moosechem
# ============================================================================

@track(name="generate_via_moosechem")
async def generate_via_moosechem(
    research_question: str,
    background_survey: Optional[str] = None,
    domain_constraints: Optional[str] = None,
    max_hypotheses: int = 5,
    temperature: Optional[float] = None,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Generate hypotheses using the MooseChem MCP pipeline.

    Delegates to MooseChemMCPTool which connects to the MOOSE-Chem MCP server
    (Docker). The server runs the original evolutionary-algorithm pipeline:
    corpus building (PubMed + OpenAlex), LLM generation, screening, scoring.

    The tool automatically receives the tool_catalog from state (populated by
    retrieve_validation_tools) and annotates each hypothesis with
    validation_tool_matching.

    Args:
        research_question: The scientific research question to generate hypotheses for.
        background_survey: Optional background context or literature survey.
        domain_constraints: Optional constraints on the applicability domain.
        max_hypotheses: Maximum number of hypotheses to generate (1-20).
        temperature: LLM temperature for generation (0.0-2.0).

    Returns:
        A dict with 'hypotheses' key containing the generated HypothesisList.
    """
    registry: Optional[HypothesisToolRegistry] = None
    audit: Optional[HypothesisAuditLogger] = None
    tool_catalog: Optional[ToolCatalog] = None
    if tool_context:
        registry = tool_context.state.get("hypothesis_registry")
        audit = tool_context.state.get("hypothesis_audit")
        tool_catalog = tool_context.state.get("tool_catalog")

    from CoScientist.hypothesis_subsystem.moosechem_mcp_tool import MooseChemMCPTool

    if registry is None:
        tool = MooseChemMCPTool()
    else:
        tool = registry.get("MooseChem")
        if tool is None:
            tool = MooseChemMCPTool()

    query = HypothesisQuery(
        research_question=research_question,
        background_survey=background_survey,
        domain_constraints=domain_constraints,
        max_hypotheses=max_hypotheses,
        temperature=temperature,
        tool_catalog=tool_catalog,
    )

    start_ts = audit.log_generation_start(research_question, "MooseChem") if audit else 0
    result: ToolResult = await tool.invoke(query)
    if audit:
        audit.log_tool_invocation(
            strategy_type="MooseChem",
            query=query,
            result=result,
        )

    hypotheses_dicts = [h.model_dump(mode="json") for h in result.hypotheses]

    # Write hypotheses directly into state so downstream consumers
    # (orchestrator, tests) find them without depending on LLM relay.
    #
    # [2026-09-14] Also stash research_question here as last_research_question.
    # Observed the LLM (gpt-4o-mini) omitting research_question when calling
    # run_critic_loop right after this tool, after a long tool chain -- ADK
    # rejects the call outright ("mandatory input parameters are not present"),
    # and on retry the model has sometimes re-run generate_via_moosechem from
    # scratch instead of just fixing the missing argument (wastes the ~20-60min
    # MooseChem run). This state entry is the fallback source run_critic_loop
    # reads from when the model doesn't pass research_question explicitly.
    # research_question itself is still required by the critic loop logic
    # downstream (loop_coordinator.run_critic_loop) -- this doesn't change that,
    # it just makes sure it always gets a value from a reliable source instead
    # of solely depending on the LLM re-stating it correctly every time.
    if tool_context is not None:
        try:
            tool_context.state["generated_hypotheses"] = {"hypotheses": hypotheses_dicts}
            tool_context.state["last_research_question"] = research_question
            if getattr(tool_context, "actions", None) is not None:
                tool_context.actions.state_delta["generated_hypotheses"] = {"hypotheses": hypotheses_dicts}
                tool_context.actions.state_delta["last_research_question"] = research_question
        except Exception:
            pass

    return {"hypotheses": hypotheses_dicts, "metadata": result.metadata}


# ============================================================================
# Function tool: run_critic_loop
# ============================================================================

@track(name="run_critic_loop")
async def run_critic_loop(
    hypotheses_json: str,
    # [2026-09-14] Made optional (was: required `str`). ADK enforces
    # "required" at the call layer, before this function body ever runs --
    # so when the LLM omitted this argument, ADK rejected the whole call
    # with "mandatory input parameters are not present" and the function
    # never got a chance to fall back to anything. Observed the model
    # (gpt-4o-mini) drop this argument after a long tool chain, then on
    # retry re-run generate_via_moosechem from scratch instead of just
    # re-calling this with the missing arg -- wasting a full MooseChem run
    # (~20-60 min). research_question is still REQUIRED by the critic loop
    # logic below (loop_coordinator.run_critic_loop) -- this does not relax
    # that. It only adds a reliable fallback source (state, written by
    # generate_via_moosechem) for the rare case the model doesn't pass it.
    research_question: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> Dict[str, Any]:
    """
    Run the Critic review loop on generated hypotheses.

    The Critic evaluates each hypothesis for rigor and falsifiability.
    Approved hypotheses pass through; those needing revision are refined;
    rejected hypotheses are marked as deferred.

    Args:
        hypotheses_json: JSON string of HypothesisList with 'hypotheses' key.
        research_question: The original research question for context.
            Optional at the call layer so ADK doesn't hard-reject the tool
            call when the model omits it; falls back to
            state["last_research_question"] (set by generate_via_moosechem)
            when not provided. Still effectively required -- if neither the
            argument nor the state fallback is available, this is an empty
            string and the critic loses question context.

    Returns:
        Dict with 'hypotheses' key containing the refined HypothesisList.
    """
    loop_coordinator = None
    audit: Optional[HypothesisAuditLogger] = None
    if tool_context:
        loop_coordinator = tool_context.state.get("loop_coordinator")
        audit = tool_context.state.get("hypothesis_audit")
        if not research_question:
            research_question = tool_context.state.get("last_research_question", "")

    # Prefer hypotheses already written to state by generate_via_moosechem —
    # more reliable than depending on the LLM to re-serialize a potentially
    # huge JSON (formulas, special chars) back into a string argument.
    # On large payloads the model sometimes breaks escaping
    # (json.JSONDecodeError: Invalid \\escape), crashing the whole critic step
    # instead of just running it.
    raw_hypotheses = None
    if tool_context:
        state_data = tool_context.state.get("generated_hypotheses")
        if isinstance(state_data, dict) and state_data.get("hypotheses"):
            raw_hypotheses = state_data["hypotheses"]

    if raw_hypotheses is None:
        try:
            parsed = json.loads(hypotheses_json) if isinstance(hypotheses_json, str) else hypotheses_json
            raw_hypotheses = parsed.get("hypotheses", []) if isinstance(parsed, dict) else []
        except json.JSONDecodeError as exc:
            logger.warning(
                "[run_critic_loop] hypotheses_json malformed (%s) and no state fallback available; proceeding with empty list",
                exc,
            )
            raw_hypotheses = []

    if loop_coordinator is None:
        # No critic loop available — return as-is (standalone/fallback).
        return {"hypotheses": raw_hypotheses}

    # Convert to Hypothesis objects. A malformed entry must not abort the whole
    # loop — build a minimal wrapper carrying the REQUIRED provenance so the
    # critic can still judge it (the previous fallback omitted provenance and
    # raised ValidationError inside the except handler; and for non-dict entries
    # it raised AttributeError on h.get before even reaching the model).
    hypotheses: List[Hypothesis] = []
    for h in raw_hypotheses:
        if not isinstance(h, dict):
            logger.warning("[run_critic_loop] skipping non-dict hypothesis entry: %r", h)
            continue
        try:
            hypotheses.append(Hypothesis(**h))
        except Exception as exc:
            logger.warning(
                "[run_critic_loop] malformed hypothesis; using minimal wrapper: %s", exc
            )
            hypotheses.append(
                Hypothesis(
                    claim=str(h.get("claim") or "Unknown hypothesis"),
                    domain=str(h.get("domain") or ""),
                    reasoning=str(h.get("reasoning") or ""),
                    strategy_type=str(h.get("strategy_type") or "unknown"),
                    verification_plan=str(h.get("verification_plan") or ""),
                    refutation_conditions=str(h.get("refutation_conditions") or ""),
                    provenance=Provenance(creator="HypothesisGenerator/fallback"),
                )
            )

    hypothesis_list = HypothesisList(hypotheses=hypotheses)

    refined = await loop_coordinator.run_critic_loop(
        hypothesis_list, research_question, tool_context=tool_context
    )

    refined_dicts = [h.model_dump(mode="json") for h in refined.hypotheses]

    # Surface the final (critic-refined) hypotheses into state as well.
    if tool_context is not None:
        try:
            tool_context.state["generated_hypotheses"] = {"hypotheses": refined_dicts}
            if getattr(tool_context, "actions", None) is not None:
                tool_context.actions.state_delta["generated_hypotheses"] = {"hypotheses": refined_dicts}
        except Exception:
            pass

    return {"hypotheses": refined_dicts}


# ============================================================================
# Builder
# ============================================================================

def build_hypothesis_generator(
    model: str,
    tool_registry: HypothesisToolRegistry,
    audit: HypothesisAuditLogger,
) -> LlmAgent:
    """
    Build the HypothesisGenerator ADK LlmAgent.

    Args:
        model: LLM model identifier (e.g., 'gpt-4').
        tool_registry: Registry of hypothesis-generation strategies.
        audit: Audit logger for observability.

    Returns:
        A configured LlmAgent ready for use in the ADK runtime.
    """
    generator_tools: List[FunctionTool] = [
        FunctionTool(retrieve_validation_tools),
        FunctionTool(generate_via_moosechem),
    ]
    # run_critic_loop is added later by add_critic_loop_tool(), after the
    # loop coordinator is constructed and injected into state.

    agent = LlmAgent(
        name="HypothesisGenerator",
        model=LiteLlm(model=model),
        instruction=GENERATOR_INSTRUCTION,
        description=(
            "Generates structured, falsifiable scientific hypotheses via the "
            "MooseChem MCP pipeline (PubMed+OpenAlex corpus → LLM generation → "
            "scoring). Discovers available validation tools and iteratively "
            "refines each hypothesis via a built-in Critic loop."
        ),
        #output_key="generated_hypotheses",
        tools=generator_tools,
    )

    return agent


def add_critic_loop_tool(agent: LlmAgent) -> None:
    """
    Add the run_critic_loop tool to an existing HypothesisGenerator agent.

    Called after the loop_coordinator is constructed, since it requires
    the coordinator to be available in state.

    Args:
        agent: The HypothesisGenerator LlmAgent to add the tool to.
    """
    agent.tools.append(FunctionTool(run_critic_loop))
