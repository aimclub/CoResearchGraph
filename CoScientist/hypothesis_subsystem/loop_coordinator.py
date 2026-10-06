"""
HypothesisLoopCoordinator — Generator↔HypothesisCriticAgent iteration loop.

Connects HypothesisGenerator to HypothesisCriticAgent from
CoScientist.hypothesis_subsystem.critic_agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import litellm
from google.adk.tools.tool_context import ToolContext
from opik import track

from CoScientist.hypothesis_subsystem.moosechem_tool import _extract_json

logger = logging.getLogger(__name__)

from CoScientist.hypothesis_subsystem.critic_agent import (
    HypothesisCriticAgent,
    HypothesisCriticResult,
    HypothesisInput,
    RAGClient,
)
from CoScientist.hypothesis_subsystem.audit import HypothesisAuditLogger
from CoScientist.hypothesis_subsystem.models import (
    CriticVerdict,
    Hypothesis,
    HypothesisList,
    HypothesisStatus,
    ProvenanceRecord,
)

_REFINEMENT_SYSTEM = """You are a scientific hypothesis refinement specialist.
Your task: improve a hypothesis based on critic feedback.

Given:
1. The ORIGINAL hypothesis (full JSON)
2. The CRITIC SCORES: verifiability, consistency, specificity, novelty (0-2 each)
3. The CRITIC FEEDBACK with specific failing dimensions
4. AVAILABLE VALIDATION TOOLS (if provided) and their input constraints

You MUST:
- Keep the hypothesis structurally identical, only modifying failing dimensions.
- If verifiability < 2: make refutation_conditions measurable and concrete.
  Prefer refutation conditions that match available tool inputs (e.g., "docking score
  < -8.0 kcal/mol in AutoDock Vina" rather than "should bind strongly").
- If specificity < 2: sharpen the claim and add distinguishing observations.
  When tools have input constraints (e.g., max 500 Da), reformulate the hypothesis
  to stay within those constraints while preserving the core scientific claim.
- If consistency < 2: align reasoning with evidence.
- If novelty < 2: differentiate from known approaches.
- WHEN TOOLS ARE AVAILABLE: prefer verifiable over novel — a testable hypothesis
  with moderate novelty beats an untestable one with high novelty.
- Return the COMPLETE revised hypothesis as JSON (all fields filled)."""


class HypothesisLoopCoordinator:
    MAX_ITERATIONS = 5  # [experiment: increased refinement budget, was 3]

    def __init__(self, model: str, audit: HypothesisAuditLogger):
        self._model = model
        self._audit = audit
        self._critic = HypothesisCriticAgent(rag_client=RAGClient(), model=self._model)

    @track(name="hypothesis_run_critic_loop")
    async def run_critic_loop(
        self,
        hypotheses: HypothesisList,
        research_question: str,
        tool_context: Optional[ToolContext] = None,
    ) -> HypothesisList:
        refined_list: List[Hypothesis] = []
        for hypothesis in hypotheses.hypotheses:
            try:
                refined_list.append(await self._iterate_one(hypothesis, research_question))
            except Exception as exc:
                self._audit.log_error("critic_loop", str(exc), hypothesis.claim)
                hypothesis.provenance.history.append(ProvenanceRecord(
                    action="critic_loop_error", agent="HypothesisLoopCoordinator",
                    detail=f"Critic loop failed: {exc}"))
                refined_list.append(hypothesis)
        final = HypothesisList(hypotheses=refined_list)
        # The hypothesis subsystem is the ONLY agent permitted to create
        # Hypothesis/VerificationMethod/ConfirmationCriteria/Tool nodes
        # (graph/research/schema.AGENT_PERMISSIONS), so the final ACTIVE
        # hypotheses must be committed here programmatically — the orchestrator
        # has no right to do it and the model cannot be relied on to emit
        # research_commit itself.
        self._commit_active_hypotheses(final, research_question, tool_context)
        return final

    def _commit_active_hypotheses(
        self,
        hypothesis_list: HypothesisList,
        research_question: str,
        tool_context: Optional[ToolContext],
    ) -> None:
        """Commit final ACTIVE hypotheses (plus their VerificationMethods,
        ConfirmationCriteria and required Tools) into the shared Research Context
        Graph. Best-effort: any failure is logged and never breaks the run."""
        active = [h for h in hypothesis_list.hypotheses if h.status == HypothesisStatus.ACTIVE]
        if not active:
            logger.info("[hypothesis] no ACTIVE hypotheses to commit to the research graph")
            return
        try:
            if tool_context is None:
                logger.warning(
                    "[hypothesis] no ToolContext available; skipping research-graph commit"
                )
                return
            from CoScientist.graph.research.store import get_research_graph
            graph = get_research_graph(tool_context)
        except Exception as exc:  # noqa: BLE001 — graph commit must never break a run
            self._audit.log_error("research_graph_commit", str(exc))
            return

        root: Optional[str] = None
        try:
            root = graph.root_id()
        except Exception:  # noqa: BLE001
            root = None

        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        for i, h in enumerate(active, 1):
            href, vmref, ccref = f"h{i}", f"vm{i}", f"cc{i}"
            nodes.append({
                "type": "Hypothesis", "ref": href,
                "attrs": {
                    "formulation": h.claim,
                    "rationale": h.reasoning,
                    "priority": "medium",
                },
            })
            nodes.append({
                "type": "VerificationMethod", "ref": vmref,
                "attrs": {
                    "method_type": "computational",
                    "inputs": "domain, variables and evidence from the hypothesis",
                    "outputs": "evidence supporting or refuting the hypothesis",
                    "limitations": h.verification_plan[:500],
                },
            })
            nodes.append({
                "type": "ConfirmationCriteria", "ref": ccref,
                "attrs": {
                    "threshold": h.refutation_conditions or "not specified",
                    "confirmations_needed": 1,
                    "reproducibility": "must be reproducible",
                },
            })
            edges.append({"type": "tested_by", "from": f"#{href}", "to": f"#{vmref}"})
            edges.append({"type": "formulated_for", "from": f"#{ccref}", "to": f"#{href}"})
            if root:
                edges.append({"type": "motivates", "from": root, "to": f"#{href}"})
            for j, tool_name in enumerate(h.tools or [], 1):
                tref = f"t{i}_{j}"
                nodes.append({
                    "type": "Tool", "ref": tref, "status": "needs_adaptation",
                    "attrs": {"name": tool_name, "tool_type": "computational"},
                })
                edges.append({"type": "requires", "from": f"#{href}", "to": f"#{tref}"})
                edges.append({"type": "uses", "from": f"#{vmref}", "to": f"#{tref}"})

        try:
            result = graph.commit(source="HypothesesAgent", nodes=nodes, edges=edges)
            if result.ok:
                logger.info(
                    "[hypothesis] committed %d ACTIVE hypotheses to the research graph",
                    len(active),
                )
            else:
                self._audit.log_error(
                    "research_graph_commit", "; ".join(result.errors)
                )
        except Exception as exc:  # noqa: BLE001
            self._audit.log_error("research_graph_commit", str(exc))

    @track(name="hypothesis_critic_iteration")
    async def _iterate_one(self, hypothesis: Hypothesis, research_question: str) -> Hypothesis:
        current = hypothesis
        for iteration in range(1, self.MAX_ITERATIONS + 1):
            critic_input = self._to_hypothesis_input(current)
            result = await self._invoke_critic(critic_input)
            verdict = self._map_verdict(result)

            if verdict == CriticVerdict.APPROVE:
                self._audit.log_status_change(current.claim, current.status, HypothesisStatus.ACTIVE, "Critic passed")
                current.status = HypothesisStatus.ACTIVE
                current.provenance.history.append(ProvenanceRecord(
                    action="critiqued", agent="HypothesisCriticAgent",
                    detail=f"Passed: scores={result.scores}"))
                break
            elif verdict == CriticVerdict.REVISE:
                self._audit.log_revision(current.claim, iteration)
                current.provenance.history.append(ProvenanceRecord(
                    action="critiqued", agent="HypothesisCriticAgent",
                    detail=f"Revise: scores={result.scores}, feedback={result.feedback}"))
                current = await self._refine_via_llm(current, result, research_question, iteration)
            else:
                self._audit.log_status_change(current.claim, current.status, HypothesisStatus.DEFERRED,
                                              f"Rejected: {result.feedback}")
                current.status = HypothesisStatus.DEFERRED
                current.provenance.history.append(ProvenanceRecord(
                    action="critiqued", agent="HypothesisCriticAgent",
                    detail=f"Rejected: tool_request={result.tool_request}"))
                break
        else:
            self._audit.log_status_change(current.claim, current.status, HypothesisStatus.DEFERRED,
                                          f"Max iterations ({self.MAX_ITERATIONS}) without pass")
            current.status = HypothesisStatus.DEFERRED
            current.provenance.history.append(ProvenanceRecord(
                action="status_changed", agent="HypothesisLoopCoordinator",
                detail=f"Deferred after {self.MAX_ITERATIONS} iterations"))
        return current

    async def _invoke_critic(self, critic_input: HypothesisInput) -> HypothesisCriticResult:
        return await asyncio.to_thread(self._critic.critique_one, critic_input)

    def _map_verdict(self, result: HypothesisCriticResult) -> CriticVerdict:
        if result.passed:
            return CriticVerdict.APPROVE

        scores = result.scores
        if not scores:
            return CriticVerdict.REVISE if result.tools_available else CriticVerdict.REJECT

        # Relaxed threshold: sum ≥ 6/10 AND min ≥ 1 → pass
        score_sum = sum(scores.values())
        score_min = min(scores.values()) if scores else 0

        if score_sum >= 6 and score_min >= 1:
            return CriticVerdict.APPROVE

        # Sum OK but has a zero → revise to fix the fatal dimension
        if score_sum >= 6 and score_min == 0:
            return CriticVerdict.REVISE

        if result.tools_available:
            return CriticVerdict.REVISE
        return CriticVerdict.REJECT

    def _to_hypothesis_input(self, h: Hypothesis) -> HypothesisInput:
        return HypothesisInput(
            id=h.claim[:80],
            claim=h.claim,
            domain=h.domain,
            variables=json.dumps({
                "independent": [{"name": v.name, "unit": v.unit} for v in h.variables.independent],
                "dependent": [{"name": v.name, "unit": v.unit} for v in h.variables.dependent],
            }, ensure_ascii=False),
            verification_plan=h.verification_plan,
            tools=h.tools,
            strategy_type=h.strategy_type,
        )

    async def _refine_via_llm(self, hypothesis: Hypothesis, result: HypothesisCriticResult,
                              research_question: str, iteration: int) -> Hypothesis:
        original_json = json.dumps(hypothesis.model_dump(mode="json"), indent=2, default=str)
        dims = [f"- {dim}: {score}/2" for dim, score in result.scores.items() if score < 2]
        user_prompt = (
            f"CRITIC SCORES:\n" + "\n".join(dims or ["all passing"]) +
            f"\n\nFEEDBACK: {result.feedback or 'None'}\n\n"
            f"ORIGINAL HYPOTHESIS:\n{original_json}\n\n"
            f"Return the COMPLETE revised hypothesis as JSON."
        )
        try:
            resp = await litellm.acompletion(
                model=self._model,
                messages=[{"role": "system", "content": _REFINEMENT_SYSTEM},
                          {"role": "user", "content": user_prompt}],
                max_tokens=4000, temperature=0.4)
            content = resp["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            if isinstance(parsed, dict) and "claim" not in parsed:
                for v in parsed.values():
                    if isinstance(v, dict) and "claim" in v:
                        parsed = v; break
            revised = Hypothesis(**parsed)
            revised.provenance.history.append(ProvenanceRecord(
                action="revised", agent="HypothesisLoopCoordinator/LLM",
                detail=f"Refinement at iteration {iteration}"))
            return revised
        except Exception as exc:
            self._audit.log_error("refine_via_llm", str(exc), hypothesis.claim)
            hypothesis.provenance.history.append(ProvenanceRecord(
                action="revised", agent="HypothesisLoopCoordinator/LLM",
                detail=f"Refinement failed: {exc}"))
            return hypothesis
