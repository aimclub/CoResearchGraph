"""Background hypothesis validator — spec Module 4, fully asynchronous.

NOT an ADK sub-agent and NOT dependent on the orchestrator. It is a plugin that
watches the graph: whenever evidence lands on a hypothesis, it fires a
fire-and-forget asyncio task that judges the hypothesis (confirmed / refuted /
inconclusive) and writes the Conclusion. The main orchestration loop never awaits
it — on the web's persistent event loop the verdict lands in the graph a moment
later and the live viewer shows it. Judgment runs in a SMALL focused context
(one hypothesis' slice), not the orchestrator's.

Best-effort throughout: any failure leaves the graph untouched and never breaks
the run. Commits are attributed to source "ValidatorAgent" (its write
permissions live in schema.AGENT_PERMISSIONS).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.agents.callbacks.report_language import (
    REPORT_LANGUAGE_STATE_KEY,
    graph_text_rule,
)
from CoScientist.graph.research import queries
from CoScientist.graph.research.store import get_research_graph, research_graph

logger = logging.getLogger(__name__)

SOURCE = "ValidatorAgent"
# Module-level refs so fire-and-forget tasks are not garbage-collected mid-flight.
_TASKS: set = set()


async def drain_validations(timeout: float = 90.0) -> None:
    """Await outstanding background validations so their verdicts + Conclusions
    land in the graph before a run reports 'done'. Best-effort and bounded — a
    stuck judgment can never hang the run."""
    pending = [t for t in list(_TASKS) if not t.done()]
    if not pending:
        return
    try:
        await asyncio.wait(pending, timeout=timeout)
    except Exception:  # noqa: BLE001 — draining must never break a run
        pass

_SYSTEM = (
    "You are a rigorous scientific hypothesis validator. You are given ONE "
    "hypothesis, its confirmation criteria, and the evidence attached to it. Some "
    "evidence is UNCLASSIFIED (its polarity is not yet set) — decide for EACH "
    "evidence item whether it supports, refutes or refines the hypothesis (or is "
    "irrelevant). Then weigh it against the criteria and return a verdict. Be "
    "conservative: confirm only when supporting evidence is strong, consistent and "
    "meets the criteria; refute when there is decisive contradicting evidence; "
    "otherwise the branch is INCONCLUSIVE — it was tested and the evidence did "
    "not settle it. Say what is missing in `reason`. Do not use inconclusive "
    "for a hypothesis nobody worked on; that is the orchestrator's backlog, not "
    "a verdict.\n"
    "\n"
    "THE CONCLUSION IS THE STUDY'S DELIVERABLE. It is the one card a reader "
    "opens to learn what came of the work, and the thing the NEXT study starts "
    "from — so it carries the chain that produced the answer, not only the "
    "answer. A single paragraph of prose is not enough: a reader had to re-walk "
    "the whole graph to find out how the answer was reached, and a follow-up "
    "study had nothing to begin with. Write these five fields, each doing its "
    "own job and none repeating another:\n"
    "  - `conclusion`: THE ANSWER, one or two sentences. It becomes the card's "
    "headline, so it must stand alone without the rest.\n"
    "  - `how_established`: the chain, BY STAGE — what the reading established, "
    "what was then run, with which instrument, and the numbers it returned. "
    "Name the evidence ids and the tools. This is the part that makes the "
    "answer checkable rather than merely stated.\n"
    "  - `against_criteria`: every criterion by id, the value actually measured "
    "against it, and whether it was met. If a criterion was not measured at "
    "all, say so — an unmeasured bar is not a met bar.\n"
    "  - `validity_bounds`: what the answer does NOT cover — the population, "
    "the conditions, the model, the routes or regimes left out.\n"
    "  - `open_questions`: what the next study should do FIRST. The measurement "
    "that was missing, the bar that was not reached, the branch nobody tested. "
    "Be concrete enough to act on.\n"
    "\n"
    "Every number you write must come from the evidence you were given. If a "
    "figure is not in the record, do not supply one — say it is missing.\n"
    "\n"
    "Reply with STRICT JSON only, no prose:\n"
    '{"evidence":{"<evidence_id>":"supports|refutes|refines|irrelevant"},'
    '"verdict":"confirmed|refuted|inconclusive",'
    '"criteria":{"<criteria_id>":"met|not_met"},'
    '"conclusion":"the answer, one or two sentences",'
    '"how_established":"the chain by stage, with instruments and numbers",'
    '"against_criteria":"each criterion id, measured value, met or not",'
    '"validity_bounds":"limits of validity",'
    '"open_questions":"what the next study should do first",'
    '"reason":"one sentence justifying the verdict"}'
)

def _enabled() -> bool:
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.enabled
    except Exception:  # noqa: BLE001
        return False


async def _complete(system: str, user: str) -> str:
    """One small LLM call via litellm (same config as the semantic layer)."""
    import litellm
    from CoScientist.config import get_settings
    s = get_settings().llm
    model = os.getenv("RESEARCH_VALIDATOR_MODEL") or s.main_model
    from CoScientist.execution_control import before_model_attempt
    await before_model_attempt(
        "research_validator", metadata={"model": model}
    )
    resp = await litellm.acompletion(
        model=model, api_base=s.main_url, api_key=s.openai_api_key,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0,
        timeout=s.request_timeout,
        num_retries=0,
    )
    # Judged in the background, off the agent tree: the ambient session binding
    # is what keeps this call attached to the run that triggered it.
    from CoScientist.logging.metrics import record_completion
    record_completion(resp, model=model, agent="ResearchValidator")
    return resp.choices[0].message.content or ""


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def _research_id(graph: Any) -> str:
    """Return the current research generation for validator isolation."""
    full = graph.full()
    if not isinstance(full, dict):
        return ""
    return str(full.get("research_id") or "")


#: Bookkeeping attrs that say nothing about what was found.
_EVIDENCE_SKIP = frozenset({"subtype", "reliability", "commit_note", "source",
                            "hypothesis", "hypothesis_id", "verification_method",
                            "confirmation_criteria", "evidence_type"})


def _evidence_text(node: Dict[str, Any], budget: int = 1200) -> str:
    """Everything an Evidence node actually says, not just its `content`.

    Workers record results under whatever key fits the result — `metrics`,
    `threshold_check`, `value`, `description` — and reading only `content` made a
    node full of measurements arrive here as an empty string. The judge then
    concluded, correctly given what it was shown, that no evidence existed, and
    a hypothesis whose criteria had in fact all passed was set aside.
    """
    attrs = node.get("attrs") or {}
    content = str(attrs.get("content") or "").strip()

    # Numbers and verdicts first: the budget must not be spent on file paths
    # before the judge has seen what was measured.
    priority = ("metrics", "threshold_check", "description", "finding", "summary",
                "value", "metric", "observation", "architecture", "generation")

    def order(key: str) -> tuple:
        return (priority.index(key) if key in priority else len(priority), key)

    rest = []
    for key in sorted(attrs, key=order):
        if key in _EVIDENCE_SKIP or key == "content":
            continue
        value = attrs[key]
        if value in (None, "", [], {}):
            continue
        rest.append(f"{key}={value}")
    text = "; ".join([t for t in [content, *rest] if t]) or "(no attributes recorded)"
    return text[:budget] + ("…" if len(text) > budget else "")


def _build_user(slice_: Dict[str, Any], hid: str) -> str:
    """Render the focused judging prompt from the hypothesis' context slice."""
    by_id = {n["id"]: n for n in slice_.get("nodes", [])}
    edges = slice_.get("edges", [])
    h = by_id.get(hid, {})
    lines = [f"HYPOTHESIS {hid}: {(h.get('attrs') or {}).get('formulation', '')}", ""]
    crit = [by_id[e["from"]] for e in edges
            if e["type"] == "formulated_for" and e["to"] == hid and e["from"] in by_id]
    lines.append("CONFIRMATION CRITERIA:")
    lines += [f"- {c['id']}: {(c.get('attrs') or {}).get('threshold', c.get('attrs'))}"
              for c in crit] or ["- (none defined)"]
    lines.append("")
    for kind, label in (("supports", "SUPPORTING"), ("refutes", "REFUTING"),
                        ("refines", "REFINING"), ("relates_to", "UNCLASSIFIED (assign polarity)")):
        evs = [by_id[e["from"]] for e in edges
               if e["type"] == kind and e["to"] == hid and e["from"] in by_id
               and by_id[e["from"]].get("type") == "Evidence"]
        if evs:
            lines.append(f"{label} EVIDENCE:")
            lines += [f"- {e['id']} ({(e.get('attrs') or {}).get('subtype','')}): "
                      f"{_evidence_text(e)}" for e in evs]
            lines.append("")
    return "\n".join(lines)


async def judge_hypothesis(
    hid: str,
    complete: Optional[Callable] = None,
    graph=None,
    expected_research_id: Optional[str] = None,
    language: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Judge one hypothesis and commit the verdict + Conclusion. Best-effort;
    returns the CommitResult dict, or None if it could not judge/commit.
    `complete` is injectable for tests (bypasses the real LLM).

    `language` is the study's report language. This is not an agent, so the
    `{report_language_block?}` substitution every agent gets never reached it:
    it wrote English conclusions into Russian studies, on the one card that
    answers the question."""
    try:
        graph = graph or research_graph
        if expected_research_id is not None \
                and _research_id(graph) != expected_research_id:
            return None
        sl = graph.get_context_slice(hid, depth=2)
        if "error" in sl:
            return None
        by_id = {n["id"]: n for n in sl.get("nodes", [])}
        h = by_id.get(hid)
        if not h or h.get("type") != "Hypothesis":
            return None
        if h.get("status") != "under_verification":
            return None  # only judge branches that are actually under verification

        system = _SYSTEM + graph_text_rule(language)
        raw = await (complete or _complete)(system, _build_user(sl, hid))
        data = _parse_json(raw)
        if not data:
            return None
        verdict = str(data.get("verdict", "")).strip().lower()
        # `postponed` used to be accepted here and written onto the hypothesis,
        # which recorded an hour of testing as "never attempted" and left the
        # study reading as idle. It is still tolerated on input, since a model
        # may reach for the old word, and mapped to the verdict it means.
        if verdict == "postponed":
            verdict = "inconclusive"
        if verdict not in ("confirmed", "refuted", "inconclusive"):
            return None

        # The LLM call yields control for long enough that the same store may
        # have been reset to a new research. Never commit an old verdict into
        # that new research generation.
        if expected_research_id is not None \
                and _research_id(graph) != expected_research_id:
            logger.info(
                "[validator] discarding stale judgment for %s from research %s",
                hid,
                expected_research_id,
            )
            return None

        status_updates: List[Dict[str, Any]] = [
            {"id": hid, "status": verdict, "reason": str(data.get("reason", ""))[:300]}]
        # criteria the model judged, restricted to this hypothesis' actual criteria
        crit_ids = {e["from"] for e in sl.get("edges", [])
                    if e["type"] == "formulated_for" and e["to"] == hid}

        # Evidence attached to the hypothesis, and its CURRENT polarity edge (if any).
        cur_pol = {}
        for e in sl.get("edges", []):
            if e["to"] == hid and e["type"] in ("supports", "refutes", "refines", "relates_to") \
                    and by_id.get(e["from"], {}).get("type") == "Evidence":
                # a real polarity edge wins over the neutral relates_to
                if e["type"] != "relates_to" or e["from"] not in cur_pol:
                    cur_pol[e["from"]] = e["type"]

        edges: List[Dict[str, Any]] = []
        polarity = {}  # ev_id -> final supports/refutes/refines
        for eid, pol in (data.get("evidence") or {}).items():
            pol = str(pol).strip().lower()
            if eid not in cur_pol or pol not in ("supports", "refutes", "refines"):
                continue
            polarity[eid] = pol
            # add the polarity edge only if it isn't already present as such
            if cur_pol[eid] != pol:
                edges.append({"type": pol, "from": eid, "to": hid})
        # evidence that already had a polarity keeps it
        for eid, et in cur_pol.items():
            if et in ("supports", "refutes", "refines"):
                polarity.setdefault(eid, et)

        # A criterion may only be marked met WITH the measurement that meets it
        # (store._commit_locked refuses a bare `met`), so the reason names the
        # evidence this judgment actually rested on. Appended after the polarity
        # pass because that is where those ids become known; the store scans the
        # whole commit for met criteria, so the order does not matter.
        supporting = sorted(e for e, pol in polarity.items() if pol == "supports")
        for ccid, met in (data.get("criteria") or {}).items():
            if ccid not in crit_ids:
                continue
            is_met = str(met).strip().lower() in ("met", "true", "yes", "1")
            cur = by_id.get(ccid, {}).get("status")
            if not ((is_met and cur == "not_met") or (not is_met and cur == "met")):
                continue
            update = {"id": ccid, "status": "met" if is_met else "not_met"}
            if is_met:
                update["reason"] = (
                    ("met by " + ", ".join(supporting)) if supporting
                    else "met by the evidence on this hypothesis")
            status_updates.append(update)

        nodes: List[Dict[str, Any]] = []
        concl = str(data.get("conclusion", "")).strip()
        if concl:
            # Every field the judge produced, so the card carries the chain
            # and not only the answer. Empty ones are dropped rather than
            # stored blank: a labelled empty row in the panel reads as "we
            # looked and there was nothing", which is a different claim.
            attrs = {"synthesis": concl[:2000]}
            for key, cap in (("how_established", 2000), ("against_criteria", 1200),
                             ("validity_bounds", 800), ("open_questions", 1200)):
                text = str(data.get(key, "") or "").strip()
                if text:
                    attrs[key] = text[:cap]
            nodes.append({"type": "Conclusion", "ref": "cl", "attrs": attrs})
            edges += [{"type": "based_on", "from": "#cl", "to": eid}
                      for eid in sorted(polarity)]
            edges += [{"type": "determines_sufficiency", "from": ccid, "to": "#cl"}
                      for ccid in sorted(crit_ids)]

        result = graph.commit(
            source=SOURCE,
            nodes=nodes,
            edges=edges,
            status_updates=status_updates,
        )
        if result.ok:
            logger.info("[validator] %s → %s%s", hid, verdict,
                        " (+Conclusion)" if concl else "")
            return result.model_dump(exclude_none=True)

        # A refusal is not a verdict. The store refuses a confirmation whose
        # criteria are still unmet, or which no Evidence supports — both of
        # which are "not yet", not "no". Writing `inconclusive` here closed a
        # branch whose missing measurement was still on its way, and discarded
        # the judge's own `met` findings with the atomic commit. The branch
        # stays under verification; what bounds the cost is the plugin's dedupe
        # key, which fingerprints the evidence set (see `_key`).
        logger.info("[validator] %s commit refused, leaving it under "
                    "verification: %s", hid, result.errors)
        return result.model_dump(exclude_none=True)
    except Exception as exc:  # noqa: BLE001 — never break a run
        logger.warning("[validator] judging %s failed: %s", hid, exc)
        return None


class BackgroundValidatorPlugin(BasePlugin):
    """Fires fire-and-forget validations when a hypothesis gains evidence.

    The callback NEVER awaits the LLM — it schedules `judge_hypothesis` on the
    event loop and returns immediately, so the orchestration loop is never
    blocked (full asynchrony). Dedup includes the research generation and exact
    evidence identities/polarities, so a hypothesis is re-judged when its
    evidence changes without leaking state into a later research."""

    def __init__(self) -> None:
        super().__init__(name="background_validator")
        self._completed: set[tuple] = set()
        self._inflight: set[tuple] = set()
        self._research_by_graph: Dict[int, str] = {}

    @staticmethod
    def _key(graph: Any, research_id: str, item: Dict[str, Any]) -> tuple:
        evidence = tuple(
            tuple(sorted(str(value) for value in (item.get(kind) or [])))
            for kind in ("supporting", "refuting", "related")
        )
        return id(graph), research_id, str(item["hypothesis"]), evidence

    def _activate_research(self, graph: Any, research_id: str) -> None:
        """Drop completed dedup entries when a store starts a new research."""
        graph_id = id(graph)
        previous = self._research_by_graph.get(graph_id)
        if previous == research_id:
            return
        self._research_by_graph[graph_id] = research_id
        self._completed = {key for key in self._completed if key[0] != graph_id}

    async def _run_validation(
        self,
        *,
        key: tuple,
        graph: Any,
        hypothesis: str,
        research_id: str,
        language: Optional[str] = None,
    ) -> None:
        try:
            result = await judge_hypothesis(
                hypothesis,
                graph=graph,
                expected_research_id=research_id,
                language=language,
            )
            # Settled for THIS evidence set whether the commit was accepted or
            # REFUSED. A refusal is deterministic in the slice it was made on,
            # so retrying it buys another LLM call and the same answer; keying
            # only on success meant a hypothesis the store would not let the
            # judge confirm was re-judged on every later commit by anyone, for
            # the rest of the run. New evidence changes the key and the branch
            # is judged again — which is the one thing that can change the
            # answer.
            if result and _research_id(graph) == research_id:
                self._completed.add(key)
        except Exception as exc:  # noqa: BLE001 -- background work is best-effort
            logger.warning("[validator] background judgment for %s failed: %s",
                           hypothesis, exc)
        finally:
            # A failed or rejected validation remains retryable on the next
            # research_commit callback.
            self._inflight.discard(key)

    def schedule_for_graph(self, graph: Any) -> int:
        """Enqueue judgments for hypotheses that have evidence but no verdict.

        Used by the ``research_commit`` plugin hook and by the Experiment
        Module graph bridge (privileged ``store.commit``, no tool call).
        Returns how many tasks were scheduled. No-op without a running loop.
        """
        if not _enabled() or graph is None:
            return 0
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return 0
        try:
            research_id = _research_id(graph)
            # The one place a tool_context is in scope, so the one place the
            # study's language can be read.
            try:
                language = tool_context.state.get(REPORT_LANGUAGE_STATE_KEY)
            except Exception:  # noqa: BLE001 — a judgment is worth more than its language
                language = None
            self._activate_research(graph, research_id)
            scheduled = 0
            for item in queries.unresolved_hypotheses(graph)["items"]:
                key = self._key(graph, research_id, item)
                if key in self._completed or key in self._inflight:
                    continue
                self._inflight.add(key)
                logger.info("[validator] scheduling background judgment for %s",
                            item["hypothesis"])
                task = loop.create_task(
                    self._run_validation(
                        key=key,
                        graph=graph,
                        hypothesis=item["hypothesis"],
                        research_id=research_id,
                        language=language,
                    )
                )
                _TASKS.add(task)
                task.add_done_callback(_TASKS.discard)
                scheduled += 1
            return scheduled
        except Exception:  # noqa: BLE001 — background work is best-effort
            return 0

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> None:
        if not _enabled() or getattr(tool, "name", "") != "research_commit":
            return None
        try:
            self.schedule_for_graph(get_research_graph(tool_context))
        except Exception:  # noqa: BLE001
            pass
        return None

    async def after_run_callback(self, *, invocation_context, **kwargs) -> None:
        """When the run finishes, wait (bounded) for outstanding validations so
        their verdicts + Conclusions are in the graph before the answer is
        reported — otherwise a completed run can still look 'all formulated'."""
        await drain_validations()
        return None


background_validator_plugin = BackgroundValidatorPlugin()
