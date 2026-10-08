"""ADK tools + prompt-injection callback for the Research Context Graph.

Two surfaces are exposed:
  * "worker"       — research_commit, research_context_slice, research_overview,
                     research_provenance (given to ResearchAgent, CoderAgent, …).
  * "orchestrator" — the worker tools PLUS research_init, research_triggers and
                     research_set_focus (given only to the root orchestrator).

The split is deliberate: the assembler renders each tool's ToolDoc into the
agent's prompt and the guard_unknown_tools callback whitelists exactly those
names, so documenting an orchestrator-only tool in a worker's prompt would let
the model call a tool it does not have and crash the run.

`source` is NEVER a tool parameter — every write is attributed to the CALLING
agent via ToolContext.agent_name, which the LLM cannot spoof. After a successful
write the tool refreshes state['research_context'] so the {research_context?}
prompt block reflects the new graph state on the agent's next LLM request
(before_agent fires once per invocation; ADK re-resolves the placeholder each
request).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext

from CoScientist.graph.research import queries
from CoScientist.graph.research.store import get_research_graph

logger = logging.getLogger(__name__)

FOCUS_STATE_KEY = "research_focus"
CONTEXT_STATE_KEY = "research_context"


def _agent(tool_context: Optional[ToolContext]) -> str:
    return getattr(tool_context, "agent_name", None) or "unknown"


def _recent_tool_calls(tool_context: Any, agent: str,
                       limit: int = 6) -> List[Dict[str, Any]]:
    """The calling agent's most recent EXECUTION tool calls (from the execution
    graph) — the concrete `tavily_search` / `execute_bash` / MCP calls that
    produced the finding. Attached to Evidence as provenance so every piece of
    evidence is traceable back to the exact call + result that yielded it.

    The graph must be the session-scoped one. Reading the module-level default
    instead took the ids from a different, usually empty graph, so every
    recorded `exec_id` pointed at a node the execution view does not contain and
    the provenance link in the UI could never resolve.
    """
    try:
        from CoScientist.graph.memory import get_knowledge_graph
        hist = get_knowledge_graph(tool_context).history(limit=60)
    except Exception:  # noqa: BLE001
        return []
    calls = [h for h in hist
             if h.get("kind") == "tool_call" and h.get("agent") == agent
             and not str(h.get("label", "")).startswith("research_")]
    out = []
    for h in calls[-limit:]:
        out.append({"exec_id": h.get("id"), "tool": h.get("label"),
                    "result": (h.get("output") or "")[:400]})
    return out


def _as_op_list(value: Any, field: str) -> Optional[List[Dict[str, Any]]]:
    """Coerce a commit argument into the list of operations it was meant to be.

    Accepts what models actually send: the list itself, a JSON string holding
    that list, or a single operation object. Raises ValueError with a message
    the agent can act on when the value cannot be one.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field}: expected a JSON array of objects, got a "
                             f"string that is not valid JSON ({exc.msg})") from exc
    if isinstance(value, dict):          # one op passed unwrapped
        value = [value]
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected a list of objects, got "
                         f"{type(value).__name__}")
    bad = [i for i, item in enumerate(value) if not isinstance(item, dict)]
    if bad:
        raise ValueError(f"{field}: items {bad} are not objects")
    return value


def _attach_provenance(nodes: Optional[List[Dict[str, Any]]], agent: str,
                       tool_context: Any = None) -> Optional[List[Dict[str, Any]]]:
    """Stamp each newly-created Evidence node with the producing tool calls."""
    if not nodes:
        return nodes
    prov = None
    out = []
    for n in nodes:
        n = dict(n)
        # only CREATE ops for Evidence (id-merges keep their existing provenance)
        if not n.get("id") and str(n.get("type", "")).strip().lower() in ("evidence", "свидетельство"):
            attrs = dict(n.get("attrs") or {})
            if "_provenance" not in attrs:
                if prov is None:
                    prov = _recent_tool_calls(tool_context, agent)
                if prov:
                    attrs["_provenance"] = prov
            n["attrs"] = attrs
        out.append(n)
    return out


def _context_budget() -> int:
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.context_char_budget
    except Exception:  # noqa: BLE001
        return 4000


def _orchestrator_digest(research_graph) -> str:
    """Overview index + active-trigger digest — what the orchestrator sees."""
    if research_graph.is_empty():
        return ("Research graph is EMPTY. If this is a research task, call "
                "research_init(question=...) before delegating.")
    budget = _context_budget()
    overview = research_graph.overview().get("rendered", "")
    triggers = queries.trigger_report(research_graph, char_budget=budget).get("rendered", "")
    parts = []
    if triggers:
        parts.append("ACTIVE TRIGGERS:\n" + triggers)
    if overview:
        parts.append("GRAPH INDEX:\n" + overview)
    text = "\n\n".join(parts)
    return text[:budget] + ("\n…[truncated]" if len(text) > budget else "")


def _worker_context(research_graph, state: Any) -> str:
    """A worker's slice of the graph: the focus node's neighborhood if the
    orchestrator set one, else the compact overview so the worker can find ids."""
    if research_graph.is_empty():
        return ""
    focus = None
    try:
        focus = (state or {}).get(FOCUS_STATE_KEY)
    except Exception:  # noqa: BLE001
        focus = None
    if focus:
        sl = research_graph.get_context_slice(focus)
        if "error" not in sl:
            return sl["rendered"]
    return "GRAPH INDEX (call research_context_slice on a node for detail):\n" \
        + research_graph.overview().get("rendered", "")


def _refresh_context_state(tool_context: Optional[ToolContext], is_root: bool) -> None:
    if tool_context is None:
        return
    try:
        research_graph = get_research_graph(tool_context)
        if is_root:
            tool_context.state[CONTEXT_STATE_KEY] = _orchestrator_digest(research_graph)
        else:
            tool_context.state[CONTEXT_STATE_KEY] = _worker_context(
                research_graph,
                tool_context.state,
            )
    except Exception:  # noqa: BLE001 — refreshing context must never break a write
        pass


# ── the toolset ───────────────────────────────────────────────────────────────

class ResearchGraphToolset(BaseToolset):
    """Research-graph tools for one agent surface ("worker" | "orchestrator" |
    "reporter"). The "reporter" surface is READ-ONLY (no research_commit) — it
    is for the Result Aggregator, which reads the finished graph to write the
    report but must never mutate it."""

    def __init__(self, surface: str = "worker", prefix: Optional[str] = None) -> None:
        super().__init__(tool_name_prefix=prefix)
        self.surface = surface
        self._is_root = surface == "orchestrator"
        self._read_only = surface == "reporter"

    async def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        tools = [
            FunctionTool(self.research_context_slice),
            FunctionTool(self.research_overview),
            FunctionTool(self.research_provenance),
        ]
        if not self._read_only:
            tools.insert(0, FunctionTool(self.research_commit))
        if self._is_root:
            tools += [
                FunctionTool(self.research_init),
                FunctionTool(self.research_triggers),
                FunctionTool(self.research_set_focus),
                FunctionTool(self.research_prior),
            ]
        return tools

    async def close(self) -> None:
        pass

    # ── writes ────────────────────────────────────────────────────────────────

    def research_commit(
        self,
        tool_context: ToolContext,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        status_updates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Record your results in the shared research graph in ONE transaction.

        Everything is validated together and applied all-or-nothing: if any item
        is invalid, NOTHING is written and you get per-item errors to fix and
        retry. You may only write node types / edges / status changes your role
        is allowed (see the RESEARCH GRAPH section of your prompt).

        Args:
            nodes: list of node ops. CREATE: {"type": "Evidence", "attrs": {...},
                "status": "obtained" (optional), "ref": "e1" (optional local
                handle)}. ENRICH an existing node's attrs: {"id": "EB1",
                "attrs": {...}} (no "type").
            edges: list of {"type": "supports", "from": "E4", "to": "H2"}. To
                point at a node created in THIS call, use its ref with a leading
                "#", e.g. "from": "#e1".
            status_updates: list of {"id": "H2", "status": "under_verification",
                "reason": "..." (optional)}.

        Returns:
            {"ok": true, "message": ..., "committed": {...}, "graph_stats": {...}}
            on success, or {"ok": false, "errors": [...], "hint": ...} — read the
            errors, fix the payload, and call research_commit again.
        """
        # If the orchestrator set a focus hypothesis before delegating, evidence
        # this worker records is auto-linked to it (relates_to) so it is never
        # orphaned — the background validator then decides its polarity.
        focus = None
        try:
            focus = (tool_context.state or {}).get(FOCUS_STATE_KEY)
        except Exception:  # noqa: BLE001
            focus = None

        # Models routinely hand structured arguments over as a JSON STRING. That
        # used to reach the store as text and raise deep inside it — and the
        # exception escaped the tool, killing the delegation chain up to the
        # orchestrator and losing a run that had already produced its result.
        # Parse what was meant, and if anything still fails, hand the agent an
        # error it can act on instead of ending the run.
        try:
            nodes, edges, status_updates = (_as_op_list(nodes, "nodes"),
                                            _as_op_list(edges, "edges"),
                                            _as_op_list(status_updates, "status_updates"))
        except ValueError as exc:
            return {"ok": False, "errors": [str(exc)],
                    "hint": "pass nodes/edges/status_updates as JSON arrays of "
                            "objects (not a string, not a single object)"}

        try:
            research_graph = get_research_graph(tool_context)
            agent = _agent(tool_context)
            result = research_graph.commit(
                source=agent, nodes=_attach_provenance(nodes, agent, tool_context),
                edges=edges,
                status_updates=status_updates, autolink_focus=focus,
            )
        except Exception as exc:  # noqa: BLE001 — a bad payload must not end the run
            logger.exception("research_commit failed")
            return {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"],
                    "hint": "the commit was rejected, nothing was written — fix "
                            "the payload and call research_commit again"}
        if result.ok:
            _refresh_context_state(tool_context, self._is_root)
        return result.model_dump(exclude_none=True)

    def research_init(
        self,
        tool_context: ToolContext,
        question: str,
        attrs: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        resources: Optional[List[Dict[str, Any]]] = None,
        empirical_bases: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Start a NEW research: create the root ResearchQuestion and its context
        star. Call this ONCE at the start of a research task, before delegating.
        Any currently-active research graph is archived first.

        Args:
            question: the root research question (its formulation).
            attrs: extra question attributes (domain, gap, research_form, …).
            constraints: [{"subtype": "ethics"|"profile"|..., "content": "..."}].
            tools: known tools [{"name": "...", "tool_type": "...", "status":
                "available"|"needs_adaptation"|"being_created"}].
            resources: budgets [{"resource_type": "GPU-hours", "remaining": 100,
                "limit": 100}].
            empirical_bases: data [{"base_type": "dataset", "volume": "...",
                "source_ref": "..."}].
        """
        research_graph = get_research_graph(tool_context)
        out = research_graph.init_research(
            source=_agent(tool_context), question=question, attrs=attrs,
            constraints=constraints, tools=tools, resources=resources,
            empirical_bases=empirical_bases,
        )
        if out.get("ok"):
            _refresh_context_state(tool_context, self._is_root)
        return out

    def research_set_focus(self, tool_context: ToolContext, node_id: str) -> Dict[str, Any]:
        """Set the graph node the NEXT delegated worker should focus on. The
        worker will receive that node's context slice automatically. Call this
        right before delegating a step about a specific hypothesis/question.

        Args:
            node_id: e.g. "H2".
        """
        sl = get_research_graph(tool_context).get_context_slice(node_id)
        if "error" in sl:
            return {"ok": False, "error": sl["error"]}
        try:
            tool_context.state[FOCUS_STATE_KEY] = node_id
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"could not set focus: {exc}"}
        return {"ok": True, "focus": node_id, "slice": sl["rendered"]}

    # ── reads ─────────────────────────────────────────────────────────────────

    def research_context_slice(self, tool_context: ToolContext, node_id: str,
                               depth: int = 1) -> Dict[str, Any]:
        """Get one node plus its neighborhood (its 1–2 hop context) — the focused
        view to work from instead of the whole graph.

        Args:
            node_id: e.g. "H2".
            depth: 1 (immediate neighbors) or 2. Capped by settings.
        """
        return get_research_graph(tool_context).get_context_slice(node_id, depth=depth)

    def research_overview(self, tool_context: ToolContext) -> Dict[str, Any]:
        """Compact index of the whole research graph: every node's id, type,
        status and label (no attribute detail). Use it to find node ids."""
        return get_research_graph(tool_context).overview()

    def research_prior(self, tool_context: ToolContext, query: str,
                       limit: int = 3) -> Dict[str, Any]:
        """Search PAST researches (previous runs) for work related to `query`.

        Use before planning: a hypothesis this system already confirmed or
        refuted, and the method/tools it used, should be reused rather than
        re-derived. Matching is deterministic token overlap, and every hit
        reports the tokens that matched so you can judge relevance yourself.

        Args:
            query: the current question or topic, in your own words.
            limit: how many prior researches to return (default 3).

        Returns:
            {"found": n, "priors": [{question, score, matched_tokens, counts,
             hypotheses[{status, formulation}], methods, conclusions, tools}]}.
        """
        try:
            from CoScientist.graph.research.index import get_research_index
            graph = get_research_graph(tool_context)
            hits = get_research_index().search(
                query, limit=max(1, min(int(limit or 3), 10)),
                exclude_id=getattr(graph, "_research_id", "") or "")
        except Exception as exc:  # noqa: BLE001 — never break a run on the index
            return {"found": 0, "priors": [], "error": str(exc)[:200]}
        slim = [{k: h.get(k) for k in ("question", "score", "matched_tokens", "counts",
                                       "hypotheses", "methods", "conclusions", "tools",
                                       "research_id")} for h in hits]
        return {"found": len(slim), "priors": slim}

    def research_provenance(self, tool_context: ToolContext, node_id: str) -> Dict[str, Any]:
        """Trace a node back to the root research question: the chain of nodes,
        edges and their sources (who produced each step)."""
        return get_research_graph(tool_context).get_provenance(node_id)

    def research_triggers(self, tool_context: ToolContext) -> Dict[str, Any]:
        """Evaluate the decision triggers over the current graph: which
        hypotheses are READY to verify (tools available), which are BLOCKED,
        REFUTE signals, CLOSABLE hypotheses (write a Conclusion), PENDING
        conclusions, TOOLS not ready, RESOURCES low, open QUESTIONS and PROGRESS.
        Consult this before deciding the next step."""
        return queries.trigger_report(
            get_research_graph(tool_context),
            char_budget=_context_budget(),
        )


# ── before_agent injection callback ────────────────────────────────────────────

def _prior_research_digest(callback_context, research_graph) -> str:
    """Digest of related PAST researches, appended to the orchestrator context."""
    try:
        from CoScientist.graph.research.index import format_priors, get_research_index
        question = ""
        try:
            root = research_graph.root_id()
            if root:
                node = research_graph.full()["nodes"]
                question = next((( n.get("attrs") or {}).get("formulation", "")
                                 for n in node if n.get("id") == root), "")
        except Exception:  # noqa: BLE001
            question = ""
        if not question:
            # Before the graph is seeded, match on the user's request itself.
            question = str(callback_context.state.get("user_query", ""))[:500]
        if not question:
            return ""
        hits = get_research_index().search(
            question, limit=3,
            exclude_id=getattr(research_graph, "_research_id", "") or "")
        if not hits:
            return ""
        return ("\n\nПРОШЛЫЕ ИССЛЕДОВАНИЯ (переиспользуй, не переоткрывай; "
                "проверь актуальность прежде чем опираться):\n"
                + format_priors(hits))
    except Exception:  # noqa: BLE001
        return ""


def make_inject_research_context(is_root: bool):
    """Build the before_agent callback that seeds state['research_context'] for
    the {research_context?} prompt placeholder. Orchestrator gets the overview +
    trigger digest; a worker gets its focus slice (or the compact overview).
    Best-effort — the graph must never break a run."""

    def inject_research_context(callback_context: CallbackContext):
        try:
            from CoScientist.config import get_settings
            if not get_settings().research_graph.enabled:
                return None
        except Exception:  # noqa: BLE001
            pass
        try:
            research_graph = get_research_graph(callback_context)
            if is_root:
                digest = _orchestrator_digest(research_graph)
                # Cross-run reuse: surface what PAST runs already settled about
                # this question, so the orchestrator builds on them instead of
                # re-deriving. Best-effort and capped; absent priors change
                # nothing.
                digest += _prior_research_digest(callback_context, research_graph)
                callback_context.state[CONTEXT_STATE_KEY] = digest
            else:
                callback_context.state[CONTEXT_STATE_KEY] = _worker_context(
                    research_graph,
                    callback_context.state,
                )
        except Exception:  # noqa: BLE001
            callback_context.state[CONTEXT_STATE_KEY] = ""
        return None

    return inject_research_context


# Shared instances (mirrors graph_reader_instance / task_tracker_instance).
research_worker_toolset = ResearchGraphToolset(surface="worker")
research_orchestrator_toolset = ResearchGraphToolset(surface="orchestrator")
# Read-only surface for the Result Aggregator (no research_commit).
research_reporter_toolset = ResearchGraphToolset(surface="reporter")
