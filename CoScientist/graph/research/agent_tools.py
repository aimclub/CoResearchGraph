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
# Canonical top-level goal the orchestrator declared via research_init. The
# module-first gate prefers it over a downstream (possibly reworded) delegation
# brief, so the Experiment Module receives the real intent, not "find literature".
ROOT_GOAL_STATE_KEY = "orchestrator_root_goal"


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


def _activations(tool_context: Any, agent: str,
                 limit: int = 6) -> List[Dict[str, str]]:
    """This agent's current activation and the ones above it, nearest first.

    Each entry is ``{agent, exec_id}``. The id is what makes a participation row
    point at a RUN rather than at a name: an agent that worked on three nodes in
    one study is three activations, and a reader following the record wants the
    one that produced this node, not the agent's whole history.

    `plugin._mint` gives every activation the id `agent:{Name}@{turn}[#n]`, and
    the before_tool hook draws a `delegated_to` edge from the caller's
    activation to the callee's before the callee runs. So the chain is on
    record and survives a restart. `plan_tracker.js` was re-deriving it live in
    the browser and losing it on replay; nothing server-side read it at all.

    The current activation is the agent's newest one by start time — activations
    are minted per run, so the newest is the one running now.
    """
    try:
        from CoScientist.graph.memory import get_knowledge_graph
        raw = get_knowledge_graph(tool_context).full()
    except Exception:  # noqa: BLE001
        return []
    try:
        mine = [n for n in raw.get("nodes") or []
                if n.get("kind") in ("agent", "agent_call")
                and n.get("executor_agent") == agent]
        if not mine:
            return []
        here = max(mine, key=lambda n: n.get("t_start") or 0)["id"]
        by_id = {n["id"]: n for n in raw.get("nodes") or [] if n.get("id")}
        parent = {e["dst"]: e["src"] for e in raw.get("edges") or []
                  if e.get("type") == "delegated_to" and e.get("dst") and e.get("src")}
        chain = [{"agent": agent, "exec_id": here}]
        seen = {here}
        node = parent.get(here)
        while node and node not in seen and len(chain) <= limit:
            seen.add(node)
            name = (by_id.get(node) or {}).get("executor_agent")
            if name and name != agent:
                chain.append({"agent": name, "exec_id": node})
            node = parent.get(node)
        return chain
    except Exception:  # noqa: BLE001
        return []


def _own_activation(tool_context: Any, agent: str) -> str:
    """The id of the run this agent is making right now, or "".

    Read once per commit, so the authorship rows the store writes point at a
    run rather than at a name.
    """
    chain = _activations(tool_context, agent, limit=0)
    return chain[0]["exec_id"] if chain else ""


def _note_participation(research_graph: Any, result: Any, agent: str,
                        tool_context: Any) -> None:
    """Record who took part in what this commit wrote, and on what basis.

    The store itself records the author and whoever moved a status. This adds
    the one basis only the calling side can know: the agents that delegated the
    work down to this one. They took part in it — the orchestrator that sent a
    worker after a hypothesis is why the evidence exists — and nothing else
    writes that down.

    No `provenance` row is manufactured here. `_recent_tool_calls` only ever
    collects the CALLING agent's own calls, so such a row would name the agent
    the `commit` row already names; the calls themselves are on the node as
    `_provenance` and the panel jumps into the log from there.
    """
    try:
        committed = getattr(result, "committed", None) or {}
        ids = [e["id"] for e in (committed.get("nodes") or []) if e.get("id")]
        ids += [e["id"] for e in (committed.get("status_updates") or [])
                if e.get("id") and not e.get("auto")]
        if not ids:
            return
        # The first entry is this agent's own activation; the rest delegated to
        # it. Each row carries the id of the RUN, so a reader can open the log
        # at the activation that produced this node instead of at whatever that
        # agent happened to do last.
        chain = _activations(tool_context, agent)
        rows = [{"node_id": nid, "agent": link["agent"],
                 "basis": "delegation", "exec_id": link["exec_id"]}
                for link in chain[1:]
                for nid in ids]
        research_graph.add_contributors(rows, source=agent)
    except Exception:  # noqa: BLE001 — bookkeeping never breaks a write
        logger.debug("could not record participation", exc_info=True)


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


_EVIDENCE_TYPES = ("evidence", "свидетельство")


def _attach_paper(attrs: Dict[str, Any], tool_context: Any) -> None:
    """Give an Evidence citing a DOI the copy of the paper the session holds.

    The agent writes the citation it read — `source_ref: 10.1021/…` — and has no
    way of knowing that `PaperCapturePlugin` already fetched that paper.
    Joining them here is what turns a DOI in the panel from a string into a file
    with a download beside it.

    Matched on identifiers and never on the title: two papers share a title far
    more often than they share a DOI. The WHOLE citation is read, not one field
    — an agent writes `source_ref` as a semicolon-separated list in which the
    only handle we hold may be a PMC id in third place. Nothing the agent wrote
    is overwritten: an attribute already present is its claim, not ours.
    """
    if attrs.get("session_artifact_id"):
        return
    try:
        from CoScientist.reporting import paper_library as pl

        cited = " ".join(str(attrs.get(k) or "")
                         for k in ("doi", "pmcid", "source_ref"))
        paper = pl.find(getattr(tool_context, "state", None), cited)
        if not paper or not paper.get("session_artifact_id"):
            return
        attrs["session_artifact_id"] = paper["session_artifact_id"]
        # The record's keys are already `kind:value`; re-parsing them as prose
        # would not match, because `pmc:12610272` is not how a PMC id is written.
        held = dict(k.split(":", 1) for k in (paper.get("refs") or []) if ":" in k)
        stamps = {
            "doi": paper.get("doi_raw") or paper.get("doi") or (
                held.get("doi") or ""),
            "pmcid": f"PMC{held['pmc']}" if held.get("pmc") else "",
            "paper_title": paper.get("title"),
            "paper_year": paper.get("year"),
        }
        for key, value in stamps.items():
            if value and not attrs.get(key):
                attrs[key] = value
    except Exception:  # noqa: BLE001 — a missing paper must not refuse a commit
        return


def _enrich_evidence(nodes: Optional[List[Dict[str, Any]]], agent: str,
                     tool_context: Any = None) -> Optional[List[Dict[str, Any]]]:
    """Stamp each newly-created Evidence with what the record already knows:
    the tool calls that produced it, and the paper it cites when we hold one."""
    if not nodes:
        return nodes
    prov = None
    out = []
    for n in nodes:
        n = dict(n)
        # only CREATE ops for Evidence (id-merges keep their existing provenance)
        if not n.get("id") and str(n.get("type", "")).strip().lower() in _EVIDENCE_TYPES:
            attrs = dict(n.get("attrs") or {})
            if "_provenance" not in attrs:
                if prov is None:
                    prov = _recent_tool_calls(tool_context, agent)
                if prov:
                    attrs["_provenance"] = prov
            _attach_paper(attrs, tool_context)
            n["attrs"] = attrs
        out.append(n)
    return out


#: The name this had while it only did the first half.
_attach_provenance = _enrich_evidence


def _context_budget() -> int:
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.context_char_budget
    except Exception:  # noqa: BLE001
        return 4000


def _orchestrator_digest(research_graph) -> str:
    """Overview index + active-trigger digest — what the orchestrator sees."""
    # Not `is_empty()`: the FIRST worker write — one Evidence, one Tool — makes
    # the graph non-empty and used to silence this line for the rest of the run,
    # so a study that never got a root question stopped being told it has none.
    # Everything downstream needs the root: the frame card, the provenance
    # chain, and the guarantee that nothing floats unattached.
    if research_graph.root_id() is None:
        return ("Research graph has NO ROOT QUESTION. If this is a research "
                "task, call research_init(question=...) before delegating.")
    budget = _context_budget()
    # The two halves are budgeted separately, and the index is cut by whole
    # LINES with a visible count. Handing the whole budget to the triggers and
    # then slicing the concatenation took every overflowing character off the
    # END — which is the index — mid-token: a real run measured 3592 chars, and
    # one more mirrored plan step tipped it over, so the index ended
    # "- VM7 [VerificationMeth …[truncated]" and the newest method simply was
    # not there. Silent, and exactly the row the orchestrator needed.
    triggers = queries.trigger_report(
        research_graph, char_budget=max(400, budget // 2)).get("rendered", "")
    overview = research_graph.overview().get("rendered", "")
    head = ("ACTIVE TRIGGERS:\n" + triggers + "\n\n") if triggers else ""
    if not overview:
        return head.rstrip()
    label = "GRAPH INDEX:\n"
    room = budget - len(head) - len(label)
    kept, dropped = [], 0
    for line in overview.split("\n"):
        if room - len(line) - 1 < 0:
            dropped += 1
            continue
        kept.append(line)
        room -= len(line) + 1
    if dropped:
        kept.append(f"…[{dropped} more row(s) not shown — call research_overview]")
    return head + label + "\n".join(kept)


def _worker_context(research_graph, state: Any) -> str:
    """A worker's slice of the graph: the focus node's neighborhood if the
    orchestrator set one, else the compact overview so the worker can find ids."""
    if research_graph.is_empty():
        # An empty string said nothing at all — not that the graph is empty, not
        # that the worker may write to it. Read as "there is no graph here", and
        # a worker that had been shown H-ids in its protocol example went on to
        # reference a hypothesis that did not exist, which rejected its whole
        # commit and lost the finding with it.
        return ("The research graph is EMPTY — nothing has been recorded yet. "
                "Record YOUR results in it (see your RESEARCH GRAPH section). "
                "Do NOT reference any node id: there are none yet. Create only "
                "what you are allowed to create, and link it with your own "
                "edges using \"#ref\" handles from the same commit.")
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
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Record your results in the shared research graph in ONE transaction.

        Everything is validated together and applied all-or-nothing: if any item
        is invalid, NOTHING is written and you get per-item errors to fix and
        retry. You may only write node types / edges / status changes your role
        is allowed (see the RESEARCH GRAPH section of your prompt).

        Args:
            nodes: list of node ops. CREATE: {"type": "<allowed_node_type>", "attrs": {...},
                "status": "..." (optional), "ref": "local_handle" (optional)}.
                CHANGE a node that is already in the graph: {"id": "H3",
                "attrs": {...}} — the id is what says WHICH node, so use the id
                the graph shows you, not the ref you once wrote. A `ref` is only
                a handle for a node created in THIS call, to point edges at
                before it has an id; writing an existing node's id there is
                refused, because it reads as an update and would record a
                duplicate. Never invent an id for a new node: the store mints
                them.
            edges: list of {"type": "supports", "from": "E4", "to": "H2"}. To
                point at a node created in THIS call, use its ref with a leading
                "#", e.g. "from": "#e_new" — a handle of your own, never the id of a
                node already in the graph.
            status_updates: list of {"id": "H2", "status": "under_verification",
                "reason": "..." (optional)}.

        Returns:
            {"ok": true, "message": ..., "committed": {...}, "graph_stats": {...}}
            on success, or {"ok": false, "errors": [...], "hint": ...} — read the
            errors, fix the payload, and call research_commit again.
        """
        # Robust fallback for LLMs that serialize the node/edge list into kwargs or keys
        if (nodes is None or edges is None) and kwargs:
            for k, v in list(kwargs.items()):
                # Check key first
                if isinstance(k, str):
                    k_str = k.strip()
                    if k_str.startswith("[") or k_str.startswith("{"):
                        try:
                            parsed = json.loads(k_str)
                            if isinstance(parsed, list):
                                if parsed and isinstance(parsed[0], dict) and ("from" in parsed[0] or "to" in parsed[0]):
                                    edges = edges or parsed
                                elif nodes is None:
                                    nodes = parsed
                            elif isinstance(parsed, dict):
                                nodes = nodes or parsed.get("nodes")
                                edges = edges or parsed.get("edges")
                                status_updates = status_updates or parsed.get("status_updates")
                        except Exception:
                            pass
                # Check val
                if isinstance(v, str):
                    v_str = v.strip()
                    if v_str.startswith("[") or v_str.startswith("{"):
                        try:
                            parsed = json.loads(v_str)
                            if isinstance(parsed, list):
                                if parsed and isinstance(parsed[0], dict) and ("from" in parsed[0] or "to" in parsed[0]):
                                    edges = edges or parsed
                                elif nodes is None:
                                    nodes = parsed
                            elif isinstance(parsed, dict):
                                nodes = nodes or parsed.get("nodes")
                                edges = edges or parsed.get("edges")
                                status_updates = status_updates or parsed.get("status_updates")
                        except Exception:
                            pass
                elif isinstance(v, list):
                    if v and isinstance(v[0], dict) and ("from" in v[0] or "to" in v[0]):
                        edges = edges or v
                    elif nodes is None and (k in ("nodes", "node_list", "hypotheses", "node_ops", "items", "args") or (len(v) > 0 and isinstance(v[0], dict) and ("type" in v[0] or "attrs" in v[0]))):
                        nodes = v

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
                source=agent, nodes=_enrich_evidence(nodes, agent, tool_context),
                edges=edges,
                status_updates=status_updates, autolink_focus=focus,
                exec_id=_own_activation(tool_context, agent),
            )
        except Exception as exc:  # noqa: BLE001 — a bad payload must not end the run
            logger.exception("research_commit failed")
            return {"ok": False, "errors": [f"{type(exc).__name__}: {exc}"],
                    "hint": "the commit was rejected, nothing was written — fix "
                            "the payload and call research_commit again"}
        if result.ok:
            _refresh_context_state(tool_context, self._is_root)
            _note_participation(research_graph, result, agent, tool_context)
        else:
            # The store logs the refusal too; this line names the agent whose
            # step was lost, which is what a reader of the run log needs.
            logger.warning("research_commit refused for %s: %s",
                           agent, "; ".join(result.errors[:3]))
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
            if self._is_root and isinstance(question, str) and question.strip():
                try:
                    tool_context.state[ROOT_GOAL_STATE_KEY] = question.strip()
                except Exception:  # noqa: BLE001
                    pass
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
