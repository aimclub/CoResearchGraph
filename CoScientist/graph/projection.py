"""Rule-based projections of the graph into agent context (Fact 2).

Deliberately rule-based (no LLM) so the same graph always yields the same
context — reproducibility is an evaluation metric. See docs/execution_graph.md.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_MAX_ITEMS = 12
_LABEL = 200


def _short(s: Optional[str], n: int = _LABEL) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def _node_line(n: dict) -> str:
    who = n.get("executor_agent") or n.get("kind", "?")
    label = _short(n.get("label"))
    out = n.get("output")
    tail = f" → {_short(out, 160)}" if out else ""
    return f"- [{who}] {label}{tail}"


def orchestrator_summary(full: dict) -> str:
    """Planning view: what already succeeded vs failed/rejected, so the
    orchestrator builds on the former and does NOT repeat the latter."""
    nodes = full.get("nodes", [])
    # Only delegations/decisions matter for planning — skip low-level tool calls.
    plany = [n for n in nodes if n.get("kind") in ("agent_call", "decision", "goal")]
    completed = [n for n in plany if n.get("status") == "success"]
    failed = [
        n for n in plany
        if n.get("status") == "failed" or (n.get("verdict") in ("reject", "wrong"))
    ]
    running = [n for n in plany if n.get("status") == "running"]

    if not (completed or failed or running):
        return ""

    blocks: List[str] = [
        "EXECUTION GRAPH — what has already happened in THIS run. Build on "
        "completed steps; do NOT repeat failed/rejected ones."
    ]
    if completed:
        blocks.append("Completed:\n" + "\n".join(_node_line(n) for n in completed[-_MAX_ITEMS:]))
    if failed:
        blocks.append(
            "Failed / rejected (do not retry as-is):\n"
            + "\n".join(_node_line(n) for n in failed[-_MAX_ITEMS:])
        )
    if running:
        blocks.append("In progress:\n" + "\n".join(_node_line(n) for n in running[-_MAX_ITEMS:]))
    return "\n\n".join(blocks)


# Toolset bindings that decide which path an agent is on. Derived from the
# config rather than hardcoded agent names, so renaming or reshaping the agent
# tree cannot silently break the signal.
_MCP_TOOLSETS = frozenset({"dynamic_tools"})
_CODER_TOOLSETS = frozenset({"coder", "sandbox"})


def _agents_bound_to(toolsets: frozenset) -> set:
    from CoScientist.assembly.schema import get_config

    return {
        name
        for name, agent in get_config().agents.items()
        if toolsets & set(agent.tools or ())
    }


def tool_vs_coder(
    full: dict,
    *,
    mcp_agents: Optional[set] = None,
    coder_agents: Optional[set] = None,
) -> dict:
    """Was the work done by a tool from the catalogue, or written from scratch?

    The system is meant to look for an existing tool first and only fall back to
    writing code. Whether that actually happened is not otherwise recorded
    anywhere, so a catalogue that has quietly stopped being used looks exactly
    like one that is working.

    Read off the execution graph the plugin already emits. The two sides are not
    symmetric, deliberately: for the tool path only a ``tool_call`` counts,
    because delegating to the executor proves nothing — the catalogue lookup may
    have come up empty. For the coder, the delegation itself is the evidence
    that code was written.

    Returns ``{"path", "mcp_tool_calls", "coder_calls"}``, where ``path`` is
    ``"mcp" | "coder" | "mixed" | "none"``. Works on any run's ``full()`` dict,
    including a re-loaded snapshot. The agent sets can be passed in for callers
    that project a graph produced by a different configuration.

    .. note::

       Checked against the 29 recorded runs under ``graph_runs/sessions``: 25
       came out ``coder``, 2 ``mcp``, 2 ``none``. No run mixed the two, and no
       run reached the catalogue by way of a built tool, which is the signal
       this is here to expose.
    """
    mcp_agents = _agents_bound_to(_MCP_TOOLSETS) if mcp_agents is None else mcp_agents
    coder_agents = (
        _agents_bound_to(_CODER_TOOLSETS) if coder_agents is None else coder_agents
    )

    mcp_tool_calls: List[str] = []
    coder_calls: List[str] = []
    for n in full.get("nodes", []):
        kind, who = n.get("kind"), n.get("executor_agent")
        if kind == "tool_call" and who in mcp_agents:
            mcp_tool_calls.append(n.get("label") or "")
        elif kind in ("agent_call", "tool_call") and who in coder_agents:
            coder_calls.append(n.get("label") or "")

    if mcp_tool_calls and coder_calls:
        path = "mixed"
    elif mcp_tool_calls:
        path = "mcp"
    elif coder_calls:
        path = "coder"
    else:
        path = "none"
    return {
        "path": path,
        "mcp_tool_calls": mcp_tool_calls,
        "coder_calls": coder_calls,
    }


def _index(full: dict) -> Dict[str, dict]:
    return {n["id"]: n for n in full.get("nodes", []) if "id" in n}


def local_view(full: dict, node_id: str) -> str:
    """Sub-agent view: the ancestral path (why this step exists) plus validated
    findings in scope. Pushed into the delegation envelope."""
    idx = _index(full)
    if node_id not in idx:
        return ""
    # walk parents up to the root
    chain: List[dict] = []
    seen = set()
    cur: Optional[dict] = idx.get(node_id)
    while cur is not None and cur["id"] not in seen:
        seen.add(cur["id"])
        chain.append(cur)
        parents = cur.get("parent_ids") or []
        cur = idx.get(parents[0]) if parents else None
    chain.reverse()
    if len(chain) <= 1:
        return ""
    path = "\n".join(f"{'  ' * i}↳ {_node_line(n)}" for i, n in enumerate(chain))
    return "REASONING PATH that led to this task (top → here):\n" + path


def _turn_resolver(nodes: List[Dict[str, Any]]):
    """Answer "which request does this node belong to" for a whole graph.

    Shared by the trace list and the per-request graph so the two can never
    disagree about where a call belongs. `turn_id` is authoritative; snapshots
    written before it existed carry the request in the old namespaced id
    ``goal:{inv}::tool:{call}``; anything still unmarked belongs to the last
    request that started before it, because a request is a stretch of time.
    """
    def tagged(node: Dict[str, Any]) -> Optional[str]:
        if node.get("turn_id"):
            return node["turn_id"]
        nid = str(node.get("id", ""))
        for prefix in ("goal:", "result:"):
            if nid.startswith(prefix):
                return nid[len(prefix):].split("::", 1)[0]
        return None

    goals = sorted(((n.get("t_start") or 0.0, tagged(n) or n["id"])
                    for n in nodes if n.get("kind") == "goal"),
                   key=lambda pair: pair[0])

    known = {turn for _, turn in goals}

    def resolve(node: Dict[str, Any]) -> str:
        marked = tagged(node)
        # A mark naming no request is worse than no mark: a delegated agent used
        # to run under its own invocation id, and trusting it split one prompt
        # into several, one of them promptless. Fall through to the clock.
        if marked and marked in known:
            return marked
        started = node.get("t_start") or 0.0
        current = None
        for goal_start, goal_turn in goals:
            if goal_start <= started:
                current = goal_turn
            else:
                break
        # Something that ran before every request still ran. Stranding it in a
        # request nobody can open loses it from every view at once — one
        # session showed 35 of its 130 calls that way, and the evidence linking
        # to the other 95 led nowhere. The first request is where it goes.
        if current is None and goals:
            current = goals[0][1]
        return current or "untagged"

    return resolve


def turns(full: Dict[str, Any]) -> Dict[str, Any]:
    """The session's execution graph as a chronological list of turns.

    The call graph answers "what is connected to what". It cannot answer "what
    happened, in what order", because agent nodes are one per agent for the
    whole session, so every turn's calls hang off the same few nodes and the
    layout is force-directed with no time axis at all.

    This regroups the same records by the prompt that caused them and sorts each
    group by start time, which is the shape a trace viewer needs: one entry per
    user request, and under it every call with its agent, its arguments, its
    result and its duration.

    Turn membership comes from ``turn_id``. Snapshots recorded before that field
    existed are reconstructed two ways: ids of the old namespaced form
    ``goal:{inv}::tool:{call}`` carry the turn in their prefix, and anything left
    is assigned to the last request that started before it. Chronology is what a
    turn is, so recovering it from time is exact wherever the ids are silent.
    """
    nodes = {n["id"]: n for n in full.get("nodes", [])}
    order = {"goal": 0, "tool_call": 1, "result": 2}

    members = [n for n in nodes.values() if n.get("kind") in order]
    members.sort(key=lambda n: (n.get("t_start") or 0.0, order[n["kind"]]))
    turn_of = _turn_resolver(list(nodes.values()))

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for node in members:
        grouped.setdefault(turn_of(node), []).append(node)

    out = []
    for turn_id, members in grouped.items():
        members.sort(key=lambda n: (n.get("t_start") or 0.0, order[n["kind"]]))
        goal = next((m for m in members if m["kind"] == "goal"), None)
        result = next((m for m in members if m["kind"] == "result"), None)
        calls = [m for m in members if m["kind"] == "tool_call"]
        started = min((m.get("t_start") or 0.0) for m in members)
        ended = max((m.get("t_end") or m.get("t_start") or 0.0) for m in members)
        out.append({
            "turn_id": turn_id,
            "prompt": (goal or {}).get("label", ""),
            "answer": (result or {}).get("output", ""),
            "status": (goal or {}).get("status", ""),
            "t_start": started,
            "t_end": ended,
            "duration": round(ended - started, 3) if ended and started else None,
            "calls": [_call_record(c) for c in calls],
        })
    out.sort(key=lambda t: t["t_start"])
    return {"turns": out, "count": len(out)}


def _call_record(c: Dict[str, Any]) -> Dict[str, Any]:
    """One tool call as a trace viewer shows it: who, what, with what, to what
    end, and how long. The same shape whether it is listed under a request or
    folded into the agent that made it, so a viewer needs one renderer."""
    return {
        "id": c["id"],
        "agent": c.get("executor_agent"),
        "tool": c.get("label"),
        "status": c.get("status"),
        "input": c.get("input"),
        "output": c.get("output"),
        "input_files": c.get("input_files") or [],
        "output_files": c.get("output_files") or [],
        "t_start": c.get("t_start"),
        "t_end": c.get("t_end"),
        "duration": (round(c["t_end"] - c["t_start"], 3)
                     if c.get("t_end") and c.get("t_start") else None),
    }


def execution_tree(full: Dict[str, Any],
                   turn: Optional[str] = None,
                   collapse_tools: bool = True) -> Dict[str, Any]:
    """The call graph of ONE user request, ready to draw left to right.

    A session's worth of requests on one canvas was the wrong unit: lanes
    collided across requests, the canvas ran to thousands of pixels, and the
    thing a reader actually wants to follow — what this prompt caused — was
    buried. One request is drawn at a time; ``turn`` picks it, and the newest
    is used when nothing is asked for. Every request in the graph is listed
    back under ``turns`` so a caller can offer the choice.

    The stored graph is not shaped for reading. Older snapshots still carry the
    seeded roster — every configured agent wired to a hub by ``has_member``,
    whether or not it was ever called — and the hub itself adds a level above
    the request that carries no information. What a reader wants is the shape
    the run actually had:

        request -> agent -> its calls -> nested agent -> its calls -> answer

    So the roster and the hub are dropped, an agent survives only if something
    invoked it, and each node is given the depth it sits at in its own request.
    Depth is measured here rather than inferred by the viewer, because an agent
    node can be shared by several requests and inference would pin it to
    whichever one reached it first.

    With ``collapse_tools`` (the default) a tool call is not a card of its own:
    it is folded into the agent that made it, under ``calls``, in the order it
    ran. Eighty tool cards on one canvas was the whole picture and none of the
    story; what a reader follows is which agents the request went through, and
    what each of them did is one click away on its card. A call whose caller is
    not in the picture keeps its own card rather than being lost.
    """
    every = full.get("nodes", [])
    every_before_scope = every
    resolve = _turn_resolver(every)

    catalogue, seen = [], set()
    for node in sorted((n for n in every if n.get("kind") == "goal"),
                       key=lambda n: n.get("t_start") or 0.0):
        key = resolve(node)
        if key in seen:
            continue
        seen.add(key)
        catalogue.append({"turn_id": key, "prompt": node.get("label") or "",
                          "t_start": node.get("t_start")})

    chosen = turn or (catalogue[-1]["turn_id"] if catalogue else None)
    if chosen is not None:
        every = _scope_to_turn(every, full.get("edges", []), resolve, chosen)

    nodes = {n["id"]: dict(n) for n in every}
    edges = [e for e in full.get("edges", [])
             if e.get("type") != "has_member"
             and e.get("src") in nodes and e.get("dst") in nodes]

    roots = [n for n in nodes.values() if n.get("kind") == "system"]
    for root in roots:                       # the hub is a fixture, not an event
        nodes.pop(root["id"], None)
    edges = [e for e in edges
             if e.get("src") in nodes and e.get("dst") in nodes]

    # Calls move onto their agent BEFORE the roster prune below, so that an
    # agent which did work in this request is visibly not roster.
    edges = _fold_calls_into_agents(nodes, edges)

    # An agent that nothing called, and that did nothing, is roster not history.
    called = {e["dst"] for e in edges}
    for node_id, node in list(nodes.items()):
        if (node.get("kind") in ("agent", "agent_call")
                and node_id not in called and not node.get("calls")):
            nodes.pop(node_id)
    edges = [e for e in edges if e["src"] in nodes and e["dst"] in nodes]

    if collapse_tools:
        edges = _fold_calls_into_agents(nodes, edges)
    _borrow_io(nodes, every_before_scope, edges)
    edges = _attach_loose_agents_to_the_request(nodes, edges)
    _keep_agents_inside_the_request(nodes)
    _number_stages(nodes)
    for node in nodes.values():
        if node.get("kind") in _AGENTS:
            node["artifacts"] = _artifacts_of(node)

    children: Dict[str, List[str]] = {}
    for edge in edges:
        children.setdefault(edge["src"], []).append(edge["dst"])

    # Breadth-first from each request; a node keeps the shallowest depth found.
    level: Dict[str, int] = {}
    goals = sorted((n for n in nodes.values() if n.get("kind") == "goal"),
                   key=lambda n: n.get("t_start") or 0.0)
    frontier = [(g["id"], 0) for g in goals]
    while frontier:
        node_id, depth = frontier.pop(0)
        if node_id in level and level[node_id] <= depth:
            continue
        level[node_id] = depth
        for child in children.get(node_id, []):
            frontier.append((child, depth + 1))

    # Anything unreachable from a request still has to be placed somewhere.
    for node_id, node in nodes.items():
        level.setdefault(node_id, 0 if node.get("kind") == "goal" else 1)

    # The answer ends its request, so it belongs to the right of everything the
    # request did rather than beside the calls that produced it.
    deepest = max(level.values(), default=0)
    for node_id, node in nodes.items():
        if node.get("kind") == "result":
            level[node_id] = deepest + 1

    for node_id, node in nodes.items():
        node["level"] = level[node_id]

    ordered = sorted(nodes.values(),
                     key=lambda n: (n.get("t_start") or 0.0, n["level"]))
    _place_in_time(ordered, edges)
    return {"run_id": full.get("run_id"), "nodes": ordered, "edges": edges,
            "turns": catalogue, "turn_id": chosen,
            "phases": _phases_of(ordered)}


_AGENTS = ("agent", "agent_call")


def _fold_calls_into_agents(nodes: Dict[str, Dict[str, Any]],
                            edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Move every tool call under the agent that made it. Mutates ``nodes``.

    The call is recorded with an edge from its caller, so the caller is read
    off the edges rather than off ``executor_agent``, which names an agent and
    not a node. An agent node is shared by every request it served and its own
    timestamps are whichever delegation wrote last, so once its calls in this
    request are known they bound its span: it started no later than its first
    call and finished no earlier than its last.
    """
    caller: Dict[str, str] = {}
    for edge in edges:
        src, dst = edge.get("src"), edge.get("dst")
        if dst not in caller and src in nodes and nodes[src].get("kind") in _AGENTS:
            caller[dst] = src

    for node_id, node in list(nodes.items()):
        if node.get("kind") != "tool_call":
            continue
        agent = nodes.get(caller.get(node_id, ""))
        if agent is None:
            continue                          # nobody to fold into: stays a card
        agent.setdefault("calls", []).append(_call_record(node))
        nodes.pop(node_id)

    for agent in nodes.values():
        calls = agent.get("calls")
        if not calls:
            continue
        calls.sort(key=lambda c: c.get("t_start") or 0.0)
        first = min((c["t_start"] for c in calls if c.get("t_start")), default=None)
        last = max((c["t_end"] for c in calls if c.get("t_end")), default=None)
        if first is not None and (agent.get("t_start") is None or agent["t_start"] > first):
            agent["t_start"] = first
        # A running agent has no end yet; giving it one would print a duration
        # for work that is still going on.
        if last is not None and agent.get("status") != "running" and (
                agent.get("t_end") is None or agent["t_end"] < last):
            agent["t_end"] = last

    return [e for e in edges if e["src"] in nodes and e["dst"] in nodes]


def _attach_loose_agents_to_the_request(
        nodes: Dict[str, Dict[str, Any]],
        edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Join an agent nothing delegated to onto the request it worked in.

    The lifecycle stages around the orchestrator — the one that seeds the
    context, the one that assembles the report at the end — are not delegated
    to by anybody: they are wired into the run itself. So no edge names them,
    and they are drawn floating beside the trace with no place in the order of
    events. Naming the agent per request does not change that: a stage nobody
    calls still has nothing pointing at it.

    They did work in this request, which is why they are here at all, so they
    hang off the request itself. The edge is marked synthetic: it says "this
    also ran", not "the request called it".
    """
    goals = sorted((n for n in nodes.values() if n.get("kind") == "goal"),
                   key=lambda n: n.get("t_start") or 0.0)
    if not goals:
        return edges

    root = goals[0]["id"]
    reached = {e["dst"] for e in edges}
    return edges + [
        {"src": root, "dst": node_id, "type": "ran_in", "synthetic": True}
        for node_id, node in nodes.items()
        if node.get("kind") in _AGENTS and node_id not in reached and node_id != root
    ]


def _keep_agents_inside_the_request(nodes: Dict[str, Dict[str, Any]]) -> None:
    """Pull an agent whose clock is from another request back into this one.

    Folding bounds an agent by its own calls, which settles anything that did
    measurable work. An agent that made none keeps whatever was written on it,
    and in a snapshot recorded before agents were named per request that is the
    last time the agent ran ANYWHERE: days past the request being drawn, which
    sorts it after everything and leaves x carrying no time at all.

    Only such an agent is moved, and only to the start of the request it is
    drawn in. An agent whose calls are here is left exactly as it is even when
    its clock looks wrong: that disagreement is a request wrongly attributed,
    and rewriting the times would hide it rather than fix it.
    """
    starts = [n.get("t_start") for n in nodes.values()
              if n.get("kind") == "goal" and n.get("t_start") is not None]
    if not starts:
        return
    opened = min(starts)
    # The far edge is read off the calls and the answer, never off an agent:
    # an agent carrying another request's clock would widen the window it is
    # meant to be caught by, and catch nothing.
    inside = [c.get("t_end") or c.get("t_start")
              for n in nodes.values() for c in (n.get("calls") or [])
              if (c.get("t_end") or c.get("t_start")) is not None]
    inside += [n.get("t_end") or n.get("t_start") for n in nodes.values()
               if n.get("kind") == "result"
               and (n.get("t_end") or n.get("t_start")) is not None]
    if not inside:
        # Nothing in the request says when it ended, so there is no window to
        # judge an agent against. The goal's own start is not one: measuring
        # against it would call every agent late and drag them all onto it.
        return
    latest = max(inside)

    for node in nodes.values():
        if node.get("kind") not in _AGENTS or node.get("calls"):
            continue
        started = node.get("t_start")
        if started is None or started < opened or started > latest:
            node["t_start"] = opened
            if node.get("t_end") is not None and node["t_end"] > latest:
                node["t_end"] = None


def _number_stages(nodes: Dict[str, Dict[str, Any]]) -> None:
    """Give each agent card its place in the request: ``stage`` is its order
    among the agents by start time, ``run`` which run of that agent it is and
    ``runs`` how many there were, so a loop reads as "3. Critic · run 2 of 4".
    """
    agents = sorted((n for n in nodes.values() if n.get("kind") in _AGENTS),
                    key=lambda n: (n.get("t_start") or 0.0, n["id"]))
    runs: Dict[str, int] = {}
    for stage, node in enumerate(agents, start=1):
        name = str(node.get("executor_agent") or node.get("label") or node["id"])
        runs[name] = runs.get(name, 0) + 1
        node["stage"], node["run"] = stage, runs[name]
    for node in agents:
        name = str(node.get("executor_agent") or node.get("label") or node["id"])
        node["runs"] = runs[name]


_LINK = re.compile(r"(?:s3://|https?://)[^\s\"'<>()\[\]]+")
_MAX_ARTIFACTS = 40


def _artifacts_of(agent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What the agent left behind: files its tools produced (the S3 references
    recorded on each call) and the files and links its report points at.
    Inputs are not artifacts, and links inside tool results are not either —
    a search result is forty links and none of them is the agent's work."""
    seen, out = set(), []

    def add(uri: str, tool: Optional[str]) -> None:
        uri = uri.rstrip(".,;:")
        if not uri or uri in seen or len(out) >= _MAX_ARTIFACTS:
            return
        seen.add(uri)
        out.append({"uri": uri, "kind": "file" if uri.startswith("s3://") else "link",
                    "tool": tool})

    for call in agent.get("calls") or []:
        for uri in call.get("output_files") or []:
            add(uri, call.get("tool"))
    for uri in agent.get("output_files") or []:
        add(uri, None)
    for uri in _LINK.findall(str(agent.get("output") or "")):
        add(uri, None)
    return out


def _borrow_io(nodes: Dict[str, Dict[str, Any]], every: List[Dict[str, Any]],
               edges: List[Dict[str, Any]]) -> None:
    """Fill an agent's task and report from where an older recorder put them.

    Snapshots written before activations existed hold the delegation's
    arguments and result on a node in the caller's request, and the callee's
    work on a second node of the same agent in a request of its own — the one
    a reader opens, which then says nothing. The delegation node is the same
    agent whose span covers this one's start; failing that, the request the
    agent served is its task and that request's answer is its report. The
    ``io_source`` field says which, so the panel can say so too.
    """
    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for node in every:
        if node.get("kind") in _AGENTS and (node.get("input") or node.get("output")):
            by_name.setdefault(str(node.get("executor_agent") or ""), []).append(node)
    parent = {e["dst"]: e["src"] for e in edges}
    answer_of = {e["src"]: nodes[e["dst"]] for e in edges
                 if e.get("type") == "produced" and nodes.get(e["dst"], {}).get("kind") == "result"}

    for node_id, node in nodes.items():
        if node.get("kind") not in _AGENTS or (node.get("input") and node.get("output")):
            continue
        started = node.get("t_start") or 0.0
        donors = [d for d in by_name.get(str(node.get("executor_agent") or ""), [])
                  if d["id"] != node_id and (d.get("t_start") or 0.0) <= started + 1
                  and (d.get("t_end") is None or d["t_end"] >= started - 1)]
        if donors:
            donor = max(donors, key=lambda d: d.get("t_start") or 0.0)
            for key in ("input", "output"):
                if not node.get(key) and donor.get(key):
                    node[key] = donor[key]
                    node["io_source"] = "delegation"
            continue
        goal = nodes.get(parent.get(node_id, ""))
        if goal is None or goal.get("kind") != "goal":
            continue
        if not node.get("input") and goal.get("label"):
            node["input"] = goal["label"]
            node["io_source"] = "request"
        answer = answer_of.get(node_id)
        if not node.get("output") and answer is not None and answer.get("output"):
            node["output"] = answer["output"]
            node["io_source"] = "request"


#: What kind of work an agent does, so a request reads as the stretches it
#: went through rather than as a row of names. The orchestrator is
#: deliberately absent: it conducts every stretch and belongs to none, so a
#: phase of its own would cut the picture into slivers.
_PHASE_OF_AGENT = {
    "ContextInitAgent": "framing",
    "PlannerAgent": "framing",
    "PlanningPipelineAgent": "framing",
    "HypothesesAgent": "framing",

    "ResearchAgent": "research",
    "MedicalAgent": "research",
    "DatasetCollectorAgent": "research",

    "TaskExecutorAgent": "experiment",
    "CoderAgent": "experiment",
    "ExperimentAgent": "experiment",
    "ExecutorSwitchAgent": "experiment",
    "FedotAgent": "experiment",
    "McpBuilderAgent": "experiment",
    "ToolPipelineAgent": "experiment",
    "ToolPreparerAgent": "experiment",

    "ResultAggregatorAgent": "report",
}

#: Shown on the band, in the language the research graph already speaks.
_PHASE_WORDS = {
    "framing": "постановка",
    "research": "поиск и данные",
    "experiment": "эксперимент",
    "report": "отчёт",
}


def _phases_of(ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The stretches a request went through, as bands along the time axis.

    Read off the agents that did the work, in the order they were placed. An
    agent with no phase of its own — the orchestrator, and anything not in the
    table — takes the phase of the next agent that has one, since a conductor
    is introducing what comes next; failing that, the one before.
    Neighbouring agents of the same phase make one band.

    Given as pixel spans rather than times because the canvas is already laid
    out: a band that agreed with the clock but not with the cards would frame
    the wrong ones.
    """
    agents = [n for n in ordered
              if n.get("kind") in _AGENTS and n.get("x") is not None]
    if not agents:
        return []
    agents.sort(key=lambda n: n["x"])

    named = [_PHASE_OF_AGENT.get(str(n.get("executor_agent") or n.get("label") or ""))
             for n in agents]
    for i in range(len(named) - 2, -1, -1):
        if named[i] is None:
            named[i] = named[i + 1]
    for i in range(1, len(named)):
        if named[i] is None:
            named[i] = named[i - 1]
    if not any(named):
        return []

    bands: List[Dict[str, Any]] = []
    for node, phase in zip(agents, named):
        if phase is None:
            continue
        right = node["x"] + (node.get("card_width") or _CARD_WIDTH)
        if bands and bands[-1]["phase"] == phase:
            bands[-1]["x1"] = max(bands[-1]["x1"], right)
            bands[-1]["agents"] += 1
        else:
            bands.append({"phase": phase, "label": _PHASE_WORDS.get(phase, phase),
                          "x0": node["x"], "x1": right, "agents": 1})
    return bands


def _scope_to_turn(every, all_edges, resolve, chosen):
    """The nodes belonging to one request, agents included.

    A goal, a tool call and an answer each happen once, so they belong to the
    request that produced them. An agent node does not: it is written once and
    reused by every request it serves. Scoping it the same way hands it to one
    request and strands the rest — their calls keep the edges that named an
    agent no longer there, the edges are dropped as dangling, and the request
    is drawn as a row of cards joined by nothing.

    So an agent joins a request when it acted in it — when an edge ties it to
    that request's own goal, call or answer — and the agents above it in the
    delegation chain come with it, since a sub-agent drawn without its caller
    hangs off the picture unreached.
    """
    by_id = {n["id"]: n for n in every}
    scoped = [n for n in every
              if n.get("kind") == "system" or resolve(n) == chosen]
    inside = {n["id"] for n in scoped}
    # Only the request's own one-off nodes may vouch for an agent; letting one
    # shared agent vouch for another would pull in the whole roster.
    anchors = {n["id"] for n in scoped if n.get("kind") not in _AGENTS}

    edges = [e for e in all_edges if e.get("type") != "has_member"]
    extra = set()
    for edge in edges:
        for near, far in ((edge.get("src"), edge.get("dst")),
                          (edge.get("dst"), edge.get("src"))):
            if near in anchors and far not in inside:
                node = by_id.get(far)
                if node is not None and node.get("kind") in _AGENTS:
                    extra.add(far)

    # Walk up the delegation chain so a nested agent keeps its caller.
    delegations = [(e["src"], e["dst"]) for e in edges
                   if e.get("type") == "delegated_to"
                   and e.get("src") in by_id and e.get("dst") in by_id]
    growing = True
    while growing:
        growing = False
        for parent, child in delegations:
            if child in extra and parent not in extra and parent not in inside:
                extra.add(parent)
                growing = True

    return scoped + [by_id[node_id] for node_id in extra]


#: A card is this wide on screen, and two of them in one lane need this much
#: clear space between their left edges or they overlap. Getting this wrong is
#: what made consecutive calls sit on top of each other: time alone decided x,
#: and a busy second put several cards inside forty pixels. Now that a request
#: is a handful of agents rather than a hundred calls, there is room for a wide
#: card and nothing left to crowd it.
_CARD_WIDTH, _CARD_GAP = 320, 40
_LANE_PITCH = _CARD_WIDTH + _CARD_GAP

#: How far the clock moves a node, and the most a single idle stretch may
#: occupy. A run waits forty minutes for a sandbox; drawn to scale that gap is
#: the whole picture and the calls either side of it are a smudge, so long
#: waits compress and the order is what survives.
_MAX_STEP, _PIXELS_PER_SECOND = 420, 8.0
_ROW_HEIGHT = 150
#: Distance between two sub-rows inside one agent's band.
_SUB_ROW_HEIGHT = 68


def _place_in_time(ordered: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> None:
    """Give every node an x from when it ran and a y from who ran it.

    Depth answers "what followed from what", which is why it stays on the node,
    but it is the wrong horizontal axis: laying calls out by depth puts thirty
    of them in one column and says nothing about the order they happened in.
    Here x advances with the clock, so the picture stretches as the run goes on
    and reading left to right is reading forward in time.

    Rows keep it legible. The request and its answer bookend row zero; every
    agent gets a row of its own in the order it first appears, and a call sits
    in the row of the agent that made it.
    """
    if not ordered:
        return

    owner = {e["dst"]: e["src"] for e in edges}

    def name_of(node: Dict[str, Any]) -> str:
        return str(node.get("executor_agent") or node.get("label")
                   or node["id"].split("::")[-1])

    def agent_of(node: Dict[str, Any]) -> Optional[str]:
        """The agent whose row this node belongs in, by walking up its callers.

        Keyed by name, not by node: the same agent called in three requests is
        three nodes, and giving each its own row would spread one participant
        across the page instead of showing it as one lane of activity.
        """
        if node.get("kind") in ("agent", "agent_call"):
            return name_of(node)
        seen, current = set(), owner.get(node["id"])
        while current and current not in seen:
            seen.add(current)
            parent = by_id.get(current)
            if parent is None:
                return None
            if parent.get("kind") in ("agent", "agent_call"):
                return name_of(parent)
            current = owner.get(current)
        return None

    by_id = {n["id"]: n for n in ordered}
    rows: Dict[str, int] = {}
    for node in ordered:                      # first appearance decides the row
        if node.get("kind") in ("goal", "result"):
            continue
        agent = agent_of(node)
        if agent is not None and agent not in rows:
            rows[agent] = len(rows) + 1

    # One pass, in time order. Each node starts at least a card's width to the
    # right of the one before it, and further when the clock says so. The floor
    # is global rather than per lane: packing lanes separately let a busy lane
    # push its cards past a quiet one, so x stopped agreeing with time. With the
    # floor applied to every step, reading left to right is reading forward in
    # time everywhere, and no two cards can overlap in any lane.
    # An agent with many calls gets a band of sub-rows rather than one long
    # line: twenty calls in a single row is a strip the reader has to scroll
    # sideways forever, and it wastes the vertical space beside it. Cards in
    # different sub-rows may sit closer horizontally, since only cards sharing
    # a sub-row can collide, so stacking also shortens the picture.
    per_agent: Dict[str, int] = {}
    for node in ordered:
        if node.get("kind") in ("goal", "result"):
            continue
        agent = agent_of(node) or ""
        per_agent[agent] = per_agent.get(agent, 0) + 1
    bands = {agent: (3 if count > 12 else 2 if count > 5 else 1)
             for agent, count in per_agent.items()}

    seen_in_lane: Dict[str, int] = {}
    previous_start, x = None, 0.0
    for node in ordered:
        started = node.get("t_start")
        is_bookend = node.get("kind") in ("goal", "result")
        agent = "" if is_bookend else (agent_of(node) or "")
        band = 1 if is_bookend else bands.get(agent, 1)

        if previous_start is not None:
            gap = (max(0.0, started - previous_start) * _PIXELS_PER_SECOND
                   if started is not None else 0.0)
            x += max(_LANE_PITCH / band, min(_MAX_STEP, gap))
        if started is not None:
            previous_start = started

        index = seen_in_lane.get(agent, 0)
        seen_in_lane[agent] = index + 1
        lane = 0 if is_bookend else rows.get(agent, 1)
        sub = 0 if is_bookend else index % band

        node["row"] = lane
        node["sub_row"] = sub
        node["x"] = round(x)
        node["y"] = round(lane * _ROW_HEIGHT + sub * _SUB_ROW_HEIGHT)
        node["card_width"] = _CARD_WIDTH

    # The answer closes the request, so it sits past everything the request did.
    # Its own timestamp cannot be trusted for this: older snapshots recorded it
    # without one, which left it drawn in the middle of the work it summarises.
    answers = [n for n in ordered if n.get("kind") == "result"]
    if answers:
        rightmost = max(n["x"] for n in ordered if n.get("kind") != "result")
        for offset, answer in enumerate(answers):
            answer["x"] = round(rightmost + _LANE_PITCH * (offset + 1))
