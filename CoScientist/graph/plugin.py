"""ADK plugin that grows the in-process knowledge graph from agent activity.

Attached to the in-process Runner (web / cli) next to the event logger. Unlike
``GraphEmitterPlugin`` (which POSTs to the A2A graph service over HTTP), this
writes straight into the graph resolved for the public user/session scope, so
every in-process agent participating in that session can read it synchronously.

Shape of one request — a tree rooted at the request, one node per agent
*activation* (a run of the agent), so a loop is a row of runs and every run
carries its own task, its report and its tool calls:

    goal (the user query)
      ├─ 1. OrchestratorAgent                     (a stage: hangs off the request)
      │     ├─ tool_call: retrieve_tools           (a tool the orchestrator ran)
      │     ├─ 2. PlannerAgent                     (a delegation: task + report)
      │     │     └─ tool_call: create_plan
      │     ├─ 3. TaskExecutorAgent   run 1 of 2
      │     │     └─ 4. CoderAgent    run 1 of 2   (nested delegation)
      │     │           └─ tool_call: execute_bash
      │     ├─ 5. TaskExecutorAgent   run 2 of 2   (the same agent, called again)
      │     │     └─ 6. CoderAgent    run 2 of 2
      │     └─ ...
      ├─ 7. ResultAggregatorAgent                 (a pipeline stage after the root)
      └─ result                                   (what the top level last said)

Sequential/parallel children hang under their composite's activation, each
pass of a loop opening a new one. Node ids are ``agent:{name}@{request}`` and
``agent:{name}@{request}#2`` from the second run on.

Node labels: tool_call shows the TOOL name, agent_call shows the AGENT name (who
called it is clear from the parent). Best-effort: a graph failure never breaks a
run. Toggle with LOG_AGENT_EVENTS=0 (shared with the event logger).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.graph.memory import ROOT_ID, get_knowledge_graph
from CoScientist.graph.session_scope import SessionKey, session_key
from CoScientist.utils.s3_refs import find_s3_uris

_agent_names_cache: Optional[set] = None
_composite_parents_cache: Optional[dict] = None

def _agent_names() -> set:
    global _agent_names_cache
    if _agent_names_cache is None:
        try:
            from CoScientist.assembly.schema import get_config
            _agent_names_cache = get_config().delegatable_names()
        except Exception:  # noqa: BLE001
            _agent_names_cache = set()
    return _agent_names_cache


def _composite_parents() -> dict:
    """child agent -> parent composite (from sequential/parallel `children`)."""
    global _composite_parents_cache
    if _composite_parents_cache is None:
        m: dict = {}
        try:
            from CoScientist.assembly.schema import get_config
            for name, a in get_config().agents.items():
                for ch in (getattr(a, "children", None) or []):
                    m[ch] = name
        except Exception:  # noqa: BLE001
            pass
        _composite_parents_cache = m
    return _composite_parents_cache


_system_root_cache: Optional[str] = None


def _system_root() -> str:
    """The system's true root agent (root: true in system.yaml). Only its
    top-level invocation opens a goal — sub-agents run in their own ADK
    invocations and must NOT each spawn a goal."""
    global _system_root_cache
    if _system_root_cache is None:
        try:
            from CoScientist.assembly.schema import get_config
            cfg = get_config()
            _system_root_cache = next(
                (n for n, a in cfg.agents.items() if getattr(a, "root", False)),
                "OrchestratorAgent",
            )
        except Exception:  # noqa: BLE001
            _system_root_cache = "OrchestratorAgent"
    return _system_root_cache


def _enabled() -> bool:
    # The web UI switch wins: with the knowledge graph off nothing is recorded,
    # matching the `graph` reader toolset dropping out of every agent.
    try:
        from CoScientist.config import get_settings
        if not get_settings().web.knowledge_graph_enabled:
            return False
    except Exception:  # noqa: BLE001 — never let config break event recording
        pass
    value = os.getenv("LOG_AGENT_EVENTS") or os.getenv("A2A_LOG_EVENTS") or "1"
    return value not in ("0", "false", "False")


#: Room for an input and for an output on a node. Generous rather than
#: unbounded: the whole graph is rewritten to disk on every single write, so a
#: half-megabyte sandbox log on one node is paid for again on every node that
#: follows it. What is cut says so, and says how much was cut.
_INPUT_LIMIT, _OUTPUT_LIMIT = 4000, 20000


def _readable(value: Any) -> str:
    """A tool's arguments or result as prose, not as a JSON dump.

    The panel used to show `{"query": "aspirin mechanism", "max_results": 5}`,
    which is the wire format and not something anyone wants to read. A mapping
    becomes one `key: value` line each, a list becomes one item per line, and a
    string is already what it should be. Nested structures fall back to JSON,
    indented, because inventing prose for them would hide their shape.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list, tuple)):
                rendered = json.dumps(item, ensure_ascii=False, indent=2,
                                      default=str)
                lines.append(f"{key}:\n" + "\n".join(
                    "  " + line for line in rendered.splitlines()))
            else:
                lines.append(f"{key}: {item}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        return "\n".join(_readable(item) for item in value)
    return json.dumps(value, ensure_ascii=False, default=str)


def _short(value: Any, limit: int = _INPUT_LIMIT) -> str:
    text = _readable(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more characters]"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    parts = getattr(content, "parts", None) or []
    return "\n".join(p.text for p in parts if getattr(p, "text", None)).strip()


def _is_error(result: Any) -> bool:
    if isinstance(result, dict):
        if result.get("status") in ("error", "failed", "timeout"):
            return True
        if result.get("error"):
            return True
        # A refused transactional write answers {"ok": false, "errors": [...]}
        # and carries neither `status` nor `error`, so a commit that saved
        # NOTHING was drawn in the log as a successful call. `is False` and not
        # `not ok`: a tool that simply has no `ok` key is not a failure.
        if result.get("ok") is False:
            return True
    return False


@dataclass
class _RunState:
    """Bookkeeping for one ADK invocation (a run of the root, or of a callee)."""
    root_agent_name: Optional[str] = None
    goal_id: str = "goal:pending"
    goal_text: str = ""
    agent_node: dict = field(default_factory=dict)   # agent name -> its activation here
    node_by_fcid: dict = field(default_factory=dict)


@dataclass
class _SessionState:
    """What one public session knows across its invocations.

    A delegated agent runs in an ADK invocation of its own, so anything that has
    to be seen from both the caller's run and the callee's — the request that
    is open, the activation a delegation minted for the callee, which
    activation of each agent is the current one — lives here, not in a run.
    """
    turn: Optional[str] = None
    goal_id: Optional[str] = None
    activations: dict = field(default_factory=dict)  # (turn, agent) -> how many so far
    current: dict = field(default_factory=dict)      # agent -> its latest activation
    pending: dict = field(default_factory=dict)      # agent -> activation a delegation minted
    stages: set = field(default_factory=set)         # activations hanging off the goal


class GraphMemoryPlugin(BasePlugin):
    """Record agent activity into the graph selected by the ADK session.

    Every run of an agent is a node of its own — an *activation* — so a request
    that went Planner → Critic → Planner → Critic is four cards in the order
    they ran, not two cards with everything merged onto them. The activation
    carries what the agent was asked (the delegation's arguments), what it
    answered (its final response: the report), and the tool calls it made.
    """

    def __init__(self, name: str = "graph_memory") -> None:
        super().__init__(name=name)
        self._runs: dict[tuple[SessionKey, str], _RunState] = {}
        self._sessions: dict[SessionKey, _SessionState] = {}

    @staticmethod
    def _ctx_agent(invocation_context) -> Optional[str]:
        agent = getattr(invocation_context, "agent", None)
        return getattr(agent, "name", None)

    @staticmethod
    def _invocation_id(context) -> str:
        invocation = getattr(context, "_invocation_context", None)
        return str(
            getattr(context, "invocation_id", None)
            or getattr(invocation, "invocation_id", None)
            or "pending"
        )

    def _run_key(self, context) -> tuple[SessionKey, str]:
        return session_key(context), self._invocation_id(context)

    def _session(self, context) -> _SessionState:
        return self._sessions.setdefault(session_key(context), _SessionState())

    def _turn(self, context) -> str:
        """The request being served, not the invocation that happens to be running."""
        return self._session(context).turn or self._invocation_id(context)

    def _state(self, context) -> _RunState:
        return self._runs.setdefault(self._run_key(context), _RunState())

    @staticmethod
    def _root_name(state: _RunState) -> str:
        return state.root_agent_name or "OrchestratorAgent"

    @staticmethod
    def _opens_a_goal(agent) -> bool:
        """Is this the agent whose invocation is the user's request?

        The Runner's root is the system root itself, or the synthesized
        pipeline wrapper around it (pre-stages → root → post-stages). Either
        one starting on a user message is the request; anything else starting
        on one is a delegated agent running in an invocation of its own.
        """
        name = getattr(agent, "name", None)
        if name == _system_root():
            return True
        stack = list(getattr(agent, "sub_agents", None) or [])
        seen = set()
        while stack:
            child = stack.pop()
            child_name = getattr(child, "name", None)
            if child_name == _system_root():
                return True
            if child_name in seen:
                continue
            seen.add(child_name)
            stack.extend(getattr(child, "sub_agents", None) or [])
        return False

    def _mint(self, sess: _SessionState, agent: str) -> str:
        """A fresh activation id: the agent, the request, and which run this is."""
        turn = sess.turn or "untagged"
        count = sess.activations.get((turn, agent), 0) + 1
        sess.activations[(turn, agent)] = count
        return f"agent:{agent}@{turn}" + (f"#{count}" if count > 1 else "")

    def _agent_node_for(
        self,
        graph,
        sess: _SessionState,
        state: _RunState,
        agent: str,
        *,
        composite_parent: Optional[str] = None,
        _seen: Optional[set] = None,
    ) -> str:
        """The activation of ``agent`` in this invocation, created on first sight.

        A delegation mints the callee's activation before the callee runs, so
        the callee's own invocation binds to that node rather than opening
        another. Otherwise the agent is running as a child of a composite (or as
        a pipeline stage) and a new activation is opened under its parent's
        current activation — or under the request itself when it has no parent
        in the picture, which is what a stage is.
        """
        nid = state.agent_node.get(agent)
        if nid:
            return nid
        nid = sess.pending.pop(agent, None)
        if nid is None:
            _seen = _seen or set()
            _seen.add(agent)
            nid = self._mint(sess, agent)
            try:
                graph.add_node(
                    id=nid, kind="agent", turn_id=sess.turn, label=agent,
                    executor_agent=agent, status="running", t_start=time.time(),
                )
                parent_agent = composite_parent or _composite_parents().get(agent)
                parent_id = None
                if parent_agent and parent_agent not in _seen:
                    parent_id = sess.current.get(parent_agent) or self._agent_node_for(
                        graph, sess, state, parent_agent, _seen=_seen)
                if parent_id:
                    graph.add_edge(parent_id, nid, type="delegated_to")
                elif sess.goal_id:
                    graph.add_edge(sess.goal_id, nid, type="caused_by")
                    sess.stages.add(nid)
            except Exception:  # noqa: BLE001
                pass
        state.agent_node[agent] = nid
        sess.current[agent] = nid
        return nid

    async def on_user_message_callback(self, *, invocation_context, user_message) -> Optional[Any]:
        if not _enabled():
            return None
        # Only the TRUE system root (or the pipeline wrapped around it) opens a
        # goal. Sub-agents run in their own ADK invocations and would otherwise
        # each spawn a separate goal + a floating duplicate of themselves.
        if not self._opens_a_goal(getattr(invocation_context, "agent", None)):
            return None
        name = _system_root()
        state = self._state(invocation_context)
        sess = self._session(invocation_context)
        state.root_agent_name = name
        inv = getattr(invocation_context, "invocation_id", "x")
        state.goal_id = f"goal:{inv}"
        state.goal_text = _content_text(user_message)
        # Everything recorded until the next prompt belongs to this one.
        sess.turn = inv
        sess.goal_id = state.goal_id
        sess.current = {}
        sess.pending = {}
        sess.stages = set()
        state.agent_node = {}
        state.node_by_fcid = {}
        try:
            graph = get_knowledge_graph(invocation_context)
            graph.add_node(
                id=state.goal_id, kind="goal", turn_id=inv, label=_short(_content_text(user_message), 200),
                status="running", parent_ids=[ROOT_ID], t_start=time.time(),
            )
            graph.add_edge(ROOT_ID, state.goal_id, type="caused_by")
        except Exception:  # noqa: BLE001
            pass
        return None

    async def before_agent_callback(self, *, agent, callback_context) -> Optional[Any]:
        """An agent starts: open its activation, under whoever runs it.

        Fires for every agent run — composites, their children, each pass of a
        loop, and delegated callees inside their own invocation — which is what
        makes the stages of a request visible in the order they happened.
        """
        if not _enabled():
            return None
        name = getattr(agent, "name", None)
        if not name:
            return None
        parent = getattr(agent, "parent_agent", None)
        parent_name = getattr(parent, "name", None)
        # The synthesized pipeline wrapper is a fixture, not a stage: it has no
        # parent, is not the system root, and only exists to hold the stages.
        if parent is None and name != _system_root() and getattr(agent, "sub_agents", None) \
                and self._opens_a_goal(agent):
            return None
        try:
            graph = get_knowledge_graph(callback_context)
            sess = self._session(callback_context)
            state = self._state(callback_context)
            composite = parent_name if parent_name in sess.current else None
            self._agent_node_for(graph, sess, state, name, composite_parent=composite)
        except Exception:  # noqa: BLE001
            pass
        return None

    async def after_agent_callback(self, *, agent, callback_context) -> Optional[Any]:
        """An agent finished normally: close its activation.

        A delegation's result, or a failure, is written afterwards by the tool
        callbacks of the caller and overrides this; an agent that raised never
        gets here and is closed as interrupted when the run ends.
        """
        if not _enabled():
            return None
        try:
            state = self._state(callback_context)
            nid = state.agent_node.get(getattr(agent, "name", None))
            if nid:
                graph = get_knowledge_graph(callback_context)
                graph.set_status(nid, status="success", t_end=time.time())
        except Exception:  # noqa: BLE001
            pass
        return None

    async def after_run_callback(self, *, invocation_context) -> None:
        """Finalize an interrupted tree and release transient bookkeeping.

        Tool and event callbacks run after ``on_user_message_callback`` and
        still need the goal id, root agent, and function-call mapping created
        there.  ADK invokes this hook after the run finishes, including runs
        that do not produce a normal final response, so it is the appropriate
        lifecycle boundary for cleanup.
        """
        state = self._runs.pop(self._run_key(invocation_context), None)
        if state is None:
            return
        try:
            graph = get_knowledge_graph(invocation_context)
            nodes = {node["id"]: node for node in graph.full().get("nodes", [])}
            # A sub-agent invocation never ran on_user_message_callback, so its
            # goal_id is still the sentinel and no goal node exists. Returning on
            # that basis left every delegated agent and its tools at "running"
            # for good; only a goal that exists and is already closed means there
            # is nothing left to finalise.
            goal = nodes.get(state.goal_id)
            if goal is not None and goal.get("status") != "running":
                return
            now = time.time()
            active_ids = {
                state.goal_id,
                *state.agent_node.values(),
                *state.node_by_fcid.values(),
            }
            for node_id in active_ids:
                if nodes.get(node_id, {}).get("status") == "running":
                    graph.set_status(
                        node_id,
                        status="interrupted",
                        output="Run ended before a final response.",
                        t_end=now,
                    )
        except Exception:  # noqa: BLE001 - cleanup must never mask run errors
            pass

    async def close(self) -> None:
        """Release per-run bookkeeping before Runner shutdown."""
        self._runs.clear()
        self._sessions.clear()

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        if not _enabled():
            return None
        try:
            graph = get_knowledge_graph(tool_context)
            sess = self._session(tool_context)
            state = self._state(tool_context)
            agent = (
                getattr(tool_context, "agent_name", None)
                or self._root_name(state)
            )
            parent = self._agent_node_for(graph, sess, state, agent)
            fcid = getattr(tool_context, "function_call_id", None) or f"{tool.name}:{time.time()}"
            if tool.name in _agent_names():
                # Delegation: a fresh activation for the callee, opened here so
                # the arguments are on it before the callee's own invocation
                # begins; that invocation binds to it (see _agent_node_for).
                nid = self._mint(sess, tool.name)
                graph.add_node(
                    id=nid, kind="agent", turn_id=sess.turn, label=tool.name,
                    executor_agent=tool.name, status="running",
                    input=_short(tool_args, 1000), t_start=time.time(),
                )
                graph.add_edge(parent, nid, type="delegated_to")
                sess.pending[tool.name] = nid
                sess.current[tool.name] = nid
            else:
                nid = f"tool:{fcid}"
                graph.add_node(
                    id=nid, kind="tool_call", turn_id=sess.turn,
                    label=tool.name, executor_agent=agent, status="running",
                    parent_ids=[parent], input=_short(tool_args),
                    input_files=find_s3_uris(tool_args), t_start=time.time(),
                )
                graph.add_edge(parent, nid, type="caused_by")
            state.node_by_fcid[fcid] = nid
        except Exception:  # noqa: BLE001
            pass
        return None

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        if not _enabled():
            return None
        try:
            graph = get_knowledge_graph(tool_context)
            sess = self._session(tool_context)
            state = self._state(tool_context)
            fcid = getattr(tool_context, "function_call_id", None)
            nid = state.node_by_fcid.get(fcid)
            if nid is None and tool.name in _agent_names():
                # A delegation whose call id ADK did not supply: the callee's
                # latest activation is the one that just returned.
                nid = sess.current.get(tool.name)
            if nid:
                graph.set_status(
                    nid, status="failed" if _is_error(result) else "success",
                    output=_short(result, _OUTPUT_LIMIT),
                    output_files=find_s3_uris(result),
                    t_end=time.time(),
                )
                # The callee never ran, or ran without binding: nothing is
                # waiting for that activation any more.
                sess.pending.pop(tool.name, None)
        except Exception:  # noqa: BLE001
            pass
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error) -> Optional[dict]:
        if not _enabled():
            return None
        try:
            graph = get_knowledge_graph(tool_context)
            sess = self._session(tool_context)
            state = self._state(tool_context)
            nid = state.node_by_fcid.get(
                getattr(tool_context, "function_call_id", None)
            )
            if nid:
                graph.set_status(
                    nid, status="failed", output=_short(str(error), _OUTPUT_LIMIT), t_end=time.time(),
                )
                sess.pending.pop(tool.name, None)
        except Exception:  # noqa: BLE001
            pass
        return None

    async def on_event_callback(self, *, invocation_context, event) -> Optional[Any]:
        if not _enabled():
            return None
        state = self._state(invocation_context)
        if not (getattr(event, "is_final_response", None) and event.is_final_response()):
            return None
        text = _content_text(getattr(event, "content", None))
        if not text:
            return None

        # The event names who spoke. The invocation context names the root of
        # the run instead, which inside the pipeline wrapper is the wrapper.
        speaker = getattr(event, "author", None) or self._ctx_agent(invocation_context)
        sess = self._session(invocation_context)
        try:
            graph = get_knowledge_graph(invocation_context)
        except Exception:  # noqa: BLE001
            return None
        # What an agent finally says is its report: it goes on its activation
        # whether it was delegated (a tool-style call is closed out again by
        # after_tool_callback with the same text) or ran as a stage.
        try:
            nid = state.agent_node.get(speaker) or sess.current.get(speaker) \
                or self._agent_node_for(graph, sess, state, speaker)
            graph.set_status(nid, status="success", output=_short(text, _OUTPUT_LIMIT),
                             t_end=time.time())
        except Exception:  # noqa: BLE001
            nid = None
        if state.root_agent_name is None:
            return None                      # a delegated run: no answer to record
        # The answer is what the request's top level last said: the root's
        # final response, or a later stage's (the report synthesized after it).
        if not (speaker == state.root_agent_name or nid in sess.stages):
            return None
        try:
            inv = getattr(invocation_context, "invocation_id", "x")
            rid = f"result:{inv}"
            now = time.time()
            graph.add_node(
                id=rid, kind="result", turn_id=inv, label=_short(text, 200),
                executor_agent=speaker, status="success",
                # The run's answer is the one thing a reader came for, and it
                # was the most tightly cut of anything recorded — 600
                # characters, where a tool result gets twenty thousand. A
                # report lost its findings a paragraph in. The label stays
                # short: that one is a card, not the text.
                parent_ids=[nid] if nid else [], output=_short(text, _OUTPUT_LIMIT),
                t_start=now, t_end=now,
            )
            if nid:
                graph.add_edge(nid, rid, type="produced")
            graph.set_status(state.goal_id, status="success", t_end=now)
        except Exception:  # noqa: BLE001
            pass
        return None
