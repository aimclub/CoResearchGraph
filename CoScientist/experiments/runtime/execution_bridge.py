"""The experiment plan, recorded in the execution (call) graph.

The research graph is the science record: it gets the plan's *methods* once a
human approves them (``graph_bridge.publish_plan_to_graph``). The execution
graph is the record of the run itself — who was called, what they did, in what
order — and until now a plan left no trace in it at all. Opening the call graph
after a run showed the planner agent as a card with a report on it and no sign
of the plan that report was about, and a plan the human sent back for revision
left no trace anywhere.

So a plan becomes a ``decision`` node of its own, hanging off the activation of
the agent that proposed it, carrying the structured plan (``plan_view``) as its
arguments. Its status follows the review: ``running`` while the human is being
asked, then ``success`` when they approve and ``failed`` when the plan is sent
back or the review is paused. A revision is a node of its own, so a plan that
took three rounds reads as three cards in the order they happened.

Best-effort by contract, like every other graph write in the module: a failure
here is logged and swallowed, never raised into a review.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from CoScientist.experiments.plan_view import plan_headline
from CoScientist.experiments.runtime.shared import audit

logger = logging.getLogger(__name__)

#: Statuses a recorded plan can end in, and the node status each maps to.
_TERMINAL_STATUS = {
    "approved": "success",
    "revision_requested": "failed",
    "paused": "failed",
    "rejected": "failed",
}
_AGENT_KINDS = ("agent", "agent_call")


def _enabled() -> bool:
    """The execution graph's own switch — the same one the plugin obeys."""
    try:
        from CoScientist.graph.plugin import _enabled as plugin_enabled

        return bool(plugin_enabled())
    except Exception:  # noqa: BLE001 — a missing switch must not break a review
        return False


def _graph(ctx: Any) -> Any:
    from CoScientist.graph.memory import get_knowledge_graph

    return get_knowledge_graph(ctx)


def _latest_activation(graph: Any, agent_name: str) -> dict[str, Any] | None:
    """The agent's current activation node, read off the graph.

    The plugin mints one activation per run of an agent and keeps the mapping
    in its own memory; reading the graph instead means this module needs no
    handle on the plugin and works the same whether the plan is proposed by the
    planner's first run or its fourth.
    """
    candidates = [
        node for node in (graph.full().get("nodes") or [])
        if isinstance(node, dict)
        and node.get("kind") in _AGENT_KINDS
        and node.get("executor_agent") == agent_name
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda n: n.get("t_start") or 0.0)


def _node_id(view: dict[str, Any], parent_id: str) -> str:
    return f"plan:{view.get('plan_id') or 'plan'}@r{view.get('revision') or 1}:{parent_id}"


def record_plan_proposed(
    ctx: Any,
    agent_name: str,
    view: dict[str, Any],
) -> str | None:
    """Add the proposed plan to the call graph; returns the node id it took.

    Called before the human is asked, so the plan is in the record even if the
    review times out, the run is stopped, or the browser was never open.
    """
    try:
        if not _enabled() or not isinstance(view, dict):
            return None
        graph = _graph(ctx)
        parent = _latest_activation(graph, agent_name)
        if parent is None:
            # No activation means the plugin recorded nothing for this run
            # (graph off mid-run, or a direct call outside a Runner). A plan
            # node with nothing to hang off would be drawn floating.
            return None
        node_id = _node_id(view, str(parent.get("id")))
        graph.add_node(
            id=node_id,
            kind="decision",
            turn_id=parent.get("turn_id"),
            label=plan_headline(view),
            executor_agent=agent_name,
            status="running",
            input=view,
            output=plan_headline(view),
            t_start=time.time(),
        )
        graph.add_edge(str(parent.get("id")), node_id, type="caused_by")
        audit(
            logger,
            f"EXPERIMENT_CALL_GRAPH_PLAN_RECORDED node={node_id} "
            f"plan_id={view.get('plan_id')} revision={view.get('revision')} "
            f"tasks={view.get('task_count')}",
        )
        return node_id
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        audit(logger, f"EXPERIMENT_CALL_GRAPH_PLAN_RECORD_FAILED error={exc}",
              level=logging.WARNING)
        return None


def close_plan_record(
    ctx: Any,
    node_id: str | None,
    outcome: str,
    *,
    reason: str | None = None,
) -> None:
    """Stamp the human's answer onto the recorded plan.

    ``outcome`` is one of ``approved`` / ``revision_requested`` / ``paused`` /
    ``rejected``; anything else leaves the node running, since a status nobody
    recognises is worse than an open one.
    """
    try:
        if not node_id or not _enabled():
            return
        status = _TERMINAL_STATUS.get(outcome)
        if status is None:
            return
        _graph(ctx).set_status(
            node_id,
            status=status,
            t_end=time.time(),
            verdict=outcome,
            output=reason or outcome,
        )
        audit(logger,
              f"EXPERIMENT_CALL_GRAPH_PLAN_CLOSED node={node_id} outcome={outcome}")
    except Exception as exc:  # noqa: BLE001 — best-effort by contract
        audit(logger, f"EXPERIMENT_CALL_GRAPH_PLAN_CLOSE_FAILED error={exc}",
              level=logging.WARNING)


__all__ = ["close_plan_record", "record_plan_proposed"]
