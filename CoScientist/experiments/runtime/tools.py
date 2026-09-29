"""ADK control tools exposing the deterministic experiment state machine."""
from __future__ import annotations

from typing import Any

from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.base_toolset import BaseToolset
from google.adk.tools.tool_context import ToolContext

from . import state_machine
from .state_machine import ExperimentRuntimeError


def _call(operation, *args, **kwargs) -> dict[str, Any]:
    try:
        return operation(*args, **kwargs)
    except ExperimentRuntimeError as exc:
        return exc.as_dict()
    except Exception as exc:
        return {"status": "error", "error_code": "validation_error", "message": str(exc)}


def _route_agents(tool_context: ToolContext) -> frozenset[str] | None:
    """Names of the agents attached as AgentTools to the executor calling us.

    That tree is what the executor can actually call; the YAML may have
    changed under a running session. None when the tree cannot be read, and
    start_task then goes by the YAML alone.
    """
    agent = getattr(getattr(tool_context, "_invocation_context", None), "agent", None)
    return state_machine.agent_tool_names(agent)


def _mirror_result_to_graph(tool_context: ToolContext, task_id: str, stored: dict[str, Any]) -> None:
    """Best-effort: mirror a recorded TaskResult into the research graph
    (Evidence/GeneratedData + VerificationMethod status). Never raises."""
    try:
        from CoScientist.experiments.runtime.graph_bridge import publish_result_to_graph
        from CoScientist.graph.research.store import get_research_graph

        task_result = stored.get("task_result")
        if not isinstance(task_result, dict):
            return
        publish_result_to_graph(
            get_research_graph(tool_context), tool_context.state, task_id, task_result,
        )
    except Exception:  # noqa: BLE001 — the recorded result always wins
        pass


def _mirror_task_state_to_graph(tool_context: ToolContext, task_id: str) -> None:
    """Best-effort: move the task's card to wherever the runtime just put it.

    Called after every control tool that starts, retries, re-routes or skips a
    task — the moments the graph never used to hear about. Reads the runtime's
    own status rather than being handed one, so it cannot disagree with it.
    Never raises.
    """
    try:
        from CoScientist.experiments.runtime.graph_bridge import publish_task_state_to_graph
        from CoScientist.graph.research.store import get_research_graph

        publish_task_state_to_graph(
            get_research_graph(tool_context), tool_context.state, task_id,
        )
    except Exception:  # noqa: BLE001 — the run always wins
        pass


class ExperimentControlToolset(BaseToolset):
    """The only write surface for experiment task/attempt lifecycle."""

    async def get_tools(self, readonly_context: ReadonlyContext | None = None) -> list[BaseTool]:
        return [
            FunctionTool(self.get_experiment_plan),
            FunctionTool(self.start_task),
            FunctionTool(self.record_result),
            FunctionTool(self.retry_task),
            FunctionTool(self.fallback_task),
            FunctionTool(self.skip_task),
            FunctionTool(self.amend_task),
        ]

    async def close(self) -> None:
        return None

    def get_experiment_plan(self, tool_context: ToolContext) -> dict[str, Any]:
        """Return the approved plan and deterministic runtime state."""
        return _call(state_machine.get_experiment_plan, tool_context.state)

    def start_task(self, task_id: str, tool_context: ToolContext) -> dict[str, Any]:
        """Start one ready task and create a fresh attempt/route envelope."""
        started = _call(
            state_machine.start_task, tool_context.state, task_id,
            route_agents=_route_agents(tool_context),
        )
        _mirror_task_state_to_graph(tool_context, task_id)
        return started

    def record_result(
        self,
        task_id: str,
        attempt_id: str,
        result: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Validate/store TaskResult; downgrade incomplete success to retryable failure.

        Durable family evidence (S3 key / workspace file / mcp_url / structured
        MCP outputs) is accepted even when planner artifact or criterion names
        differ — name mismatch alone must not fallback to Coder.
        """
        route_agents = _route_agents(tool_context)
        try:
            stored = state_machine.record_result(
                tool_context.state, task_id, attempt_id, result, route_agents=route_agents,
            )
            _mirror_result_to_graph(tool_context, task_id, stored)
            return stored
        except ExperimentRuntimeError as exc:
            if exc.code != "result_incomplete" or result.get("status") not in {"success", "partial"}:
                return exc.as_dict()
            downgraded = {
                **result,
                "status": "failure",
                "error_code": "result_incomplete",
                "error_message": str(exc),
                "retryable": True,
            }
            stored = state_machine.record_result(
                tool_context.state, task_id, attempt_id, downgraded, route_agents=route_agents,
            )
            stored.update({"downgraded_from": result.get("status"), "downgrade_reason": "result_incomplete"})
            _mirror_result_to_graph(tool_context, task_id, stored)
            return stored
        except Exception as exc:
            return {"status": "error", "error_code": "validation_error", "message": str(exc)}

    def retry_task(self, task_id: str, tool_context: ToolContext) -> dict[str, Any]:
        """Permit a retryable failure to create a new attempt on the same route."""
        retried = _call(state_machine.retry_task, tool_context.state, task_id)
        _mirror_task_state_to_graph(tool_context, task_id)
        return retried

    def fallback_task(self, task_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
        """Move a failed task to the next route in its finite fallback chain."""
        moved = _call(
            state_machine.fallback_task, tool_context.state, task_id, reason,
            route_agents=_route_agents(tool_context),
        )
        _mirror_task_state_to_graph(tool_context, task_id)
        return moved

    def skip_task(self, task_id: str, reason: str, tool_context: ToolContext) -> dict[str, Any]:
        """Skip an optional task and create its terminal skipped TaskResult."""
        skipped = _call(state_machine.skip_task, tool_context.state, task_id, reason)
        _mirror_task_state_to_graph(tool_context, task_id)
        return skipped

    def amend_task(
        self,
        task_id: str,
        patch: dict[str, Any],
        reason: str,
        tool_context: ToolContext,
    ) -> dict[str, Any]:
        """Amend an unstarted task; material criteria changes return to review."""
        return _call(state_machine.amend_task, tool_context.state, task_id, patch, reason)


experiment_control_toolset = ExperimentControlToolset()


__all__ = ["ExperimentControlToolset", "experiment_control_toolset"]
