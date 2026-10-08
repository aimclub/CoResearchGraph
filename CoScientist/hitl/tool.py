"""HITL Toolset — tools that agents call to request human input."""

import os
from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.config import get_settings
from CoScientist.hitl.models import HITLRequest, HITLAction
from CoScientist.hitl.handler import AbstractHITLHandler, ConsoleHITLHandler
from CoScientist.graph.session_scope import session_key

settings = get_settings()

def a2a_mode() -> bool:
    """True when this process serves agents over A2A (set by a2a/serve.py).

    Over A2A there is no console/websocket to the human, so the blocking tools
    below would hang the server; the A2A-native long-running variants are used
    instead (see CoScientist/hitl/a2a_tools.py).
    """
    return os.getenv("COSCIENTIST_A2A_MODE", "") not in ("", "0", "false", "False")


def get_hitl_tools(a2a_root: Optional[bool] = None) -> list:
    """HITL tools for an agent, picking the transport that actually reaches a human.

    - in-process / web: the handler-driven (blocking) tools — unchanged.
    - A2A **root** (the agent the client talks to): the native long-running tools,
      which pause the run and put the task into `input-required` for the caller.
    - A2A **non-root** (reached through the orchestrator's AgentTool): a pause
      cannot reach the caller. AgentTool runs the sub-agent and returns its final
      text; a paused sub-agent produces none, so the parent gets an EMPTY result
      and carries on — the human review silently vanishes. Keep the handler path
      there; the headless guard in ConsoleHITLHandler answers explicitly instead
      of hanging.
    """
    if a2a_mode() and a2a_root:
        from CoScientist.hitl.a2a_tools import get_a2a_hitl_tools
        return get_a2a_hitl_tools()
    return [
        FunctionTool(hitl_toolset.request_approval),
        FunctionTool(hitl_toolset.request_selection)
    ]

class HITLToolset(BaseToolset):
    """Toolset providing HITL tools to agents.

    Agents call these tools when they need human confirmation,
    selection, or input before proceeding.
    """

    def __init__(self, handler: AbstractHITLHandler, prefix: str = "hitl_"):
        self._handler = handler
        self.tool_name_prefix = prefix

    async def get_tools(
        self, readonly_context: Optional[ReadonlyContext] = None
    ) -> List[BaseTool]:
        return [
            FunctionTool(self.request_approval),
            FunctionTool(self.request_selection)
        ]

    async def close(self) -> None:
        pass

    async def request_approval(
        self,
        agent_name: str,
        message: str,
        tool_context: ToolContext,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request human approval for an action.

        Use this tool when you need the user to confirm or reject
        a proposed action before proceeding.

        Args:
            agent_name: Name of the agent requesting approval.
            message: Description of what needs approval.
            context: Additional context for the human.

        Returns:
            Dictionary with 'approved' (bool) and optional 'feedback' (str).
        """
        if context is None:
            request_context = {}
        elif isinstance(context, dict):
            request_context = dict(context)
        else:
            # LLM may pass a string or other non-dict value
            request_context = {"details": context}
        user_id, session_id = session_key(tool_context)
        request_context["_session"] = {
            "user_id": user_id,
            "session_id": session_id,
        }
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.APPROVE,
            message=f"Agent '{agent_name}' requests approval for the following action: {message}",
            context=request_context,
            invoked_via="tool"
        )
        response = await self._handler.handle_request(request)
        return {
            "approved": response.approved,
            "feedback": response.instructions or response.free_input or "No feedback provided.",
        }

    async def request_selection(
        self,
        agent_name: str,
        message: str,
        options: List[str],
        tool_context: ToolContext,
    ) -> Dict[str, Any]:
        """Ask the human to select from a list of options.

        Use this tool when you have generated multiple proposals
        (e.g. hypotheses, plans) and need the user to choose the best one.

        Args:
            agent_name: Name of the agent requesting selection.
            message: Explanation of what to select and why.
            options: List of options for the human to choose from.

        Returns:
            Dictionary with 'selected' (str) and 'approved' (bool).
        """
        user_id, session_id = session_key(tool_context)
        request = HITLRequest(
            agent_name=agent_name,
            action_type=HITLAction.SELECT,
            message=message,
            options=options,
            context={
                "_session": {
                    "user_id": user_id,
                    "session_id": session_id,
                }
            },
            invoked_via="tool"
        )
        response = await self._handler.handle_request(request)
        return {
            "selected": response.selected_option,
            "approved": response.approved,
            "feedback": response.instructions or response.free_input or "No feedback provided.",
        }

hitl_toolset = HITLToolset(handler=ConsoleHITLHandler())
