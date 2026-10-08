"""Custom (non-LLM) agent classes referenced from system.yaml via custom:<name>."""
import logging
from typing import AsyncGenerator, List, Dict, Any
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.utils.context_utils import Aclosing

logger = logging.getLogger(__name__)

class WebToolsDeployerAgent(BaseAgent):
    """
    Custom agent for deploying found web mcp servers.
    """

    # --- Field Declarations for Pydantic ---
    # model_config allows setting Pydantic configurations if needed, e.g., arbitrary_types_allowed
    model_config = {"arbitrary_types_allowed": True}


    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Implements the custom orchestration logic for the story workflow.
        Uses the instance attributes assigned by Pydantic.
        """

        current_state = ctx.session.state

        filtered_mcps: List[Dict[str, Any]] = current_state.get('filtered_mcps', [])

        #TODO: Implement full deploying strategy after side tools are ready
        deployed_mcps = []
        ctx.session.state['deployed_mcps'] = deployed_mcps
        yield Event(author=self.name, invocation_id=ctx.invocation_id)

        # if not filtered_mcps:
        #     return


class ExecutorSwitchAgent(BaseAgent):
    """Runs exactly ONE of its children: the primary executor, or the fallback.

    Declared as ``children: [<primary>, <fallback>]``. The fallback is chosen
    only when the tool-prep pipeline's reranker never actually JUDGED the
    retrieved candidates — an unreadable answer, or one carrying no usable
    ``{index, score}`` pair — and the local cross-encoder could not stand in for
    it either. In that state ``filtered_tools`` is empty for a reason that says
    nothing about relevance, and abstaining on it throws away tools retrieval
    found correctly; FEDOT.MAS instead takes the whole unfiltered candidate set
    and does its own selection over it.

    Why a switch rather than two siblings in the sequence: ``AgentTool`` returns
    the LAST content-bearing event of the agent it wraps, and a ``before_agent``
    callback that short-circuits its own agent still emits one. A sibling that
    stood down after the real executor ran would therefore overwrite the
    pipeline's answer with its own "skipped" note (measured, both with and
    without text). Running one child and only one child removes that hazard.
    """

    model_config = {"arbitrary_types_allowed": True}

    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        from CoScientist.agents.callbacks.tool_callbacks import rerank_fallback_active

        children: List[BaseAgent] = list(self.sub_agents or [])
        if not children:
            logger.warning("%s: no children declared — nothing to run", self.name)
            return

        chosen = children[0]
        if len(children) > 1 and rerank_fallback_active(ctx.session.state):
            chosen = children[1]
            verdict = (ctx.session.state.get("executor_tool_match") or {})
            logger.info(
                "%s: reranker verdict=%s over %s candidate(s) — routing to %s",
                self.name, verdict.get("reason"), verdict.get("candidates"), chosen.name,
            )

        async with Aclosing(chosen.run_async(ctx)) as agen:
            async for event in agen:
                yield event
