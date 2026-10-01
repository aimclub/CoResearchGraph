"""A nested AgentTool run is linked back to the call that launched it, so the
ToolsViewer nests the delegated agent under its own `delegates` card."""
import asyncio

from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

from CoScientist.logging import tool_activity


async def _delegate_and_start() -> list[dict]:
    records = []

    async def sink(key, payload):
        records.append(payload)

    tool_activity.set_tool_activity_sink(sink)
    try:
        sessions = InMemorySessionService()
        outer = await sessions.create_session(app_name="app", user_id="u")
        inner = await sessions.create_session(app_name="app", user_id="u")
        caller = LlmAgent(name="LiteratureOrchestrator", model="gemini-2.0-flash")
        child = LlmAgent(name="ResearchAgent", model="gemini-2.0-flash")
        plugin = tool_activity.ToolActivityPlugin()

        async def one_call():
            outer_ctx = InvocationContext(
                session_service=sessions, invocation_id="inv", agent=caller, session=outer,
            )
            await plugin.before_tool_callback(
                tool=AgentTool(child), tool_args={"request": "q"},
                tool_context=ToolContext(outer_ctx, function_call_id="fc_1"),
            )
            # What the AgentTool's nested Runner does, in the same task.
            inner_ctx = InvocationContext(
                session_service=sessions, invocation_id="inv2", agent=child, session=inner,
            )
            await plugin.before_agent_callback(
                agent=child, callback_context=CallbackContext(inner_ctx),
            )

        # ADK runs each call in a task of its own.
        await asyncio.create_task(one_call())
        # Outside that task nothing is left behind.
        assert tool_activity._delegation_caller.get() is None
        return records, outer.id
    finally:
        tool_activity.set_tool_activity_sink(None)


def test_nested_agent_start_names_its_delegation():
    records, outer_id = asyncio.run(_delegate_and_start())
    start = next(r for r in records if r["phase"] == "agent_start")

    assert start["parent"] == "LiteratureOrchestrator"
    assert start["parent_instance"] == outer_id
    assert start["spawn_call_id"] == "fc_1"
