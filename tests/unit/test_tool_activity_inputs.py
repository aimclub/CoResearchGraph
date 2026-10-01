"""The ToolsViewer shows a call's complete inputs, not only the model's args:
the session-state keys the tool read and its arguments as callbacks rewrote them."""
import asyncio

from google.adk.agents import LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool, ToolContext

from CoScientist.logging import tool_activity


def _design(x: int, tool_context: ToolContext) -> dict:
    tz = tool_context.state.get("structured_tz")
    tool_context.state["design_out"] = {"x": x}
    tool_context.state.get("design_out")  # its own output, not an input
    tool_context.state.get("absent_key")
    tz["mutated"] = True  # the viewer must show what the tool saw
    return {"x": x}


async def _run_call(rewrite: bool) -> tuple[dict, ToolContext]:
    records = []

    async def sink(key, payload):
        records.append(payload)

    tool_activity.set_tool_activity_sink(sink)
    try:
        sessions = InMemorySessionService()
        session = await sessions.create_session(
            app_name="app", user_id="u",
            state={"structured_tz": {"target": "vanillin"}, "guard_key": 1},
        )
        context = InvocationContext(
            session_service=sessions, invocation_id="inv",
            agent=LlmAgent(name="DesignAgent", model="gemini-2.0-flash"), session=session,
        )
        tool_context = ToolContext(context, function_call_id="fc_1")
        tool = FunctionTool(_design)
        plugin = tool_activity.ToolActivityPlugin()
        args = {"x": 1}

        await plugin.before_tool_callback(tool=tool, tool_args=args, tool_context=tool_context)
        tool_context.state.get("guard_key")  # a before-tool callback's read
        if rewrite:
            args["x"] = 2
        result = await tool.run_async(args=args, tool_context=tool_context)
        await plugin.after_tool_callback(
            tool=tool, tool_args=args, tool_context=tool_context, result=result,
        )
    finally:
        tool_activity.set_tool_activity_sink(None)
    return records[-1], tool_context


def test_result_record_carries_state_the_tool_read():
    record, tool_context = asyncio.run(_run_call(rewrite=False))

    assert record["state_inputs"] == {
        "structured_tz": {"target": "vanillin"},
        "absent_key": None,
    }
    assert record["state_inputs_truncated"] is False
    assert "effective_args" not in record
    # The call's own State is back in place and the tool's write landed.
    assert type(tool_context.state).__name__ == "State"
    assert tool_context.state["design_out"] == {"x": 1}


def test_result_record_carries_args_rewritten_by_callbacks():
    record, _ = asyncio.run(_run_call(rewrite=True))

    assert record["effective_args"] == {"x": 2}


def test_secret_looking_values_are_masked():
    from CoScientist.logging.tool_activity import _redact

    state = {
        "temp:adk_oauth_credential": {"oauth2": {"access_token": "ya29.x"}},
        "structured_tz": {"target": "vanillin", "api_key": "sk-1", "token_budget": None},
        "runs": [{"password": "p", "value": 1}],
    }
    assert _redact(state) == {
        "temp:adk_oauth_credential": "***",
        "structured_tz": {"target": "vanillin", "api_key": "***", "token_budget": None},
        "runs": [{"password": "***", "value": 1}],
    }
