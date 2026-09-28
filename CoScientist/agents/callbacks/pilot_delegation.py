"""Verify the isolated pilot from observed ADK tool events.

The pilot keeps the normal tool roster and lets the orchestrator choose its
order. A final answer is valid only after each required tool has actually
been called and returned a successful response in the current invocation.
"""

import json
import re

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

_REQUIRED = ("retrieve_tools", "ResearchAgent", "TaskExecutorAgent")
_PILOT_SCIENCE_TOOLS = (
    "dataset_overview_heracleum_tox",
    "chemical_space_clustering",
    "predict_ld50",
    "predict_molecule_profile",
)


def enforce_pilot_science_handoff(tool, args, tool_context):
    """Carry discovered pilot MCP names into the remote executor request."""
    if not isinstance(tool, AgentTool) or tool.name != "TaskExecutorAgent":
        return None
    request = args.get("request") if isinstance(args, dict) else None
    if not isinstance(request, str):
        return None
    retrieved = tool_context.state.get("accumulated_tools") or []
    science = {
        item["tool"]: item.get("server_id")
        for item in retrieved
        if isinstance(item, dict) and item.get("tool") in _PILOT_SCIENCE_TOOLS
    }
    if not science or (
        all(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request) for name in science)
        and all(server_id in request for server_id in science.values() if server_id)
    ):
        return None
    available = ", ".join(
        f"{name} (server_id={server_id})" if server_id else name
        for name, server_id in science.items()
    )
    args["request"] = (
        f"{request}\n\nDiscovered pilot MCP tools: {available}. "
        "This list is routing context; execute only the computation requested "
        "above through ToolPipelineAgent. heracleum-tox is a prepared MCP "
        "service, not a GitHub repository."
    )
    return None


def enforce_pilot_executor_route(tool, args, tool_context):
    """Do not turn a named ready MCP computation into sandbox code."""
    if not isinstance(tool, AgentTool) or tool.name != "CoderAgent":
        return None
    subtask = args.get("request") if isinstance(args, dict) else None
    if not isinstance(subtask, str):
        return None
    invocation = getattr(tool_context, "_invocation_context", None)
    content = getattr(invocation, "user_content", None)
    request = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    names_ready_tool = any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", subtask)
        for name in _PILOT_SCIENCE_TOOLS
    )
    names_prepared_service = re.search(
        r"\b(?:heracleum[-_ ]tox|server_id)\b", subtask, re.IGNORECASE
    )
    reimplements_science = re.search(
        r"\b(?:predict|compute|calculate|estimate|run)\b.{0,100}?"
        r"\b(?:LD50|hepatotoxicity|DILI|cardiotoxicity|carcinogenicity|"
        r"chemical[ -]space|clustering|molecule[ -]profile)\b",
        subtask, re.IGNORECASE,
    ) and any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request)
        for name in _PILOT_SCIENCE_TOOLS
    )
    ambiguous_clone = (
        re.search(r"\b(?:clone|checkout|check out)\b", subtask, re.IGNORECASE)
        and not re.search(r"https?://github\.com/[^\s,;)]+", subtask, re.IGNORECASE)
        and any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request)
                for name in _PILOT_SCIENCE_TOOLS)
    )
    if names_ready_tool or names_prepared_service or reimplements_science or ambiguous_clone:
        return {"error": (
            "The request names a ready scientific MCP tool. Delegate this "
            "computation to ToolPipelineAgent; CoderAgent cannot call MCP tools. "
            "A prepared MCP server label is not a GitHub repository."
        )}
    return None


def _completed_delegations(callback_context):
    invocation = callback_context._invocation_context
    called = set()
    completed = {}
    for event in invocation.session.events:
        if event.invocation_id != invocation.invocation_id:
            continue
        called.update(call.name for call in event.get_function_calls())
        for response in event.get_function_responses():
            if response.name not in called:
                continue
            body = response.response
            if isinstance(body, dict) and (
                body.get("error") or body.get("status") in {"error", "failed"}
            ):
                continue
            completed.setdefault(response.name, []).append(body)
    return completed


def _scientific_receipt(bodies):
    calls = []
    for body in bodies:
        if not isinstance(body, dict):
            continue
        result = body.get("result")
        try:
            receipt = json.loads(result) if isinstance(result, str) else result
        except ValueError:
            continue
        if not isinstance(receipt, dict) or receipt.get("status") != "computed":
            continue
        if isinstance(receipt.get("scientific_mcp_calls"), list):
            calls.extend(receipt["scientific_mcp_calls"])
    names = {
        call.get("tool") for call in calls
        if isinstance(call, dict)
        and call.get("result") not in (None, "", {})
        and not (
            isinstance(call.get("result"), dict)
            and call["result"].get("truncated")
        )
    }
    if not set(_PILOT_SCIENCE_TOOLS) <= names:
        return None
    return {"status": "computed", "scientific_mcp_calls": calls}


def require_pilot_delegations(callback_context, llm_response):
    """Reject a final answer until all pilot tools have real call/response pairs."""
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(part.function_call for part in parts):
        return None
    completed = _completed_delegations(callback_context)
    missing = [name for name in _REQUIRED if name not in completed]
    if missing:
        raise RuntimeError(
            "Pilot delegation contract: final response before observed "
            "call and successful response for " + ", ".join(missing)
        )
    receipt = _scientific_receipt(completed["TaskExecutorAgent"])
    if receipt is None:
        raise RuntimeError(
            "Pilot scientific computation contract: TaskExecutorAgent did not "
            "return verified results for the required MCP tools"
        )
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(
        text="Observed scientific MCP results:\n" + json.dumps(
            receipt, ensure_ascii=False, indent=2
        )
    )]))
