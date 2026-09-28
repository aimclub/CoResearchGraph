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
    """Keep discovered pilot MCP names in the outbound executor request."""
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
    invocation = getattr(tool_context, "_invocation_context", None)
    invocation_id = getattr(invocation, "invocation_id", None)
    prior = tool_context.state.get("_pilot_science_catalog") or {}
    known = dict(prior.get("tools") or {}) if prior.get("invocation_id") == invocation_id else {}
    known.update(science)
    tool_context.state["_pilot_science_catalog"] = {
        "invocation_id": invocation_id, "tools": known,
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
    return {"error": (
        "Pilot scientific handoff: TaskExecutorAgent request must carry all "
        f"retrieved MCP tool names and server ids: {available}. Resubmit the "
        "computation naming its target tool and available tools; send it "
        "through ToolPipelineAgent. "
        "heracleum-tox is a prepared MCP service, not a GitHub repository."
    )}


def enforce_pilot_executor_route(tool, args, tool_context):
    """Do not turn a named ready MCP computation into sandbox code."""
    if not isinstance(tool, AgentTool) or tool.name != "CoderAgent":
        return None
    invocation = getattr(tool_context, "_invocation_context", None)
    content = getattr(invocation, "user_content", None)
    request = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    subtask = args.get("request") if isinstance(args, dict) else None
    if isinstance(subtask, str) and not any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", subtask)
        for name in _PILOT_SCIENCE_TOOLS
    ):
        explicit_repo = re.search(
            r"\b(?:clone|inspect|read|checkout|check out)\b.{0,100}?"
            r"(https?://github\.com/[^\s,;)]+)", request, re.IGNORECASE,
        )
        if (explicit_repo and explicit_repo.group(1) in subtask
                and re.search(r"\b(?:clone|inspect|read|checkout|check out)\b",
                              subtask, re.IGNORECASE)):
            return None
    if any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request)
        for name in _PILOT_SCIENCE_TOOLS
    ):
        return {"error": (
            "The request names a ready scientific MCP tool. Delegate this "
            "computation to ToolPipelineAgent; CoderAgent cannot call MCP tools. "
            "A prepared MCP server label is not a GitHub repository."
        )}
    return None


def preserve_pilot_target(tool, args, tool_context):
    """Keep an explicit science target when the executor delegates downstream."""
    if not isinstance(tool, AgentTool) or tool.name != "ToolPipelineAgent":
        return None
    request = args.get("request") if isinstance(args, dict) else None
    if not isinstance(request, str):
        return None
    invocation = getattr(tool_context, "_invocation_context", None)
    content = getattr(invocation, "user_content", None)
    source = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    target = re.search(r"\bTarget tool:\s*([A-Za-z_][A-Za-z_0-9]*)\b", source)
    if not target or target.group(1) not in _PILOT_SCIENCE_TOOLS:
        return None
    if re.search(r"\bTarget tool:\s*" + re.escape(target.group(1)) + r"\b", request):
        return None
    args["request"] = f"{source}\n\nExecutor subtask: {request}"
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


def _scientific_calls(bodies):
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
    return calls


def _verified_science_names(calls):
    return {
        call.get("tool") for call in calls
        if isinstance(call, dict)
        and call.get("result") not in (None, "", {})
        and not (
            isinstance(call.get("result"), dict)
            and call["result"].get("truncated")
        )
    }


def _request_missing_science(callback_context, name):
    state = getattr(callback_context, "state", None)
    invocation_id = callback_context._invocation_context.invocation_id
    discovered = (state.get("accumulated_tools") or []) if state is not None else []
    science = {
        item["tool"]: item.get("server_id")
        for item in discovered
        if isinstance(item, dict) and item.get("tool") in _PILOT_SCIENCE_TOOLS
    }
    prior = state.get("_pilot_science_catalog") or {}
    if prior.get("invocation_id") == invocation_id:
        science.update(prior.get("tools") or {})
    if not science.get(name):
        raise RuntimeError(
            "Pilot scientific computation contract: required MCP tool "
            f"{name} was not discovered with a server id"
        )

    progress = state.get("_pilot_science_recovery") or {}
    attempted = (
        list(progress.get("attempted") or [])
        if progress.get("invocation_id") == invocation_id else []
    )
    if name in attempted:
        raise RuntimeError(
            "Pilot scientific computation contract: no verified result from "
            f"{name} after a targeted executor attempt"
        )
    attempted.append(name)
    state["_pilot_science_recovery"] = {
        "invocation_id": invocation_id, "attempted": attempted,
    }

    available = ", ".join(
        f"{tool} (server_id={server_id})"
        for tool, server_id in science.items()
        if server_id
    )
    arguments = " with name_or_smiles=xanthotoxin" if name == "predict_molecule_profile" else ""
    cluster_member = (
        " xanthotoxin is a member of Heracleum cluster E."
        if name == "predict_molecule_profile" else ""
    )
    request = (
        f"Target tool: {name} (server_id={science[name]}). "
        f"Call {name}{arguments} through ToolPipelineAgent and return its actual "
        f"MCP result.{cluster_member} "
        f"Available pilot MCP tools: {available}. The other names are routing "
        "context, not additional targets for this request."
    )
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(
            name="TaskExecutorAgent", args={"request": request}
        )
    ]))


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
    calls = _scientific_calls(completed["TaskExecutorAgent"])
    verified_names = _verified_science_names(calls)
    missing_science = [name for name in _PILOT_SCIENCE_TOOLS if name not in verified_names]
    if missing_science:
        return _request_missing_science(callback_context, missing_science[0])
    receipt = {"status": "computed", "scientific_mcp_calls": calls}
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(
        text="Observed scientific MCP results:\n" + json.dumps(
            receipt, ensure_ascii=False, indent=2
        )
    )]))
