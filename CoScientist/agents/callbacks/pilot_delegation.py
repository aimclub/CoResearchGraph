"""Verify the isolated pilot from observed ADK tool events.

The pilot keeps the normal tool roster and lets the orchestrator choose its
order. A final answer is valid only after each required tool has actually
been called and returned a successful response in the current invocation.
"""

import json

from google.adk.models.llm_response import LlmResponse
from google.genai import types

_REQUIRED = ("retrieve_tools", "ResearchAgent", "TaskExecutorAgent")
_PILOT_SCIENCE_TOOLS = (
    "dataset_overview_heracleum_tox",
    "butina_clustering",
    "predict_ld50",
    "predict_molecule_profile",
)


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
