import re

from google.adk.models import LlmResponse
from google.genai import types


_TARGET = re.compile(
    r"\bTarget tool:\s*([A-Za-z_][A-Za-z_0-9]*)\b"
    r"(?:\s*\(server_id=([A-Za-z0-9_-]+)\))?"
)
_ATTEMPT_KEY = "_pilot_exact_tool_attempt"


def ensure_pilot_exact_tool_retrieved(callback_context, llm_response):
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(getattr(part, "function_call", None) for part in parts):
        return None
    content = getattr(callback_context, "user_content", None)
    task = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    target_match = _TARGET.search(task)
    if target_match is None:
        return None
    target = target_match.group(1)
    server_id = target_match.group(2)
    inventory = callback_context.state.get("accumulated_tools") or []
    if any(
        isinstance(row, dict)
        and row.get("tool") == target
        and row.get("server_id")
        and (server_id is None or str(row["server_id"]) == server_id)
        for row in inventory
    ):
        return None
    attempt_key = (
        f"{_ATTEMPT_KEY}:{getattr(callback_context, 'invocation_id', '')}:{target}"
    )
    if callback_context.state.get(attempt_key):
        raise RuntimeError(
            f"Explicit target tool {target} was not retrieved after exact lookup"
        )
    callback_context.state[attempt_key] = True
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(name="retrieve_tools", args={"query": target})
    ]))
