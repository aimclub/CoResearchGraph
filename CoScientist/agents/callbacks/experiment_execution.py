"""Execution evidence for the direct MCP experiment agent."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from urllib.parse import urlparse

import requests

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.mcp_tool import McpTool
from google.genai import types


_STATE_KEY = "_experiment_mcp_receipt"
_EXECUTOR_STATE_KEY = "_executor_science_receipt"
_PILOT_JSON_TOOL_KEY = "_pilot_json_science_tool"
_MAX_RESULT_BYTES = 16_384
_ARTIFACT_EXTENSIONS = {
    ".csv", ".tsv", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".pdf"
}
_ARTIFACT_KEYS = {
    "artifact", "csv", "table", "plot", "figure", "image", "file", "path",
    "download_url", "presigned_url",
}


def reset_scientific_execution(callback_context):
    """Keep a second ExperimentAgent task from inheriting the first task's proof."""
    callback_context.state[_STATE_KEY] = {
        "invocation_id": callback_context._invocation_context.invocation_id,
        "calls": [],
    }
    callback_context.state[_PILOT_JSON_TOOL_KEY] = None
    return None


def reset_executor_science_receipt(callback_context):
    callback_context.state[_EXECUTOR_STATE_KEY] = None
    return None


def require_first_scientific_tool_call(callback_context, llm_request):
    """Make the first matched MCP operation a real function call."""
    receipt = callback_context.state.get(_STATE_KEY) or {}
    if (receipt.get("invocation_id") == callback_context._invocation_context.invocation_id
            and receipt.get("calls")):
        return None
    if not (callback_context.state.get("executor_tool_match") or {}).get("matched"):
        return None
    selected = {
        item.get("tool") for item in callback_context.state.get("filtered_tools") or []
        if isinstance(item, dict) and item.get("tool")
    }
    target = callback_context.state.get("explicit_tool_target")
    if target and target not in selected:
        raise RuntimeError(f"Explicit target tool {target} was not selected")
    if target:
        selected = {target}
    if not selected:
        return None
    declarations = [
        declaration
        for tool in llm_request.config.tools or []
        for declaration in tool.function_declarations or []
        if declaration.name in selected
    ]
    if not declarations:
        if target:
            raise RuntimeError(f"Explicit target tool {target} is not available to ExperimentAgent")
        return None
    callback_context.state[_PILOT_JSON_TOOL_KEY] = {
        "invocation_id": callback_context._invocation_context.invocation_id,
        "name": declarations[0].name if len(declarations) == 1 else None,
    }
    llm_request.config.tools = [types.Tool(function_declarations=declarations)]
    llm_request.config.tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.ANY
        )
    )
    return None


def recover_pilot_json_science_call(callback_context, llm_response):
    """Route a single JSON-shaped call through ADK when the gateway omits function_call."""
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(getattr(part, "function_call", None) for part in parts):
        return None
    receipt = callback_context.state.get(_STATE_KEY) or {}
    invocation_id = callback_context._invocation_context.invocation_id
    if receipt.get("invocation_id") == invocation_id and receipt.get("calls"):
        return None
    offered = callback_context.state.get(_PILOT_JSON_TOOL_KEY) or {}
    name = offered.get("name") if offered.get("invocation_id") == invocation_id else None
    if not name or not (callback_context.state.get("executor_tool_match") or {}).get("matched"):
        return None
    target = callback_context.state.get("explicit_tool_target")
    if target and target != name:
        return None
    text_parts = [part.text for part in parts if part.text and not getattr(part, "thought", False)]
    if len(text_parts) != 1:
        return None
    try:
        payload = json.loads(text_parts[0])
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, list) or len(payload) != 1:
        return None
    call = payload[0]
    if (not isinstance(call, dict) or set(call) != {"name", "parameters"}
            or call.get("name") != name or not isinstance(call.get("parameters"), dict)):
        return None
    return LlmResponse(content=types.Content(role="model", parts=[
        *(part for part in parts if getattr(part, "thought", False)),
        types.Part.from_function_call(name=name, args=call["parameters"]),
    ]))


def capture_scientific_pipeline_receipt(tool, args, tool_context, tool_response):
    """Accept a receipt only from the real local ToolPipelineAgent call."""
    if not isinstance(tool, AgentTool) or tool.name != "ToolPipelineAgent":
        return None
    if isinstance(tool_response, dict) and tool_response.get("error"):
        return None
    payload = (
        tool_response.get("result") if isinstance(tool_response, dict)
        else tool_response
    )
    try:
        receipt = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, ValueError):
        return None
    if not isinstance(receipt, dict) or receipt.get("status") != "computed":
        return None
    if not isinstance(receipt.get("scientific_mcp_calls"), list):
        return None
    if not receipt["scientific_mcp_calls"]:
        return None
    verdict = tool_context.state.get("executor_tool_match") or {}
    if not verdict.get("matched"):
        return None
    tool_context.state[_EXECUTOR_STATE_KEY] = {
        "invocation_id": tool_context._invocation_context.invocation_id,
        "receipt": receipt,
    }
    return None


def attest_executor_science(callback_context, llm_response):
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(getattr(part, "function_call", None) for part in parts):
        return None
    item = callback_context.state.get(_EXECUTOR_STATE_KEY) or {}
    if item.get("invocation_id") != callback_context._invocation_context.invocation_id:
        item = {}
    receipt = item.get("receipt")
    if not receipt:
        text = "\n".join(part.text or "" for part in parts)
        if "scientific_mcp_calls" in text:
            raise RuntimeError(
                "TaskExecutorAgent cannot claim scientific results without an "
                "observed ToolPipelineAgent receipt"
            )
        return None
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(
        text=json.dumps(receipt, ensure_ascii=False, default=str)
    )]))


def _result_data(response: dict):
    structured = response.get("structuredContent") or response.get("structured_content")
    if structured is not None:
        return structured
    text = "\n".join(
        str(block.get("text"))
        for block in response.get("content") or []
        if isinstance(block, dict) and block.get("text")
    )
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


def _bounded_result(value):
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    if len(encoded.encode("utf-8")) <= _MAX_RESULT_BYTES:
        return value
    return {"truncated": True, "sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _is_artifact_reference(key: str, value: str) -> bool:
    parsed = urlparse(value)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in _ARTIFACT_EXTENSIONS:
        return True
    artifact_key = key.lower() in _ARTIFACT_KEYS or key.lower().endswith(
        ("_artifact", "_file", "_path", "_url")
    )
    return artifact_key and ("/" in value or "\\" in value)


def _artifact_exists(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        try:
            with requests.get(html.unescape(value), stream=True, timeout=5) as response:
                response.raise_for_status()
                return bool(next(response.iter_content(chunk_size=1), b""))
        except (OSError, requests.RequestException):
            return False
    try:
        return Path(value).is_file()
    except OSError:
        return False


def _verified_result(value, key: str = ""):
    """Remove artifact references that cannot be opened at receipt time."""
    if isinstance(value, dict):
        return {
            field: _verified_result(item, str(field))
            for field, item in value.items()
            if not (isinstance(item, str) and _is_artifact_reference(str(field), item)
                    and not _artifact_exists(item))
        }
    if isinstance(value, list):
        return [
            _verified_result(item, key)
            for item in value
            if not (isinstance(item, str) and _is_artifact_reference(key, item)
                    and not _artifact_exists(item))
        ]
    return value


def _has_scientific_result(value) -> bool:
    if value in (None, "", [], {}):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"ok", "success", "done", "completed"}
    if isinstance(value, dict):
        if value.get("isError") or value.get("is_error") or value.get("error"):
            return False
        if value.get("status") in {"error", "failed"}:
            return False
        return any(
            key not in {"status", "isError", "is_error"}
            and _has_scientific_result(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_scientific_result(item) for item in value)
    return True


def record_scientific_mcp_result(tool, args, tool_context, tool_response):
    """Record only a real, nonempty result from a selected MCP tool."""
    if not isinstance(tool, McpTool) or not isinstance(tool_response, dict):
        return None
    if tool_response.get("isError") or tool_response.get("is_error"):
        return None
    if tool_response.get("error") or tool_response.get("status") in {"error", "failed"}:
        return None
    result = _result_data(tool_response)
    if not _has_scientific_result(result):
        return None
    result = _verified_result(result)
    if not _has_scientific_result(result):
        return None

    invocation_id = tool_context._invocation_context.invocation_id
    receipt = tool_context.state.get(_STATE_KEY) or {}
    calls = (
        list(receipt.get("calls") or [])
        if receipt.get("invocation_id") == invocation_id else []
    )
    calls.append({
        "tool": tool.name,
        "args": args,
        "result": _bounded_result(result),
    })
    tool_context.state[_STATE_KEY] = {"invocation_id": invocation_id, "calls": calls}
    return None


def require_scientific_execution(callback_context, llm_response):
    """A narrated tool call is not a completed experiment."""
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(getattr(part, "function_call", None) for part in parts):
        return None
    text = "\n".join(part.text or "" for part in parts)
    if text.strip().startswith("NO_MATCHING_TOOL:"):
        return None
    invocation = callback_context._invocation_context
    receipt = callback_context.state.get(_STATE_KEY) or {}
    if receipt.get("invocation_id") != invocation.invocation_id or not receipt.get("calls"):
        raise RuntimeError(
            "ExperimentAgent cannot finish without a successful scientific MCP call"
        )
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part(text=json.dumps(
                {"status": "computed", "scientific_mcp_calls": receipt["calls"]},
                ensure_ascii=False,
                default=str,
            ))],
        )
    )
