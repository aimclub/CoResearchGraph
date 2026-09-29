"""after_model callback: reduce a model response to its first valid JSON payload.

Ported from VibePAV's ``llm._extract_json``. Agents constrained by an ADK
``output_schema`` hard-fail (``ValidationError: trailing characters``) when the
model emits ANYTHING besides the JSON object — markdown fences, prose around
it, duplicated objects, or trailing text. Not every provider honours
``response_format`` strictly, so this callback rewrites the response text to
exactly the extracted JSON before ADK's strict validation sees it.

Attach as ``after_model: [sanitize_json_output]`` on schema-constrained agents
(see CoScientist/microfluidics/microfluidics.yaml). Agent-specific repairs
(schema, fallback, normalization) are registered with
:func:`register_structured_answer`.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

import yaml

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.genai import types

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _try_loads(text: str) -> Optional[Any]:
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _unwrap_completion_state(obj: Any) -> Any:
    """Flatten GLM/OpenRouter ``completionState``/``entries`` envelopes.

    Some providers emit list items as JSON *strings* like
    ``{"completionState":"complete","entries":[["index",{"value":1}], ...],
    "type":"Object"}`` instead of a plain ``{"index": 1, ...}``. Schema
    validation (ToolRanking/MCPRanking, FEDOT MASConfig) then fails.
    """
    if isinstance(obj, str):
        parsed = _try_loads(obj)
        if parsed is None:
            return obj
        obj = parsed
    if not isinstance(obj, dict):
        return obj
    if obj.get("type") == "Object" and isinstance(obj.get("entries"), list):
        out: dict[str, Any] = {}
        for entry in obj["entries"]:
            if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
                continue
            key, val = entry
            out[str(key)] = _unwrap_completion_state(
                val["value"] if isinstance(val, dict) and "value" in val else val
            )
        return out
    return {k: _unwrap_completion_state(v) for k, v in obj.items()}


def _normalize_ranking_payload(data: Any) -> Any:
    """Coerce ranking list items that arrived as completionState strings."""
    if not isinstance(data, dict):
        return data
    out = dict(data)
    for key in ("tools", "mcp_scores"):
        items = out.get(key)
        if isinstance(items, list):
            out[key] = [_unwrap_completion_state(item) for item in items]
    return out


def _extract_json(text: str) -> Optional[Any]:
    """First JSON object/array in the text: fenced block, whole text, or the
    first balanced ``{...}`` candidate that parses."""
    text = text.strip()

    match = _JSON_BLOCK_RE.search(text)
    if match:
        parsed = _try_loads(match.group(1).strip())
        if parsed is not None:
            return parsed

    parsed = _try_loads(text)
    if parsed is not None:
        return parsed

    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    parsed = _try_loads(text[start : i + 1])
                    if parsed is not None:
                        return parsed
                    break
        start = text.find("{", start + 1)
    return None


def _maybe_apply_tool_rerank(callback_context: CallbackContext, payload: Any) -> None:
    """Fold ToolReranker score apply into sanitize (avoids a separate after_model cb).

    ``collect_reranked_tools`` (after_agent) remains the fallback when scores only
    land in ``output_key`` state. Lazy import avoids a cycle with tool_callbacks.
    """
    if not isinstance(payload, dict) or payload.get("tools") is None:
        return
    try:
        from CoScientist.agents.callbacks.tool_callbacks import (
            _TOOL_RERANK_APPLIED_KEY,
            _score_items_from_reranked_state,
            apply_tool_rerank_scores,
        )
    except Exception:  # noqa: BLE001
        return
    state = callback_context.state
    if state.get(_TOOL_RERANK_APPLIED_KEY):
        return
    items = _score_items_from_reranked_state(payload)
    if not items:
        return
    apply_tool_rerank_scores(state, items)
    logger.info(
        "[%s] applied tool rerank during sanitize (%d scores)",
        getattr(callback_context, "agent_name", "?"),
        len(items),
    )

@dataclass(frozen=True)
class StructuredAnswer:
    """How one schema-constrained agent's answer is repaired.

    schema      pydantic model the payload is validated against (and dumped
                through) when an ADK callback context is present
    fallback    ``state -> payload``: a conservative schema-valid answer for
                when the model's text cannot be parsed or fails the schema
    normalize   ``payload -> payload``: fix-ups applied before validation
    contextless the answer for unparseable text when no callback context is
                given (older direct callers)
    """
    schema: Optional[type] = None
    fallback: Optional[Callable[[Mapping[str, Any]], Optional[dict]]] = None
    normalize: Optional[Callable[[Any], Any]] = None
    contextless: Optional[dict] = None


# agent name -> StructuredAnswer. Profiles register their own agents (e.g.
# CoScientist/microfluidics/json_answers.py); an agent with no entry gets the
# generic extraction only.
_STRUCTURED_ANSWERS: dict[str, StructuredAnswer] = {}


def register_structured_answer(agent_name: str, **spec: Any) -> None:
    """Register how ``agent_name``'s JSON answer is normalized / validated."""
    _STRUCTURED_ANSWERS[agent_name] = StructuredAnswer(**spec)


def _fallback_payload(agent_name: str, callback_context: Any) -> Optional[dict[str, Any]]:
    """Build a conservative schema-valid result when the model fails."""
    spec = _STRUCTURED_ANSWERS.get(agent_name)
    if spec is None or spec.fallback is None:
        return None
    state = getattr(callback_context, "state", {}) or {}
    return spec.fallback(state)


def _extract_structured_payload(
    text: str, agent_name: str = "", callback_context: Any = None,
) -> Optional[Any]:
    """Extract JSON, with a conservative YAML fallback for schema answers."""
    payload = _extract_json(text)
    if payload is None:
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError:
            payload = None
        if not isinstance(payload, (dict, list)):
            payload = None
    if payload is None and callback_context is not None and hasattr(callback_context, "state"):
        fallback = _fallback_payload(agent_name, callback_context)
        if fallback is not None:
            logger.warning("[%s] non-JSON response; using safe fallback", agent_name)
            return fallback
    spec = _STRUCTURED_ANSWERS.get(agent_name)
    if payload is None:
        # Callers that give no ADK callback context get the agent's plain default.
        return dict(spec.contextless) if spec and spec.contextless is not None else None
    if spec is None:
        return payload
    if spec.normalize is not None:
        payload = spec.normalize(payload)
    if spec.schema is not None and callback_context is not None and hasattr(callback_context, "state"):
        try:
            payload = spec.schema.model_validate(payload).model_dump()
        except Exception:  # noqa: BLE001
            fallback = _fallback_payload(agent_name, callback_context)
            if fallback is not None:
                logger.warning("[%s] invalid JSON shape; using safe fallback", agent_name)
                payload = fallback
    return payload


def sanitize_json_output(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> Optional[LlmResponse]:
    """Rewrite the response to exactly its JSON payload (or pass through)."""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None
    # Function calls are not JSON answers — leave them alone.
    if any(getattr(p, "function_call", None) for p in parts):
        return None

    text = "".join(
        p.text for p in parts
        if getattr(p, "text", None) and not getattr(p, "thought", False)
    )
    thought_text = "".join(
        p.text for p in parts
        if getattr(p, "text", None) and getattr(p, "thought", False)
    )

    if not text.strip() and not thought_text.strip():
        fallback = _fallback_payload(getattr(callback_context, "agent_name", ""), callback_context)
        if fallback is None:
            return None
        return LlmResponse(content=types.Content(
            role="model", parts=[types.Part(text=json.dumps(fallback, ensure_ascii=False))]
        ))

    extracted = _extract_structured_payload(
        text, getattr(callback_context, "agent_name", ""), callback_context
    ) if text.strip() else None
    if extracted is None and thought_text.strip():
        # GLM often parks ToolRanking JSON in a thought part.
        extracted = _extract_json(thought_text)
    if extracted is None:
        # Last resort: for agents with an output_schema, returning None lets
        # the raw (non-JSON) text reach ADK's validate_schema, which crashes
        # with "Invalid JSON".  Build a safe fallback instead.
        agent_name = getattr(callback_context, "agent_name", "")
        fallback = _fallback_payload(agent_name, callback_context)
        if fallback is not None:
            logger.warning(
                "[%s] could not extract any JSON; using safe fallback", agent_name,
            )
            extracted = fallback
        else:
            return None  # unknown agent — nothing to fix

    normalized = _normalize_ranking_payload(extracted)
    _maybe_apply_tool_rerank(callback_context, normalized)

    if not text.strip():
        return None

    clean = json.dumps(normalized, ensure_ascii=False)
    if clean == text.strip():
        return None

    logger.info(
        "[%s] sanitized JSON output (%d -> %d chars)",
        getattr(callback_context, "agent_name", "?"), len(text), len(clean),
    )
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=clean)])
    )


def unwrap_model_response_args(
    tool: Any = None,
    args: Any = None,
    tool_context: Any = None,
    **_: Any,
) -> None:
    """before_tool callback: lift a ``set_model_response`` answer out of one
    extra wrapper key before ADK validates it.

    Some models call ``set_model_response({"params": {...}})`` instead of
    passing the schema fields at the top level. ADK validates the arguments
    as they come, and a schema whose fields all have defaults accepts the
    unknown key silently — the answer becomes an EMPTY object and whatever
    the agent produced (the literature routes, in the run this guards) is
    lost. The arguments are rewritten in place: ADK hands the tool the same
    dict it gave the callbacks.
    """
    if getattr(tool, "name", "") != "set_model_response" or not isinstance(args, dict):
        return None
    model_type = getattr(tool, "_model_type", None)
    fields = set(getattr(model_type, "model_fields", None) or ())
    if not fields or len(args) != 1 or set(args) & fields:
        return None
    (key, inner), = args.items()
    if isinstance(inner, str):
        inner = _try_loads(inner)
    if not isinstance(inner, dict) or not set(inner) & fields:
        return None
    logger.warning(
        "[%s] set_model_response answer was wrapped in %r — unwrapped",
        getattr(tool_context, "agent_name", "?"), key,
    )
    args.clear()
    args.update(inner)
    return None
