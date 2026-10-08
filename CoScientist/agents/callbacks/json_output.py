"""after_model callback: reduce a model response to its first valid JSON payload.

Ported from VibePAV's ``llm._extract_json``. Agents constrained by an ADK
``output_schema`` hard-fail (``ValidationError: trailing characters``) when the
model emits ANYTHING besides the JSON object — markdown fences, prose around
it, duplicated objects, or trailing text. Not every provider honours
``response_format`` strictly, so this callback rewrites the response text to
exactly the extracted JSON before ADK's strict validation sees it.

Attach as ``after_model: [sanitize_json_output]`` on schema-constrained agents
(see CoScientist/agents/microfluidics.yaml).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

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
    if not text.strip():
        return None

    extracted = _extract_json(text)
    if extracted is None:
        return None  # nothing to fix — let schema validation report it

    clean = json.dumps(extracted, ensure_ascii=False)
    if clean == text.strip():
        return None

    logger.info(
        "[%s] sanitized JSON output (%d -> %d chars)",
        getattr(callback_context, "agent_name", "?"), len(text), len(clean),
    )
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=clean)])
    )
