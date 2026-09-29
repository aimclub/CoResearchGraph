"""ADK plugin: cap large tool results before they enter the LLM context.

A single tool call returning a huge payload (``tavily_extract`` of full web
pages → ~240k tokens) can blow past the model's context window and crash the
run. The cap keeps each call bounded.

WHERE THE CUT IS MADE, AND WHY IT MOVED
---------------------------------------
It used to be made in ``after_tool_callback``, which returned the bounded copy.
That silently disabled every agent's own ``after_tool`` chain. ADK runs PLUGIN
callbacks first and the agent's only ``if altered_function_response is None``
(`flows/llm_flows/_tool_caller.py`, steps 4 and 5), and the plugin manager
early-exits on the first plugin that answers with anything. So on every result
longer than the cap — which is to say on every result worth looking at — the
agent's callbacks never ran at all.

Measured on a live session: seven Tavily calls, the agent chain logged for four,
and the three it skipped were the ones whose results carried the paper links
the paper capture exists to collect. `register_tool_result_links` and
the research logger were lost the same way. Silent, and worst exactly where it
mattered most.

So the cut happens at the boundary it was always about: ``before_model``, where
the request is assembled. ``after_tool`` keeps only a safety net at a far higher
ceiling — enough to stop a pathological payload becoming a session event, never
low enough to touch an ordinary one — and it ALWAYS returns None.

MUTATION RULES (ADK ≥ 1.25)
---------------------------
``flows/llm_flows/contents._copy_content_for_request`` shallow-copies ``Content``
and each ``Part`` and shares the nested payloads with the session events, with a
docstring that spells out the contract: *"Downstream processors must therefore
only replace Part objects or set top-level Part fields; mutating a nested field
in place would corrupt session history."* Capping at ``before_model`` therefore
REPLACES ``part.function_response`` with a ``model_copy`` — the same move ADK
makes two lines below that docstring — and never writes into it.

At ``after_tool`` time the object is not yet shared with anything (the event is
built from it afterwards), so the net may work in place.

Caps: TOOL_RESULT_MAX_CHARS (default 12000 ≈ 3k tokens) is what the model may
see per string; TOOL_RESULT_EVENT_MAX_CHARS (default 200000) is the net.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Tuple

from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CHARS = 12000
#: The net, not the cap. Twenty times the model's share: an ordinary web-search
#: result (~20k) passes through untouched, and the 240k-token page dump that
#: this plugin was written for is still bounded before it becomes an event.
_DEFAULT_EVENT_MAX_CHARS = 200000


def _truncate(obj: Any, cap: int) -> Tuple[Any, bool]:
    """Recursively cap every string in a JSON-like value. Returns (value, changed)."""
    if isinstance(obj, str):
        if len(obj) > cap:
            return obj[:cap] + f"\n…[truncated {len(obj) - cap} chars to fit the context window]", True
        return obj, False
    if isinstance(obj, dict):
        out, changed = {}, False
        for k, v in obj.items():
            nv, ch = _truncate(v, cap)
            out[k] = nv
            changed = changed or ch
        return out, changed
    if isinstance(obj, list):
        out, changed = [], False
        for v in obj:
            nv, ch = _truncate(v, cap)
            out.append(nv)
            changed = changed or ch
        return out, changed
    return obj, False


def _truncate_in_place(obj: Any, cap: int) -> bool:
    """Cap the strings inside a dict/list, writing back into it. True if cut.

    Only containers can be capped this way. A result that IS a string, or a
    pydantic object, is left alone rather than quietly replaced — the caller
    returns None either way, and a net that misses a rare shape is better than
    one that hands back an object the rest of the turn did not expect.
    """
    changed = False
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, list):
        items = enumerate(obj)
    else:
        return False
    for key, value in list(items):
        if isinstance(value, str):
            if len(value) > cap:
                obj[key] = value[:cap] + (
                    f"\n…[truncated {len(value) - cap} chars]")
                changed = True
        elif isinstance(value, (dict, list)):
            changed = _truncate_in_place(value, cap) or changed
    return changed


def _cap_tool_results(llm_request: Any, cap: int) -> int:
    """Bound every tool result in the request. Returns how many parts were cut.

    Replaces the ``FunctionResponse`` rather than writing into it: the parts of
    a request are shallow copies whose payloads are shared with the session
    events, so an in-place edit here would rewrite the stored record of what the
    tool actually returned.
    """
    if cap <= 0:
        return 0
    cut = 0
    for content in getattr(llm_request, "contents", None) or []:
        for part in getattr(content, "parts", None) or []:
            response = getattr(part, "function_response", None)
            if response is None:
                continue
            capped, changed = _truncate(getattr(response, "response", None), cap)
            if changed:
                part.function_response = response.model_copy(
                    update={"response": capped})
                cut += 1
    return cut


class ToolResultTruncationPlugin(BasePlugin):
    def __init__(self, max_chars: Optional[int] = None,
                 event_max_chars: Optional[int] = None,
                 name: str = "tool_truncation") -> None:
        super().__init__(name=name)
        self.max_chars = _cap_from_env(
            "TOOL_RESULT_MAX_CHARS", max_chars, _DEFAULT_MAX_CHARS)
        self.event_max_chars = _cap_from_env(
            "TOOL_RESULT_EVENT_MAX_CHARS", event_max_chars,
            _DEFAULT_EVENT_MAX_CHARS)

    async def after_tool_callback(self, *, tool, tool_args, tool_context,
                                  result) -> None:
        """The safety net, and nothing else.

        ALWAYS returns None. Answering here is what made ADK skip the agent's
        own after_tool callbacks, and those are where this system records what
        a tool returned — the papers it found, the links it produced.
        """
        if self.event_max_chars > 0 and _truncate_in_place(result, self.event_max_chars):
            logger.warning(
                "tool result from %s exceeded %d chars per string and was cut "
                "before it reached the session event",
                getattr(tool, "name", "?"), self.event_max_chars)
        return None

    async def before_model_callback(self, *, callback_context,
                                    llm_request) -> None:
        """Bound what the model sees. Never returns a response."""
        try:
            _cap_tool_results(llm_request, self.max_chars)
        except Exception:  # noqa: BLE001 — a cap must never fail a request
            logger.debug("could not cap tool results", exc_info=True)
        return None


def _cap_from_env(var: str, given: Optional[int], fallback: int) -> int:
    try:
        return int(os.getenv(var, given or fallback))
    except (TypeError, ValueError):
        return fallback
