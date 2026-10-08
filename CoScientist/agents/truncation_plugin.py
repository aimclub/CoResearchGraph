"""ADK plugin: cap large tool results before they enter the LLM context.

A single tool call returning a huge payload (e.g. ``tavily_extract`` of full web
pages → ~240k tokens) can blow past the model's context window and crash the
whole run. This recursively truncates long strings in any tool result so each
call stays bounded.

MUST be the LAST plugin on the Runner: ADK stops at the first plugin whose
``after_tool_callback`` returns non-None, so the event logger / graph-memory
plugins (which return None) still observe the FULL result first, and only then
does this plugin hand a bounded copy to the model.

Per-string cap is configurable via TOOL_RESULT_MAX_CHARS (default 12000 ≈ 3k tokens).
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

from google.adk.plugins.base_plugin import BasePlugin

_DEFAULT_MAX_CHARS = 12000


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


class ToolResultTruncationPlugin(BasePlugin):
    def __init__(self, max_chars: Optional[int] = None, name: str = "tool_truncation") -> None:
        super().__init__(name=name)
        try:
            self.max_chars = int(os.getenv("TOOL_RESULT_MAX_CHARS", max_chars or _DEFAULT_MAX_CHARS))
        except (TypeError, ValueError):
            self.max_chars = _DEFAULT_MAX_CHARS

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result) -> Optional[dict]:
        if not isinstance(result, (dict, list, str)) or self.max_chars <= 0:
            return None
        truncated, changed = _truncate(result, self.max_chars)
        # Return the bounded copy only if we actually shortened something — so we
        # don't needlessly early-exit the (None-returning) observer plugins... they
        # already ran before us, so returning non-None here is safe regardless.
        return truncated if changed else None
