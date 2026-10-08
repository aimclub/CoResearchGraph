"""JSON-repair shim for the LiteLlm tool-call boundary (F015a.A4 failure #1).

qwen sometimes emits malformed JSON in `tool_call.function.arguments` (truncation /
missing comma / extra data — worse for big payloads like a `submit_plan` plan). ADK
parses it in `google.adk.models.lite_llm._message_to_generate_content_response` with
`json.loads(...)` UNGUARDED in the non-streaming path (lite_llm.py:1630) — a
`JSONDecodeError` there kills the WHOLE run (the streaming path :2158 is guarded).

`install_json_repair()` swaps the module-global `json` inside lite_llm with a shim
whose `.loads()` repairs ON FAILURE ONLY (json_repair → valid-prefix salvage →
bracket-balancing heuristic → `{}`), covering both parse sites. SUCCESS behavior is
unchanged. This is process-wide once installed.
"""
from __future__ import annotations

import json as _json
import re


def repair_json_loads(s):
    """Parse JSON, repairing common LLM malformations on failure. Last resort: {}."""
    if not isinstance(s, str):
        return s
    try:
        return _json.loads(s)
    except _json.JSONDecodeError:
        pass
    # 1) json_repair — purpose-built for malformed LLM JSON (truncation, commas, quotes)
    try:
        import json_repair
        out = json_repair.loads(s)
        if out not in (None, ""):
            return out
    except Exception:
        pass
    # 2) salvage a valid prefix (handles trailing garbage / "Extra data")
    try:
        obj, _end = _json.JSONDecoder().raw_decode(s.strip())
        return obj
    except Exception:
        pass
    # 3) heuristic: drop a trailing comma, balance unclosed brackets (handles truncation)
    t = re.sub(r",\s*$", "", s.strip())
    t = t + "]" * max(0, t.count("[") - t.count("]")) + "}" * max(0, t.count("{") - t.count("}"))
    try:
        return _json.loads(t)
    except Exception:
        return {}  # better empty args (tool re-decides / re-prompts) than a dead run


class _RepairJsonModule:
    """Drop-in for the `json` module name used inside lite_llm; repairs on failure."""

    JSONDecodeError = _json.JSONDecodeError
    JSONDecoder = _json.JSONDecoder

    def loads(self, s, *args, **kwargs):
        try:
            return _json.loads(s, *args, **kwargs)
        except _json.JSONDecodeError:
            return repair_json_loads(s)

    def dumps(self, *args, **kwargs):
        return _json.dumps(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(_json, name)


_installed = False


def install_json_repair() -> bool:
    """Patch lite_llm's `json` with the repair shim. Idempotent; returns True if applied."""
    global _installed
    if _installed:
        return True
    try:
        import google.adk.models.lite_llm as _ll
        _ll.json = _RepairJsonModule()
        _installed = True
        return True
    except Exception:
        return False
