"""Fill in the session scope of a tool call, so no agent has to.

The MCP servers write their objects under a scoped key:

    ephemeral/<user_id>/<session_id>/<feature>/<filename>

The tools that store something therefore declare ``user_id`` and ``session_id``.
Neither is a decision the model should make. A model that copies the wrong id
writes into another session's prefix, and the mistake is silent.

This plugin fills both in at the tool boundary, from the same scope the graph
uses. It touches an argument only when the tool declares it and the caller left
it empty, so a tool that does not take the pair sees no change. An explicit
value from the caller wins, which keeps a deliberate cross-session read working.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional, Set

from google.adk.plugins import BasePlugin

from CoScientist.graph.session_scope import session_key

logger = logging.getLogger(__name__)

SCOPE_ARGS = ("user_id", "session_id")

# The vault rejects an id that does not match this, and it builds every key from
# the pair, so send something it accepts.
_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_id(value: Any, default: str) -> str:
    cleaned = _ID_RE.sub("_", str(value or "")).strip("_")
    return cleaned[:64] or default


def _declared_params(tool: Any) -> Set[str]:
    """The argument names a tool accepts.

    An MCP tool carries a JSON schema; a native FunctionTool builds a
    FunctionDeclaration. Read whichever is there, and return an empty set when
    neither is, so an unknown tool shape is left alone.
    """
    schema = getattr(getattr(tool, "_mcp_tool", None), "inputSchema", None)
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return set(props)

    try:
        declaration = tool._get_declaration()  # noqa: SLF001 — the only accessor ADK offers
        params = getattr(declaration, "parameters", None)
        props = getattr(params, "properties", None)
        if props:
            return set(props)
    except Exception:  # noqa: BLE001 — a tool without a declaration is normal
        pass
    return set()


class SessionScopePlugin(BasePlugin):
    """Inject ``user_id`` / ``session_id`` into the tool calls that declare them."""

    def __init__(self, name: str = "session_scope") -> None:
        super().__init__(name)

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[dict]:
        try:
            if not isinstance(tool_args, dict):
                return None
            declared = _declared_params(tool)
            wanted = [a for a in SCOPE_ARGS if a in declared and not tool_args.get(a)]
            if not wanted:
                return None

            user_id, session_id = session_key(tool_context)
            values = {
                "user_id": _safe_id(user_id, "unknown_user"),
                "session_id": _safe_id(session_id, "unknown_session"),
            }
            for arg in wanted:
                tool_args[arg] = values[arg]
            logger.debug(
                "session_scope: %s <- %s", getattr(tool, "name", "?"), wanted,
            )
        except Exception:  # noqa: BLE001 — scoping must never break a tool call
            pass
        return None  # never replace the tool result
