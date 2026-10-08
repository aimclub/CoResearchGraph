"""The plugin that fills user_id / session_id into scoped tool calls.

An MCP server builds its S3 key from the pair. A model that copies the wrong id
writes into another session's prefix, silently, so the framework supplies both.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from CoScientist.tools.session_scope_plugin import SessionScopePlugin


class _McpTool:
    """An MCP tool: its arguments come from a JSON schema."""

    def __init__(self, name, properties):
        self.name = name
        self._mcp_tool = SimpleNamespace(inputSchema={"properties": dict.fromkeys(properties, {})})


class _NativeTool:
    """A native tool: its arguments come from a FunctionDeclaration."""

    def __init__(self, name, properties):
        self.name = name
        self._properties = properties

    def _get_declaration(self):
        return SimpleNamespace(parameters=SimpleNamespace(properties=dict.fromkeys(self._properties, {})))


def _context(user_id="alice", session_id="sess-1"):
    session = SimpleNamespace(id=session_id, user_id=user_id, state={})
    return SimpleNamespace(_invocation_context=SimpleNamespace(session=session), state={})


def _run(tool, args, context=None):
    """Drive the callback. The repo has no pytest-asyncio, so sibling tests run
    the loop by hand (see test_hitl_agenttool_scope.py)."""
    plugin = SessionScopePlugin()
    result = asyncio.run(plugin.before_tool_callback(
        tool=tool, tool_args=args, tool_context=context or _context()
    ))
    assert result is None, "the plugin must never replace a tool result"
    return args


@pytest.mark.parametrize("tool_cls", [_McpTool, _NativeTool])
def test_scope_is_injected_for_a_tool_that_declares_it(tool_cls):
    tool = tool_cls("visualize_molecule", ["smiles", "user_id", "session_id"])
    args = _run(tool, {"smiles": "CCO"})
    assert args == {"smiles": "CCO", "user_id": "alice", "session_id": "sess-1"}


def test_a_tool_that_does_not_declare_the_pair_is_left_alone():
    """Injecting into a tool without these parameters would break the call."""
    tool = _McpTool("name2smiles", ["mol"])
    assert _run(tool, {"mol": "aspirin"}) == {"mol": "aspirin"}


def test_an_explicit_value_from_the_caller_wins():
    """A deliberate cross-session read must keep working."""
    tool = _McpTool("list_artifacts", ["user_id", "session_id"])
    args = _run(tool, {"user_id": "bob", "session_id": "other"})
    assert args == {"user_id": "bob", "session_id": "other"}


def test_only_the_missing_half_is_filled():
    tool = _McpTool("t", ["user_id", "session_id"])
    args = _run(tool, {"user_id": "bob"})
    assert args == {"user_id": "bob", "session_id": "sess-1"}


def test_ids_are_made_to_fit_the_vault_key_layout():
    """The vault rejects an id outside ^[a-zA-Z0-9_-]{1,64}$ and builds every
    key from the pair, so send something it accepts."""
    tool = _McpTool("t", ["user_id", "session_id"])
    args = _run(tool, {}, _context(user_id="alice@example.com", session_id="a/b c"))
    # A dot is outside the vault regex too, so it goes as well.
    assert args == {"user_id": "alice_example_com", "session_id": "a_b_c"}


def test_an_unreadable_tool_or_context_never_breaks_the_call():
    args = _run(SimpleNamespace(name="odd"), {"x": 1})
    assert args == {"x": 1}
