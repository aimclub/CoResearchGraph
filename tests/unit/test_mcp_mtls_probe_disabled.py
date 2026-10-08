"""Importing the MCP toolset must turn ADK's Google-mTLS probe off.

ADK runs google.auth.default() + configure_mtls_channel() before every MCP
connection unless GOOGLE_API_USE_CLIENT_CERTIFICATE is "false". For plain MCP
servers that probe can only fail, and it costs ~12s per connect while the
session lock is held. A deployment that really does use GCP client certificates
must still be able to say so.
"""

import os

from CoScientist.tools.dynamic_tools import _disable_adk_mcp_mtls_probe

_VAR = "GOOGLE_API_USE_CLIENT_CERTIFICATE"


def test_the_probe_is_off_once_the_toolset_module_is_imported():
    assert os.environ[_VAR] == "false"


def test_the_default_is_applied_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv(_VAR, raising=False)

    _disable_adk_mcp_mtls_probe()

    assert os.environ[_VAR] == "false"


def test_an_explicit_choice_is_left_alone(monkeypatch):
    """A deployment that really uses GCP client certificates keeps its setting."""
    monkeypatch.setenv(_VAR, "true")

    _disable_adk_mcp_mtls_probe()

    assert os.environ[_VAR] == "true"
