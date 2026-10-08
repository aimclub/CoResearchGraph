"""HTTP-transport wrapper around the generated FastMCP server.

The coder agent emits ``mcp.run()`` (stdio default), which would block
imports. We monkey-patch ``FastMCP.run`` to a no-op for the duration of
the import, then re-arm it and run the server with the HTTP transport so
the container port can be mapped to the host.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

SERVER_PATH = Path(os.environ["SERVER_PATH"]).resolve()
sys.path.insert(0, str(SERVER_PATH.parent))

import fastmcp  # type: ignore  # installed in the generated venv

_orig_run = fastmcp.FastMCP.run
fastmcp.FastMCP.run = lambda *a, **k: None  # type: ignore[assignment]
try:
    spec = importlib.util.spec_from_file_location("server", SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load server module from {SERVER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
finally:
    fastmcp.FastMCP.run = _orig_run  # type: ignore[assignment]

port = int(os.environ.get("MCP_PORT", "8000"))
host = os.environ.get("MCP_HOST", "0.0.0.0")

if not hasattr(module, "mcp"):
    raise RuntimeError(
        f"{SERVER_PATH} does not expose a top-level `mcp` FastMCP instance"
    )

module.mcp.run(transport="streamable-http", host=host, port=port)
