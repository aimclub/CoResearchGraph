"""Coder toolsets: the per-session code workspace and the OpenHands sandbox.

* :mod:`coder_tools` — bash / file I/O / package installs in the session's own
  code-exec workspace.
* :mod:`sandbox_tools` — ADK tools delegating heavy jobs to the OpenHands
  sandbox (a separate machine).
* :mod:`openhands_sandbox` — the framework-agnostic sandbox client the adapter
  is built on.

The names of the code workspace are re-exported here so that the historic
``from CoScientist.tools.coder_tools import CoderToolset`` keeps working now
that this is a package. The sandbox modules are deliberately NOT re-exported:
importing them constructs settings and must stay opt-in (the assembly registry
imports them lazily).
"""

from CoScientist.tools.coder_tools.coder_tools import (
    CoderToolset,
    _WORKSPACE_STATE_KEY,
    coder_toolset,
    coder_toolset_instance,
    seed_coder_workspace,
)

__all__ = [
    "CoderToolset",
    "coder_toolset",
    "coder_toolset_instance",
    "seed_coder_workspace",
    "_WORKSPACE_STATE_KEY",
]
