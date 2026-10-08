"""YAML-driven assembly of the CoScientist multi-agent system.

The single source of truth for the system layout is ``CoScientist/agents/system.yaml``:
which agents exist, who is subordinate to whom, which tools / callbacks / prompts
each agent uses, whether it uses HITL, and how it is exposed over A2A.

Modules:
  * registry   — name -> object registries (tools, callbacks, prompts, classes, ...)
  * schema     — pydantic models for the YAML + loading/validation
  * prompting  — PromptContext: renders the unified prompt sections (<<TOOLS>>, ...)
  * bindings   — registers every concrete tool/callback/class/schema in the registries
  * assembler  — builds the agent tree from the validated config

Typical use:
    from CoScientist.assembly import build_system
    system = build_system()                       # in-process sub-agents
    system = build_system(remote_subagents=True)  # sub-agents over A2A
"""
from CoScientist.assembly.assembler import (
    AgentSystem,
    build_system,
    delegatable_agent_names,
    load_config,
)

__all__ = [
    "AgentSystem",
    "build_system",
    "delegatable_agent_names",
    "load_config",
]
