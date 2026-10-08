"""Orchestrator agent that calls sub-agents via the A2A protocol.

Assembled from the SAME system.yaml as the in-process orchestrator
(:mod:`CoScientist.agents`) — identical roster, prompt and critic wiring — but
every subordinate with an ``a2a`` section is attached as a ``RemoteA2aAgent``
(HTTP service) instead of an in-process ``AgentTool``.
"""
from CoScientist.assembly import build_system

orchestrator_a2a_agent = build_system(remote_subagents=True).root
