"""Research Context Graph — the typed, schema-validated blackboard agents share.

Implements developer/research_context_graph_spec.md: agents write typed
epistemic nodes (Hypothesis, Evidence, Conclusion, …) under per-agent write
permissions instead of exchanging dialog history; the orchestrator routes work
via named trigger queries; workers receive context slices, not the whole graph.

Distinct from the sibling execution graph (CoScientist.graph.*), which is an
auto-recorded observational trace. This package is deliberately import-light —
the ADK toolsets live in .agent_tools and are imported lazily by the assembly
bindings.
"""
from CoScientist.graph.research.store import (
    ResearchGraphStore,
    get_research_graph,
    research_graph,
)

__all__ = ["ResearchGraphStore", "get_research_graph", "research_graph"]
