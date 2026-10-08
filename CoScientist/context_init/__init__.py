"""Context initialization: draft, confirm (HITL), and seed the research frame.

The 6-layer scientific meta-model is already the schema of the Research Context
Graph (``graph/research/``). This package fills its FRAMING entities on each run:
a pre-stage agent drafts a structured ``ResearchFrame`` from the raw question,
the operator confirms/edits it through a structured web form, and the confirmed
frame is seeded into the graph BEFORE the orchestrator picks a research strategy.
"""
