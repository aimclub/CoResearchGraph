"""Questions, hypotheses and deliverables — one statement, one graph, one completion.

The research graph remains the store. This package is the contract for what the
user asked, what the agent only proposed, and whether that ask was actually met.
It does not introduce a second obligation database or a subtype taxonomy.
"""
from CoScientist.requirements.completion import evaluate_fulfillment, evaluate_nodes
from CoScientist.requirements.models import NormalizedStatement
from CoScientist.requirements.routing import (
    merge_statements,
    modules_for,
    normalize_statement,
    retire_parts,
)

__all__ = [
    "NormalizedStatement",
    "evaluate_fulfillment",
    "evaluate_nodes",
    "merge_statements",
    "modules_for",
    "normalize_statement",
    "retire_parts",
]
