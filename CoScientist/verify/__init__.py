"""Deterministic, system-run validators for agent deliverables.

The system does NOT trust the coder's self-reported results — it independently
verifies the actual artifacts with code (not an LLM judge). A fabricated dataset
(toy integers, one repeated row, non-molecules) fails validation, so fabricating
gains the agent nothing: downstream steps are gated on real artifacts.
"""
from CoScientist.verify.molecular_dataset import validate_molecular_dataset
from CoScientist.verify.training import validate_training

__all__ = ["validate_molecular_dataset", "validate_training"]
