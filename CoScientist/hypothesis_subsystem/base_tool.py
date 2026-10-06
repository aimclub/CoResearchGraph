"""
Base hypothesis tool contract.

Every hypothesis-generation strategy must subclass BaseHypothesisTool
and implement strategy_type, invoke(), and validate_query().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CoScientist.hypothesis_subsystem.models import HypothesisQuery, ToolResult


class BaseHypothesisTool(ABC):
    """
    Abstract contract for hypothesis-generation strategies.

    Each tool is a self-contained strategy (MooseChem, literature mining,
    GNN-based, retrosynthesis, etc.) that accepts a HypothesisQuery and
    returns a structured ToolResult with Hypothesis objects.
    """

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """Unique strategy identifier used for registration and logging."""
        ...

    @abstractmethod
    async def invoke(self, query: "HypothesisQuery") -> "ToolResult":
        """Execute the hypothesis generation strategy."""
        ...

    @abstractmethod
    def validate_query(self, query: "HypothesisQuery") -> bool:
        """Validate that the query is suitable for this tool."""
        ...

    def describe(self) -> str:
        """Human-readable description for tool selection, logging, and UI display."""
        return (
            f"{self.strategy_type}: "
            f"{self.__class__.__doc__ or 'No description available.'}"
        )[:250]
