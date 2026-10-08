"""
CoScientist Module - A multiagent system for solving scientific tasks.
"""

from CoScientist.main import (
    CoScientistManager,
    create_manager
)

from CoScientist.tools import FedotMASToolset

from CoScientist.agents import (
    orchestrator_agent,
    hypotheses_agent, 
    research_agent, 
    fedot_agent,
    tool_retriever_agent,
    task_execution_agent,
    tool_websearcher_agent,
    tool_agent
)

from CoScientist.storage import RetrievalFinalResult, RetrievalToolResult

from CoScientist.hitl import (
    HITLAction,
    HITLRequest,
    HITLResponse,
    AbstractHITLHandler,
    ConsoleHITLHandler,
    HITLToolset,
)

__version__ = "1.0.0"

__all__ = [
    # Main classes
    "CoScientistManager",
    # Models
    "RetrievalFinalResult",
    "RetrievalToolResult",
    # Tools
    "FedotMASToolset",
    # Agents
    "orchestrator_agent",
    "hypotheses_agent",
    "research_agent",
    "fedot_agent",
    "tool_retriever_agent",
    "task_execution_agent",
    'tool_websearcher_agent',
    "tool_agent",
    # HITL
    "HITLAction",
    "HITLRequest",
    "HITLResponse",
    "AbstractHITLHandler",
    "ConsoleHITLHandler",
    "HITLToolset",
    # Functions
    "create_manager",
]
