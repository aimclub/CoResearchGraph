"""Toolset module."""
# Must run before any MCP toolset is used: fail-fast backport for truncated SSE
# frames from remote MCP servers (see mcp_patches docstring).
import CoScientist.tools.mcp_patches  # noqa: F401

from CoScientist.tools.fedotmas_tools import FedotMASToolset, fedot_toolset_instance
from CoScientist.tools.research_tools import (
    websearch_toolset_instance,
    paper_analysis_toolset_instance,
    papers_search_toolset_instance,
    vault_toolset_instance,
)
from CoScientist.tools.retrieval_tools import RetrievalToolSet, retrieval_toolset_instance
from CoScientist.tools.servers_web_search import search_mcp_servers
from CoScientist.tools.med_tools import med_toolset_instance
from CoScientist.tools.coder_tools.coder_tools import CoderToolset, coder_toolset_instance
from CoScientist.tools.task_tracker import TaskTrackerToolset, task_tracker_instance
from CoScientist.tools.result_formatter_tool import (
    ResultFormatterToolset,
    result_formatter_tool,
    result_formatter_toolset_instance,
)
from CoScientist.tools.dynamic_tools import DynamicMCPToolset, dynamic_mcp_toolset_instance
from CoScientist.tools.alembic_tools import ALEMBIC_TOOLS

__all__ = [
    "FedotMASToolset",
    "fedot_toolset_instance",
    "DynamicMCPToolset",
    "dynamic_mcp_toolset_instance",
    "websearch_toolset_instance",
    "paper_analysis_toolset_instance",
    "papers_search_toolset_instance",
    "vault_toolset_instance",
    "RetrievalToolSet",
    "retrieval_toolset_instance",
    "search_mcp_servers",
    "med_toolset_instance",
    "CoderToolset",
    "coder_toolset_instance",
    "TaskTrackerToolset",
    "task_tracker_instance",
    "ResultFormatterToolset",
    "result_formatter_tool",
    "result_formatter_toolset_instance",
    "ALEMBIC_TOOLS",
]
