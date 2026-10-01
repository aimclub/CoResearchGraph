"""Risk tiers for Work Order tools and tool calls.

A Work Order is confirmed by risk (see hitl/work_order_tools.py):

  read         looking things up — the human is informed, the run goes on;
  compute      doing work in a sandbox or on a shared resource — a veto window;
  side_effect  something that outlives the run or leaves the machine (installs,
               pushes, deletions, long jobs, sharing links) — a blocking review.

Tiers are keyed by the REAL tool names the bindings attach. An unknown tool is
treated as ``compute``: guessing "read" for a tool nobody classified would let it
through without a human ever seeing it. The system's ``internal_tools`` (see
system.yaml) are never tiered: a Work Order leaves them out of its tools.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Iterable


class Tier(str, Enum):
    READ = "read"
    COMPUTE = "compute"
    SIDE_EFFECT = "side_effect"


_RANK = {Tier.READ: 0, Tier.COMPUTE: 1, Tier.SIDE_EFFECT: 2}


def max_tier(tiers: Iterable[Tier]) -> Tier:
    return max(tiers, key=lambda t: _RANK[t], default=Tier.READ)


class SideEffectKind(str, Enum):
    PACKAGE_INSTALL = "package_install"
    NETWORK_DOWNLOAD = "network_download"
    GIT_WRITE = "git_write"
    LONG_JOB = "long_job"
    FILE_DELETE = "file_delete"
    EXTERNAL_SHARE = "external_share"


TOOL_TIERS: dict[str, Tier] = {
    # web / literature
    "tavily_search": Tier.READ,
    "tavily_extract": Tier.READ,
    "tavily_crawl": Tier.COMPUTE,
    "search_papers": Tier.READ,
    "download_papers_from_search": Tier.COMPUTE,
    "explore_scientific_database": Tier.READ,
    # The old name of the one above. PR 367 renamed it; the prompt still
    # offers it as the fallback for a server that predates the rename, and an
    # unlisted tool is priced COMPUTE — an approval prompt for a RAG read.
    "explore_scientific_database": Tier.READ,
    "explore_my_papers": Tier.READ,
    # medical
    "search_pubmed": Tier.READ,
    "get_pico": Tier.READ,
    "get_study_taxonomy": Tier.READ,
    "analyze_medical_image": Tier.COMPUTE,
    # graphs / tasks
    "get_active_tasks": Tier.READ,
    "get_agents_info": Tier.READ,
    # sandbox / coder
    "read_file": Tier.READ,
    "list_directory": Tier.READ,
    "check_job": Tier.READ,
    "list_sandbox_files": Tier.READ,
    "check_sandbox_task": Tier.READ,
    "execute_bash": Tier.COMPUTE,
    "write_file": Tier.COMPUTE,
    "run_sandbox_task": Tier.COMPUTE,
    "validate_dataset": Tier.COMPUTE,
    "validate_training": Tier.COMPUTE,
    "get_download_link": Tier.READ,
    "install_package": Tier.SIDE_EFFECT,
    "get_upload_link": Tier.SIDE_EFFECT,
}

# Side effect a tool has by its very nature, whatever its arguments.
TOOL_SIDE_EFFECTS: dict[str, SideEffectKind] = {
    "install_package": SideEffectKind.PACKAGE_INSTALL,
    "get_upload_link": SideEffectKind.EXTERNAL_SHARE,
}

# Reading the agent's own context — allowed before a Work Order is declared, so
# the agent can orient itself and plan from what is already known. Graph reads
# are not listed: they are internal tools and pass anyway.
ORIENTATION_TOOLS = frozenset({
    "get_active_tasks",
    "get_agents_info",
    "list_directory",
    "list_sandbox_files",
})

# Never blocked: the Work Order protocol itself and the human channel. Other
# system tools (bookkeeping, waiting on a job) are listed in the system YAML as
# `internal_tools` and join this set per agent — see exempt_tools().
EXEMPT_TOOLS = frozenset({
    "declare_work_order",
    "update_work_order",
    "update_work_step",
    "submit_work_report",
    "request_approval",
    "request_selection",
    # ADK's own tool for an agent with an output schema AND tools: it carries the
    # final structured answer, not work.
    "set_model_response",
})


def exempt_tools(internal_tools: Iterable[str] = ()) -> frozenset:
    """Tools a Work Order neither blocks nor shows: the protocol plus the
    system's ``internal_tools``."""
    return EXEMPT_TOOLS | frozenset(internal_tools)


def register_tool_tiers(tiers: dict[str, Tier]) -> None:
    """Add the tiers of a profile's own tools (unlisted tools are COMPUTE)."""
    TOOL_TIERS.update(tiers)


def tool_tier(tool_name: str) -> Tier:
    return TOOL_TIERS.get(tool_name, Tier.COMPUTE)


def order_tier(planned_tools: Iterable[str], side_effect_kinds: Iterable[Any]) -> Tier:
    """The tier a Work Order is confirmed at: its riskiest declared element."""
    tiers = [tool_tier(t) for t in planned_tools]
    if any(True for _ in side_effect_kinds):
        tiers.append(Tier.SIDE_EFFECT)
    return max_tier(tiers)
