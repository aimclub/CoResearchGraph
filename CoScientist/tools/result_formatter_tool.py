"""In-process ``format_results`` tool for the Result Aggregator agent.

Collects the run's figures and data tables into the per-run report folder and
returns ready-to-embed markdown blocks. Runs in-process (not via the
result-aggregator MCP server) so it can read ADK session state and the sandbox
workspace directly; it needs no running container.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from google.adk.tools import FunctionTool
from google.adk.tools.tool_context import ToolContext

from CoScientist.config.report import ReportConfig
from CoScientist.graph.session_scope import session_key
from CoScientist.reporting.collect import collect_artifacts
from CoScientist.tools.vault_client import call_vault_sync
from CoScientist.tools.workspace_sync import sync_workspace_to_s3
from CoScientist.utils.s3_refs import split_s3_uri

logger = logging.getLogger(__name__)


def _session_id(tool_context: ToolContext) -> str:
    inv = (getattr(tool_context, "_invocation_context", None)
           or getattr(tool_context, "invocation_context", None))
    session = getattr(inv, "session", None) if inv is not None else None
    sid = getattr(session, "id", None) if session else None
    return sid or "default_session"


def _graph_nodes(tool_context: ToolContext) -> list:
    """Research-graph nodes (with full, untruncated attrs) for the current
    session, so artifact URLs an agent committed to Evidence nodes can be
    downloaded. Best-effort — never break report generation if the graph is
    disabled/empty."""
    try:
        from CoScientist.graph.research.agent_tools import get_research_graph
        graph = get_research_graph(tool_context)
        return graph.full().get("nodes", []) or []
    except Exception as exc:  # noqa: BLE001
        logger.info("format_results: no research graph nodes (%s)", exc)
        return []


def _state_to_dict(state: Any) -> Dict[str, Any]:
    """ADK ``tool_context.state`` is a ``State`` wrapper (merged session + delta),
    not a plain dict — ``dict(state)`` mis-reads it as a sequence. Convert safely."""
    if state is None:
        return {}
    if isinstance(state, dict):
        return dict(state)
    to_dict = getattr(state, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            pass
    try:
        return {k: state[k] for k in state.keys()}
    except Exception:
        return dict(getattr(state, "_value", {}) or {})


def _download_link(uri: str) -> Optional[str]:
    """A fresh download URL for an ``s3://bucket/key``.

    A presigned URL lives one hour. The index caches the one that came back with
    the artifact, so a long run — or one that was restarted — reaches the report
    holding dead links. The object is still there, and the vault mints a new URL
    for it.

    ``collect_artifacts`` stays synchronous, so this cannot await. The whole
    collection runs in a worker thread instead (see below), which is why
    ``call_vault_sync`` is safe here: that thread has no running loop.

    The vault resolves the key against its own bucket, and ``get_download_link``
    takes no bucket. A deployment runs one bucket, so this holds. An artifact
    from a server pointed at a different bucket would resolve to nothing, and
    the vault reports that as an error the caller logs.
    """
    split = split_s3_uri(uri)
    if not split:
        return None
    result = call_vault_sync("get_download_link", s3_key=split[1])
    return (result or {}).get("presigned_url")


async def format_results(tool_context: ToolContext) -> Dict[str, Any]:
    """Collect every figure and data table this run produced into the report
    folder and return markdown blocks (image embeds + tables) to embed verbatim
    in the final report. Call this FIRST, before writing the report."""
    state = _state_to_dict(getattr(tool_context, "state", {}))
    session_id = _session_id(tool_context)
    cfg = ReportConfig.from_mapping(state.get("report_config"))

    # With a remote code-exec host the sandbox files are not on this disk, so
    # collect_artifacts would find nothing there. Push them to S3 first. This has
    # to run BEFORE collection, not at finalize: a figure reaches the report only
    # through the call below.
    synced = set()
    try:
        synced = await sync_workspace_to_s3(tool_context, state)
    except Exception as exc:  # noqa: BLE001 - a failed sync must not stop a report
        logger.warning("format_results: workspace sync failed (%s)", exc)

    # Off the event loop. Collection downloads every artifact over the network,
    # and each dead link costs a vault round trip on top. On the loop thread all
    # of that would stall every other agent and every open websocket.
    result = await asyncio.to_thread(
        collect_artifacts,
        session_id=session_id,
        state=state,
        reports_root=cfg.reports_root,
        graph_nodes=_graph_nodes(tool_context),
        # The capture plugin writes the index under the same scope the graph
        # uses, which is the public web session even inside an AgentTool child.
        index_key=session_key(tool_context),
        resolve_url=_download_link,
        # What the sync uploaded arrives through the artifact index. The walk
        # would find the same files again whenever the code-exec server runs on
        # this host, because both sides use code_exec.workspace_root. Naming them
        # keeps a file whose upload failed reachable from disk.
        synced_files=synced,
    )
    logger.info(
        "format_results: session=%s figures=%d tables=%d",
        session_id, len(result["figures"]), len(result["tables"]),
    )
    return {
        "status": "success",
        "report_dir": result["report_dir"],
        "figures_count": len(result["figures"]),
        "tables_count": len(result["tables"]),
        "formatted_markdown": result["blocks_markdown"],
    }


# Registered under the "result_formatter" tool key (see assembly/bindings.py).
result_formatter_tool = FunctionTool(format_results)

# Back-compat alias: assembly bindings import ``result_formatter_toolset_instance``.
result_formatter_toolset_instance = result_formatter_tool


class ResultFormatterToolset:  # kept for import compatibility; no longer used
    """Deprecated: the formatter is now an in-process FunctionTool."""


__all__ = [
    "format_results",
    "result_formatter_tool",
    "result_formatter_toolset_instance",
    "ResultFormatterToolset",
]
