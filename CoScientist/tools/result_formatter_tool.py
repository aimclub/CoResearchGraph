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
from CoScientist.tools.sandbox_sweep import sweep_sandbox_workspace
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


#: The directory the collector walks for a session's workspace. Named here
#: because the sweep stages into it and the collector reads it — the two have
#: to agree, and agreeing by default value is how they came apart before.
_WORKSPACE_ROOT = "workspace"


def _indexed_filenames(tool_context: ToolContext) -> set:
    """Names already in the artifact index, so the sweep does not duplicate them.

    A figure the sandbox agent published itself is collected from its S3
    reference. Staging the same file from the container would put it in the
    report twice, under two labels.
    """
    try:
        from CoScientist.reporting.artifact_index import load as load_index

        user_id, session_id = session_key(tool_context)
        return {
            str(entry.get("label") or "").rsplit("/", 1)[-1]
            for entry in load_index(session_id, user_id)
            if entry.get("label")
        }
    except Exception:  # noqa: BLE001 - no index yet is normal
        return set()


async def format_results(tool_context: ToolContext) -> Dict[str, Any]:
    """Collect every figure, data table and downloadable file this run produced
    into the report folder and return markdown blocks (image embeds, tables,
    download links) to embed verbatim in the final report. Call this FIRST,
    before writing the report."""
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

    # The OpenHands sandbox is a third place again: another host, its own API,
    # and nothing on this disk. Until now a report saw only the files that
    # sandbox agent chose to upload mid-task — so a run whose plots were never
    # published reached the reader with none of them. Stage the rest into the
    # directory the collector walks, so everything downstream is unchanged.
    try:
        await asyncio.to_thread(
            sweep_sandbox_workspace, tool_context, state, session_id,
            _WORKSPACE_ROOT, _indexed_filenames(tool_context),
        )
    except Exception as exc:  # noqa: BLE001 - as above, attachments are not the report
        logger.warning("format_results: sandbox sweep failed (%s)", exc)

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
        workspace_root=_WORKSPACE_ROOT,
    )
    markdown = result["blocks_markdown"]
    # The counts the model is shown come from the BLOCKS, not from the files on
    # disk. Reporting "figures_count: 3" beside an empty string is what misled a
    # live run: the model concluded the figures must be somewhere and invented
    # three `figures/<name>.png` paths that no route serves. A number here now
    # means "this many are embedded in the markdown below", which is a promise
    # the markdown itself keeps.
    blocks = result.get("block_counts") or {}
    counts = (
        int(blocks.get("figures", len(result["figures"]))),
        int(blocks.get("tables", len(result["tables"]))),
        int(blocks.get("files", len(result["files"]))),
    )
    on_disk = (len(result["figures"]), len(result["tables"]), len(result["files"]))
    logger.info(
        "format_results: session=%s embedded=%d/%d/%d collected=%d/%d/%d markdown=%d chars",
        session_id, *counts, *on_disk, len(markdown),
    )
    payload = {
        "status": "success",
        "report_dir": result["report_dir"],
        "figures_count": counts[0],
        "tables_count": counts[1],
        "files_count": counts[2],
        "formatted_markdown": markdown,
    }
    # Should the two ever diverge again, say so in the envelope rather than
    # leaving the model to guess what happened to the difference.
    if on_disk != counts:
        payload["status"] = "partial"
        payload["collected_but_not_embedded"] = {
            "figures": on_disk[0] - counts[0],
            "tables": on_disk[1] - counts[1],
            "files": on_disk[2] - counts[2],
        }
        payload["note"] = (
            "Some collected artifacts could not be turned into embeddable "
            "markdown. Do NOT construct links for them yourself — say in the "
            "report that they could not be embedded."
        )
        logger.warning(
            "format_results: session=%s collected %d/%d/%d but embedded %d/%d/%d",
            session_id, *on_disk, *counts,
        )
    return payload


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
