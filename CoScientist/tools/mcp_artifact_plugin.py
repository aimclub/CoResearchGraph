"""Generic capture of artifact links returned by MCP tools.

Many MCP tools (e.g. the tox-antitargets suite) render a figure/table server-side
and return a *presigned URL* to it inside the tool result — commonly under a
``metadata.figure.artifact`` key or a ``*_presigned_url`` key. That link only
lives in the tool result envelope: with the Result Aggregator running graph-first
(``include_contents: none``) it never reaches the report unless it is captured.

This App-level plugin fires at every tool-call boundary and stashes each artifact
URL it finds into ``state["mcp_artifacts"]`` (a ``*_artifacts`` key the report
collector already downloads). It must run BEFORE the tool-result truncation plugin
so it still sees the full, untruncated URL.

Distinct from :mod:`CoScientist.tools.fedot_artifact_plugin`, which captures S3
CSV links *inside* a nested FEDOT.MAS run; this one is generic and global.
"""
from __future__ import annotations

import logging
from typing import Any

from google.adk.plugins import BasePlugin

from CoScientist.reporting.artifact_index import record
from CoScientist.reporting.collect import (
    _IMAGE_EXTS,
    _TABLE_EXTS,
    _looks_like,
    find_artifact_urls,
)
from CoScientist.utils.s3_refs import find_s3_artifacts

logger = logging.getLogger(__name__)

_STATE_KEY = "mcp_artifacts"


def _is_report_artifact(key: str) -> bool:
    """Whether this object belongs in a report at all."""
    return _looks_like(key, _IMAGE_EXTS + _TABLE_EXTS)


class McpArtifactCapturePlugin(BasePlugin):
    """Capture artifact (figure/table) URLs from tool results into session state."""

    def __init__(self, name: str = "mcp_artifact_capture") -> None:
        super().__init__(name)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):  # noqa: ANN001
        try:
            urls = find_artifact_urls(result)
            artifacts = find_s3_artifacts(result)
        except Exception:  # noqa: BLE001 — capture must never break a tool call
            return None
        if not urls and not artifacts:
            return None

        # Durable copy first. Session state lives in an InMemorySessionService and
        # does not survive a restart, so the file on disk is what the report finds
        # afterwards.
        self._record_index(tool, tool_context, urls, artifacts)

        if not urls:
            return None
        try:
            existing = list(tool_context.state.get(_STATE_KEY) or [])
            seen = {a.get("url") for a in existing if isinstance(a, dict)}
            tool_name = getattr(tool, "name", None)
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                existing.append({"url": url, "tool": tool_name})
            tool_context.state[_STATE_KEY] = existing
            logger.info(
                "mcp_artifact_capture: %s → %d artifact URL(s) (%d total)",
                tool_name, len(urls), len(existing),
            )
        except Exception:  # noqa: BLE001
            pass
        return None  # never mutate the tool result itself

    @staticmethod
    def _record_index(tool, tool_context, urls, artifacts) -> None:
        """Append this call's artifacts to the on-disk index of the session."""
        tool_name = getattr(tool, "name", None)
        # An upload link is a PUT capability. It cannot fetch the object, and the
        # object may not even exist yet — the agent was only handed somewhere to
        # put one. Keep the reference, drop the URL, and let the collector mint a
        # download link once there is something to download.
        is_upload = tool_name == "get_upload_link"
        entries = [
            {
                "bucket": a["bucket"],
                "s3_key": a["s3_key"],
                "tool": tool_name,
                "label": a["s3_key"].rsplit("/", 1)[-1],
                # A capability with an expiry, kept so the report can download
                # inside the same run. The bucket and key above are the reference.
                "url": None if is_upload else a.get("url"),
            }
            for a in artifacts
            # A worker parks whatever it likes in the vault — a checkpoint, a
            # pickle. The report holds figures and tables, and an unknown
            # extension there is collected as a table.
            if not is_upload or _is_report_artifact(a["s3_key"])
        ]
        # A server that returns a bare presigned URL and no key. Keep the URL so
        # nothing is lost, and change 2 gives these servers the full contract.
        claimed = {a.get("url") for a in artifacts if a.get("url")}
        entries.extend(
            {"bucket": None, "s3_key": None, "tool": tool_name, "label": "artifact", "url": url}
            for url in urls if url not in claimed
        )
        record(entries, tool_context)
