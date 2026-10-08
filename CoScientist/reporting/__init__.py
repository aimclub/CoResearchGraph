"""In-process reporting: turn a finished run into a folder deliverable.

Two responsibilities, split by *when* they run relative to the aggregator LLM:

* :func:`collect_artifacts` — runs when the aggregator agent calls the
  ``format_results`` tool, mid-turn. It copies/downloads every figure and data
  table produced by the run into the per-run report folder and hands the agent
  ready-to-embed markdown blocks. It does NOT need the LLM's narrative.

* :func:`finalize_report` — runs in the manager driver AFTER the aggregator
  stage finishes, when the assembled narrative markdown exists. It writes
  ``report.md``, renders LaTeX (per :class:`ReportConfig`), and writes
  ``MANIFEST.json`` describing everything on disk.

Kept in-process (rather than behind the result-aggregator MCP server) so it can
read ADK session state and the sandbox workspace directly and never depends on a
running container.
"""
from CoScientist.reporting.collect import collect_artifacts, report_dir_for
from CoScientist.reporting.finalize import finalize_report, RunResult
from CoScientist.reporting.latex import render_latex

__all__ = [
    "collect_artifacts",
    "report_dir_for",
    "finalize_report",
    "RunResult",
    "render_latex",
]
