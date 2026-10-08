"""Run-time report configuration — how the system renders its final deliverable.

Deliberately source-agnostic: the manager takes a :class:`ReportConfig` and does
not care whether it came from CLI args (today) or HTTP params (later). A thin
adapter per surface builds one of these — see :meth:`ReportConfig.from_cli`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

# LaTeX generation modes for the final report:
#   skip       — markdown deliverable only (default; .tex is experimental)
#   standalone — one compilable .tex with a minimal preamble (pdflatex-ready)
#   body       — body fragment only, for pasting into an existing Overleaf project
#   tree       — multi-file LaTeX project: main.tex + \input sections + references.bib
LATEX_MODES = ("skip", "standalone", "body", "tree")


@dataclass
class ReportConfig:
    """Everything the Result Aggregator needs to render the deliverable."""

    latex: str = "skip"
    # Base directory under which per-run report folders are written.
    reports_root: Path = field(default_factory=lambda: Path("logs/reports"))

    def __post_init__(self) -> None:
        if self.latex not in LATEX_MODES:
            raise ValueError(
                f"latex must be one of {LATEX_MODES}, got {self.latex!r}"
            )
        self.reports_root = Path(self.reports_root)

    # ── adapters (surface -> config) ─────────────────────────────────────────
    @classmethod
    def from_cli(cls, args: Any) -> "ReportConfig":
        """Build from an argparse Namespace (or anything with these attrs)."""
        return cls(
            latex=getattr(args, "latex", None) or "skip",
            reports_root=Path(getattr(args, "reports_root", None) or "logs/reports"),
        )

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> "ReportConfig":
        """Build from a dict (e.g. HTTP JSON params). Unknown keys ignored."""
        data = data or {}
        return cls(
            latex=data.get("latex") or "skip",
            reports_root=Path(data.get("reports_root") or "logs/reports"),
        )

    # ── serialisation (config -> session state, for the aggregator tool) ─────
    def to_state(self) -> Dict[str, Any]:
        d = asdict(self)
        d["reports_root"] = str(self.reports_root)
        return d


__all__ = ["ReportConfig", "LATEX_MODES"]
