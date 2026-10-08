"""ArtifactGatePlugin — a deterministic gate that blocks the CoderAgent from
TRAINING on a fabricated dataset.

When execute_bash is about to run a training command, the gate independently
checks that a REAL molecular dataset exists in the sandbox (RDKit-valid, diverse
SMILES). If the only data is toy/placeholder/degenerate — the fabrication pattern
we keep catching — the command is refused with actionable reasons, so faking the
dataset buys the agent nothing.

Enforcement lives here (a plugin), not in a prompt the model can ignore. Disable
with ARTIFACT_GATE=0.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.verify.molecular_dataset import validate_molecular_dataset
from CoScientist.verify.tools import workspace_dir, VALIDATED_MARKER

# A command counts as "training" if it runs a train*.py / known training script,
# a `-m ...train` module, or calls .fit(. Data-generation / extraction commands
# (GOLEM run_experiment, trajectory extraction) do NOT match, so they run freely.
_TRAIN_RE = re.compile(
    r"python\d?\b[^\n;|&]*?\b\S*train[\w-]*\.py"
    r"|python\d?\b[^\n;|&]*?-m\s+\S*train\w*"
    r"|\btrain_model\b|\btrain_transformer\b|\btrain_and_generate\b|\brun_train\b"
    r"|\.fit\(",
    re.IGNORECASE,
)
_CSV_RE = re.compile(r"[\w./\-]+\.csv")
_EXCLUDE = ("/MCPhub/", "/GOLEM/test", "/.git/", "/node_modules/", "/site-packages/")
_SCAN_SUBDIRS = ("", "data", "datasets", "dataset", "optimization_histories", "trajectories", "results")


def _enabled() -> bool:
    return os.getenv("ARTIFACT_GATE", "1") not in ("0", "false", "False")


def _is_training(cmd: str) -> bool:
    return bool(_TRAIN_RE.search(cmd or ""))


def _candidate_csvs(cmd: str, wd: Optional[Path]) -> List[str]:
    """CSV paths named in the command, then a bounded scan of the workspace."""
    found: List[str] = []
    if wd is not None:
        for tok in _CSV_RE.findall(cmd or ""):
            p = tok if os.path.isabs(tok) else str(wd / tok)
            if os.path.exists(p):
                found.append(p)
        for sub in _SCAN_SUBDIRS:
            base = wd / sub if sub else wd
            if not base.exists():
                continue
            try:
                it = base.rglob("*.csv") if sub in ("results", "optimization_histories") else base.glob("*.csv")
                for p in it:
                    sp = str(p)
                    if any(x in sp for x in _EXCLUDE):
                        continue
                    found.append(sp)
                    if len(found) > 40:
                        break
            except Exception:
                pass
    # de-dup, keep order
    return list(dict.fromkeys(found))


def _marker_has_real(wd: Optional[Path]) -> bool:
    if wd is None:
        return False
    marker = wd / VALIDATED_MARKER
    if not marker.exists():
        return False
    try:
        for line in marker.read_text(encoding="utf-8").split("\n"):
            line = line.strip()
            if line and os.path.exists(line) and validate_molecular_dataset(line).get("ok"):
                return True
    except Exception:
        pass
    return False


class ArtifactGatePlugin(BasePlugin):
    """Refuse a training run unless a real molecular dataset exists."""

    def __init__(self, name: str = "artifact_gate") -> None:
        super().__init__(name=name)

    async def before_tool_callback(self, *, tool, tool_args, tool_context) -> Optional[Dict[str, Any]]:
        if not _enabled():
            return None
        if getattr(tool, "name", "") and not str(tool.name).endswith("execute_bash"):
            return None
        cmd = (tool_args or {}).get("command", "") if isinstance(tool_args, dict) else ""
        if not cmd or not _is_training(cmd):
            return None

        wd = workspace_dir(tool_context)
        # a previously-validated real dataset is enough
        if _marker_has_real(wd):
            return None

        candidates = _candidate_csvs(cmd, wd)
        checked = []
        for p in candidates:
            r = validate_molecular_dataset(p)
            checked.append((p, r))
            if r.get("ok"):
                return None  # a real dataset exists → allow training

        # No real dataset found → BLOCK.
        detail = ""
        if checked:
            worst = checked[0][1]
            detail = (f" Nearest candidate '{os.path.basename(checked[0][0])}' failed: "
                      f"{'; '.join(worst.get('reasons', [])[:1])}")
        return {
            "status": "blocked",
            "blocked_by": "artifact_gate",
            "message": (
                "TRAINING BLOCKED: no REAL molecular dataset was found in the "
                "workspace." + detail + " A training dataset must contain RDKit-valid, "
                "diverse SMILES with a fitness/SA column. Toy/placeholder data "
                "(integers like [1,2,3,4,5], one repeated molecule, or a hand-made "
                "synthetic fallback) is refused. Produce REAL data first: run GOLEM's "
                "`run_experiment(molecule_search_setup, ..., metrics=['norm_sa_score'], "
                "save_history=True)` to get trajectories, parse the real SMILES + SA "
                "into a CSV, then call `validate_dataset(path)` — once it passes, "
                "training will be allowed."
            ),
        }


artifact_gate_plugin = ArtifactGatePlugin()
