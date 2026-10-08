"""Verification tools the CoderAgent can call to self-check its artifacts.

These wrap the deterministic validators and resolve paths inside the coder's
per-session sandbox workspace. ``validate_dataset`` also records a passing path
so the training gate (gate_plugin) can see a real dataset exists.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.tools import BaseTool, FunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.base_toolset import BaseToolset
from google.adk.agents.readonly_context import ReadonlyContext

from CoScientist.verify.molecular_dataset import validate_molecular_dataset
from CoScientist.verify.training import validate_training

VALIDATED_MARKER = ".validated_datasets"


def resolve_ws_path(tool_context: Optional[ToolContext], path: str) -> str:
    """Resolve a (possibly relative) path against the coder's sandbox workspace."""
    if os.path.isabs(path):
        return path
    try:
        from CoScientist.tools.coder_tools import CoderToolset, _CFG
        ws = CoderToolset._workspace_id(tool_context)
        return str(Path(_CFG.workspace_root) / ws / path)
    except Exception:
        return path


def workspace_dir(tool_context: Optional[ToolContext]) -> Optional[Path]:
    try:
        from CoScientist.tools.coder_tools import CoderToolset, _CFG
        ws = CoderToolset._workspace_id(tool_context)
        return Path(_CFG.workspace_root) / ws
    except Exception:
        return None


def _record_validated(tool_context: Optional[ToolContext], real_path: str) -> None:
    wd = workspace_dir(tool_context)
    if wd is None:
        return
    try:
        wd.mkdir(parents=True, exist_ok=True)
        marker = wd / VALIDATED_MARKER
        existing = set()
        if marker.exists():
            existing = set(marker.read_text(encoding="utf-8").split("\n"))
        existing.add(real_path)
        marker.write_text("\n".join(sorted(x for x in existing if x)), encoding="utf-8")
    except Exception:
        pass


class VerifyToolset(BaseToolset):
    """Deterministic self-check tools for the CoderAgent."""

    def __init__(self, prefix: str = "verify_"):
        super().__init__()
        self.tool_name_prefix = prefix

    def get_tools(self, readonly_context: Optional[ReadonlyContext] = None) -> List[BaseTool]:
        return [FunctionTool(self.validate_dataset), FunctionTool(self.validate_training)]

    async def close(self) -> None:
        pass

    async def validate_dataset(self, path: str, tool_context: ToolContext = None) -> Dict[str, Any]:
        """Deterministically check that a dataset file contains REAL, diverse
        molecules (RDKit-valid SMILES + a fitness/property column). Call this on
        your training dataset BEFORE training — training is blocked until a real
        dataset validates. Toy/placeholder data (integers, one repeated molecule,
        a synthetic fallback) is rejected.

        Args:
            path: path to the dataset CSV (relative to your sandbox is fine).

        Returns:
            {ok, reasons, n_valid_smiles, n_unique_valid, smiles_col, has_fitness, sample}.
        """
        real = resolve_ws_path(tool_context, path)
        res = validate_molecular_dataset(real)
        if res.get("ok"):
            _record_validated(tool_context, real)
        return res

    async def validate_training(self, checkpoint_path: str, loss_log: str = "",
                                tool_context: ToolContext = None) -> Dict[str, Any]:
        """Deterministically check a training result: a real saved checkpoint and
        a loss that actually decreased.

        Args:
            checkpoint_path: path to the saved model (.pt/.pth).
            loss_log: the training stdout/log text (or a path to it) with per-epoch loss.
        """
        cp = resolve_ws_path(tool_context, checkpoint_path)
        ll = resolve_ws_path(tool_context, loss_log) if loss_log else ""
        return validate_training(cp, ll)


verify_toolset = VerifyToolset()
