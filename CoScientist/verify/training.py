"""Deterministic validator for a training result: a real checkpoint whose loss
actually went down. Rejects "I trained the model" claims with no saved weights
or a flat/absent loss curve.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional


def _extract_losses(text: str) -> List[float]:
    # matches "Epoch 3/10, Loss: 0.0245" / "epoch 3 loss 0.0245" / "loss=0.02"
    vals = []
    for m in re.finditer(r"loss[^0-9\-]{0,8}(-?\d+\.\d+)", text, re.IGNORECASE):
        try:
            vals.append(float(m.group(1)))
        except ValueError:
            pass
    return vals


def validate_training(
    checkpoint_path: str,
    loss_log: Optional[str] = None,
    min_epochs: int = 1,
) -> Dict[str, Any]:
    """Validate a training run.

    Args:
        checkpoint_path: path to the saved model (.pt/.pth/.ckpt) — must exist and
            be non-trivially sized.
        loss_log: text (stdout or a log file's contents/path) containing the
            per-epoch loss; must show a decreasing loss over >= min_epochs.
    """
    out: Dict[str, Any] = {
        "ok": False, "checkpoint": checkpoint_path, "reasons": [],
        "checkpoint_bytes": 0, "n_loss_points": 0,
        "loss_start": None, "loss_end": None, "decreased": False,
    }

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        out["reasons"].append(f"no checkpoint file at {checkpoint_path}")
    else:
        sz = os.path.getsize(checkpoint_path)
        out["checkpoint_bytes"] = sz
        if sz < 1024:
            out["reasons"].append(f"checkpoint suspiciously small ({sz} bytes)")

    text = ""
    if loss_log:
        if os.path.exists(loss_log):
            try:
                text = open(loss_log, encoding="utf-8", errors="replace").read()
            except Exception:  # noqa: BLE001
                text = loss_log
        else:
            text = loss_log
    losses = _extract_losses(text)
    out["n_loss_points"] = len(losses)
    if len(losses) < min_epochs:
        out["reasons"].append(
            f"loss logged for < {min_epochs} epoch(s) — no evidence training ran.")
    else:
        out["loss_start"], out["loss_end"] = losses[0], losses[-1]
        out["decreased"] = losses[-1] < losses[0]
        if not out["decreased"]:
            out["reasons"].append(
                f"loss did not decrease ({losses[0]:.4f} -> {losses[-1]:.4f}).")

    out["ok"] = not out["reasons"]
    return out
