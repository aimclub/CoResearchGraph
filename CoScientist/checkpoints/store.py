"""Local checkpoint storage: one zip bundle per checkpoint.

Layout (root configurable via CHECKPOINTS__DIR, default ./checkpoints_data):

    checkpoints_data/
      <run_id>/
        ckpt_20260721T140211_T2_after_hypotheses_a3f9.zip
        ...

Inside each zip: ``manifest.json`` + ``blobs/sha256-<hash>``. Blobs are named
by content hash so a future MinIO/platform backend can reuse the same
addressing; the 5-method surface (put/save, load, list, latest, bundle_path)
is the seam where that backend slots in.

Writes are atomic (tmp file + os.replace): a crash mid-save leaves either the
previous complete set of bundles or a stray ``*.tmp`` — never a torn zip.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from CoScientist.checkpoints.model import CheckpointManifest

logger = logging.getLogger(__name__)

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(name: str) -> str:
    return _SAFE.sub("_", name)[:120] or "run"


def new_checkpoint_id(label: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"ckpt_{stamp}_{_safe(label)}_{uuid.uuid4().hex[:8]}"


class LocalZipStore:
    def __init__(self, root: Optional[str] = None) -> None:
        if root is None:
            from CoScientist.config import get_settings

            root = get_settings().checkpoints.dir
        self.root = Path(root)

    # ── write ────────────────────────────────────────────────────────────────
    def save(self, manifest: CheckpointManifest, parts: Dict[str, bytes]) -> CheckpointManifest:
        """Write one bundle. ``parts`` maps logical names to raw bytes; the
        blob map in the manifest is filled here (content-addressed)."""
        run_dir = self.root / _safe(manifest.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        blobs: Dict[str, Tuple[str, bytes]] = {}
        for name, data in parts.items():
            digest = hashlib.sha256(data).hexdigest()
            blobs[name] = (f"blobs/sha256-{digest}", data)
        manifest.blobs = {name: path for name, (path, _) in blobs.items()}

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(
                "manifest.json",
                manifest.model_dump_json(indent=2),
            )
            written = set()
            for path, data in blobs.values():
                if path in written:  # identical content stored once
                    continue
                zf.writestr(path, data)
                written.add(path)

        final = run_dir / f"{manifest.checkpoint_id}.zip"
        tmp = final.with_suffix(".zip.tmp")
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, final)
        logger.info(
            "Checkpoint %s (%s) saved: %s (%d KB)",
            manifest.checkpoint_id, manifest.label, final, final.stat().st_size // 1024,
        )
        return manifest

    # ── read ─────────────────────────────────────────────────────────────────
    def _find(self, checkpoint_id: str) -> Optional[Path]:
        if not self.root.exists():
            return None
        target = f"{_safe(checkpoint_id)}.zip"
        for run_dir in self.root.iterdir():
            if run_dir.is_dir():
                candidate = run_dir / target
                if candidate.exists():
                    return candidate
        return None

    def bundle_path(self, checkpoint_id: str) -> Optional[Path]:
        return self._find(checkpoint_id)

    def load(self, checkpoint_id: str) -> Tuple[CheckpointManifest, Dict[str, bytes]]:
        """Return (manifest, {logical_name: bytes})."""
        path = self._find(checkpoint_id)
        if path is None:
            raise FileNotFoundError(f"checkpoint {checkpoint_id} not found under {self.root}")
        with zipfile.ZipFile(path) as zf:
            manifest = CheckpointManifest.model_validate_json(zf.read("manifest.json"))
            parts = {name: zf.read(blob) for name, blob in manifest.blobs.items()}
        return manifest, parts

    def list(self, run_id: Optional[str] = None) -> List[dict]:
        """Compact listing (id, label, run, ts, parent), newest first.

        Rebuilt from the bundles themselves — no separate index to corrupt.
        """
        out: List[dict] = []
        if not self.root.exists():
            return out
        run_dirs = (
            [self.root / _safe(run_id)] if run_id else
            [d for d in self.root.iterdir() if d.is_dir()]
        )
        for run_dir in run_dirs:
            if not run_dir.is_dir():
                continue
            for path in run_dir.glob("*.zip"):
                try:
                    with zipfile.ZipFile(path) as zf:
                        m = json.loads(zf.read("manifest.json"))
                    out.append({
                        "checkpoint_id": m["checkpoint_id"],
                        "label": m["label"],
                        "run_id": m["run_id"],
                        "created_at": m["created_at"],
                        "parent_checkpoint_id": m.get("parent_checkpoint_id"),
                        "hitl_pending": bool(m.get("hitl_pending")),
                        "size_kb": path.stat().st_size // 1024,
                    })
                except Exception as exc:  # noqa: BLE001 — one bad bundle must not hide the rest
                    logger.warning("Unreadable checkpoint bundle %s: %s", path, exc)
        out.sort(key=lambda r: r["created_at"], reverse=True)
        return out

    def latest(self, run_id: str) -> Optional[str]:
        rows = self.list(run_id)
        return rows[0]["checkpoint_id"] if rows else None


_default_store: Optional[LocalZipStore] = None


def get_default_store() -> LocalZipStore:
    global _default_store
    if _default_store is None:
        _default_store = LocalZipStore()
    return _default_store
