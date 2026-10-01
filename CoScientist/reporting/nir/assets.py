"""Turn the run's collected figures into the ``assets`` map the MCP accepts.

The normcontrol MCP takes images one way only: ``assets`` maps a bare filename
to base64 PNG/JPEG bytes, and every ``figure.path`` in the document must be one
of those keys. Local paths, ``generated:`` paths and URLs are all rejected by
name, so a presigned S3 link — the form every other consumer in this codebase
uses — is not an option here.

That costs nothing, because ``collect_artifacts`` has already downloaded each
figure into ``<report_dir>/figures/``. This module reads those files, discards
the ones the server would refuse, renames what it keeps to satisfy the server's
ASCII pattern, and stops before the size budget.

Nothing here raises. A figure that cannot be encoded is dropped and *named* in
the result: a report that silently lost an illustration reads exactly like one
that never had it, and the author has no way to tell.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from CoScientist.reporting.nir import contract

logger = logging.getLogger(__name__)

#: Everything that is not an ASCII letter, digit, dot, dash or underscore. The
#: collector names files after the tool that produced them, so a Cyrillic or
#: space-bearing name reaches us intact and would fail the server's check.
_UNSAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Leading characters the server's pattern forbids (it demands alphanumeric).
_LEADING_RE = re.compile(r"^[^A-Za-z0-9]+")

_MAX_STEM = 96


@dataclass
class AssetBundle:
    """Encoded images plus an honest account of what did not make it."""

    #: asset filename -> base64 payload, ready to pass as ``assets``.
    assets: Dict[str, str] = field(default_factory=dict)
    #: original local path -> asset filename, so the builder can point a
    #: ``figure`` block at the right key.
    names: Dict[str, str] = field(default_factory=dict)
    #: human-readable reasons, one per dropped file.
    dropped: List[str] = field(default_factory=list)
    encoded_bytes: int = 0

    def name_for(self, local_path: Path | str) -> str | None:
        return self.names.get(str(local_path))


def _probe(data: bytes, suffix: str) -> str | None:
    """The image format Pillow reads, when it matches ``suffix``. Else None.

    The server runs this same check and answers ``invalid_asset``; doing it here
    turns a failed render into a dropped figure with a reason. Pillow ships with
    the environment but is not a declared dependency of every deployment, so an
    ImportError degrades to a magic-byte check rather than failing the report.
    """
    expected = contract.ASSET_FORMATS
    try:
        from io import BytesIO

        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            fmt = image.format
            image.verify()
    except ImportError:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            fmt = "PNG"
        elif data.startswith(b"\xff\xd8\xff"):
            fmt = "JPEG"
        else:
            return None
    except Exception:  # noqa: BLE001 - a broken image is a dropped figure
        return None
    if fmt not in expected or suffix not in expected[fmt]:
        return None
    return fmt


def _safe_name(path: Path, taken: Iterable[str]) -> str:
    """A filename the server's ``ASSET_NAME_RE`` accepts, unique among ``taken``.

    Keeps the extension, because the server checks it against the real format.
    """
    suffix = path.suffix.lower()
    stem = _UNSAFE_RE.sub("_", path.stem)
    stem = _LEADING_RE.sub("", stem)[:_MAX_STEM]
    if not stem:
        stem = "figure"
    used = set(taken)
    candidate = f"{stem}{suffix}"
    index = 2
    while candidate in used:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def build_assets(
    figure_paths: Iterable[Path | str],
    budget_bytes: int = contract.ASSET_BUDGET_BYTES,
) -> AssetBundle:
    """Encode what the MCP will accept, in order, until the budget runs out.

    ``figure_paths`` is the ``figures`` list ``collect_artifacts`` returned (or
    any iterable of local paths). Order is preserved and matters: the caller has
    already decided which illustration belongs where, and the budget is spent
    from the front so the most relevant figures survive a squeeze.
    """
    bundle = AssetBundle()
    for raw in figure_paths:
        path = Path(raw)
        suffix = path.suffix.lower()
        if suffix not in contract.ASSET_EXTENSIONS:
            # SVG/GIF/WEBP reach the report folder (collect.py accepts them) but
            # the DOCX renderer takes PNG and JPEG only.
            bundle.dropped.append(f"{path.name}: формат {suffix or '?'} не поддерживается (нужен PNG или JPEG)")
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            bundle.dropped.append(f"{path.name}: файл не прочитан ({exc})")
            continue
        if _probe(data, suffix) is None:
            bundle.dropped.append(f"{path.name}: содержимое не является корректным PNG/JPEG")
            continue

        encoded = base64.b64encode(data).decode("ascii")
        if bundle.encoded_bytes + len(encoded) > budget_bytes:
            bundle.dropped.append(
                f"{path.name}: не поместился в бюджет вложений "
                f"({budget_bytes // (1024 * 1024)} МБ)"
            )
            continue

        name = _safe_name(path, bundle.assets)
        bundle.assets[name] = encoded
        bundle.names[str(path)] = name
        bundle.encoded_bytes += len(encoded)

    if bundle.dropped:
        logger.info("nir assets: kept %d, dropped %d", len(bundle.assets), len(bundle.dropped))
    return bundle


def request_size(values: Dict, assets: Dict[str, str]) -> int:
    """Bytes the server will measure against ``MAX_FILE_SIZE``.

    It sizes ``json.dumps({'values': ..., 'assets': ...})`` with
    ``ensure_ascii=False``, so Cyrillic prose counts as UTF-8 and not as
    six-byte escapes. Measured the same way here, or the check would pass
    locally and fail on the server.
    """
    try:
        payload = json.dumps(
            {"values": values, "assets": assets}, ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError):
        return contract.MAX_REQUEST_BYTES + 1
    return len(payload.encode("utf-8"))


def fits(values: Dict, assets: Dict[str, str]) -> Tuple[bool, int]:
    """``(within the limit, measured size)``."""
    size = request_size(values, assets)
    return size <= contract.MAX_REQUEST_BYTES, size


__all__ = ["AssetBundle", "build_assets", "request_size", "fits"]
