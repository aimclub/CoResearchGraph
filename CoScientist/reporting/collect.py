"""Deterministic artifact collection for the final report.

Gathers every figure and data table a run produced — from captured artifacts in
session state and from files left in the sandbox workspace — into the per-run
report folder, and returns ready-to-embed markdown blocks. General on purpose:
it knows nothing about any specific paper or task, only about *artifacts*.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from CoScientist.reporting.artifact_index import load as load_artifact_index
from CoScientist.utils.s3_refs import s3_uri

from CoScientist.reporting.s3_upload import upload_and_presign

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")
_TABLE_EXTS = (".csv", ".tsv")
# Source material a tool pulled in, not a result the run produced. The papers
# search server uploads every PDF it downloads and returns a link to each one.
# The report shows figures and tables, so these only add noise to it.
_SOURCE_EXTS = (".pdf", ".doc", ".docx", ".zip", ".tar", ".gz")
_MAX_TABLE_ROWS = 15

# Workspace scan guards: dependency/VCS/cache dirs that carry bundled example
# assets (never run outputs), and a cap so a stray clone can't flood the report.
_WORKSPACE_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "node_modules", "site-packages", "__pycache__",
    ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache", ".cache",
    ".ipynb_checkpoints", "build", "dist",
})
_MAX_WORKSPACE_FILES = 40

# Where a collected file came from, written beside the report folder for
# finalize_report. Collection knows the bucket and the key and then throws
# them away for a local path, and finalize runs later and needs them back to
# promote the object out of ephemeral/. Dot-prefixed: it is plumbing between
# two stages, not part of the deliverable.
SOURCES_FILENAME = ".artifact_sources.json"

# Any http(s) URL whose path ends in a known media extension (the presigned query
# string is optional). Bulletproof fallback: matches an artifact link embedded in
# ANY stringified payload — a Pydantic tool-result object, a Python-repr blob an
# agent stored on a graph node, or prose — regardless of structure.
_MEDIA_URL_RE = re.compile(
    r"""https?://[^\s"'<>]+?\.(?:png|jpe?g|svg|gif|webp|csv|tsv|pdf)(?:\?[^\s"'<>]*)?""",
    re.IGNORECASE,
)


def report_dir_for(session_id: str, reports_root: Path | str = "logs/reports") -> Path:
    """The per-run report folder for a session."""
    return Path(reports_root) / session_id


def _url_filename(url: str, default_ext: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path).split("?")[0]
    if not name or "." not in name:
        name = f"{uuid.uuid4().hex}{default_ext}"
    return name


def _looks_like(url_or_name: str, exts: tuple) -> bool:
    low = url_or_name.lower()
    return any(low.endswith(e) or f"{e}?" in low or f"{e}&" in low for e in exts)


_ARTIFACT_KEY_SUFFIXES = ("artifact", "presigned_url")


def _is_artifact_url(key: str, value: str) -> bool:
    """A tool/graph attr value is a downloadable artifact when it is an http(s)
    URL that either sits under an artifact-ish key (``artifact`` /
    ``*presigned_url``) or whose path ends in a known media extension. This keeps
    figure/table links (e.g. MinIO/S3 presigned PNGs) while ignoring plain
    reference links (PubChem pages, DOIs)."""
    if not (isinstance(value, str) and value.startswith(("http://", "https://"))):
        return False
    if isinstance(key, str) and key.lower().endswith(_ARTIFACT_KEY_SUFFIXES):
        return True
    path = value.split("?", 1)[0].lower()  # drop presigned query string
    return path.endswith(_IMAGE_EXTS + _TABLE_EXTS)


def find_artifact_urls(obj: Any, _out: Optional[List[str]] = None) -> List[str]:
    """Recursively collect artifact URLs from a nested structure (a tool result
    envelope or a graph node's ``attrs``). Parses JSON-looking strings on the way
    down so a URL nested inside a ``content[].text`` blob is still found."""
    out = [] if _out is None else _out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_artifact_url(k, v):
                out.append(v)
            else:
                find_artifact_urls(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_artifact_urls(v, out)
    elif isinstance(obj, str):
        if "http" in obj:
            # Structured JSON-looking payloads: parse and recurse (keeps keyed
            # `artifact` links even without a media extension).
            if obj.lstrip()[:1] in "{[":
                try:
                    import json
                    find_artifact_urls(json.loads(obj), out)
                except Exception:
                    pass
            # Regex fallback: media URLs embedded in any string (Python-repr
            # blobs, prose, non-JSON) that the parse above would miss.
            out.extend(_MEDIA_URL_RE.findall(obj))
    elif obj is not None and not isinstance(obj, (int, float, bool)):
        # Non-JSON object (e.g. a Pydantic CallToolResult): scan its repr.
        out.extend(_MEDIA_URL_RE.findall(str(obj)))
    # De-dup while preserving order.
    if _out is None:
        seen: set = set()
        return [u for u in out if not (u in seen or seen.add(u))]
    return out


def _table_to_markdown(path: Path) -> Optional[str]:
    """Render the first rows of a CSV/TSV as a markdown table, or None on failure.

    Uses the stdlib ``csv`` module so it never depends on pandas/tabulate being
    installed in the host environment.
    """
    import csv

    def clean(v: Any) -> str:
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ")

    try:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            rows = list(csv.reader(f, delimiter=delimiter))
    except Exception as exc:  # pragma: no cover - depends on file contents
        logger.warning("collect: could not parse table %s (%s)", path, exc)
        return None
    if not rows:
        return None

    header, body = rows[0], rows[1:]
    ncol = len(header)
    shown = body[:_MAX_TABLE_ROWS]
    lines = ["| " + " | ".join(clean(c) for c in header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    for r in shown:
        r = (r + [""] * ncol)[:ncol]
        lines.append("| " + " | ".join(clean(c) for c in r) + " |")
    md = "\n".join(lines)
    if len(body) > len(shown):
        md += f"\n\n*… showing first {len(shown)} of {len(body)} rows.*"
    return md


def _download(url: str, dest: Path, timeout: int = 30) -> bool:
    try:
        import html
        import requests

        # MCP servers often HTML-escape the presigned URL (``&amp;`` for ``&``);
        # downloading that literal string corrupts the AWS SigV4 query params and
        # MinIO answers 403. Unescape before the request.
        url = html.unescape(url)
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as exc:
        logger.warning("collect: failed to download %s (%s)", url, exc)
        return False


def _artifact_link(dest: Path, session_id: str, kind: str, fallback: str) -> str:
    """Presigned S3 URL for a collected artifact, else the local fallback path.

    ``kind`` is ``figures`` or ``tables``; the object lands under
    ``reports/<session_id>/<kind>/<name>``. With S3 off or on any upload
    failure the returned markdown is byte-identical to the local-path form.
    """
    url = upload_and_presign(dest, f"reports/{session_id}/{kind}")
    return url or fallback


def collect_artifacts(
    session_id: str,
    state: Optional[Dict[str, Any]] = None,
    reports_root: Path | str = "logs/reports",
    workspace_root: Path | str = "workspace",
    graph_nodes: Optional[List[Dict[str, Any]]] = None,
    index_key: Optional[tuple] = None,
    resolve_url: Optional[Callable[[str], Optional[str]]] = None,
    synced_files: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Copy/download run artifacts into the report folder; return markdown blocks.

    Args:
        session_id:    used to name the report folder and locate the workspace.
        state:         ADK session state; ``fedot_artifacts`` (and any list under
                       a ``*_artifacts`` key holding ``{"url": ...}`` dicts) are
                       downloaded.
        reports_root:  base dir for per-run report folders.
        workspace_root: base dir for sandbox workspaces (``<root>/ws_<session>``).
        graph_nodes:   research-graph nodes (``{"id","type","attrs"}``); any
                       artifact URL recorded on a node's ``attrs`` (e.g. a MinIO
                       presigned figure URL an agent committed to an Evidence node)
                       is downloaded too.
        index_key:     ``(user_id, session_id)`` scope of the on-disk artifact
                       index. It is not always the ``session_id`` above: inside
                       an AgentTool child the raw session id is a transient
                       random one, while the index keeps the public web scope.
                       Without it the index is found by ``session_id`` alone.
        resolve_url:   turns an ``s3://bucket/key`` into a fresh download URL.
                       A presigned URL cached in the index expires in an hour,
                       so a run that is restarted, or simply slow, arrives here
                       holding dead links. This function is called for those.
                       ``None`` leaves them unresolved, as before.
        synced_files:  workspace-relative paths the vault sync already uploaded.
                       Those arrive through the index, so the walk below skips
                       them. Without this they would be collected twice whenever
                       the code-exec server runs on this host, because both sides
                       use ``code_exec.workspace_root``. A file the sync failed
                       to upload is NOT in this set and is still walked, which is
                       the point of naming them instead of skipping the walk.

    Returns a dict with ``report_dir``, ``figures``, ``tables``, and
    ``blocks_markdown`` (the concatenation the agent should embed).
    """
    state = state or {}
    report_dir = report_dir_for(session_id, reports_root)
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    sections_dir = report_dir / "sections"
    for d in (figures_dir, tables_dir, sections_dir):
        d.mkdir(parents=True, exist_ok=True)

    figure_blocks: List[str] = []
    table_blocks: List[str] = []
    figures: List[str] = []
    tables: List[str] = []

    # 1) Captured artifacts. The on-disk index comes first: it survives a restart,
    #    while session state does not. Session state is the fallback, and both are
    #    read because a run that started before this index existed has state only.
    #    Then any artifact URL an agent recorded on a research-graph node.
    index_user, index_session = index_key or (None, session_id)
    artifact_lists: List[Dict[str, Any]] = list(load_artifact_index(index_session, index_user))
    for key, val in state.items():
        if key == "fedot_artifacts" or (isinstance(key, str) and key.endswith("_artifacts")):
            if isinstance(val, list):
                artifact_lists.extend(a for a in val if isinstance(a, dict))
    for node in (graph_nodes or []):
        if not isinstance(node, dict):
            continue
        label = " ".join(str(node.get(k)) for k in ("id", "type") if node.get(k)).strip()
        for url in find_artifact_urls(node.get("attrs") or {}):
            artifact_lists.append({"url": url, "tool": label or "graph"})
    seen_urls = set()
    seen_refs = set()
    unresolved = 0
    sources: Dict[str, Dict[str, str]] = {}

    def _fresh_url(art: Dict[str, Any]) -> Optional[str]:
        """A download URL for this artifact, minted from the durable reference
        when the cached one is missing or dead."""
        bucket, key = art.get("bucket"), art.get("s3_key")
        if not (resolve_url and bucket and key):
            return None
        try:
            return resolve_url(s3_uri(bucket, key))
        except Exception as exc:  # noqa: BLE001 - a report outranks one figure
            logger.warning("collect: cannot resolve s3://%s/%s (%s)", bucket, key, exc)
            return None

    def _note_source(art: Dict[str, Any], dest: Path) -> None:
        """Remember which object this local file came from, for finalize."""
        bucket, key = art.get("bucket"), art.get("s3_key")
        if bucket and key:
            sources[_rel(dest, report_dir)] = {"bucket": bucket, "s3_key": key}

    for art in artifact_lists:
        # Dedupe on the durable reference where there is one. Two entries for one
        # object can hold two different presigned URLs and still be one file.
        bucket, key = art.get("bucket"), art.get("s3_key")
        ref = s3_uri(bucket, key) if bucket and key else None
        if ref:
            if ref in seen_refs:
                continue
            seen_refs.add(ref)

        url = art.get("url")
        if not url or url in seen_urls:
            # No URL at all, or one already spent on an earlier entry. A durable
            # reference can still produce a working link.
            if url and not ref:
                continue
            url = _fresh_url(art) or url
            if not url:
                if art.get("s3_key"):
                    unresolved += 1
                continue
        seen_urls.add(url)
        if _looks_like(url, _SOURCE_EXTS):
            continue
        label = art.get("tool") or art.get("name") or "artifact"
        if _looks_like(url, _IMAGE_EXTS):
            name = f"{label}_{_url_filename(url, '.png')}"
            dest = figures_dir / name
            if not _download(url, dest):
                # The cached URL expired. Mint one and try once more.
                retry = _fresh_url(art)
                if not (retry and retry != url and _download(retry, dest)):
                    if art.get("s3_key"):
                        unresolved += 1
                    continue
            figures.append(str(dest))
            link = _artifact_link(dest, session_id, "figures", f"figures/{name}")
            figure_blocks.append(f"### {label}\n\n![{label}]({link})")
            _note_source(art, dest)
        else:  # default remote artifacts to tabular
            name = f"{label}_{_url_filename(url, '.csv')}"
            dest = tables_dir / name
            if not _download(url, dest):
                retry = _fresh_url(art)
                if not (retry and retry != url and _download(retry, dest)):
                    if art.get("s3_key"):
                        unresolved += 1
                    continue
            tables.append(str(dest))
            md = _table_to_markdown(dest)
            link = _artifact_link(dest, session_id, "tables", _rel(dest, report_dir))
            head = f"### {label} — [download]({link})"
            table_blocks.append(f"{head}\n\n{md}" if md else head)
            _note_source(art, dest)

    # 2) Files the run itself LEFT in the sandbox workspace. Prune vendored trees
    #    aggressively: a coder step may `git clone` a whole library (e.g. the RDKit
    #    repo) or create a venv into the sandbox, and its bundled example
    #    images/CSVs are NOT run outputs — collecting them buries the real figures.
    workspace_dir = Path(workspace_root) / f"ws_{session_id}"
    already_synced = synced_files or set()
    ws_figures = ws_tables = 0
    if workspace_dir.exists():
        for root, dirs, files in os.walk(workspace_dir):
            # Skip a cloned-repo subtree (a dir that contains .git) and any known
            # dependency/VCS/cache dir — modifying `dirs` in place prunes descent.
            if ".git" in dirs:
                dirs[:] = []
                continue
            dirs[:] = [
                d for d in dirs
                if d not in _WORKSPACE_SKIP_DIRS
                and not d.endswith((".dist-info", ".egg-info"))
            ]
            for fname in sorted(files):
                src = Path(root) / fname
                if _rel(src, workspace_dir) in already_synced:
                    continue  # the vault sync sent this one; it comes back above
                stem = src.stem
                if _looks_like(fname, _IMAGE_EXTS):
                    if ws_figures >= _MAX_WORKSPACE_FILES:
                        continue
                    dest = figures_dir / fname
                    _safe_copy(src, dest)
                    figures.append(str(dest))
                    link = _artifact_link(dest, session_id, "figures", f"figures/{fname}")
                    figure_blocks.append(f"### {stem}\n\n![{stem}]({link})")
                    ws_figures += 1
                elif _looks_like(fname, _TABLE_EXTS):
                    if ws_tables >= _MAX_WORKSPACE_FILES:
                        continue
                    dest = tables_dir / fname
                    _safe_copy(src, dest)
                    tables.append(str(dest))
                    md = _table_to_markdown(dest)
                    link = _artifact_link(dest, session_id, "tables", f"tables/{fname}")
                    head = f"### {stem} — [download]({link})"
                    table_blocks.append(f"{head}\n\n{md}" if md else head)
                    ws_tables += 1

    # 3) Persist the building blocks as section files (for reference / LaTeX tree).
    if figure_blocks:
        (sections_dir / "figures.md").write_text(
            "## Figures\n\n" + "\n\n".join(figure_blocks) + "\n", encoding="utf-8"
        )
    if table_blocks:
        (sections_dir / "tables.md").write_text(
            "## Data tables\n\n" + "\n\n".join(table_blocks) + "\n", encoding="utf-8"
        )

    # 4) Where each collected file came from. finalize_report reads this to
    #    promote the objects out of ephemeral/ before the lifecycle rule takes
    #    them. It goes on disk rather than into session state because one of the
    #    two finalize call sites (web/app.py) passes no state at all.
    if sources:
        try:
            (report_dir / SOURCES_FILENAME).write_text(
                json.dumps(sources, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 - promotion is not the report
            logger.warning("collect: failed to write %s (%s)", SOURCES_FILENAME, exc)

    blocks: List[str] = []
    if figure_blocks:
        blocks.append("## Figures\n\n" + "\n\n".join(figure_blocks))
    if table_blocks:
        blocks.append("## Data tables\n\n" + "\n\n".join(table_blocks))

    logger.info(
        "collect: session=%s figures=%d tables=%d -> %s",
        session_id, len(figures), len(tables), report_dir,
    )
    if unresolved:
        logger.warning(
            "collect: %d indexed artifact(s) have a key but no usable URL. "
            "They need the vault download tool to reach the report.", unresolved,
        )
    return {
        "report_dir": str(report_dir),
        "figures": figures,
        "tables": tables,
        "blocks_markdown": "\n\n".join(blocks),
    }


def _safe_copy(src: Path, dest: Path) -> None:
    try:
        shutil.copy2(src, dest)
    except Exception as exc:  # pragma: no cover
        logger.warning("collect: failed to copy %s -> %s (%s)", src, dest, exc)


def _rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return path.name


__all__ = ["collect_artifacts", "report_dir_for", "SOURCES_FILENAME"]
