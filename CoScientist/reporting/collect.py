"""Deterministic artifact collection for the final report.

Gathers every artifact a run produced — figures, data tables and downloadable
files, from captured artifacts in session state and from files left in the
sandbox workspace — into the per-run report folder, and returns ready-to-embed
markdown blocks. General on purpose: it knows nothing about any specific paper
or task, only about *artifacts*.
"""
from __future__ import annotations

import html
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
from CoScientist.utils.report_links import (
    KIND_FIGURE,
    KIND_TABLE,
    artifact_citation,
    artifact_link,
    artifact_markdown,
)
from CoScientist.utils.s3_refs import s3_uri

from CoScientist.reporting.s3_upload import upload_and_ref

logger = logging.getLogger(__name__)

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp")
_TABLE_EXTS = (".csv", ".tsv")
# Downloadable run outputs that are neither figures nor tables: archives,
# checkpoints, a PDF the run produced. The workspace walk collects these into
# the Files section. Indexed artifacts need no such list — a tool returned
# them as results, so they are curated already.
_FILE_EXTS = (".pdf", ".zip", ".tar", ".tar.gz", ".tgz", ".gz",
              ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".onnx",
              ".h5", ".hdf5", ".joblib", ".parquet",
              # Metrics and arrays a run writes beside its figures. Boilerplate
              # that shares these extensions is excluded by name below — a
              # package.json is furniture, a metrics.json is the result.
              ".json", ".npz", ".npy", ".xlsx")
#: Files that carry a result extension and are not results.
_BOILERPLATE_NAMES = frozenset({
    "package.json", "package-lock.json", "tsconfig.json", "composer.json",
    "manifest.json", "meta.json", "base_state.json", "owner_lease.json",
    "tasks.json", "config.json", "settings.json",
})
# Ingest material, never a deliverable: source documents a search or parse
# step pulled in. Everything else the run uploaded lands in Files.
_SOURCE_EXTS = (".doc", ".docx")
# Key markers of bulk source material. The papers server uploads every PDF it
# finds under one prefix; those are inputs to the run, not results of it.
_SOURCE_KEY_MARKERS = ("papers_search_results",)
#: Material that came off a page the run READ. A figure in someone else's
#: article is not an illustration of this study, and putting it in the report's
#: Figures says it is — a journal's own plate, uncredited, in a document the
#: operator hands to someone else. `paper` is the same judgement about the
#: article itself: the marker above catches the papers server's own upload
#: prefix, and this catches the copy the session mirrored, which is
#: content-addressed and has no prefix left to match on.
SOURCE_ASSET_KIND = "source_asset"
_SOURCE_KINDS = frozenset({"paper", SOURCE_ASSET_KIND})

#: The tools that READ the web rather than produce anything. This is the honest
#: discriminator, and it is provenance rather than the shape of a URL: measured
#: over the recorded sessions, 101 of 111 mirrored web artifacts came from these
#: two and 10 came from the analysis tools that actually drew something
#: (`predict_ld50`, `chemical_space_tsne`, `logp_confounder_analysis`, …). No
#: rule over hosts or file names separates those sets — the publisher's CDN
#: serves a 22 007-byte journal banner and a 22 477-byte figure alike.
WEB_READING_TOOLS = ("tavily_search", "tavily_extract", "tavily_crawl")


def is_reading_tool(name: Any) -> bool:
    """Whether this tool brings back what someone else published."""
    return str(name or "") in WEB_READING_TOOLS

# Furniture, not content. Reading one article page brought home eighteen viewer
# icons of 141–720 bytes, two journal logos, a publisher logo, an avatar
# placeholder and an author's photograph — and every one of them would have been
# offered to the reader as an illustration of the run.
#
# The cut is by PATH SEGMENT, not by size, because size does not separate them:
# in that same session a journal banner was 22 007 bytes and a real figure from
# the paper was 22 477. What does separate them is where a publisher keeps each
# — chrome under `/img/`, `/static/`, `/banners/`, `/bundles/`, `/profiles/`;
# content under the article's own path. Note `img` and `images` are different
# segments, and that difference is doing real work here:
#
#   pub.mdpi-res.com/img/journals/separations-logo.png              → chrome
#   pub.mdpi-res.com/separations/…/html/images/separations-…-g001.png → a figure
_CHROME_SEGMENTS = frozenset({
    "img", "static", "banners", "bundles", "profiles", "thumb", "thumbs",
    "icons", "icon", "logos", "assets", "design", "sprites", "avatars",
})
#: A second net, for hosts that do not sort their furniture into directories.
#: Matched as WORDS, not as substrings: `logo` inside `logotype-analysis.png`
#: is a figure's own name, and dropping it would lose real content to a rule
#: meant for a site's letterhead.
_CHROME_NAMES = ("favicon", "sprite", "placeholder", "unknown-user",
                 "logo", "avatar", "banner", "watermark")
_CHROME_NAME_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:" + "|".join(_CHROME_NAMES) + r")(?:[^a-z0-9]|$)")


def _is_page_chrome(url: str) -> bool:
    """A site's own furniture, pulled in with the page that was being read."""
    from urllib.parse import unquote, urlsplit

    try:
        path = unquote(urlsplit(str(url or "")).path).lower()
    except Exception:  # noqa: BLE001
        return False
    if not path:
        return False
    segments = [s for s in path.split("/") if s]
    if not segments:
        return False
    if _CHROME_SEGMENTS.intersection(segments[:-1]):
        return True
    return bool(_CHROME_NAME_RE.search(segments[-1]))


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
#
# The character classes exclude the punctuation that *frames* a URL rather than
# belonging to it. Without that, a link written as markdown — ``![fig](URL)`` —
# hands back ``URL)``, and one written in backticks hands back ``URL` ``. Both
# shapes are in the live indexes: of 82 captured artifacts across six sessions,
# 36 carried a trailing ``)``, ``` ` ```, ``.``, ``,``, ``**`` or ``****``, and
# each one is a link that resolves to nothing.
_MEDIA_URL_RE = re.compile(
    r"""https?://[^\s"'<>)\]}`*]+?\.(?:png|jpe?g|svg|gif|webp|csv|tsv|pdf)(?:\?[^\s"'<>)\]}`*]*)?""",
    re.IGNORECASE,
)

#: Punctuation that can only be prose, never the last character of a URL.
#: Superset of ``report_links._TRAILING_PUNCTUATION``, which guards the same
#: mistake for a narrower pattern.
_URL_TRAILING_JUNK = ".,;:!?*`'\"()[]{}<>"


def clean_url(url: Any) -> str:
    """A captured URL with the prose that framed it removed.

    Applied on capture *and* on read, so the entries already written with a
    trailing ``)`` collapse onto their clean twin the next time the index is
    loaded — no migration, and the duplicate count drops on its own.

    ``html.unescape`` because MCP servers hand back ``&amp;`` between query
    parameters, which breaks the signature. ``_download`` unescapes for the same
    reason; doing it here means the stored string is the one that works.
    """
    text = html.unescape(str(url or "")).strip()
    # A truncated URL is not a URL. `_short()` marks its cut with an ellipsis,
    # and "http://10.3…" has been captured as an artifact that can never resolve.
    if "…" in text:
        return ""
    return text.rstrip(_URL_TRAILING_JUNK)


def _default_reports_root_str() -> str:
    return os.getenv("REPORTS_ROOT") or os.getenv("EXPERIMENTS__REPORTS_DIR") or "logs/reports"


def report_dir_for(session_id: str, reports_root: Path | str | None = None) -> Path:
    """The per-run report folder for a session."""
    root = reports_root or _default_reports_root_str()
    return Path(root) / session_id


def _url_filename(url: str, default_ext: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = os.path.basename(path).split("?")[0]
    if not name or "." not in name:
        name = f"{uuid.uuid4().hex}{default_ext}"
    return name


def _looks_like(url_or_name: str, exts: tuple) -> bool:
    low = url_or_name.lower()
    return any(low.endswith(e) or f"{e}?" in low or f"{e}&" in low for e in exts)


def _unique_dest(folder: Path, name: str, artifact_id: str) -> Path:
    """``folder/name``, disambiguated when two artifacts share a file name.

    Six ``family_outputs.json`` in one run is ordinary — they come from six
    different tasks. Naming the copies after the tool avoided the clash and
    cost the real name; a short suffix from the (content-addressed) id keeps
    both.
    """
    dest = folder / name
    if not dest.exists():
        return dest
    stem, suffix = Path(name).stem, Path(name).suffix
    return folder / f"{stem}-{Path(artifact_id).stem[-8:]}{suffix}"


def _is_session_paperwork(record: Dict[str, Any]) -> bool:
    """What the run wrote ABOUT itself, as opposed to what it produced.

    The chat's document panel (``reporting/documents.py``) keeps its paperwork
    — work orders, agents' reports and answers, the research frame, the plan
    review, a node's write-up — in the same session store as the run's outputs,
    because that store already exports and deduplicates. And ``finalize``
    mirrors ``report.md`` there too. Read as artifacts, they filled a live
    report's Files section: fifteen entries, fourteen of them paperwork and one
    of them a result.

    The kinds come from the documents module itself, so a new document kind is
    excluded here without anyone remembering to add it.
    """
    kind = str(record.get("source_kind") or "")
    if not kind:
        return False
    if kind == "report":            # finalize's copy of report.md
        return True
    try:
        from CoScientist.reporting.documents import KIND_PREFIX, UNLISTED_KINDS
    except Exception:  # noqa: BLE001 — a report is not worth an import error
        KIND_PREFIX, UNLISTED_KINDS = "doc:", frozenset({"node_report"})
    return kind.startswith(KIND_PREFIX) or kind in UNLISTED_KINDS


def _kind_from_name(name: str) -> str:
    """Fallback classification when the store has no record to read."""
    if _looks_like(name, _IMAGE_EXTS):
        return KIND_FIGURE
    if _looks_like(name, _TABLE_EXTS):
        return KIND_TABLE
    return "file"


def _is_source_material(art: Dict[str, Any], url: str) -> bool:
    """Ingest material a search or parse step pulled in, not a run output.

    Matches on what the file was filed as, on the extension (source documents)
    and on the key or URL (the bulk prefix a search server uploads under). A PDF
    the run itself produced matches none of the three and lands in Files like
    any other deliverable.
    """
    if str(art.get("source_kind") or "") in _SOURCE_KINDS:
        return True
    if _looks_like(url, _SOURCE_EXTS):
        return True
    key = str(art.get("s3_key") or "")
    return any(marker in key or marker in url for marker in _SOURCE_KEY_MARKERS)


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
    # Clean, then de-dup while preserving order. Cleaning first is what collapses
    # the five captures of one figure that differ only by the punctuation each
    # was written next to.
    if _out is None:
        seen: set = set()
        cleaned = (clean_url(u) for u in out)
        return [u for u in cleaned if u and not (u in seen or seen.add(u))]
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
    """A durable link for a collected artifact, else the local fallback path.

    ``kind`` is ``figures``, ``tables`` or ``files``; the object lands under
    ``reports/<session_id>/<kind>/<name>``. With S3 off or on any upload
    failure the returned markdown is byte-identical to the local-path form.

    Deliberately ``upload_and_ref`` and not ``upload_and_presign``: the URL a
    presign returns carries a signature that dies in an hour and an endpoint
    only this network can reach, and that string was going straight into report
    prose a person opens later. ``/api/artifact/<bucket>/<key>`` is the same
    object addressed through a route that mints a fresh signature per request.
    """
    reference = upload_and_ref(dest, f"reports/{session_id}/{kind}")
    return artifact_link(*reference) if reference else fallback


def collect_artifacts(
    session_id: str,
    state: Optional[Dict[str, Any]] = None,
    reports_root: Path | str | None = None,
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

    Returns a dict with ``report_dir``, ``figures``, ``tables``, ``files``, and
    ``blocks_markdown`` (the concatenation the agent should embed).
    """
    state = state or {}
    report_dir = report_dir_for(session_id, reports_root)
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    files_dir = report_dir / "files"
    sections_dir = report_dir / "sections"
    for d in (figures_dir, tables_dir, files_dir, sections_dir):
        d.mkdir(parents=True, exist_ok=True)

    figure_blocks: List[str] = []
    table_blocks: List[str] = []
    file_blocks: List[str] = []
    figures: List[str] = []
    tables: List[str] = []
    files: List[str] = []

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

    # 0) What this session already mirrored. These need no network at all: the
    #    bytes are on disk beside the graph, captured while the tool's link was
    #    still alive. Taken first so the URL pass below never re-downloads one.
    skipped: List[Dict[str, Any]] = []
    mirrored_by_url: Dict[str, str] = {}
    mirrored_ids: Set[str] = set()
    try:
        from CoScientist.reporting import session_files

        manifest = session_files.load_manifest(index_session, index_user)
        scope = (index_user or "", index_session or "")
        for artifact_id, record in manifest.items():
            if _is_session_paperwork(record):
                # Claimed, so the URL pass never fetches it either — and not
                # listed as "not collected" when it failed: it was never a
                # result the reader was owed.
                mirrored_ids.add(artifact_id)
                continue
            if record.get("state") != session_files.STATE_STORED:
                # Never silent: a figure that did not survive says why, in the
                # report, instead of simply not being there.
                skipped.append(record)
                continue
            source_url = record.get("source_url")
            if source_url:
                mirrored_by_url[source_url] = artifact_id
            mirrored_ids.add(artifact_id)
            # Registered above, copied below — and a source is registered but
            # not copied. The URL pass has always had this check; this one never
            # did, because until papers were mirrored deliberately nothing that
            # reached the session store was ingest material. Claiming it here
            # (rather than skipping the record outright) is what stops the URL
            # pass from fetching the same paper again by its publisher link.
            if _is_source_material(record, str(source_url or record.get("filename") or "")):
                continue
            local = session_files.resolve_path(scope, artifact_id)
            if local is None:
                continue
            name = Path(str(record.get("filename") or artifact_id)).name
            citation = artifact_citation(scope, artifact_id)
            kind = citation["kind"] if citation else _kind_from_name(name)
            if kind == KIND_FIGURE:
                bucket_dir, sink, blocks = figures_dir, figures, figure_blocks
            elif kind == KIND_TABLE:
                bucket_dir, sink, blocks = tables_dir, tables, table_blocks
            else:
                bucket_dir, sink, blocks = files_dir, files, file_blocks
            # Named after the file, not after the tool that made it. The old
            # `<tool>_<name>` prefix was invented here and returned nowhere, so
            # the on-disk name was unknowable to anything downstream — and it
            # leaks into the GOST report's figure captions, which are derived
            # from file names.
            dest = _unique_dest(bucket_dir, name, artifact_id)
            try:
                bucket_dir.mkdir(parents=True, exist_ok=True)
                if dest.resolve() != local.resolve():
                    shutil.copy2(local, dest)
            except OSError as exc:
                logger.warning("collect: cannot copy %s (%s)", local, exc)
                continue
            sink.append(str(dest))
            # The block, not just the file. Without this the pass reports
            # `figures_count: 3, formatted_markdown: ""` — which is what a live
            # run produced, and the aggregator then invented three
            # `figures/<name>.png` paths that no route serves.
            #
            # Emitted unconditionally, including when the store could not be
            # asked: a CLI run and most of the tests call this with no user in
            # the scope, and making the block conditional on a citation would
            # reproduce that same silent gap for exactly those callers. The
            # invariant is simple — an entry in a path list always has a block.
            caption = str(record.get("label") or "").strip() or Path(name).stem
            if citation:
                block = artifact_markdown(citation, caption)
            else:
                link = (
                    artifact_link(record["bucket"], record["s3_key"])
                    if record.get("bucket") and record.get("s3_key")
                    else _rel(dest, report_dir)
                )
                block = (
                    f"### {caption}\n\n![{caption}]({link})\n\n`{name}`"
                    if kind == KIND_FIGURE
                    else f"### {caption} — [{name}]({link})"
                )
            blocks.append(block)
            # A table is worth reading inline, the way the URL pass below does.
            if kind == KIND_TABLE and (md := _table_to_markdown(dest)):
                blocks[-1] = f"{blocks[-1]}\n\n{md}"
            if record.get("bucket") and record.get("s3_key"):
                sources[_rel(dest, report_dir)] = {
                    "bucket": record["bucket"], "s3_key": record["s3_key"],
                }
        if manifest:
            logger.info(
                "collect: %d mirrored artifact(s), %d not stored",
                len(manifest) - len(skipped), len(skipped),
            )
    except Exception as exc:  # noqa: BLE001 — the old path still works
        logger.warning("collect: could not read the session store (%s)", exc)

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
        # Already taken from the store above, bytes and all. Downloading it
        # again would at best duplicate the file and at worst fail, because the
        # link that worked at capture time has had a run's worth of time to die.
        if url and url in mirrored_by_url:
            continue
        # Against every id the store holds, not just the ones that arrived with
        # a source URL. An artifact the experiment runtime mirrored from a local
        # path has no `source_url`, so it was absent from that map and came
        # through here a second time — thirteen of eighteen records in one live
        # session were exactly that shape.
        if art.get("artifact_id") in mirrored_ids:
            continue
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
        # Both rules, here at the boundary that decides what the reader sees.
        # Refusing chrome at MIRROR time is not enough and was in fact worse
        # than nothing: no mirror record means no entry in `mirrored_by_url`,
        # and that entry is exactly what stops this pass fetching the file
        # again. So the icons were declined once, downloaded a second time from
        # the publisher during collection, and printed as illustrations of the
        # run — while the paper's real figures, which DID mirror, were excluded
        # as source material. The change inverted its own intent.
        if _is_page_chrome(url) or _is_source_material(art, url):
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
        elif _looks_like(url, _TABLE_EXTS):
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
        else:  # anything else the run produced is a downloadable file
            name = f"{label}_{_url_filename(url, '.bin')}"
            dest = files_dir / name
            if not _download(url, dest):
                retry = _fresh_url(art)
                if not (retry and retry != url and _download(retry, dest)):
                    if art.get("s3_key"):
                        unresolved += 1
                    continue
            files.append(str(dest))
            # The object already lives in S3: link it through the artifact
            # route instead of re-uploading the copy we just downloaded.
            if bucket and key:
                link = artifact_link(bucket, key)
            else:
                link = _artifact_link(dest, session_id, "files", f"files/{name}")
            file_blocks.append(f"### {label} — [download]({link})")
            _note_source(art, dest)

    # 2) Files the run itself LEFT in the sandbox workspace. Prune vendored trees
    #    aggressively: a coder step may `git clone` a whole library (e.g. the RDKit
    #    repo) or create a venv into the sandbox, and its bundled example
    #    images/CSVs are NOT run outputs — collecting them buries the real figures.
    workspace_dir = Path(workspace_root) / f"ws_{session_id}"
    already_synced = synced_files or set()
    ws_figures = ws_tables = ws_files = 0
    if workspace_dir.exists():
        for root, dirs, files_on_disk in os.walk(workspace_dir):
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
            for fname in sorted(files_on_disk):
                src = Path(root) / fname
                if _rel(src, workspace_dir) in already_synced:
                    continue  # the vault sync sent this one; it comes back above
                if fname.lower() in _BOILERPLATE_NAMES:
                    continue
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
                elif _looks_like(fname, _FILE_EXTS):
                    if ws_files >= _MAX_WORKSPACE_FILES:
                        continue
                    dest = files_dir / fname
                    _safe_copy(src, dest)
                    files.append(str(dest))
                    link = _artifact_link(dest, session_id, "files", f"files/{fname}")
                    file_blocks.append(f"### {stem} — [download]({link})")
                    ws_files += 1

    # 3) Persist the building blocks as section files (for reference / LaTeX tree).
    if figure_blocks:
        (sections_dir / "figures.md").write_text(
            "## Figures\n\n" + "\n\n".join(figure_blocks) + "\n", encoding="utf-8"
        )
    if table_blocks:
        (sections_dir / "tables.md").write_text(
            "## Data tables\n\n" + "\n\n".join(table_blocks) + "\n", encoding="utf-8"
        )
    if file_blocks:
        (sections_dir / "files.md").write_text(
            "## Files\n\n" + "\n\n".join(file_blocks) + "\n", encoding="utf-8"
        )
    if skipped:
        # An artifact that did not survive is reported, not omitted. A report
        # missing a figure silently is indistinguishable from a run that never
        # produced one, and the reader has no way to tell which happened.
        rows = "\n".join(
            f"- `{r.get('filename') or r.get('artifact_id')}` — {r.get('reason') or r.get('state')}"
            for r in skipped
        )
        (sections_dir / "skipped.md").write_text(
            "## Не собрано\n\n" + rows + "\n", encoding="utf-8"
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
    if file_blocks:
        blocks.append("## Files\n\n" + "\n\n".join(file_blocks))

    logger.info(
        "collect: session=%s figures=%d tables=%d files=%d -> %s",
        session_id, len(figures), len(tables), len(files), report_dir,
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
        "files": files,
        "blocks_markdown": "\n\n".join(blocks),
        # How many artifacts actually made it into the markdown, as opposed to
        # onto disk. The tool envelope reports these, because a caller told
        # "3 figures" while handed an empty string once concluded the figures
        # must be somewhere and invented paths to them.
        "block_counts": {
            "figures": len(figure_blocks),
            "tables": len(table_blocks),
            "files": len(file_blocks),
        },
    }


def _safe_copy(src: Path, dest: Path) -> None:
    try:
        shutil.copy2(src, dest)
    except Exception as exc:  # pragma: no cover
        logger.warning("collect: failed to copy %s -> %s (%s)", src, dest, exc)


def _rel(path: Path, base: Path) -> str:
    """POSIX separators: the result goes into markdown links."""
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


__all__ = ["collect_artifacts", "report_dir_for", "SOURCES_FILENAME"]
