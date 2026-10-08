"""Push the remote sandbox workspace into S3 at the end of a run.

``collect_artifacts`` walks ``workspace/ws_<session>`` on the local disk. With
``CODE_EXEC__URL`` set, the files sit on the code-exec host instead, the walk
finds nothing, and the report loses every figure the run drew.

The code-exec API is submit and result only. It cannot serve a file. So the file
has to leave from the inside: the framework mints a presigned PUT link with the
vault, and the sandbox uploads to S3 with it. The vault itself stays private to
CoScientist — the signed URL points at S3.

The upload runs through ``python3`` and ``urllib``, not ``curl``. The code-exec
image (``docker/Dockerfile``) is ``python:3.12-slim`` plus build-essential, vim
and git. It has no curl, and it always has python3.

Nothing here raises. A sync that fails leaves the report exactly as it is today.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
from typing import Any, Dict, List, Optional, Set

from CoScientist.config import get_settings
from CoScientist.reporting import artifact_index
from CoScientist.reporting.collect import (
    _IMAGE_EXTS,
    _MAX_WORKSPACE_FILES,
    _TABLE_EXTS,
    _WORKSPACE_SKIP_DIRS,
    _looks_like,
)
from CoScientist.graph.session_scope import session_key
from CoScientist.tools.session_scope_plugin import _safe_id
from CoScientist.tools.vault_client import call_vault, vault_url

logger = logging.getLogger(__name__)

# The same key the CoderToolset pins its sandbox under. Read it, never mint one:
# no id in state means no coder step ran, and there is nothing to sync.
_WORKSPACE_STATE_KEY = "coder_workspace_id"

# The vault filename rule: ^[a-zA-Z0-9_.-]{1,128}$ (vault_server._FILENAME_RE).
_FILENAME_SUB = re.compile(r"[^a-zA-Z0-9_.-]+")
_MAX_FILENAME = 128

_LIST_TIMEOUT = 30
_UPLOAD_TIMEOUT = 180

# Read the file, PUT it, print nothing. No headers: the vault deliberately signs
# no content type, so any header it did not sign would break the signature.
_PUT_SNIPPET = (
    "import sys,urllib.request;"
    "d=open(sys.argv[1],'rb').read();"
    "r=urllib.request.Request(sys.argv[2],data=d,method='PUT');"
    "urllib.request.urlopen(r,timeout=120)"
)


def _workspace_id(state: Dict[str, Any]) -> Optional[str]:
    """The sandbox this session used, by the same priority the CoderToolset uses.

    Note this is NOT ``ws_<session_id>``. Every CoderAgent delegation runs in its
    own AgentTool sub-session with a random id, so the toolset pins one id in
    session state and reuses it. Reading the session id here would name a
    workspace that never existed.
    """
    # An explicit pin (the A2A shared workspace) wins, from either source, and
    # the toolset never writes it to state. Reading state alone would miss it.
    fixed = get_settings().web.coder_workspace_id or os.getenv("CODER_WORKSPACE_ID")
    if fixed:
        return f"ws_{re.sub(r'[^A-Za-z0-9_-]', '', fixed)[:48] or 'shared'}"
    existing = state.get(_WORKSPACE_STATE_KEY)
    return str(existing) if existing else None


def _flatten(rel_path: str) -> str:
    """``plots/loss.png`` -> ``plots_loss.png``.

    The vault key is ``ephemeral/<user>/<session>/workspace/<filename>``. A
    sandbox often holds two files with one basename (``plots/loss.png`` and
    ``runs/2/loss.png``), and a bare basename would let the second upload
    overwrite the first. The vault regex does accept a slash here, but a nested
    key would put the workspace two levels deep for no gain.

    The result must match the vault filename rule, ``^[a-zA-Z0-9_.-]{1,128}$``.
    A sandbox writes names the rule rejects — ``roc curve (fold 1).png`` — and
    the vault answers with an error the caller can only log.
    """
    flat = _FILENAME_SUB.sub("_", rel_path.lstrip("./").replace("/", "_")).strip("_")
    if len(flat) <= _MAX_FILENAME:
        return flat or "artifact"
    # Keep the extension: the report sorts a figure from a table by it.
    stem, dot, ext = flat.rpartition(".")
    if dot and len(ext) < 16:
        return stem[: _MAX_FILENAME - len(ext) - 1] + "." + ext
    return flat[:_MAX_FILENAME]


def _is_pruned(rel_path: str, repo_roots: List[str]) -> bool:
    parts = rel_path.split("/")
    if any(p in _WORKSPACE_SKIP_DIRS for p in parts):
        return True
    if any(p.endswith((".dist-info", ".egg-info")) for p in parts):
        return True
    # A cloned repo brings its own example images. collect.py drops the whole
    # subtree when it sees a .git inside, and so does this.
    return any(rel_path.startswith(root + "/") for root in repo_roots)


async def _list_files(toolset, workspace_id: str) -> List[str]:
    """Every figure and table in the sandbox, workspace-relative, pruned."""
    repos = await toolset._run_sync(
        "find . -type d -name .git 2>/dev/null", workspace_id, max_wait=_LIST_TIMEOUT
    )
    if repos.get("status") != "success":
        # Without this list nothing prunes a cloned library, and its bundled
        # example images would fill the report. Sync nothing rather than that.
        logger.warning("workspace sync: cannot probe %s for repositories (%s)",
                       workspace_id, repos.get("stderr"))
        return []
    repo_roots = [
        line.rstrip("/").removesuffix("/.git").lstrip("./")
        for line in (repos.get("stdout") or "").splitlines()
        if line.strip()
    ]

    listed = await toolset._run_sync(
        "find . -type f 2>/dev/null", workspace_id, max_wait=_LIST_TIMEOUT
    )
    if listed.get("status") != "success":
        logger.warning("workspace sync: cannot list %s (%s)",
                       workspace_id, listed.get("stderr"))
        return []

    figures, tables = [], []
    for line in sorted((listed.get("stdout") or "").splitlines()):
        rel = line.strip().lstrip("./")
        if not rel or _is_pruned(rel, repo_roots):
            continue
        if _looks_like(rel, _IMAGE_EXTS) and len(figures) < _MAX_WORKSPACE_FILES:
            figures.append(rel)
        elif _looks_like(rel, _TABLE_EXTS) and len(tables) < _MAX_WORKSPACE_FILES:
            tables.append(rel)
    return figures + tables


async def _upload(toolset, workspace_id: str, rel_path: str,
                  scope: tuple) -> Optional[Dict[str, Any]]:
    """Mint a link, push the file from the sandbox, return the index entry."""
    # The scope is passed explicitly. get_upload_link requires user_id and
    # session_id, and SessionScopePlugin fills those only at the ADK tool
    # boundary — this call does not go through one.
    link = await call_vault(
        "get_upload_link", user_id=scope[0], session_id=scope[1],
        filename=_flatten(rel_path), feature="workspace",
    )
    if not link or not link.get("upload_url"):
        return None

    command = "python3 -c {} {} {}".format(
        shlex.quote(_PUT_SNIPPET), shlex.quote(rel_path), shlex.quote(link["upload_url"])
    )
    res = await toolset._run_sync(command, workspace_id, max_wait=_UPLOAD_TIMEOUT)
    if res.get("status") != "success":
        logger.warning("workspace sync: upload of %s failed (%s)",
                       rel_path, res.get("stderr"))
        return None

    # No ``url``. The upload link is a PUT capability and cannot fetch anything,
    # and a download link minted now would expire before the report reads it.
    # The collector mints a fresh one from the key through ``resolve_url``.
    return {
        "bucket": link.get("bucket"),
        "s3_key": link.get("s3_key"),
        "tool": "workspace",
        "label": rel_path,
    }


async def sync_workspace_to_s3(tool_context: Any, state: Dict[str, Any]) -> Set[str]:
    """Upload the remote sandbox artifacts and index them.

    Returns the workspace-relative paths that reached S3. The collector skips
    exactly those in its disk walk, so a file whose upload failed is still
    collected from disk when the two share a host.
    """
    settings = get_settings()
    if not settings.code_exec.url:
        # Local mode. The files are already on this disk, and collect.py walks
        # them. Uploading would put every figure in the report twice.
        return set()
    if not vault_url():
        logger.info("workspace sync: MCP__VAULT_URL is not set, skipping")
        return set()

    workspace_id = _workspace_id(state)
    if not workspace_id:
        return set()

    from CoScientist.tools.coder_tools.coder_tools import CoderToolset

    toolset = CoderToolset()
    files = await _list_files(toolset, workspace_id)
    if not files:
        return set()

    # The same pair the vault builds its key from, sanitized the way
    # SessionScopePlugin sanitizes it for every other vault call.
    user_id, session_id = session_key(tool_context)
    scope = (_safe_id(user_id, "unknown_user"), _safe_id(session_id, "unknown_session"))

    entries, uploaded = [], set()
    for rel in files:
        entry = await _upload(toolset, workspace_id, rel, scope)
        if entry:
            entries.append(entry)
            uploaded.add(rel)

    artifact_index.record(entries, tool_context)
    logger.info("workspace sync: uploaded %d of %d file(s) from %s",
                len(uploaded), len(files), workspace_id)
    return uploaded


__all__ = ["sync_workspace_to_s3"]
