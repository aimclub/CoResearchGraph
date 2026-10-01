"""Session bundle — export and import complete session state as a ZIP archive.

A session bundle (``*.cossession.zip``) captures every in-memory and on-disk
layer that constitutes a CoScientist session so it can be:

* **saved** to disk and **restored** after a process restart;
* **downloaded** and **shared** with another person who imports it.

Bundle contents
~~~~~~~~~~~~~~~
::

    manifest.json                   — format version, export timestamp, original IDs, title
    adk_session.json                — ADK Session (state + events) via pydantic model_dump
    agent_events.json               — WebRuntime.agent_events log (chat + tool activity)
    tool_full_values.json           — untruncated tool args/results behind the previews, by call_id
    metrics.json                    — WebRuntime.metrics (cost snapshot)
    dataset_url.json                — attached dataset URL
    report_language.json            — report language chosen for the session
    settings_snapshot.json          — settings at export time (read-only, informational)
    graphs/execution.json           — execution graph snapshot
    graphs/research_active.json     — research graph snapshot
    graphs/agent_summaries.json     — per-agent run accounts already written
    knowledge_memory_snapshot.json  — global memory snapshot (read-only, informational)
    sandbox_trajectory.json         — full OpenHands sandbox trace, if any (read-only, best-effort)

Sandbox trajectories can run to gigabytes of raw tool output and live on a
separate server, so fetching one is a best-effort step: any failure (timeout,
the container already gone, a bad response) is logged and skipped rather than
failing the whole export — every other section is still saved.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from CoScientist.agents.callbacks.report_language import REPORT_LANGUAGES

logger = logging.getLogger("CoScientist.web.session_bundle")

BUNDLE_VERSION = 1
BUNDLE_EXTENSION = ".cossession.zip"

# Keys inside the ZIP
_MANIFEST = "manifest.json"
_ADK_SESSION = "adk_session.json"
_AGENT_EVENTS = "agent_events.json"
_TOOL_FULL_VALUES = "tool_full_values.json"
_METRICS = "metrics.json"
_DATASET_URL = "dataset_url.json"
_REPORT_LANGUAGE = "report_language.json"
_SETTINGS = "settings_snapshot.json"
_GRAPH_EXECUTION = "graphs/execution.json"
_GRAPH_RESEARCH = "graphs/research_active.json"
#: What a small model already wrote about each agent's run. Carried so an
#: imported study opens with its accounts intact instead of re-buying every
#: one of them, card by card, from a model.
_GRAPH_SUMMARIES = "graphs/agent_summaries.json"
_KNOWLEDGE_MEMORY = "knowledge_memory_snapshot.json"
_SANDBOX_TRAJECTORY = "sandbox_trajectory.json"
_MCP_BUILDS_JOBS = "mcp_builds/jobs.json"
_MCP_BUILDS_LOGS = "mcp_builds/logs/"
_MCP_BUILDS_BUNDLES = "mcp_builds/bundles/"
# The files the run produced, and the index describing them. Without these a
# bundle opened on another machine shows a graph full of artifacts it cannot
# display — which is most of what a reader wanted to see.
_ARTIFACT_MANIFEST = "artifacts/manifest.json"
_ARTIFACT_INDEX = "artifacts/index.json"
_ARTIFACT_FILES = "artifacts/files/"

#: A bundle is a file the operator can edit, so an entry name from inside one is
#: checked before it becomes a path — the same argument the report-language
#: block makes about its own member.
_ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _pack_artifacts(zf: "zipfile.ZipFile", user_id: str, session_id: str) -> None:
    """Put the run's own files in the bundle, with the manifest describing them.

    Best-effort, like the sandbox trajectory: a session that cannot be read
    still exports everything else. What does not fit the budget is written into
    the exported manifest as ``skipped/bundle_quota``, so the importer can say
    *why* a figure is missing instead of rendering a hole.
    """
    try:
        from CoScientist.reporting import artifact_index, session_files

        key = (user_id, session_id)
        manifest = session_files.load_manifest(session_id, user_id)
        if not manifest:
            return

        budget = int(os.getenv("BUNDLE__MAX_ARTIFACT_MB", "512")) * 1024 * 1024
        spent, packed = 0, 0
        exported = dict(manifest)
        for artifact_id, record in manifest.items():
            if record.get("state") != session_files.STATE_STORED:
                continue
            path = session_files.resolve_path(key, artifact_id)
            if path is None:
                continue
            size = path.stat().st_size
            if spent + size > budget:
                exported[artifact_id] = {
                    **record,
                    "state": session_files.STATE_SKIPPED,
                    "reason": session_files.REASON_BUNDLE_QUOTA,
                }
                continue
            zf.write(path, f"{_ARTIFACT_FILES}{artifact_id}")
            spent += size
            packed += 1

        zf.writestr(_ARTIFACT_MANIFEST, _json_bytes({"artifacts": exported}))
        index = artifact_index.load(session_id, user_id)
        if index:
            zf.writestr(_ARTIFACT_INDEX, _json_bytes({"artifacts": index}))
        logger.info(
            "bundle: packed %d/%d artifacts (%.1f MB)",
            packed, len(manifest), spent / (1024 * 1024),
        )
    except Exception as exc:  # noqa: BLE001 — never sink an export over this
        logger.warning("Could not include session artifacts in export: %s", exc)


def _restore_artifacts(zf: "zipfile.ZipFile", user_id: str, session_id: str) -> int:
    """Unpack a bundle's files under the NEW session, and return how many.

    Nothing is rewritten on the way in. Graph attrs and report markdown hold
    ``cos-artifact:<id>`` references that carry no session, and the link is
    built from whoever is reading — so the same bytes answer under the new id.
    """
    try:
        from CoScientist.reporting import artifact_index, session_files
    except Exception:  # noqa: BLE001
        return 0

    key = (user_id, session_id)
    restored = 0
    try:
        raw = json.loads(zf.read(_ARTIFACT_MANIFEST))
    except (KeyError, json.JSONDecodeError):
        return 0  # a bundle from before artifacts travelled

    try:
        target = session_files.files_dir(key)
        target.mkdir(parents=True, exist_ok=True)
        for entry in zf.namelist():
            if not entry.startswith(_ARTIFACT_FILES) or entry.endswith("/"):
                continue
            name = Path(entry).name
            if not _ARTIFACT_NAME_RE.match(name):
                logger.warning("bundle: refusing artifact name %r", name)
                continue
            (target / name).write_bytes(zf.read(entry))
            restored += 1

        records = raw.get("artifacts") if isinstance(raw, dict) else raw
        if isinstance(records, dict):
            session_files.record_entries(
                list(records.values()), user_id=user_id, session_id=session_id
            )
        try:
            index = json.loads(zf.read(_ARTIFACT_INDEX))
            entries = index.get("artifacts") if isinstance(index, dict) else index
            if isinstance(entries, list):
                artifact_index.record(entries, user_id=user_id, session_id=session_id)
        except (KeyError, json.JSONDecodeError):
            pass
        logger.info("bundle: restored %d artifact(s) for %s", restored, session_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not restore session artifacts: %s", exc)
    return restored


def _json_bytes(obj: Any) -> bytes:
    """Serialize *obj* to compact UTF-8 JSON bytes.

    Uses ``default=str`` so non-serializable leaves (pydantic models, enums,
    datetimes …) degrade to strings rather than blowing up the whole export.
    """
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=str,
        indent=2,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

async def export_session(
    runtime: Any,          # WebRuntime
    key: tuple[str, str],  # (user_id, session_id)
) -> bytes:
    """Collect every layer of a session and pack them into a ZIP archive.

    Returns the raw ZIP bytes (ready for streaming / writing to disk).
    """
    from CoScientist.web.app import APP_NAME

    user_id, session_id = key

    # 1. Registry metadata
    session_meta = runtime.registry.get_session(user_id, session_id) or {}
    user_meta = runtime.registry.get_user(user_id) or {}
    title = session_meta.get("title", "session")

    # 2. ADK session (pydantic → dict)
    adk_session_data: Optional[Dict[str, Any]] = None
    try:
        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is not None:
            adk_session_data = adk_session.model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not serialize ADK session: %s", exc)

    # 3. Agent events log, plus the full values its tool previews were cut
    # from — without them "Show more" in the imported session has nothing
    # to fetch.
    agent_events = list(runtime.agent_events.get(key, []))
    tool_full_values = dict(runtime.tool_full_values.get(key, {}))

    # 4. Metrics
    metrics = runtime.metrics.get(key)

    # 5. Dataset URL
    dataset_url = runtime.dataset_urls.get(key, "")
    report_language = runtime.report_languages.get(key, "")

    # 6. Settings snapshot
    settings_snapshot: Dict[str, Any] = {}
    try:
        from CoScientist.web.app import _settings_payload
        settings_snapshot = _settings_payload()
    except Exception:  # noqa: BLE001
        pass

    # 7. Execution graph
    execution_graph: Optional[Dict[str, Any]] = None
    try:
        from CoScientist.graph.memory import get_knowledge_graph
        kg = get_knowledge_graph(user_id=user_id, session_id=session_id)
        execution_graph = kg.full()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read execution graph: %s", exc)

    # 7b. Accounts already written about the agents of this run.
    agent_summaries: Optional[Dict[str, Any]] = None
    try:
        from CoScientist.graph.summary_store import all_entries
        agent_summaries = all_entries((user_id, session_id)) or None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read agent summaries: %s", exc)

    # 8. Research graph
    research_graph: Optional[Dict[str, Any]] = None
    try:
        from CoScientist.graph.research.store import get_research_graph
        rg = get_research_graph(user_id=user_id, session_id=session_id)
        research_graph = rg.full()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read research graph: %s", exc)

    # 9. Knowledge memory snapshot (global, read-only)
    knowledge_memory: Optional[Dict[str, Any]] = None
    try:
        from CoScientist.graph.memory_store import get_global_knowledge_memory
        km = get_global_knowledge_memory()
        knowledge_memory = km.full()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read knowledge memory: %s", exc)

    # 10. Sandbox trajectory (best-effort — can be huge, or the container
    # behind it may already be gone; must never sink the rest of the export)
    sandbox_trajectory: Optional[Dict[str, Any]] = None
    try:
        from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
        sandbox_id = None
        if adk_session_data and isinstance(adk_session_data.get("state"), dict):
            sandbox_id = adk_session_data["state"].get(sandbox.SESSION_STATE_KEY)
        if sandbox_id:
            sandbox_trajectory = await sandbox.aget_sandbox_trajectory(
                sandbox_id=str(sandbox_id),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch sandbox trajectory: %s", exc)
        sandbox_trajectory = None

    # 11. MCP builds (Alembic): job metadata, build logs, docker-buildable bundles
    mcp_jobs: Optional[list] = None
    mcp_bundles: Dict[str, bytes] = {}  # repo_name -> zip bytes
    mcp_logs: Dict[str, str] = {}       # job_id -> log text
    try:
        from CoScientist.tools.alembic_tools import (
            export_jobs_snapshot, LOG_DIR,
        )
        snapshot = export_jobs_snapshot()
        if snapshot:
            mcp_jobs = snapshot
            # Collect build logs from disk
            if LOG_DIR.exists():
                for log_file in LOG_DIR.glob("*.log"):
                    mcp_logs[log_file.stem] = log_file.read_text(
                        encoding="utf-8", errors="replace"
                    )
            # Collect docker-buildable bundles for completed builds
            done = [r for r in snapshot
                    if r.get("status") == "done" and r.get("repo_url")]
            if done:
                try:
                    from CoScientist.alembic.web.artifacts import _repo_name, bundle_zip
                    from CoScientist.tools.alembic_tools import web_build_workdir
                    for r in done:
                        workdir = web_build_workdir(r["job_id"])
                        bundle_data = bundle_zip(workdir, r["repo_url"]) if workdir else None
                        if bundle_data:
                            mcp_bundles[_repo_name(r["repo_url"])] = bundle_data
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not collect MCP bundles: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not collect MCP builds: %s", exc)

    # --- Build manifest ---
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "original_user_id": user_id,
        "original_session_id": session_id,
        "original_nickname": user_meta.get("nickname", ""),
        "title": title,
        "session_meta": session_meta,
    }

    # --- Pack ZIP ---
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        from CoScientist.tools.fedot_runs import snapshot as fedot_snapshot
        # Optional v1 member: old bundles remain importable. Graph configs,
        # inputs, outputs and tool traces are carried with every session run.
        zf.writestr("fedot/runs.json", _json_bytes(fedot_snapshot(key)))
        zf.writestr(_MANIFEST, _json_bytes(manifest))
        if adk_session_data is not None:
            zf.writestr(_ADK_SESSION, _json_bytes(adk_session_data))
        zf.writestr(_AGENT_EVENTS, _json_bytes(agent_events))
        # No BUNDLE_VERSION bump, same as report_language below.
        if tool_full_values:
            zf.writestr(_TOOL_FULL_VALUES, _json_bytes(tool_full_values))
        if metrics is not None:
            zf.writestr(_METRICS, _json_bytes(metrics))
        zf.writestr(_DATASET_URL, _json_bytes({"dataset_url": dataset_url}))
        # No BUNDLE_VERSION bump: an older bundle without this member reads
        # back as {}, and older code ignores a zip entry it does not know.
        zf.writestr(_REPORT_LANGUAGE, _json_bytes({"report_language": report_language}))
        zf.writestr(_SETTINGS, _json_bytes(settings_snapshot))
        if execution_graph is not None:
            zf.writestr(_GRAPH_EXECUTION, _json_bytes(execution_graph))
        if research_graph is not None:
            zf.writestr(_GRAPH_RESEARCH, _json_bytes(research_graph))
        if agent_summaries:
            zf.writestr(_GRAPH_SUMMARIES,
                        _json_bytes({"version": 1, "entries": agent_summaries}))
        if knowledge_memory is not None:
            zf.writestr(_KNOWLEDGE_MEMORY, _json_bytes(knowledge_memory))
        if sandbox_trajectory is not None:
            try:
                zf.writestr(_SANDBOX_TRAJECTORY, _json_bytes(sandbox_trajectory))
            except Exception as exc:  # noqa: BLE001 - a huge/odd trace must not sink the export
                logger.warning("Could not include sandbox trajectory in export: %s", exc)
        _pack_artifacts(zf, user_id, session_id)
        if mcp_jobs is not None:
            zf.writestr(_MCP_BUILDS_JOBS, _json_bytes(mcp_jobs))
        for jid, log_text in mcp_logs.items():
            zf.writestr(f"{_MCP_BUILDS_LOGS}{jid}.log", log_text.encode("utf-8"))
        for name, bundle_data in mcp_bundles.items():
            zf.writestr(f"{_MCP_BUILDS_BUNDLES}{name}.zip", bundle_data)

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def preview_bundle(bundle_bytes: bytes) -> Dict[str, Any]:
    """Read a bundle's manifest and MCP build info without importing.

    Returns ``{title, exported_at, mcp_builds, has_mcp_builds}``.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(bundle_bytes), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid bundle file: {exc}") from exc

    manifest: Dict[str, Any] = {}
    try:
        manifest = json.loads(zf.read(_MANIFEST))
    except (KeyError, json.JSONDecodeError):
        pass

    mcp_builds: list = []
    try:
        mcp_builds = json.loads(zf.read(_MCP_BUILDS_JOBS))
    except (KeyError, json.JSONDecodeError):
        pass

    zf.close()
    return {
        "title": manifest.get("title", "session"),
        "exported_at": manifest.get("exported_at", ""),
        "mcp_builds": mcp_builds,
        "has_mcp_builds": bool(mcp_builds),
    }


async def import_session(
    runtime: Any,           # WebRuntime
    target_user_id: str,    # user to assign the session to
    bundle_bytes: bytes,
    rebuild_mcp: bool = False,
) -> Dict[str, Any]:
    """Unpack a session bundle and restore it into the running process.

    Creates a *new* session (new session_id) under ``target_user_id``.
    Returns ``{"user": {...}, "session": {...}}`` on success.
    """
    from CoScientist.web.app import APP_NAME
    from CoScientist.graph.session_scope import (
        GRAPH_SCOPE_SESSION_KEY,
        GRAPH_SCOPE_USER_KEY,
    )

    # --- Unpack ---
    try:
        zf = zipfile.ZipFile(io.BytesIO(bundle_bytes), "r")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"Invalid bundle file: {exc}") from exc

    def _read_json(name: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(zf.read(name))
        except (KeyError, json.JSONDecodeError):
            return None

    manifest = _read_json(_MANIFEST)
    if manifest is None:
        raise ValueError("Bundle has no manifest.json — not a valid session bundle.")
    if manifest.get("bundle_version", 0) > BUNDLE_VERSION:
        raise ValueError(
            f"Bundle version {manifest.get('bundle_version')} is newer than "
            f"supported ({BUNDLE_VERSION}). Please update CoScientist."
        )

    adk_session_data = _read_json(_ADK_SESSION)
    agent_events = _read_json(_AGENT_EVENTS) or []
    tool_full_values = _read_json(_TOOL_FULL_VALUES) or {}
    metrics = _read_json(_METRICS)
    dataset_url_data = _read_json(_DATASET_URL) or {}
    report_language_data = _read_json(_REPORT_LANGUAGE) or {}
    research_graph_data = _read_json(_GRAPH_RESEARCH)
    execution_graph_data = _read_json(_GRAPH_EXECUTION)
    # Absent from a bundle written before this existed; reads back as {}.
    agent_summaries_data = _read_json(_GRAPH_SUMMARIES)
    fedot_data = _read_json("fedot/runs.json")
    if "fedot/runs.json" in zf.namelist() and fedot_data is None:
        raise ValueError("Invalid fedot/runs.json in session bundle")
    if fedot_data is not None:
        from CoScientist.tools.fedot_runs import validate_snapshot
        validate_snapshot(fedot_data)

    title = manifest.get("title", "Imported session")
    session_id = f"session_{uuid4().hex}"

    # --- Ensure user exists in registry ---
    # ensure_user accepts a nickname, not an ID. Prefer the chosen owner;
    # otherwise import silently creates a different user with the ID as name.
    user = runtime.registry.get_user(target_user_id) or runtime.registry.ensure_user(target_user_id)
    user_id = user["id"]

    # Delivered text (chat events, the Report node) carries links already
    # resolved against the exporting scope. Point them at the new one.
    rebase = _session_link_rebaser(
        manifest.get("original_user_id"), manifest.get("original_session_id"),
        user_id, session_id,
    )
    adk_session_data = rebase(adk_session_data)
    agent_events = rebase(agent_events)
    tool_full_values = rebase(tool_full_values)
    research_graph_data = rebase(research_graph_data)
    execution_graph_data = rebase(execution_graph_data)
    agent_summaries_data = rebase(agent_summaries_data)

    # --- Create ADK session ---
    initial_state: Dict[str, Any] = {
        "active_tasks": [],
        GRAPH_SCOPE_USER_KEY: user_id,
        GRAPH_SCOPE_SESSION_KEY: session_id,
    }
    # Merge saved state (active_tasks, report_config, …)
    if adk_session_data and isinstance(adk_session_data.get("state"), dict):
        saved_state = dict(adk_session_data["state"])
        # Override scope keys to match the new user/session
        saved_state[GRAPH_SCOPE_USER_KEY] = user_id
        saved_state[GRAPH_SCOPE_SESSION_KEY] = session_id
        initial_state.update(saved_state)

    adk_session = await runtime.session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state=initial_state,
    )

    # Replay events into the ADK session so the model sees conversation history
    if adk_session_data and isinstance(adk_session_data.get("events"), list):
        from google.adk.events.event import Event as ADKEvent
        for raw_event in adk_session_data["events"]:
            try:
                event = ADKEvent.model_validate(raw_event)
                await runtime.session_service.append_event(adk_session, event)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skipping unrestorable event: %s", exc)

    # --- Create registry session ---
    session_meta_data = manifest.get("session_meta", {})
    session = runtime.registry.import_session(
        user_id=user_id,
        session_id=session_id,
        title=title,
        created_at=session_meta_data.get("created_at"),
        updated_at=session_meta_data.get("updated_at"),
    )

    key = (user_id, session_id)

    if fedot_data is not None:
        from CoScientist.tools.fedot_runs import restore as restore_fedot
        restore_fedot(key, fedot_data)

    # --- Restore agent events ---
    if isinstance(agent_events, list) and agent_events:
        runtime.agent_events[key] = agent_events
        # Also on disk, like a live session's transcript: otherwise the imported
        # chat (HITL cards included) is gone after the next server restart.
        from CoScientist.web.session_store import append_event
        for event in agent_events:
            if isinstance(event, dict):
                append_event(user_id, session_id, event)

    if isinstance(tool_full_values, dict):
        store = runtime.tool_full_values[key]
        for call_id, entry in tool_full_values.items():
            if isinstance(entry, dict):
                store[str(call_id)] = entry

    # --- Restore metrics ---
    if metrics is not None:
        runtime.metrics[key] = metrics

    # --- Restore dataset URL ---
    dataset_url = dataset_url_data.get("dataset_url", "")
    if dataset_url:
        runtime.dataset_urls[key] = dataset_url
    report_language = report_language_data.get("report_language", "")
    # A bundle is a file the user can edit, so it does not get to bypass the
    # enum the socket enforces. An unknown value leaves the session with no
    # choice, and the callback normalizer supplies the default.
    if report_language in REPORT_LANGUAGES:
        runtime.report_languages[key] = report_language
    elif report_language:
        logger.warning(
            "Session bundle carries an unknown report language %r. Ignored.",
            report_language,
        )

    # --- Restore graphs ---
    _restore_graph_files(user_id, session_id, execution_graph_data,
                         research_graph_data, agent_summaries_data)

    # --- Restore the run's own files, so the graph has something to show ---
    _restore_artifacts(zf, user_id, session_id)

    # --- Restore MCP builds ---
    _restore_mcp_builds(zf, rebuild_mcp)

    zf.close()

    return {"user": user, "session": session}


def _session_link_rebaser(
    old_user_id: Any, old_session_id: Any, user_id: str, session_id: str
):
    """A function that rewrites old-scope session links inside JSON data.

    ``cos-artifact:`` references need no help, but anything resolved before
    export — ``/api/users/<u>/sessions/<s>/...``, relative or absolute — names
    the exporting user and session, which do not exist here.
    """
    from urllib.parse import quote

    if not old_user_id or not old_session_id:
        return lambda data: data

    def prefix(u: str, s: str) -> str:
        return f"/api/users/{quote(str(u), safe='')}/sessions/{quote(str(s), safe='')}/"

    old, new = prefix(old_user_id, old_session_id), prefix(user_id, session_id)
    if old == new:
        return lambda data: data

    def rebase(data: Any) -> Any:
        if isinstance(data, str):
            return data.replace(old, new) if old in data else data
        if isinstance(data, list):
            return [rebase(item) for item in data]
        if isinstance(data, dict):
            return {key: rebase(value) for key, value in data.items()}
        return data

    return rebase


def _restore_graph_files(
    user_id: str,
    session_id: str,
    execution_data: Optional[Dict[str, Any]],
    research_data: Optional[Dict[str, Any]],
    summaries_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Write graph snapshots to the on-disk location the stores expect."""
    from CoScientist.graph.session_scope import storage_dir

    graph_dir = os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")
    session_dir = storage_dir(graph_dir, (user_id, session_id))
    session_dir.mkdir(parents=True, exist_ok=True)

    if execution_data is not None:
        _write_json(session_dir / "execution.json", execution_data)
    if research_data is not None:
        from CoScientist.graph.research.store import _default_file
        _write_json(session_dir / _default_file(), research_data)
    if summaries_data:
        from CoScientist.graph.summary_store import STORE_FILENAME
        _write_json(session_dir / STORE_FILENAME, summaries_data)


def _restore_mcp_builds(zf: zipfile.ZipFile, rebuild: bool) -> None:
    """Restore MCP build artefacts from the bundle.

    Always: restore build logs to LOG_DIR + repopulate the in-memory _JOBS.
    If ``rebuild`` is True: launch ``build_mcp_server`` for every repo that
    had a completed build in the exported session.
    """
    # 1. Read job metadata
    try:
        jobs = json.loads(zf.read(_MCP_BUILDS_JOBS))
    except (KeyError, json.JSONDecodeError):
        return  # no MCP builds in this bundle

    # 2. Restore build logs
    try:
        from CoScientist.tools.alembic_tools import LOG_DIR, import_jobs_snapshot
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        for entry in zf.namelist():
            if entry.startswith(_MCP_BUILDS_LOGS) and entry.endswith(".log"):
                log_name = Path(entry).name
                (LOG_DIR / log_name).write_bytes(zf.read(entry))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not restore MCP build logs: %s", exc)

    # 3. Repopulate in-memory job registry
    try:
        import_jobs_snapshot(jobs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not restore MCP job records: %s", exc)

    # 4. Optionally rebuild
    if rebuild:
        done_repos = list({r["repo_url"] for r in jobs
                          if r.get("status") == "done" and r.get("repo_url")})
        for repo_url in done_repos:
            try:
                import asyncio
                from CoScientist.tools.alembic_tools import build_mcp_server
                asyncio.ensure_future(build_mcp_server(repo_url, force_rebuild=True))
                logger.info("MCP rebuild launched for %s", repo_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not launch MCP rebuild for %s: %s", repo_url, exc)


def _write_json(path: Path, data: Any) -> None:
    """Atomic JSON write."""
    tmp = path.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Saved-sessions management (on-disk snapshots)
# ---------------------------------------------------------------------------

def _snapshots_dir() -> Path:
    """Directory where manually saved sessions live."""
    try:
        from CoScientist.config import get_settings
        return Path(get_settings().web.session_snapshots_dir)
    except Exception:  # noqa: BLE001
        return Path("session_snapshots")


def save_to_disk(bundle_bytes: bytes, title: str, user_id: str, session_id: str) -> str:
    """Write a bundle to the snapshots directory. Returns the filename."""
    directory = _snapshots_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:60].strip()
    filename = f"{safe_title}_{stamp}{BUNDLE_EXTENSION}"
    (directory / filename).write_bytes(bundle_bytes)
    return filename


def list_saved_sessions() -> list[Dict[str, Any]]:
    """Scan the snapshots directory and read each bundle's manifest."""
    directory = _snapshots_dir()
    if not directory.is_dir():
        return []
    result = []
    for path in sorted(directory.glob(f"*{BUNDLE_EXTENSION}"), reverse=True):
        try:
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read(_MANIFEST))
            result.append({
                "filename": path.name,
                "title": manifest.get("title", path.stem),
                "exported_at": manifest.get("exported_at", ""),
                "original_nickname": manifest.get("original_nickname", ""),
                "size_bytes": path.stat().st_size,
            })
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping unreadable snapshot %s: %s", path.name, exc)
    return result


def read_saved_bundle(filename: str) -> bytes:
    """Read a saved bundle by filename (no path traversal)."""
    safe = Path(filename).name
    path = _snapshots_dir() / safe
    if not path.is_file():
        raise FileNotFoundError(f"No saved session '{safe}'.")
    return path.read_bytes()


def read_saved_events(filename: str) -> list[Dict[str, Any]]:
    """Return just the ``agent_events.json`` log of a saved bundle.

    Reading the whole bundle to get at its event log means shipping the ADK
    session, the graphs and possibly a sandbox trajectory to the browser — tens
    of megabytes for a list the status indicator's demo mode replays in a few
    seconds. This reads the one member and nothing else.
    """
    safe = Path(filename).name
    path = _snapshots_dir() / safe
    if not path.is_file():
        raise FileNotFoundError(f"No saved session '{safe}'.")
    with zipfile.ZipFile(path, "r") as zf:
        try:
            events = json.loads(zf.read(_AGENT_EVENTS))
        except (KeyError, json.JSONDecodeError):
            return []
    return events if isinstance(events, list) else []


def delete_saved_session(filename: str) -> bool:
    """Delete a saved bundle by filename."""
    safe = Path(filename).name
    path = _snapshots_dir() / safe
    if path.is_file():
        path.unlink()
        return True
    return False
