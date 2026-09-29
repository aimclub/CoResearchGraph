from CoScientist.agents.common import is_proxy_error
import asyncio
import copy
import json
import logging
import mimetypes
import os
import re
import time
from collections import defaultdict, OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote, urlparse
from uuid import uuid4
from weakref import WeakKeyDictionary

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from CoScientist.agents.callbacks.tool_callbacks import DATASET_URL_STATE_KEY
from CoScientist.agents.callbacks.report_language import (
    REPORT_LANGUAGES,
    REPORT_LANGUAGE_STATE_KEY,
)
from CoScientist.main import CoScientistManager
from CoScientist.web.handler import WebHITLHandler, hitl_response_event
from CoScientist.web.session_registry import LocalSessionRegistry
from CoScientist.web.run_control import ExecutionControlMixin, register_execution_routes
from CoScientist.agents import agent_system, planner_agent
from CoScientist.config import ReportConfig, settings_scope
from CoScientist.config.settings import Settings
from CoScientist.reporting import finalize_report
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.config import get_settings
from CoScientist.paper_parser.s3_connection import s3_service
from CoScientist.tools.coder_tools.coder_tools import coder_toolset
from CoScientist.tools.vault_client import call_vault_sync, vault_url
from CoScientist.agents.common import sync_proxy_session
from CoScientist.utils.report_links import remint_report_urls
from CoScientist.utils.text import strip_thinking

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from CoScientist.web.durable_sessions import DurableSessionService
from google.genai import types
from google.adk.agents.run_config import RunConfig
from google.adk.workflow.utils._workflow_hitl_utils import (
    has_request_input_function_call,
    get_request_input_interrupt_ids,
    create_request_input_response,
    REQUEST_INPUT_FUNCTION_CALL_NAME,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json_safe(value):
    """Return a deeply JSON-serializable copy of ``value``.

    ws.send_json() calls json.dumps() WITHOUT a ``default`` hook, so any object
    it can't natively encode (e.g. a pydantic ``MCPServer`` nested inside a tool
    result) raises and kills the whole event stream. Round-tripping through
    json.dumps(default=str) coerces such leaves to strings while preserving the
    dict/list structure the frontend expects — matching the project's existing
    logging/graph serialisation convention.
    """
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


_STATIC_REF = re.compile(r'((?:src|href)="/static/)([^"?#]+)(")')


def _versioned_static_refs(html: str, static_dir: Path) -> str:
    """Append ``?v=<mtime>`` to every /static/ asset the page references.

    The page itself is no-store, so a changed asset gets a new URL and is
    fetched on the next load — even from a browser that cached the old one
    before the assets were served with Cache-Control."""

    def version(match: re.Match) -> str:
        try:
            stamp = (static_dir / match.group(2)).stat().st_mtime_ns
        except OSError:
            return match.group(0)
        return f"{match.group(1)}{match.group(2)}?v={stamp}{match.group(3)}"

    return _STATIC_REF.sub(version, html)


def _pipeline_stages(run_settings: Settings | None = None) -> list:
    """The stages of a linear pipeline (see ``SystemConfig.linear_stages``) —
    the status indicator counts them; [] when the run is not a linear one."""
    try:
        from CoScientist.agents import config_for_mode

        with settings_scope(run_settings):
            return config_for_mode().linear_stages()
    except Exception as exc:  # noqa: BLE001 — a status line must not break the socket
        logging.getLogger(__name__).warning("pipeline stages unavailable: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Runtime types and constants
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"
APP_NAME = "coscientist_app"


SessionKey = tuple[str, str]
SOCKET_SEND_TIMEOUT_SECONDS = 5.0


def _web_session_service() -> DurableSessionService:
    """Create the durable ADK session store used by the web runtime.

    Session files live beside the web registry, so state, ADK events and
    checkpoint state are kept on the same durable volume.
    """
    from CoScientist.web.session_store import state_dir
    return DurableSessionService(state_dir())


# Tool records replayed to a reconnecting tab. Chat messages live in the same
# log and are never dropped, so only the tool stream is capped.
MAX_TOOL_ACTIVITY_EVENTS = 600
TOOL_ACTIVITY_TRIM_SLACK = 200
# Untruncated tool args/results kept for on-demand fetch (ToolsViewer's "Show
# full result"), keyed by call_id. Independent of MAX_TOOL_ACTIVITY_EVENTS
# since most calls never need an entry here at all — only ones whose preview
# was actually truncated get one.
MAX_TOOL_FULL_VALUES = 300
DATASET_URL_MAX_LENGTH = 2048
# Graph stores the Settings modal can wipe. The derived ``knowledge`` view is
# absent on purpose: it is a projection of ``execution`` plus ``memory``.
GRAPH_DELETE_TARGETS = ("execution", "research")
# Freshly minted artifact download URLs, so a burst of clicks on one report
# does not mint on every request. The URLs are minted for one hour; the cache
# lets go of them ten minutes earlier.
ARTIFACT_URL_CACHE_TTL_SECONDS = 50 * 60
_ARTIFACT_URL_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_ARTIFACT_URL_CACHE_MAX = 1000

#: A direct download out of the sandbox is held in memory on the way through,
#: so it is bounded — beyond this, the transfer through storage is the route.
_DOWNLOAD_MAX_BYTES = 256 * 1024 * 1024


def _ascii_name(name: str) -> str:
    """A filename safe to put in a header: no quotes, no newlines, no non-ASCII.

    Non-ASCII is dropped rather than replaced: `отчёт_v2.txt` keeps `_v2.txt`,
    which is at least a name, where a row of question marks is not.
    """
    cleaned = "".join(ch for ch in str(name) if ch.isprintable() and ch not in '"\\')
    return cleaned.encode("ascii", "ignore").decode("ascii")


def _disposition(kind: str, name: str) -> str:
    """``Content-Disposition`` that keeps the name a Russian-speaking run gives.

    The header is ASCII, so a wholly Cyrillic name has no ASCII form worth
    offering. RFC 5987's ``filename*`` carries the real one, and the plain
    ``filename`` stays only for a client that cannot read it — as a fallback
    that is still a filename. This is the shape the session-file endpoint
    settled on; the two should not answer differently.
    """
    stem = _ascii_name(Path(str(name)).stem).strip(" ._-")
    suffix = _ascii_name(Path(str(name)).suffix)
    ascii_name = f"{stem or 'file'}{suffix}"
    return (f'{kind}; filename="{ascii_name}"; '
            f"filename*=UTF-8''{quote(str(name))}")


def _readable_links(text: str, scope: SessionKey) -> str:
    """Every link in one piece of agent prose, made openable by the reader.

    The single boundary between what an agent wrote and what a person sees.
    Both chat paths pass through it, so the live socket and the replayed
    transcript can never disagree about a link.

    Two rewrites, in order:

    * ``cos-artifact:<id>`` becomes ``/api/users/<u>/sessions/<s>/artifacts/<id>``.
      The stored form carries no session on purpose — that is what lets an
      exported bundle open elsewhere — but no browser knows the scheme, and the
      chat's sanitizer drops an ``src`` it cannot classify. So it is resolved
      here, from the scope that is asking, and nowhere earlier.
    * a raw S3 URL becomes ``/api/artifact/<bucket>/<key>``. Report prose used
      to carry presigned URLs verbatim: four hundred characters of signature
      that stop working within the hour, which is exactly what a user reported
      seeing in the chat.

    Never raises, and never shortens: a message must still arrive.
    """
    if not text:
        return text
    try:
        from CoScientist.utils.report_links import resolve_artifact_refs

        return remint_report_urls(resolve_artifact_refs(text, scope), scope)
    except Exception:  # noqa: BLE001 — a link is not worth a lost message
        logging.getLogger("CoScientist.web").warning(
            "could not rewrite links for delivery", exc_info=True
        )
        return text


_MARKDOWN_PAGE = """<!DOCTYPE html>
<html class="dark" lang="en">
<head>
  <meta charset="utf-8" />
  <meta content="width=device-width, initial-scale=1.0" name="viewport" />
  <title>__TITLE__</title>
  <link rel="stylesheet" href="/static/css/fonts.css" />
  <script src="/static/js/appearance.js"></script>
  <link rel="stylesheet" href="/static/css/main.css" />
  <script src="https://cdnjs.cloudflare.com/ajax/libs/marked/4.3.0/marked.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/dompurify/3.1.6/purify.min.js"></script>
  <style>
    body { margin: 0; background: rgb(var(--c-background)); color: rgb(var(--c-on-surface));
           font-family: var(--font-ui); }
    main { max-width: 860px; margin: 0 auto; padding: 32px 16px 64px; }
    .bar { display: flex; justify-content: flex-end; font-size: 12px; margin-bottom: 16px; }
    .bar a, .md-body a { color: rgb(var(--c-primary-text)); }
    .md-body img { max-width: 100%; }
    .md-body pre { overflow-x: auto; }
    .md-body table { display: block; overflow-x: auto; border-collapse: collapse; }
    .md-body th, .md-body td { border: 1px solid rgb(var(--c-outline-variant) / .35); padding: 4px 8px; }
  </style>
</head>
<body>
  <main>
    <div class="bar"><a href="__RAW__">Raw .md</a></div>
    <article id="doc" class="md-body"></article>
  </main>
  <script id="doc-source" type="application/json">__SOURCE__</script>
  <script>
    const source = JSON.parse(document.getElementById('doc-source').textContent);
    const doc = document.getElementById('doc');
    if (window.marked && window.DOMPurify) {
      marked.setOptions({ gfm: true });
      doc.innerHTML = DOMPurify.sanitize(marked.parse(source));
    } else {
      // Offline: the CDN is out of reach, and plain text still beats nothing.
      doc.style.whiteSpace = 'pre-wrap';
      doc.textContent = source;
    }
  </script>
</body>
</html>
"""


def _markdown_page(text: str, *, title: str, raw_href: str) -> str:
    """A markdown artifact as a page a person can read, in the app's theme.

    Rendered in the browser with the same marked + DOMPurify the chat uses, so
    the server needs no markdown library. The source travels as JSON with every
    ``<`` escaped, which is what keeps a ``</script>`` in it from closing the tag.
    """
    from html import escape

    source = json.dumps(text).replace("<", "\\u003c")
    return (
        _MARKDOWN_PAGE
        .replace("__TITLE__", escape(title))
        .replace("__RAW__", escape(raw_href))
        .replace("__SOURCE__", source)
    )


def _mint_artifact_url(bucket: str, key: str) -> str | None:
    """Mint a fresh download URL for one object. Runs in a worker thread.

    Vault objects (``ephemeral/`` and ``permanent/`` keys in the vault bucket)
    go through the vault so its key rules and external signing endpoint apply.
    Every other bucket is signed directly with the shared credentials.
    """
    vault_bucket = os.getenv("S3__BUCKET_NAME", "")
    if (
        bucket == vault_bucket
        and key.startswith(("ephemeral/", "permanent/"))
        and vault_url()
    ):
        payload = call_vault_sync("get_download_link", s3_key=key)
        if payload is None:
            return None
        return payload.get("presigned_url") or payload.get("url")
    content_type = mimetypes.guess_type(key)[0]
    return s3_service.generate_presigned_url(
        key, "get_object", expiration=3600, bucket_name=bucket,
        response_content_type=content_type,
    )


def _validated_dataset_url(raw: Any) -> str:
    """Normalize the dataset link the user attached in the chat; "" clears it.

    Only an http(s) ``.zip`` is accepted, because that is what the sandbox does
    with it: download the URL and unpack the archive into /workspace. Rejecting
    anything else here turns a typo into an immediate message instead of a run
    that fails minutes later inside the container.
    """
    url = str(raw or "").strip()
    if not url:
        return ""
    if len(url) > DATASET_URL_MAX_LENGTH:
        raise ValueError("Dataset link is too long.")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Dataset link must be an http(s) URL.")
    if not parsed.path.lower().endswith(".zip"):
        raise ValueError("Dataset link must point to a .zip archive.")
    return url


def _validated_report_language(raw: Any) -> str:
    """Normalize the report language the user picked; "" clears the choice.

    A closed enum, so a typo becomes an immediate message instead of a finished
    report in the wrong language. Clearing it hands the session back to the
    default that ``normalize_report_language`` applies.
    """
    lang = str(raw or "").strip().lower()
    if not lang:
        return ""
    if lang not in REPORT_LANGUAGES:
        raise ValueError(
            "Report language must be one of: " + ", ".join(REPORT_LANGUAGES) + "."
        )
    return lang


class _RevalidatingStatic(StaticFiles):
    """Serve the front end with "ask me every time".

    `index()` below already sends `no-store` for exactly this reason — but the
    dozen modules the page then pulls in went out with no `Cache-Control` at
    all, so a browser applied heuristic freshness to them. A restarted server
    could therefore still be driving yesterday's JavaScript, and that failure
    is invisible: nothing errors, the new thing simply is not there. Seen on
    2026-09-21, when two rail rows merged half an hour earlier were reported
    missing from a page whose server had them.

    `no-cache` is not `no-store`: the browser still keeps the file and still
    asks, and the ETag turns the answer into a 304 with no body. The 1.6 MB of
    vendored ELK is downloaded once, not once per page.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response


def _apply_frontend_settings(frontend: dict) -> None:
    """Map the frontend JS ``appSettings`` object to ``settings.web``.

    Called before every ``_get_manager()`` invocation so the config singleton
    is always up-to-date when the system is (re)built.
    """
    from CoScientist.config import get_settings
    _startup_settings()  # pin the launch values before the first write
    web = get_settings().web

    general = frontend.get("general", {})
    if "openrouterProviderSort" in general:
        web.openrouter_provider_sort = str(general["openrouterProviderSort"]).strip().lower()
    if "openrouterProviderOrder" in general:
        web.openrouter_provider_order = str(general["openrouterProviderOrder"]).strip()
    if "startMode" in general:
        web.start_mode = general["startMode"]
    if "maxRetries" in general:
        web.max_retries = int(general["maxRetries"])
    if "hitlEnabled" in general:
        val = bool(general["hitlEnabled"])
        web.hitl_enabled = val
        get_settings().hitl.enabled = val
    # The one knob that decides whether this run asks a human and for how long.
    # Read at call time everywhere, so it applies to the next card rather than
    # the next restart — but HITL__MODE in the environment still wins, and the
    # tab is told so rather than silently losing the change.
    if "hitlMode" in general:
        from CoScientist.hitl.mode import MODES

        chosen = str(general["hitlMode"] or "").strip().lower()
        if chosen in MODES:
            web.hitl_mode = chosen
    if "hitlAutoApproveTimeout" in general:
        web.hitl_auto_approve_timeout = int(general["hitlAutoApproveTimeout"])
    if "workOrderEnabled" in general:
        web.work_order_enabled = bool(general["workOrderEnabled"])
    if "workOrderVetoSeconds" in general:
        val = int(general["workOrderVetoSeconds"])
        web.work_order_veto_seconds = val if val > 0 else -1
    if "usePlanner" in general:
        web.use_planner = bool(general["usePlanner"])
    if "useProxy" in general:
        web.use_proxy = bool(general["useProxy"])
        sync_proxy_session()
    if "opikEnabled" in general:
        val = bool(general["opikEnabled"])
        web.opik_enabled = val
        get_settings().opik.enabled = val
    if "autoNamingEnabled" in general:
        web.auto_naming_enabled = bool(general["autoNamingEnabled"])
    if "coscientistUsername" in general:
        val = str(general["coscientistUsername"]).strip()
        web.coscientist_username = val if val else None
    if "contextInitEnabled" in general:
        val = bool(general["contextInitEnabled"])
        web.context_init_enabled = val
        get_settings().context_init.enabled = val
    if "knowledgeGraphEnabled" in general:
        web.knowledge_graph_enabled = bool(general["knowledgeGraphEnabled"])
    if "researchGraphEnabled" in general:
        # The research blackboard reads its own settings block (bindings and
        # graph/research/* both go through settings.research_graph.enabled).
        get_settings().research_graph.enabled = bool(general["researchGraphEnabled"])

    # The experiment module's own two reviews. They are in the Approvals tab
    # but NOT under `hitlEnabled`: ExperimentReviewSessionAgent._should_run_review
    # ignores the global switch, so these flags are the only way past them.
    experiment = frontend.get("experimentModule", {})
    if experiment:
        exp = get_settings().experiments
        if "planAutoApprove" in experiment:
            exp.plan_auto_approve = bool(experiment["planAutoApprove"])
        if "resultAutoApprove" in experiment:
            exp.result_auto_approve = bool(experiment["resultAutoApprove"])
        # FEDOT.MAS route (EXPERIMENTS__ROUTE_FEDOT). The next session's tree
        # attaches FedotAgent on it and builds the planner prompt with or
        # without the route. Off also stops a running session at once: the
        # critique and start_task read this flag on every call. On cannot reach
        # a running session - its executor was built without FedotAgent.
        if "routeFedot" in experiment:
            exp.route_fedot = bool(experiment["routeFedot"])
        # Alembic is already attached inside ExperimentExecutorAgent; this
        # switch only makes the reviewed reuse→Coder/Alembic choice available.
        if "routeAlembic" in experiment:
            exp.route_alembic = bool(experiment["routeAlembic"])
        # Both windows fail closed, so a nonsense value must not become
        # "wait forever" by accident: the model declares gt=0 but a BaseModel
        # does not validate assignment, and handler.py reads <= 0 as no
        # deadline at all.
        for key, attr in (("planReviewTimeoutS", "plan_review_timeout_s"),
                          ("resultReviewTimeoutS", "result_review_timeout_s")):
            if key in experiment:
                try:
                    val = float(experiment[key])
                except (TypeError, ValueError):
                    continue
                if val > 0:
                    setattr(exp, attr, val)

    planner = frontend.get("plannerAgent", {})
    if "retrievalEnabled" in planner:
        web.planner_retrieval_enabled = bool(planner["retrievalEnabled"])
    if "graphEnabled" in planner:
        web.planner_graph_enabled = bool(planner["graphEnabled"])
    if "criticEnabled" in planner:
        web.planner_critic_enabled = bool(planner["criticEnabled"])
    if "criticRounds" in planner:
        # A zero-round critic is just a critic that never runs — that is what
        # the switch above is for, so keep the budget at one round minimum.
        val = int(planner["criticRounds"])
        if val >= 1:
            web.planner_critic_rounds = val
    if "mergeTasksEnabled" in planner:
        web.merge_tasks_enabled = bool(planner["mergeTasksEnabled"])

    if "callGraphCollapseSeconds" in general:
        val = int(general["callGraphCollapseSeconds"])
        if 1 <= val <= 600:
            web.call_graph_collapse_seconds = val

    research = frontend.get("researchAgent", {})
    if "maxSearches" in research:
        val = int(research["maxSearches"])
        if val >= 0:
            web.max_searches = val

    task_exec = frontend.get("taskExecutorAgent", {})
    if "keepScore" in task_exec:
        web.executor_tool_keep_score = float(task_exec["keepScore"])
    if "abstainScore" in task_exec:
        web.executor_tool_abstain_score = float(task_exec["abstainScore"])
    # EXECUTOR__FEDOT_FALLBACK: FedotAgent's switch in the main profile (Settings →
    # Agents). Attached when the next tree is built; ExecutorSwitchAgent reads
    # it on every call, so off also stops the fallback in a running session.
    if "fedotFallback" in task_exec:
        web.fedot_fallback_enabled = bool(task_exec["fedotFallback"])

    # MEDICAL__ENABLED. The next session's tree attaches MedicalAgent on it
    # (orchestrator roster, experiment executor, planner prompt). Off also takes
    # the medical route out of a running experiment at once: the critique and
    # start_task ask medical_route_available on every call.
    medical = frontend.get("medicalAgent", {})
    if "enabled" in medical:
        web.medical_agent_enabled = bool(medical["enabled"])

    # NIR__ENABLED. Purely a runtime gate: NirReportAgent is attached whenever
    # MCP__NORMCONTROL_URL is set, and this decides whether ask_nir_report
    # offers the GOST 7.32-2017 report at all. It reads settings on every call,
    # so the switch takes effect on the next aggregator run without a restart.
    nir = frontend.get("nirReport", {})
    if "enabled" in nir:
        get_settings().nir.enabled = bool(nir["enabled"])

    hypotheses = frontend.get("hypothesesAgent", {})
    if "maxActiveHypotheses" in hypotheses:
        val = int(hypotheses["maxActiveHypotheses"])
        if 1 <= val <= 5:
            web.max_active_hypotheses = val
    if "nodeReportAuto" in hypotheses:
        web.node_report_auto = bool(hypotheses["nodeReportAuto"])

    # Settings → Agents: per-agent on/off, reasoning and model over system.yaml.
    # Read when the next session's agent tree is assembled.
    agents = frontend.get("agents")
    if isinstance(agents, dict):
        from CoScientist.web.agent_settings import apply_agent_settings
        apply_agent_settings(agents)

    coder = frontend.get("coderAgent", {})
    if "sandboxUrl" in coder:
        web.sandbox_url = coder["sandboxUrl"]
    if "workspaceId" in coder:
        val = coder["workspaceId"]
        web.coder_workspace_id = val if val else None
    if "mode" in coder:
        web.coder_mode = str(coder["mode"])


_startup_settings_snapshot: dict | None = None


def _startup_settings() -> dict:
    """The settings the server was launched with (.env and settings.py).

    Captured before the UI writes anything, so "reset to defaults" in the
    settings modal restores the launch values rather than the last save.
    """
    global _startup_settings_snapshot
    if _startup_settings_snapshot is None:
        _startup_settings_snapshot = _current_settings()
    return _startup_settings_snapshot


def _settings_payload() -> dict:
    """The frontend ``appSettings`` shape plus the launch values as ``defaults``.

    Both /api/settings endpoints answer with it, so a GET and the echo of a
    POST can never drift apart.
    """
    return {**_current_settings(), "defaults": _startup_settings()}


def _effective_hitl_mode() -> str:
    """The mode actually in force, so the tab shows what is happening and not
    what was last saved. `HITL__MODE` in the environment outranks the setting,
    and a tab that hid that would report a change it did not make."""
    try:
        from CoScientist.hitl.mode import hitl_mode

        return hitl_mode()
    except Exception:  # noqa: BLE001
        return "basic"


def _current_settings() -> dict:
    """The frontend ``appSettings`` shape, read back off the config singleton."""
    from CoScientist.config import get_settings
    from CoScientist.web.agent_settings import current_agent_settings
    settings = get_settings()
    web = settings.web
    return {
        "general": {
            "openrouterProviderSort": web.openrouter_provider_sort,
            "openrouterProviderOrder": web.openrouter_provider_order,
            "startMode": web.start_mode,
            "maxRetries": web.max_retries,
            "hitlEnabled": web.hitl_enabled,
            "hitlMode": _effective_hitl_mode(),
            "hitlModePinnedByEnv": bool(os.getenv("HITL__MODE", "").strip()),
            "hitlAutoApproveTimeout": web.hitl_auto_approve_timeout,
            "workOrderEnabled": web.work_order_enabled,
            "workOrderVetoSeconds": web.work_order_veto_seconds,
            "usePlanner": web.use_planner,
            "useProxy": web.use_proxy,
            "opikEnabled": web.opik_enabled,
            "autoNamingEnabled": web.auto_naming_enabled,
            # Read-only here: the browser stores its own choice over this default.
            "showInternal": web.show_internal_enabled,
            "callGraphCollapseSeconds": web.call_graph_collapse_seconds,
            "coscientistUsername": web.coscientist_username or "",
            "contextInitEnabled": settings.context_init.enabled,
            "knowledgeGraphEnabled": web.knowledge_graph_enabled,
            "autoClearGraphEnabled": web.auto_clear_graph_enabled,
            "researchGraphEnabled": settings.research_graph.enabled,
        },
        "experimentModule": {
            "planAutoApprove": settings.experiments.plan_auto_approve,
            "resultAutoApprove": settings.experiments.result_auto_approve,
            "planReviewTimeoutS": settings.experiments.plan_review_timeout_s,
            "resultReviewTimeoutS": settings.experiments.result_review_timeout_s,
            "routeFedot": settings.experiments.route_fedot,
            "routeAlembic": settings.experiments.route_alembic,
        },
        "plannerAgent": {
            "retrievalEnabled": web.planner_retrieval_enabled,
            "graphEnabled": web.planner_graph_enabled,
            "criticEnabled": web.planner_critic_enabled,
            "criticRounds": web.planner_critic_rounds,
            "mergeTasksEnabled": web.merge_tasks_enabled,
        },
        "researchAgent": {"maxSearches": web.max_searches},
        "medicalAgent": {"enabled": web.medical_agent_enabled},
        # The GOST 7.32-2017 report. `available` is not a setting but a fact
        # about the deployment: with no MCP__NORMCONTROL_URL there is nothing to
        # submit a document to, and the switch has nothing to switch.
        "nirReport": {
            "enabled": settings.nir.enabled,
            "available": settings.nir_buildable,
        },
        "hypothesesAgent": {
            "maxActiveHypotheses": web.max_active_hypotheses,
            "nodeReportAuto": web.node_report_auto,
        },
        "taskExecutorAgent": {
            "keepScore": web.executor_tool_keep_score,
            "abstainScore": web.executor_tool_abstain_score,
            "fedotFallback": web.fedot_fallback_enabled,
        },
        "coderAgent": {
            "sandboxUrl": web.sandbox_url,
            "workspaceId": web.coder_workspace_id or "",
            "mode": web.coder_mode,
        },
        "agents": current_agent_settings(),
    }


_AGENT_CONFIGURATION_FIELDS = {
    "general": ("startMode", "contextInitEnabled"),
    "experimentModule": ("routeFedot", "routeAlembic"),
    "medicalAgent": ("enabled",),
    "nirReport": ("enabled",),
    "taskExecutorAgent": ("fedotFallback",),
    "agents": ("defaultReasoning", "overrides"),
}


def _agent_configuration_payload(frontend: dict) -> dict:
    """Only settings that change which agents are built or can be routed to."""
    result: dict[str, dict[str, Any]] = {}
    for group, fields in _AGENT_CONFIGURATION_FIELDS.items():
        source = frontend.get(group)
        if not isinstance(source, dict):
            continue
        selected = {
            field: copy.deepcopy(source[field])
            for field in fields
            if field in source
        }
        if selected:
            result[group] = selected
    return result


def _without_agent_configuration(frontend: dict) -> dict:
    """Return the process-wide part of a settings form submission."""
    result = copy.deepcopy(frontend)
    for group, fields in _AGENT_CONFIGURATION_FIELDS.items():
        section = result.get(group)
        if not isinstance(section, dict):
            continue
        for field in fields:
            section.pop(field, None)
        if not section:
            result.pop(group, None)
    return result



class WebRuntime(ExecutionControlMixin):
    """Process-local users, ADK sessions, managers, sockets, and event logs."""

    def __init__(self) -> None:
        # Identifies this server process for diagnostics and frontend cache
        # compatibility. It no longer controls session selection: sessions are
        # durable and must reopen after a restart.
        self.boot_id = uuid4().hex
        # The UI catalogue and graph files are durable, so the ADK session must
        # be durable as well.  InMemorySessionService made a browser reconnect
        # look healthy while the next invocation silently received an empty
        # state/event history after a server restart.
        self.session_service = _web_session_service()
        self.registry = LocalSessionRegistry()
        self.managers: dict[SessionKey, CoScientistManager] = {}
        # Each session owns a small overlay containing only settings that alter
        # the agent topology. A manager holds the immutable snapshot used by its
        # current invocation; a newer desired revision is applied next time.
        self.session_settings_cache: dict[SessionKey, tuple[int, Settings]] = {}
        self.manager_agent_revisions: dict[SessionKey, int] = {}
        self.stale_manager_trees: set[SessionKey] = set()
        self.manager_lock = asyncio.Lock()
        self.control_locks: dict[SessionKey, asyncio.Lock] = {}
        self.execution_locks: dict[SessionKey, asyncio.Lock] = {}
        self.socket_locks = WeakKeyDictionary()
        self.run_versions: dict[SessionKey, int] = defaultdict(int)
        self.stopping_runs: set[SessionKey] = set()
        self._closing = False
        self.agent_events: dict[SessionKey, list[dict[str, Any]]] = defaultdict(list)
        # Full (untruncated) tool args/results, keyed by call_id — see
        # MAX_TOOL_FULL_VALUES. Never broadcast; only served on demand.
        self.tool_full_values: dict[SessionKey, "OrderedDict[str, dict[str, Any]]"] = (
            defaultdict(OrderedDict)
        )
        # Latest usage/cost snapshot per session — cumulative, so one entry is
        # the whole history and a reconnecting tab needs nothing older.
        self.metrics: dict[SessionKey, dict[str, Any]] = {}
        # Latest ТЗ snapshot per session (ТЗ panel) — each one is
        # the whole ТЗ, so a reconnecting tab needs only the last.
        self.tz_snapshots: dict[SessionKey, dict[str, Any]] = {}
        # Dataset archive attached to a session from the chat's "+" menu. Kept
        # here as well as in ADK state so a reconnecting tab and a session whose
        # manager has not been built yet both see the same link.
        self.dataset_urls: dict[SessionKey, str] = {}
        # Report language chosen for a session from the settings language
        # toggle. Holds ONLY explicit choices — an absent key means the browser
        # has not spoken yet, which is what lets the UI default follow its
        # interface language.
        self.report_languages: dict[SessionKey, str] = {}
        self.pending_hitl: dict[str, dict[str, Any]] = {}
        self.hitl_handler = WebHITLHandler()
        self.hitl_handler.set_sender(self.send_socket)
        self.hitl_handler.set_recorder(self.record_event)
        self.sockets: dict[SessionKey, list[WebSocket]] = defaultdict(list)
        self.active_runs: dict[SessionKey, asyncio.Task] = {}
        # Metadata for the invocation currently producing automatic stage
        # checkpoints.  The checkpoint plugin itself is Web-agnostic; its sink
        # reads this table to attach the original query and event baseline.
        self.run_contexts: dict[SessionKey, dict[str, Any]] = {}
        # Installed by _wire_checkpoints; also used for HITL interrupt pauses
        # which do not pass through the stage checkpoint plugin.
        self.checkpoint_hitl = None
        # Run execution times (start to finish) per session
        self.run_times: dict[SessionKey, dict[str, Any]] = {}
        self.init_execution_control()

    def agent_configuration(self, key: SessionKey) -> tuple[int, dict[str, Any]]:
        try:
            session = self.registry.require_session(*key) or {}
        except KeyError:
            # Internal/unit callers can exercise the runtime before registering
            # the public session. HTTP and WebSocket entry points validate it.
            return 0, {}
        config = session.get("agent_configuration") or {}
        return int(config.get("revision") or 0), copy.deepcopy(config.get("settings") or {})

    def settings_snapshot(self, key: SessionKey) -> tuple[int, Settings]:
        """Effective immutable settings for the next invocation of ``key``."""
        revision, overlay = self.agent_configuration(key)
        cached = self.session_settings_cache.get(key)
        if cached is not None and cached[0] == revision:
            return cached
        _startup_settings()
        snapshot = get_settings().model_copy(deep=True)
        with settings_scope(snapshot):
            _apply_frontend_settings(overlay)
        current = (revision, snapshot)
        self.session_settings_cache[key] = current
        return current

    def save_agent_configuration(
        self,
        key: SessionKey,
        settings: dict[str, Any],
    ) -> dict[str, Any]:
        config = self.registry.set_agent_configuration(*key, settings)
        self.session_settings_cache.pop(key, None)
        self.stale_manager_trees.add(key)
        return config

    def agent_configuration_state(self, key: SessionKey) -> dict[str, Any]:
        desired_revision, _ = self.agent_configuration(key)
        active_revision = self.manager_agent_revisions.get(key)
        running = bool(
            key in self.active_runs and not self.active_runs[key].done()
        )
        return {
            "desiredRevision": desired_revision,
            "activeRevision": active_revision,
            "pending": active_revision is not None and active_revision != desired_revision,
            "running": running,
        }

    def record_event(self, key: SessionKey, event: dict[str, Any]) -> None:
        """Append a UI event to memory and to the session's on-disk transcript,
        so the conversation can be reopened after a restart."""
        self.agent_events[key].append(event)
        try:
            from CoScientist.web.session_store import append_event
            append_event(key[0], key[1], event)
        except Exception:  # noqa: BLE001 — persistence must never break a run
            pass

    def control_lock(self, key: SessionKey) -> asyncio.Lock:
        """Serialize start/stop ownership changes for one public session."""
        lock = self.control_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self.control_locks[key] = lock
        return lock

    def _next_run_version(self, key: SessionKey) -> int:
        self.run_versions[key] += 1
        return self.run_versions[key]

    def status_payload(
        self,
        key: SessionKey,
        status: str,
        message: str,
        *,
        version: int | None = None,
    ) -> dict[str, Any]:
        timing = self.run_times.get(key)
        return {
            "type": "status",
            "status": status,
            "message": message,
            "run_status_version": (
                self.run_versions[key] if version is None else version
            ),
            "started_at": timing.get("started_at") if timing else None,
            "finished_at": timing.get("finished_at") if timing else None,
        }

    async def start_run(self, key: SessionKey, data: dict[str, Any]) -> bool:
        """Start one run, rejecting concurrent messages from other tabs."""
        async with self.control_lock(key):
            if self._closing or key in self.stopping_runs:
                return False
            current = self.active_runs.get(key)
            if current is not None and not current.done():
                return False

            version = self._next_run_version(key)
            run_data = dict(data)
            run_data["_run_status_version"] = version
            agent_revision, run_settings = self.settings_snapshot(key)
            # Capture before creating the task: saving settings after the user
            # pressed Send must not alter this already accepted invocation.
            run_data["_agent_revision"] = agent_revision
            run_data["_settings_snapshot"] = run_settings
            try:
                handle = self.prepare_execution(key, run_data, run_settings)
            except ValueError:
                return False
            # Continuation is the same durable research run, not a new timer.
            # Set timing only after acceptance, so a rejected chat cannot reset it.
            self.run_times[key] = {
                "started_at": self._public_execution(handle.status())["started_at"],
                "finished_at": None,
            }
            task = asyncio.create_task(self.run_controlled_chat(key, run_data))
            self.active_runs[key] = task

            def schedule_discard(finished: asyncio.Task) -> None:
                finished.get_loop().create_task(self.discard_run(key, finished))

            task.add_done_callback(schedule_discard)
        await self.send(key, self.status_payload(
            key,
            "processing",
            f"Processing query: {data.get('message', '').strip()}",
            version=version,
        ))
        return True

    async def discard_run(self, key: SessionKey, task: asyncio.Task) -> bool:
        """Drop a finished run only if it still owns the session slot."""
        async with self.control_lock(key):
            if (
                self.active_runs.get(key) is not task
                or key in self.stopping_runs
            ):
                return False
            self.active_runs.pop(key, None)
            version = self._next_run_version(key)
            control = self.execution_snapshot(key)
            terminal_status = "paused" if control and control.get("pause_causes") else "idle"
            timing = self.run_times.get(key)
            if timing and terminal_status == "paused":
                timing["finished_at"] = None
            elif timing and timing.get("finished_at") is None:
                timing["finished_at"] = datetime.now().astimezone().isoformat()
        await self.send(key, self.status_payload(
            key,
            terminal_status,
            "Research is paused; its progress is preserved." if terminal_status == "paused"
            else "Session is ready for the next request.",
            version=version,
        ))
        return True

    async def stop_run(self, key: SessionKey) -> bool:
        """Cancel and remove the exact run currently owning this session."""
        async with self.control_lock(key):
            handle = self.execution_handle(key)
            if handle is not None and not self._closing:
                self.execution_controller.stop(handle.run_id, "operator_stopped")
            task = self.active_runs.get(key)
            stopped = task is not None
            self.stopping_runs.add(key)
            if task is not None:
                if not task.done():
                    task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

        # Keep the stopping gate set through scoped cleanup so a new run cannot
        # create HITL state that belongs to the next owner and then lose it here.
        _cancel_pending_hitl(self, key)
        self.hitl_handler.reset(key)
        try:
            self.registry.touch_session(*key, status="idle")
        except KeyError:
            pass

        async with self.control_lock(key):
            if task is not None and self.active_runs.get(key) is task:
                self.active_runs.pop(key, None)
            self.stopping_runs.discard(key)
            version = self._next_run_version(key)
            timing = self.run_times.get(key)
            if timing and timing.get("finished_at") is None:
                timing["finished_at"] = datetime.now().astimezone().isoformat()

        # Network I/O happens outside the ownership lock so a slow browser
        # cannot block future control operations for the session.
        await self.send(key, self.status_payload(
            key,
            "idle",
            (
                "Agent execution stopped. Session history was preserved."
                if stopped else "Session is already idle."
            ),
            version=version,
        ))
        if stopped:
            await self.send(key, {
                "type": "final_response",
                "content": "Stopped",
            })
        return stopped

    async def get_manager(
        self,
        user_id: str,
        session_id: str,
        *,
        settings_snapshot: Settings | None = None,
        agent_revision: int | None = None,
    ) -> CoScientistManager:
        self.registry.require_session(user_id, session_id)
        key = (user_id, session_id)
        if settings_snapshot is None or agent_revision is None:
            desired_revision, desired_settings = self.settings_snapshot(key)
            if settings_snapshot is None:
                settings_snapshot = desired_settings
            if agent_revision is None:
                agent_revision = desired_revision
        desired_revision, _ = self.agent_configuration(key)
        manager = self.managers.get(key)
        if (
            manager is not None
            and self.manager_agent_revisions.get(key) == agent_revision
            and (key not in self.stale_manager_trees or agent_revision != desired_revision)
        ):
            return manager
        async with self.manager_lock:
            manager = self.managers.get(key)
            if (
                manager is not None
                and (
                    self.manager_agent_revisions.get(key) != agent_revision
                    or (key in self.stale_manager_trees and agent_revision == desired_revision)
                )
            ):
                manager.settings_override = settings_snapshot
                await manager.rebuild_agent_tree()
            elif manager is None:
                # NOT the place to wipe the session's graphs. A missing manager
                # means "no manager since this process started", which is a very
                # different thing from "a new session": continuing yesterday's
                # study after a server restart came through here too, and
                # archived the whole record before the first message of the day.
                # A session starts clean where it is CREATED (create_user_session),
                # and the research graph is explicitly meant to accumulate across
                # every prompt in one session — that is what makes it readable as
                # one investigation rather than a pile of single turns.
                manager = CoScientistManager(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    session_service=self.session_service,
                    settings_override=settings_snapshot,
                )
                await manager.initialize()
                self.managers[key] = manager
                self.execution_locks[key] = asyncio.Lock()
            self.manager_agent_revisions[key] = agent_revision
            if desired_revision == agent_revision:
                self.stale_manager_trees.discard(key)
            else:
                self.stale_manager_trees.add(key)
        return manager

    def invalidate_agent_trees(self) -> int:
        """Apply tree-level settings on each session's next request."""
        keys = set(self.managers)
        self.stale_manager_trees.update(keys)
        return len(keys)

    def attach_socket(self, key: SessionKey, ws: WebSocket) -> None:
        if ws not in self.sockets[key]:
            self.sockets[key].append(ws)

    def detach_socket(self, key: SessionKey, ws: WebSocket) -> None:
        sockets = self.sockets.get(key)
        if not sockets:
            return
        try:
            sockets.remove(ws)
        except ValueError:
            pass
        if not sockets:
            self.sockets.pop(key, None)

    def socket_lock(self, ws: WebSocket) -> asyncio.Lock:
        lock = self.socket_locks.get(ws)
        if lock is None:
            lock = asyncio.Lock()
            self.socket_locks[ws] = lock
        return lock

    async def _send_json_unlocked(self, ws: WebSocket, payload: dict) -> None:
        await asyncio.wait_for(
            ws.send_json(payload),
            timeout=SOCKET_SEND_TIMEOUT_SECONDS,
        )

    async def send_socket(
        self,
        ws: WebSocket,
        payload: dict,
        key: SessionKey | None = None,
    ) -> None:
        """Serialize writes to one socket and bound backpressure time."""
        try:
            async with self.socket_lock(ws):
                await self._send_json_unlocked(ws, payload)
        except Exception:
            if key is not None:
                self.detach_socket(key, ws)
            raise

    async def attach_with_snapshot(
        self,
        key: SessionKey,
        ws: WebSocket,
        *,
        user: dict[str, Any],
        session: dict[str, Any],
        active_tasks: Any,
    ) -> None:
        """Attach a tab with an ordered snapshot before any live broadcasts."""
        async with self.socket_lock(ws):
            async with self.control_lock(key):
                self.attach_socket(key, ws)
                current_run = self.active_runs.get(key)
                status = (
                    "processing"
                    if current_run is not None and not current_run.done()
                    else "idle"
                )
                execution = self.execution_snapshot(key)
                if execution and execution.get("pause_causes"):
                    status = "paused"
                version = self.run_versions[key]
                messages = list(self.agent_events[key])
            try:
                await self._send_json_unlocked(ws, {
                    "type": "connected",
                    "timestamp": datetime.now().isoformat(),
                    "message": f"Connected as {user['nickname']}",
                })
                await self._send_json_unlocked(ws, {
                    "type": "session_snapshot",
                    "user": user,
                    "session": session,
                    "messages": messages,
                    "active_tasks": _json_safe(active_tasks),
                    "status": status,
                    "run_control": execution,
                    "run_status_version": version,
                    "metrics": self.metrics.get(key),
                    "tz": self.tz_snapshots.get(key),
                    "run_times": self.run_times.get(key),
                    "dataset_url": self.dataset_urls.get(key, ""),
                    "report_language": self.report_languages.get(key, ""),
                    "pipeline_stages": _pipeline_stages(self.settings_snapshot(key)[1]),
                    "agent_configuration": self.agent_configuration_state(key),
                    "checkpoints": self.list_checkpoints(key),
                })
            except Exception:
                self.detach_socket(key, ws)
                raise

    async def apply_dataset_url(
        self,
        key: SessionKey,
        url: str | None = None,
    ) -> str:
        """Attach/clear the session's dataset archive and mirror it into ADK state.

        ``url=None`` re-syncs the stored link without changing it — called at the
        start of every run, because the ADK session may not have existed yet when
        the user attached the archive.
        """
        if url is not None:
            if url:
                self.dataset_urls[key] = url
            else:
                self.dataset_urls.pop(key, None)
        current = self.dataset_urls.get(key, "")

        user_id, session_id = key
        adk_session = await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is None:
            return current
        if adk_session.state.get(DATASET_URL_STATE_KEY, "") == current:
            return current
        await self.session_service.append_event(
            adk_session,
            Event(
                invocation_id=f"dataset_{uuid4().hex}",
                author="user",
                actions=EventActions(
                    state_delta={DATASET_URL_STATE_KEY: current},
                ),
            ),
        )
        return current

    async def apply_report_language(
        self,
        key: SessionKey,
        lang: str | None = None,
    ) -> str:
        """Set/clear the session's report language and mirror it into ADK state.

        ``lang=None`` re-syncs the stored choice without changing it — called at
        the start of every run, because the ADK session may not have existed yet
        when the user picked the language.

        The mirror holds only explicit choices, and an unset session mirrors ""
        into state rather than a language. That keeps the one fallback site in
        ``normalize_report_language``. Writing a default here would turn "the UI
        default follows its interface language" into "the default is always
        Russian".
        """
        if lang is not None:
            if lang:
                self.report_languages[key] = lang
            else:
                self.report_languages.pop(key, None)
        current = self.report_languages.get(key, "")
        if lang is None and not current:
            # A re-sync must never downgrade a language already in state to "".
            # Only an explicit clear does that, so a session whose state carries
            # a choice the mirror has lost keeps it.
            return current

        user_id, session_id = key
        adk_session = await self.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is None:
            return current
        if adk_session.state.get(REPORT_LANGUAGE_STATE_KEY, "") == current:
            return current
        await self.session_service.append_event(
            adk_session,
            Event(
                invocation_id=f"reportlang_{uuid4().hex}",
                author="user",
                actions=EventActions(
                    state_delta={REPORT_LANGUAGE_STATE_KEY: current},
                ),
            ),
        )
        return current

    async def send(self, key: SessionKey, payload: dict[str, Any]) -> None:
        """Broadcast an event only to tabs viewing this session."""
        sockets = list(self.sockets.get(key, []))
        if not sockets:
            return

        async def deliver(socket: WebSocket) -> None:
            try:
                await self.send_socket(socket, payload, key)
            except Exception:
                self.detach_socket(key, socket)

        await asyncio.gather(*(deliver(socket) for socket in sockets))

    async def close(self) -> None:
        self._closing = True
        await asyncio.gather(
            *(self.stop_run(key) for key in list(self.active_runs)),
            return_exceptions=True,
        )
        _cancel_pending_hitl(self)
        self.hitl_handler.reset()
        await asyncio.gather(
            *(manager.close() for manager in self.managers.values()),
            return_exceptions=True,
        )

    @staticmethod
    def list_checkpoints(key: SessionKey) -> list[dict[str, Any]]:
        from CoScientist.web.checkpoints import list_checkpoints

        return list_checkpoints(*key)


def _wire_hitl(runtime: WebRuntime) -> None:
    """Wire the routing Web handler once for this application runtime."""

    # SessionAgent handlers are delegates because workflow assembly deep-copies
    # their references.
    wired = []
    for name, agent in agent_system.agents.items():
        handler = getattr(agent, "hitl_handler", None)
        if handler is not None and hasattr(handler, "set_delegate"):
            handler.set_delegate(runtime.hitl_handler)
            wired.append(name)

    if hasattr(hitl_toolset._handler, "set_delegate"):
        hitl_toolset._handler.set_delegate(runtime.hitl_handler)
    else:
        hitl_toolset._handler = runtime.hitl_handler

    # CoderToolset owns its approval handler separately from the agent-level
    # delegates. Preserve HITL__ENABLED semantics while routing Web approvals.
    from CoScientist.tools.coder_tools import coder_toolset
    if coder_toolset._hitl_handler is not None:
        coder_toolset._hitl_handler = runtime.hitl_handler

    from CoScientist.agents.common import hitl_handler as common_hitl_handler
    if hasattr(common_hitl_handler, "set_delegate"):
        common_hitl_handler.set_delegate(runtime.hitl_handler)

    logging.getLogger("CoScientist.web").info(
        "Session-routing WebHITLHandler wired into: %s", wired,
    )


def _wire_sandbox_links(runtime: WebRuntime) -> None:
    """Deliver the sandbox console links while the sandbox is still working.

    ``run_sandbox_task`` waits inline for as long as the job runs (up to
    ``SANDBOX_RUN_WAIT`` seconds), so the only other carrier of ``watch_url`` /
    ``vscode_url`` — the tool's function_response — reaches the browser when
    the run is already over. The sandbox client therefore calls this sink the
    moment the container answers, and the links land in the tab as an ordinary
    agent event.
    """
    from CoScientist.tools.coder_tools import sandbox_tools

    # sandbox_id per session: a follow-up task reuses the container and would
    # otherwise repost the same two links on every single call.
    announced: dict[SessionKey, str] = {}

    async def deliver(key: SessionKey | None, info: dict[str, Any]) -> None:
        urls = [url for url in (info.get("watch_url"), info.get("vscode_url")) if url]
        sandbox_id = str(info.get("sandbox_id") or "")
        if key is None or not urls or announced.get(key) == sandbox_id:
            return
        announced[key] = sandbox_id

        event = {
            "type": "agent_event",
            "author": "CoderAgent",
            "is_final": False,
            "timestamp": datetime.now().isoformat(),
            "content": "\n".join([f"Sandbox is up:", *urls]),
        }
        runtime.agent_events[key].append(event)
        await runtime.send(key, event)

    sandbox_tools.set_sandbox_start_sink(deliver)


def _wire_sandbox_plan(runtime: WebRuntime) -> None:
    """Relay the sandbox agent's own plan to the tabs watching the session.

    A sandbox task is the longest single thing the system does — one tool call
    that can run for hours — and it reports nothing until it is over. The agent
    in the container does keep a task list, though, and rewrites it as it goes;
    the client relays each new revision here, which is what lets the status
    line say *which step* is running instead of "writing and running code" for
    the whole job.

    Live-only, deliberately: the plan describes what is happening right now, a
    replayed one would describe a container that is already gone. A tab that
    reconnects mid-run picks the current revision up on the next poll.
    """
    from CoScientist.tools.coder_tools import sandbox_tools

    async def deliver(key: SessionKey | None, info: dict[str, Any]) -> None:
        if key is None or (key not in runtime.sockets and key not in runtime.active_runs):
            return
        await runtime.send(key, {"type": "sandbox_plan", **_json_safe(info)})

    sandbox_tools.set_sandbox_plan_sink(deliver)


def _wire_metrics(runtime: WebRuntime) -> None:
    """Stream the running cost of a session to the tabs watching it.

    The ledger is updated on every model call; the sink is rate-limited on the
    producing side (``METRICS_PUSH_INTERVAL``) so a hundred-call run does not
    turn into a hundred websocket frames. Only the latest snapshot is kept for
    replay — it is cumulative, so history would be the same number repeated.
    """
    from CoScientist.logging.metrics import set_metrics_sink

    async def deliver(key: SessionKey, payload: dict[str, Any]) -> None:
        if key not in runtime.sockets and key not in runtime.active_runs:
            return
        event = {"type": "metrics", **_json_safe(payload)}
        runtime.metrics[key] = event
        await runtime.send(key, event)

    set_metrics_sink(deliver)


def _wire_tool_activity(runtime: WebRuntime) -> None:
    """Stream every tool call — including those inside AgentTool sub-agents.

    The top-level ``run_async`` stream only carries a delegation's own
    function_call/function_response, so a subordinate's inner tools (e.g.
    ``ResearchAgent`` → ``tavily_search``) are invisible to the browser. The
    plugin sink below fires at every nesting level and routes each call to the
    tabs watching the owning session.
    """
    from CoScientist.logging.tool_activity import set_tool_activity_sink

    def is_delegation_record(event: dict[str, Any]) -> bool:
        """A hand-off between agents, or the start of the run it spawned.

        These few records are what the side-nav call graph is drawn from, so
        they outlive the trim and the process, unlike the bulk of tool calls.
        """
        return bool(event.get("is_delegation")) or (
            event.get("phase") == "agent_start" and bool(event.get("spawn_call_id"))
        )

    def trim(events: list[dict[str, Any]]) -> None:
        """Bound replay history: drop the oldest tool records, keep the chat
        and the delegations."""
        if len(events) <= MAX_TOOL_ACTIVITY_EVENTS + TOOL_ACTIVITY_TRIM_SLACK:
            return
        indexes = [
            index for index, event in enumerate(events)
            if event.get("type") == "tool_activity" and not is_delegation_record(event)
        ]
        excess = len(indexes) - MAX_TOOL_ACTIVITY_EVENTS
        if excess <= 0:
            return
        dropped = set(indexes[:excess])
        events[:] = [
            event for index, event in enumerate(events) if index not in dropped
        ]

    def stash_full_values(key: SessionKey, event: dict[str, Any]) -> None:
        """Pull the untruncated `*_full` fields out of the broadcast event and
        keep them server-side under the call's id, so a client can fetch the
        whole thing later without it ever going out over every tab's socket.
        """
        call_id = event.get("call_id")
        full_fields = {
            field: event.pop(key_name)
            for field in ("args", "result", "error", "effective_args", "state_inputs")
            if (key_name := f"{field}_full") in event
        }
        if not call_id or not full_fields:
            return
        store = runtime.tool_full_values[key]
        entry = store.setdefault(call_id, {})
        entry.update(full_fields)
        store.move_to_end(call_id)
        while len(store) > MAX_TOOL_FULL_VALUES:
            store.popitem(last=False)

    async def deliver(key: SessionKey, payload: dict[str, Any]) -> None:
        if key not in runtime.sockets and key not in runtime.active_runs:
            # A key we never served (e.g. the CLI default scope) has nowhere to go.
            return
        event = {"type": "tool_activity", **_json_safe(payload)}
        stash_full_values(key, event)
        if is_delegation_record(event):
            runtime.record_event(key, event)
        else:
            runtime.agent_events[key].append(event)
        trim(runtime.agent_events[key])
        await runtime.send(key, event)

    set_tool_activity_sink(deliver)


async def _attach_document(
    payload: dict[str, Any],
    key: SessionKey,
    *,
    markdown: str,
    kind: str,
    agent: str = "",
    summary: str = "",
) -> None:
    """Publish one long body as a session document and point the message at it.

    Silent on failure: a message with no document is a message that shows its
    body the way it always did. Losing the text is not on the table.
    """
    try:
        from CoScientist.reporting.documents import publish_document

        document = await asyncio.to_thread(
            publish_document, key,
            markdown=markdown, kind=kind, agent=agent, summary=summary,
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("CoScientist.web").warning(
            "chat document not published (%s): %s", kind, exc
        )
        return
    if document:
        payload["document"] = document.as_payload()
        payload["summary"] = document.summary


def _wire_checkpoints(runtime: WebRuntime) -> None:
    """Persist the state emitted before every configured pipeline stage."""
    from CoScientist.agents.checkpoint_plugin import set_checkpoint_sink
    from CoScientist.web.checkpoints import save_checkpoint

    async def deliver(key: SessionKey, payload: dict[str, Any]) -> None:
        # A CLI/A2A runner in the same process has no Web-owned invocation.
        run = runtime.run_contexts.get(key)
        if run is None:
            return
        payload = dict(payload)
        if not isinstance(payload.get("state"), dict):
            session = await runtime.session_service.get_session(
                app_name=APP_NAME, user_id=key[0], session_id=key[1],
            )
            if session is None:
                return
            payload["state"] = dict(session.state)

        stages = _pipeline_stages()
        agent = payload.get("agent") or payload.get("agent_name")
        if payload.get("stage_index") is None:
            payload["stage_index"] = next(
                (i for i, stage in enumerate(stages)
                 if stage.get("agent") == agent),
                -1,
            )
        payload.setdefault("stage_count", len(stages))
        payload.setdefault("title", f"HITL: {payload.get('message') or agent or 'decision'}")
        payload.setdefault("agent", agent)
        payload.setdefault("created_at", datetime.now().isoformat())
        record = {
            **payload,
            "run_version": run.get("run_version", 0),
            "ui_event_index": len(runtime.agent_events.get(key, [])),
            "adk_event_baseline": run.get("adk_event_baseline", 0),
            "root_query": run.get("root_query", ""),
        }
        # State and graph blackboards form one logical stage boundary.  Keeping
        # only state would let a resumed agent read findings produced after the
        # selected checkpoint through graph tools.
        try:
            from CoScientist.graph.memory import get_knowledge_graph
            record["execution_graph"] = get_knowledge_graph(
                user_id=key[0], session_id=key[1],
            ).full()
        except Exception:  # noqa: BLE001 - state checkpoint is still useful
            pass
        try:
            from CoScientist.graph.research.store import get_research_graph
            record["research_graph"] = get_research_graph(
                user_id=key[0], session_id=key[1],
            ).full()
        except Exception:  # noqa: BLE001
            pass
        public = await asyncio.to_thread(save_checkpoint, key[0], key[1], record)
        await runtime.send(key, {"type": "checkpoint_created", **public})

    runtime.checkpoint_hitl = deliver
    runtime.hitl_handler.set_checkpoint_sink(deliver)
    set_checkpoint_sink(deliver)


def _wire_tz_snapshots(runtime: WebRuntime) -> None:
    """Stream a ТЗ into the ТЗ panel while it is being built.

    The ТЗ agent (microfluidics profile) runs inside an AgentTool, in a child
    session this web session cannot read until the whole module returns — so
    the agent pushes a snapshot after every change and the sink below routes it
    to the session's tabs.
    """
    from CoScientist.hitl.tz_panel import set_tz_sink

    async def deliver(key: SessionKey, payload: dict[str, Any]) -> None:
        if key not in runtime.sockets and key not in runtime.active_runs:
            return
        event = {"type": "tz_snapshot", **_json_safe(payload)}
        runtime.tz_snapshots[key] = event
        await runtime.send(key, event)

    set_tz_sink(deliver)


def _wire_agent_output(runtime: WebRuntime) -> None:
    """Post the final answer of the key agents into the chat.

    A subordinate runs as an ``AgentTool``, so its deliverable — the hypotheses,
    the research summary, the execution report — arrives as the caller's
    function_response and is never spoken in the top-level stream. The plugin
    sink below routes the answer of every agent flagged ``report_output`` in
    ``system.yaml`` to the tabs watching the session, as a message of its own.
    """
    from CoScientist.logging.agent_output import set_agent_output_sink

    async def deliver(key: SessionKey, payload: dict[str, Any]) -> None:
        if key not in runtime.sockets and key not in runtime.active_runs:
            # A key we never served (e.g. the CLI default scope) has nowhere to go.
            return
        if "content" in payload and isinstance(payload["content"], str):
            payload["content"] = _readable_links(strip_thinking(payload["content"]), key)
            if not payload["content"].strip():
                return
            # The deliverable is a document; the feed gets its first paragraph
            # and a button. `content` stays on the event — the transcript and
            # any tab that predates the panel still read it.
            await _attach_document(
                payload, key,
                markdown=payload["content"], kind="answer",
                agent=str(payload.get("agent") or ""),
            )
        event = {"type": "agent_output", **_json_safe(payload)}
        runtime.agent_events[key].append(event)
        await runtime.send(key, event)

    set_agent_output_sink(deliver)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[CoScientist Web] Starting up …")
    try:
        from CoScientist.agents.common import verify_proxy_reachable
        await verify_proxy_reachable()
    except Exception as exc:
        logging.getLogger("CoScientist.web").warning(
            "Proxy pre-flight check failed on startup: %s", exc
        )
    yield
    print("[CoScientist Web] Shutting down …")
    from CoScientist.tools.mcp_catalog import close_catalog

    await close_catalog()
    await app.state.runtime.close()


def _clear_session_graphs(
    user_id: str,
    session_id: str,
    view: str = "all",
) -> tuple[dict[str, Any], bool]:
    """Wipe graph data for a user/session.

    Returns a tuple of (deleted_details_dict, has_failed_boolean).
    """
    if view == "all":
        targets = GRAPH_DELETE_TARGETS
    elif view == "session":
        targets = ("execution", "research")
    elif view in GRAPH_DELETE_TARGETS:
        targets = (view,)
    else:
        targets = ()

    deleted: dict[str, Any] = {}
    failed = False
    for name in targets:
        try:
            if name == "execution":
                from CoScientist.graph.memory import reset_knowledge_graph
                reset_knowledge_graph(user_id=user_id, session_id=session_id)
                deleted[name] = {"scope": "session", "cleared": True}
            elif name == "research":
                from CoScientist.graph.research.store import get_research_graph
                archived = get_research_graph(
                    user_id=user_id,
                    session_id=session_id,
                ).reset(archive=True)
                deleted[name] = {
                    "scope": "session",
                    "cleared": True,
                    "archived": archived,
                }
            elif name == "memory":
                from CoScientist.graph.memory_store import (
                    get_global_knowledge_memory,
                )
                result = get_global_knowledge_memory().clear()
                deleted[name] = {
                    "scope": "global",
                    "cleared": bool(result.get("persisted")),
                    **result,
                }
                failed = failed or not result.get("persisted")
        except Exception as exc:  # noqa: BLE001 — report every target's fate
            logging.getLogger("CoScientist.web").warning(
                "Graph deletion failed for %s: %s", name, exc
            )
            deleted[name] = {"cleared": False, "error": str(exc)}
            failed = True
    return deleted, failed


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def _esc(text: str) -> str:
    """Minimal XML escape for text embedded in an SVG error card."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def create_app() -> FastAPI:
    os.environ["COSCIENTIST_WEB_MODE"] = "true"
    _startup_settings()
    runtime = WebRuntime()
    _wire_hitl(runtime)
    _wire_sandbox_links(runtime)
    _wire_sandbox_plan(runtime)
    _wire_tool_activity(runtime)
    _wire_checkpoints(runtime)
    _wire_agent_output(runtime)
    _wire_metrics(runtime)
    _wire_tz_snapshots(runtime)
    app = FastAPI(
        title="CoScientist Web UI",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    register_execution_routes(app, runtime)

    # Vendored JS/CSS (e.g. vis-network for the live graph) so the UI works
    # offline / behind a VPN without any CDN.
    _static_dir = WEB_DIR / "static"
    if _static_dir.exists():
        app.mount("/static", _RevalidatingStatic(directory=str(_static_dir)),
                  name="static")

    if get_settings().checkpoints.enabled:
        from CoScientist.checkpoints import make_checkpoint_router

        async def resolve_checkpoint_session():
            return runtime.session_service, APP_NAME

        app.include_router(make_checkpoint_router(
            session_resolver=resolve_checkpoint_session,
        ))

    # --- HTML endpoint ---
    @app.get("/", response_class=HTMLResponse)
    async def index():
        # no-store: a cached index.html silently serves an OLD frontend — HITL
        # review cards then render without controls/content.
        # The asset URLs carry their file's version, so a changed script is
        # fetched right away rather than served from the browser's cache.
        return HTMLResponse(
            _versioned_static_refs(TEMPLATE_PATH.read_text(encoding="utf-8"), _static_dir),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/downloads/logs")
    async def proxy_download_logs(wait_seconds: float = 60.0):
        """Relay the sandbox's download SSE stream to the browser.

        The sandbox lives on another origin, so a page served from here cannot
        open an ``EventSource`` against it without CORS headers we do not
        control. Proxying keeps the browser same-origin and lets the sandbox
        URL stay a server-side setting.
        """
        import httpx

        from CoScientist.tools.coder_tools.openhands_sandbox import resolve_sandbox_url

        try:
            base = resolve_sandbox_url()
        except Exception as exc:  # sandbox_url not configured
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        upstream = f"{base.rstrip('/')}/api/v1/downloads/logs"

        async def relay():
            # No read timeout: an SSE stream is idle by design between events.
            timeout = httpx.Timeout(10.0, read=None)
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "GET", upstream, params={"wait_seconds": wait_seconds},
                    ) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            yield chunk
            except Exception as exc:
                logging.getLogger("CoScientist.web").warning(
                    "Download log proxy failed (%s): %s", upstream, exc
                )
                yield f"event: status\ndata: unreachable: {exc}\n\n".encode()

        return StreamingResponse(
            relay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/v1/downloads/cancel")
    @app.post("/api/downloads/cancel")
    async def proxy_download_cancel(request: Request):
        """Relay download cancellation POST request to the sandbox."""
        import httpx

        from CoScientist.tools.coder_tools.openhands_sandbox import resolve_sandbox_url

        try:
            body = await request.json()
        except Exception:
            body = {}

        try:
            base = resolve_sandbox_url()
            upstream = f"{base.rstrip('/')}/api/v1/downloads/cancel"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(upstream, json=body)
                return Response(
                    content=resp.content,
                    status_code=resp.status_code,
                    headers={"Content-Type": resp.headers.get("Content-Type", "application/json")},
                )
        except Exception as exc:
            logging.getLogger("CoScientist.web").warning(
                "Download cancel proxy failed: %s", exc
            )
            return JSONResponse({"status": "cancelled", "detail": str(exc)}, status_code=200)

    # --- Artifact delivery (report links) ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/sandbox/tasks")
    async def list_session_sandbox_tasks(user_id: str, session_id: str):
        """Every workspace the sandbox still holds, so the reader can choose one.

        The session's own container is one of many on that machine, and the
        sandbox addresses each by task id — the same id its console carries in
        `?task_id=`. Offering the list turns "look at ours" into "look at any".
        """
        from CoScientist.tools.coder_tools import openhands_sandbox as sandbox

        result = await asyncio.to_thread(
            sandbox.list_sandbox_tasks, session_id=session_id,
        )
        if result.get("status") != "ok":
            return JSONResponse(
                {"status": "error",
                 "message": result.get("error") or "Песочница не ответила.",
                 "workspaces": []},
                status_code=502,
            )
        return JSONResponse({"status": "ok",
                             "sandbox_id": result.get("sandbox_id"),
                             "workspaces": result.get("workspaces", [])})

    @app.get("/api/users/{user_id}/sessions/{session_id}/sandbox/files")
    async def list_session_sandbox_files(user_id: str, session_id: str,
                                         path: str = "/workspace",
                                         sandbox_id: Optional[str] = None):
        """What a sandbox workspace actually holds, for a human to see.

        The agents have had `list_sandbox_files` all along; the person watching
        the run had no way to look, and no way to reach a file the agent never
        mentioned. Without ``sandbox_id`` this reads the session's own sandbox.
        """
        from CoScientist.tools.coder_tools import openhands_sandbox as sandbox

        result = await asyncio.to_thread(
            sandbox.list_sandbox_files, path, session_id=session_id,
            sandbox_id=(sandbox_id or None),
        )
        if result.get("status") != "ok":
            return JSONResponse(
                {"status": "error",
                 "message": result.get("error") or "Песочница не ответила.",
                 "path": path, "entries": []},
                status_code=502,
            )
        from CoScientist.web.preview import kind_of

        entries = []
        for entry in result.get("entries", []):
            # The sandbox answers "dir"/"file"; normalise once, here, so the
            # panel has a single word to test against.
            directory = str(entry.get("type", "")).lower() in ("dir", "directory")
            entries.append({
                **entry,
                "type": "dir" if directory else "file",
                "kind": "dir" if directory else kind_of(entry.get("name", "")),
            })
        return JSONResponse({"status": "ok", "path": result.get("path", path),
                             "entries": entries})

    @app.get("/api/users/{user_id}/sessions/{session_id}/sandbox/view")
    async def view_session_sandbox_file(user_id: str, session_id: str,
                                        path: str,
                                        sandbox_id: Optional[str] = None,
                                        download: int = 0,
                                        dir: int = 0):
        """Serve one workspace file to the browser, for reading rather than keeping.

        The transfer through S3 is for files that must outlive the container —
        a link in a report. Looking at a log or a plot wants neither a bucket
        nor durability, so these bytes go straight through: the sandbox is on
        another network, the browser is same-origin with us, and nothing is
        stored on the way.
        """
        from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
        from CoScientist.web.preview import cap_for, kind_of, media_type_of

        name = path.rsplit("/", 1)[-1] or "file"
        # A directory comes back from the sandbox as one ZIP, so that is what
        # the reader is saving and what it should be called.
        if dir:
            name, download = f"{name}.zip", 1
        kind = kind_of(name)
        # A checkpoint is not something to look at, and reading one to say so
        # would move gigabytes. Downloading it, however, is fair.
        cap = cap_for(name) if not download else _DOWNLOAD_MAX_BYTES
        if cap <= 0:
            raise HTTPException(
                status_code=415,
                detail="Этот файл можно только забрать — показать его нечем.",
            )

        result = await asyncio.to_thread(
            sandbox.read_sandbox_file, path, max_bytes=cap,
            session_id=session_id, sandbox_id=(sandbox_id or None),
        )
        if result.get("status") != "ok":
            raise HTTPException(
                status_code=502,
                detail=result.get("error") or "Песочница не отдала файл.",
            )

        disposition = "attachment" if download else "inline"
        headers = {
            "Content-Disposition": _disposition(disposition, name),
            # Sent as text, read as text: no sniffing an .html back into
            # something the browser would run on our origin.
            "X-Content-Type-Options": "nosniff",
            "X-Preview-Kind": kind,
            "X-Preview-Truncated": "1" if result.get("truncated") else "0",
            "Cache-Control": "no-store",
        }
        if kind != "pdf":
            # A PDF is shown by the browser's own viewer, which a bare `sandbox`
            # policy stops from loading at all; it gets no DOM of ours either
            # way. Everything else is locked down.
            headers["Content-Security-Policy"] = (
                "default-src 'none'; style-src 'unsafe-inline'; sandbox")
        return Response(
            content=result["data"],
            media_type=("application/octet-stream" if download
                        else media_type_of(name)),
            headers=headers,
        )

    @app.post("/api/users/{user_id}/sessions/{session_id}/sandbox/fetch")
    async def fetch_session_sandbox_file(user_id: str, session_id: str,
                                         request: Request):
        """Copy one sandbox path into storage and answer with its durable link.

        The same transfer the agents use, so a file a person pulls across and a
        file an agent pulls across are the same object with the same link.
        """
        from CoScientist.tools.coder_tools.sandbox_artifacts import (
            transfer_sandbox_artifact,
        )

        body = await request.json()
        path = str((body or {}).get("path") or "").strip()
        if not path:
            raise HTTPException(status_code=400, detail="path is required")
        target = str((body or {}).get("sandbox_id") or "").strip() or None

        result = await asyncio.to_thread(
            transfer_sandbox_artifact, path, session_id=session_id,
            sandbox_id=target,
        )
        return JSONResponse(result,
                            status_code=200 if result.get("status") == "success" else 502)

    @app.get("/api/artifact/{bucket}/{key:path}")
    async def get_artifact(bucket: str, key: str):
        """Redirect the browser to a fresh download URL for one S3 object.

        Report markdown carries these local links instead of raw presigned
        URLs (see ``utils/report_links.py``), so a link in an old report still
        works and the internal S3 endpoint never reaches the browser.
        """
        if not bucket or not key or ".." in key.split("/"):
            raise HTTPException(status_code=404, detail="Invalid artifact reference.")
        cache_key = (bucket, key)
        now = time.monotonic()
        cached = _ARTIFACT_URL_CACHE.get(cache_key)
        if cached is not None and cached[1] > now:
            return RedirectResponse(cached[0], status_code=302)
        try:
            url = await asyncio.to_thread(_mint_artifact_url, bucket, key)
        except Exception as exc:
            logging.getLogger("CoScientist.web").warning(
                "Artifact minting failed for %s/%s: %s", bucket, key, exc
            )
            raise HTTPException(
                status_code=502, detail="Artifact storage is unreachable."
            ) from exc
        if not url:
            raise HTTPException(
                status_code=404, detail="Artifact not found or no longer available."
            )
        if len(_ARTIFACT_URL_CACHE) >= _ARTIFACT_URL_CACHE_MAX:
            _ARTIFACT_URL_CACHE.clear()
        _ARTIFACT_URL_CACHE[cache_key] = (url, now + ARTIFACT_URL_CACHE_TTL_SECONDS)
        return RedirectResponse(url, status_code=302)

    @app.get("/api/users/{user_id}/sessions/{session_id}/artifacts")
    async def list_session_artifacts(user_id: str, session_id: str):
        """Everything this run produced, whether or not a node points at it.

        A graph node links an artifact only when something recorded the link,
        and plenty is captured that nothing ever attaches — a figure an MCP tool
        rendered mid-task is in the store with its bytes, and reachable from
        nowhere on the page. This is the flat answer to "what did the run make",
        including what did NOT survive and why, so a missing figure is a fact
        the reader can see rather than an absence they have to infer.
        """
        from CoScientist.reporting import session_files

        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        from CoScientist.utils.report_links import session_artifact_link

        items = []
        for artifact_id, record in session_files.load_manifest(session_id, user_id).items():
            stored = record.get("state") == session_files.STATE_STORED
            items.append({
                "artifact_id": artifact_id,
                "name": record.get("filename") or artifact_id,
                # `label` titles it, `source_kind` is how the document panel
                # tells a written document from a captured figure.
                "label": record.get("label"),
                "source_kind": record.get("source_kind"),
                "agent": record.get("agent") or record.get("source_tool"),
                "media_type": record.get("media_type"),
                "size_bytes": record.get("size_bytes"),
                "tool": record.get("source_tool"),
                "state": record.get("state"),
                "reason": record.get("reason"),
                "captured_at": record.get("captured_at"),
                "href": (
                    session_artifact_link(user_id, session_id, artifact_id)
                    if stored else None
                ),
            })
        items.sort(key=lambda i: (i["state"] != session_files.STATE_STORED,
                                  -(i.get("captured_at") or 0)))
        return JSONResponse({
            "artifacts": items,
            "stored": sum(1 for i in items if i["state"] == session_files.STATE_STORED),
            "total": len(items),
        })

    @app.get("/api/users/{user_id}/sessions/{session_id}/artifacts/{artifact_id}")
    async def get_session_artifact(
        request: Request,
        user_id: str, session_id: str, artifact_id: str, fallback: str = "",
        raw: bool = False,
    ):
        """Serve one file this session mirrored into its own directory.

        The sibling of ``/api/artifact/<bucket>/<key>``, and deliberately not a
        replacement for it:

        =================  ============================  ==========================
        ..                 /api/artifact/<b>/<k>          this route
        =================  ============================  ==========================
        storage            S3/MinIO                       this host's disk
        scope              none                           one session, guarded
        answers            302 to a fresh presigned URL   the bytes
        needs              S3 reachable                   nothing
        survives export    only if the bucket is shared   yes
        =================  ============================  ==========================

        Graph attrs and report markdown store ``cos-artifact:<id>``, which
        carries no session — so an imported bundle resolves through whatever
        scope is asking, with nothing rewritten.

        Traversal is refused three times over: ``artifact_id`` is a plain path
        parameter, so Starlette will not match a ``/`` at all; ``resolve_path``
        re-checks the name against ``ARTIFACT_ID_RE`` and then confirms the
        resolved file still sits inside this session's directory.
        """
        from CoScientist.reporting import session_files

        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        key = (user_id, session_id)
        record = session_files.load_manifest(session_id, user_id).get(artifact_id) or {}
        path = session_files.resolve_path(key, artifact_id)

        if path is None:
            # The bundle was exported without its files, or the copy was pruned.
            # A second address beats a 404 when the record kept one.
            if fallback == "s3" and record.get("bucket") and record.get("s3_key"):
                from CoScientist.utils.report_links import artifact_link

                return RedirectResponse(
                    artifact_link(record["bucket"], record["s3_key"]), status_code=302
                )
            raise HTTPException(status_code=404, detail="No such artifact in this session.")

        media_type = (
            record.get("media_type")
            or mimetypes.guess_type(artifact_id)[0]
            or "application/octet-stream"
        )
        # A link to a markdown file is clicked by a person, who wants it read,
        # not its `**` and `#`. Only a navigation asks for text/html; the doc
        # panel's fetch() sends */* and keeps getting the bytes, as does ?raw=1.
        is_markdown = (
            media_type in ("text/markdown", "text/x-markdown")
            or artifact_id.lower().endswith((".md", ".markdown"))
        )
        if is_markdown and not raw and "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(
                _markdown_page(
                    # `cos-artifact:` refs inside resolve against this session,
                    # exactly as the same text would in the chat.
                    _readable_links(path.read_text(encoding="utf-8", errors="replace"), key),
                    title=record.get("label") or record.get("filename") or artifact_id,
                    raw_href="?raw=1",
                ),
                headers={"Cache-Control": "private, no-cache", "Vary": "Accept"},
            )

        # Show what a browser can show; hand the rest over as a download under
        # the name the tool gave it, not the hashed id.
        inline = media_type.startswith(("image/", "text/")) or media_type == "application/pdf"
        name = record.get("filename") or artifact_id
        # A wholly Cyrillic name folds to nothing, and `filename=" .png"` is
        # worse than no fallback at all. The id is ASCII by construction.
        ascii_name = name.encode("ascii", "ignore").decode().strip()
        if len(ascii_name.strip(" ._-")) < 3:
            ascii_name = artifact_id
        disposition = (
            f"{'inline' if inline else 'attachment'}; filename=\"{ascii_name}\"; "
            f"filename*=UTF-8''{quote(name)}"
        )
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Content-Disposition": disposition,
                # The id is the content hash, so the bytes behind it can never
                # change. Unlike /api/artifact/, whose 302 target expires.
                "Cache-Control": "private, max-age=31536000, immutable",
                # The same URL answers a navigation to a markdown file with a
                # rendered page; a cache must not hand one to the other.
                "Vary": "Accept",
            },
        )

    # --- Local users and sessions (process lifetime only) ---
    @app.get("/api/users")
    async def list_users():
        from CoScientist.config import get_settings
        web = get_settings().web
        default_username = web.coscientist_username.strip() if web.coscientist_username else None
        if default_username:
            users = runtime.registry.list_users()
            existing = [u for u in users if u["nickname"].casefold() == default_username.casefold()]
            if not existing:
                try:
                    user = runtime.registry.create_user(default_username)
                    runtime.registry.create_session(user["id"], title="New session")
                except ValueError:
                    pass
        return JSONResponse({
            "users": runtime.registry.list_users(),
            "defaultUsername": default_username,
            "serverBootId": runtime.boot_id,
        })

    @app.post("/api/users")
    async def create_user(data: dict):
        try:
            user = runtime.registry.create_user(data.get("nickname", ""))
        except ValueError as exc:
            status_code = 409 if "already registered" in str(exc) else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        return JSONResponse({"user": user}, status_code=201)

    @app.get("/api/users/{user_id}/sessions")
    async def list_user_sessions(user_id: str):
        try:
            sessions = runtime.registry.list_sessions(user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        from CoScientist.web.session_store import has_events
        for session in sessions:
            key = (user_id, session["id"])
            session["empty"] = not runtime.agent_events.get(key) and not has_events(*key)
        return JSONResponse({"sessions": sessions})

    @app.post("/api/users/{user_id}/sessions")
    async def create_user_session(user_id: str, data: dict):
        try:
            runtime.registry.require_user(user_id)
            raw_title = data.get("title", "")
            if not isinstance(raw_title, str):
                raise ValueError("Session title must be a string.")
            title = " ".join(raw_title.strip().split()) or "New session"
            if len(title) > 120:
                raise ValueError("Session title must be at most 120 characters.")
            session_id = f"session_{uuid4().hex}"
            from CoScientist.graph.session_scope import (
                GRAPH_SCOPE_SESSION_KEY,
                GRAPH_SCOPE_USER_KEY,
            )
            await runtime.session_service.create_session(
                app_name=APP_NAME,
                user_id=user_id,
                session_id=session_id,
                state={
                    "active_tasks": [],
                    GRAPH_SCOPE_USER_KEY: user_id,
                    GRAPH_SCOPE_SESSION_KEY: session_id,
                },
            )
            session = runtime.registry.create_session(
                user_id,
                title,
                session_id=session_id,
            )
            if get_settings().web.auto_clear_graph_enabled:
                _clear_session_graphs(user_id, session_id, view="all")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"session": session}, status_code=201)

    @app.post("/api/demo/replay")
    async def start_replay(data: dict):
        """Replay a recorded session into a fresh one, for a demonstration.

        The study being shown took hours; a booth has minutes, and re-running it
        on stage is not an option. This plays back what that run actually
        recorded — its events and the growth of its research graph — into a new
        session, at a chosen speed, so the screen shows the real run on a
        different clock. Every replayed event carries a marker saying so.
        """
        from CoScientist.web.replay import ReplaySession, load_recording

        bundle = str(data.get("bundle") or "").strip()
        if not bundle:
            raise HTTPException(status_code=400, detail="'bundle' is required")
        user_id = str(data.get("user_id") or "").strip()
        speed = float(data.get("speed") or 120)
        max_gap = float(data.get("max_gap") or 2.5)
        min_gap = float(data.get("min_gap") or 0.4)
        warmup = float(data.get("warmup") if data.get("warmup") is not None else 25.0)
        chat_gap = float(data.get("chat_gap") if data.get("chat_gap") is not None else 10.0)
        thoughts = bool(data.get("thoughts", True))
        title = str(data.get("title") or "Recorded study (replay)")

        try:
            events, graph = load_recording(bundle)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            if user_id:
                runtime.registry.require_user(user_id)
            else:
                users = runtime.registry.list_users() or []
                if not users:
                    raise HTTPException(
                        status_code=400,
                        detail="no user exists yet — open the UI once, then replay")
                user_id = users[0]["id"]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        session_id = f"session_{uuid4().hex}"
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )
        await runtime.session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
            state={"active_tasks": [], GRAPH_SCOPE_USER_KEY: user_id,
                   GRAPH_SCOPE_SESSION_KEY: session_id},
        )
        session = runtime.registry.create_session(user_id, title,
                                                  session_id=session_id)

        replay = ReplaySession(runtime, user_id, session_id, events=events,
                               graph=graph, speed=speed, max_gap=max_gap,
                               min_gap=min_gap, warmup=warmup, chat_gap=chat_gap,
                               thoughts=thoughts, source=bundle)
        asyncio.create_task(replay.run())
        return JSONResponse({
            "session": session, "user_id": user_id, "session_id": session_id,
            "events": len(events), "nodes": len(graph.get("nodes") or []),
            "speed": speed,
            "projected_seconds": round(replay._event_wall_clock(), 1),
            "open": f"/?user_id={user_id}&session_id={session_id}",
            "graph": f"/graph?user_id={user_id}&session_id={session_id}",
        }, status_code=201)

    @app.get("/api/users/{user_id}/sessions/{session_id}")
    async def get_user_session(user_id: str, session_id: str):
        try:
            session = runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"session": session})

    @app.patch("/api/users/{user_id}/sessions/{session_id}")
    async def rename_user_session(user_id: str, session_id: str, data: dict):
        try:
            session = runtime.registry.rename_session(
                user_id, session_id, data.get("title", "")
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse({"session": session})

    # Hiding only takes sessions out of the picker; nothing is deleted.
    @app.post("/api/users/{user_id}/sessions/hide-old")
    async def hide_old_user_sessions(user_id: str, data: dict):
        keep = data.get("keep", [])
        if not isinstance(keep, list) or not all(isinstance(item, str) for item in keep):
            raise HTTPException(status_code=400, detail="'keep' must be a list of session ids.")
        try:
            hidden = runtime.registry.hide_old_sessions(user_id, keep)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"hidden": hidden})

    @app.post("/api/users/{user_id}/sessions/unhide-all")
    async def unhide_user_sessions(user_id: str):
        try:
            shown = runtime.registry.unhide_sessions(user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"shown": shown})

    # --- Automatic stage checkpoints ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/checkpoints")
    async def get_session_checkpoints(user_id: str, session_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({
            "checkpoints": runtime.list_checkpoints((user_id, session_id)),
            "pipeline_stages": _pipeline_stages(),
        })

    @app.post(
        "/api/users/{user_id}/sessions/{session_id}/checkpoints/"
        "{checkpoint_id}/restore"
    )
    async def restore_session_checkpoint(
        user_id: str,
        session_id: str,
        checkpoint_id: str,
        data: dict,
    ):
        """Restore one stage boundary and optionally continue immediately.

        Later ADK events are removed from model context, but the UI transcript
        remains append-only for auditability and receives an explicit rollback
        marker.  Graph state is restored alongside ADK state.
        """
        from CoScientist.agents.checkpoint_plugin import CHECKPOINT_RESUME_STATE_KEY
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )
        from CoScientist.web.checkpoints import load_checkpoint

        key = (user_id, session_id)
        try:
            runtime.registry.require_session(user_id, session_id)
            checkpoint = await asyncio.to_thread(
                load_checkpoint, user_id, session_id, checkpoint_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        continue_run = bool(data.get("continue", True))
        async with runtime.control_lock(key):
            current = runtime.active_runs.get(key)
            if current is not None and not current.done():
                raise HTTPException(
                    status_code=409,
                    detail="Stop the active run before restoring a checkpoint.",
                )

            manager = await runtime.get_manager(user_id, session_id)
            adk_session = await runtime.session_service.get_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id,
            )
            if adk_session is None:
                raise HTTPException(status_code=404, detail="ADK session not found.")

            restored_state = dict(checkpoint["state"])
            restored_state[GRAPH_SCOPE_USER_KEY] = user_id
            restored_state[GRAPH_SCOPE_SESSION_KEY] = session_id
            restored_state[CHECKPOINT_RESUME_STATE_KEY] = {
                "checkpoint_id": checkpoint_id,
                "stage_index": checkpoint.get("stage_index", 0),
                "agent": checkpoint.get("agent"),
            }
            adk_session.state.clear()
            adk_session.state.update(restored_state)

            # All events from the old invocation are unsafe after a rollback:
            # they can mention outputs from stages later than this snapshot.
            baseline = max(0, int(checkpoint.get("adk_event_baseline") or 0))
            del adk_session.events[min(baseline, len(adk_session.events)):]
            replace_session = getattr(runtime.session_service, "replace_session", None)
            if replace_session is not None:
                await replace_session(adk_session)

            execution_graph = checkpoint.get("execution_graph")
            if isinstance(execution_graph, dict):
                from CoScientist.graph.memory import get_knowledge_graph
                await asyncio.to_thread(
                    get_knowledge_graph(
                        user_id=user_id, session_id=session_id,
                    ).restore,
                    execution_graph,
                )
            research_graph = checkpoint.get("research_graph")
            if isinstance(research_graph, dict):
                from CoScientist.graph.research.store import get_research_graph
                await asyncio.to_thread(
                    get_research_graph(
                        user_id=user_id, session_id=session_id,
                    ).restore,
                    research_graph,
                )

            runtime.tz_snapshots.pop(key, None)
            runtime.tool_full_values.pop(key, None)
            dataset_url = str(restored_state.get(DATASET_URL_STATE_KEY) or "")
            if dataset_url:
                runtime.dataset_urls[key] = dataset_url
            else:
                runtime.dataset_urls.pop(key, None)
            language = str(restored_state.get(REPORT_LANGUAGE_STATE_KEY) or "")
            if language in REPORT_LANGUAGES:
                runtime.report_languages[key] = language
            else:
                runtime.report_languages.pop(key, None)
            runtime.registry.touch_session(user_id, session_id, status="idle")

        stage_index = int(checkpoint.get("stage_index") or 0)
        marker = {
            "type": "checkpoint_restored",
            "checkpoint_id": checkpoint_id,
            "stage_index": stage_index,
            "stage_count": int(checkpoint.get("stage_count") or 0),
            "agent": checkpoint.get("agent"),
            "title": checkpoint.get("title") or checkpoint.get("agent"),
            "pipeline_stages": _pipeline_stages(),
            "dataset_url": runtime.dataset_urls.get(key, ""),
            "report_language": runtime.report_languages.get(key, ""),
            "continue": continue_run,
            "timestamp": datetime.now().isoformat(),
        }
        runtime.record_event(key, marker)
        await runtime.send(key, marker)

        started = False
        if continue_run:
            root_query = str(checkpoint.get("root_query") or "").strip()
            resume_message = (
                "Continue the original task from the restored pipeline "
                f"checkpoint immediately before stage {stage_index + 1} "
                f"({marker['title']}). Use the restored session state as the "
                "authoritative output of all earlier stages. Do not recreate "
                "earlier work; the runtime will fast-forward any earlier "
                "stages that a containing workflow enters. Execute the target "
                "stage and every later required stage, then produce the final "
                "deliverable."
            )
            if root_query:
                resume_message += f"\n\nOriginal user request:\n{root_query}"
            started = await runtime.start_run(key, {
                "message": resume_message,
                "_checkpoint_resume": True,
                "_checkpoint_root_query": root_query or resume_message,
            })
            if not started:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "Checkpoint was restored, but another run claimed the "
                        "session before continuation could start."
                    ),
                )

        return JSONResponse({
            "status": "restored",
            "checkpoint": {
                key: marker[key]
                for key in (
                    "checkpoint_id", "stage_index", "stage_count", "agent", "title"
                )
            },
            "continued": started,
        })

    # --- Session export / import / save / restore ---
    @app.post("/api/users/{user_id}/sessions/{session_id}/export")
    async def export_session_endpoint(user_id: str, session_id: str):
        """Download a full session bundle as a .cossession.zip file."""
        from urllib.parse import quote
        from CoScientist.web.session_bundle import export_session, BUNDLE_EXTENSION
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        bundle_bytes = await export_session(runtime, (user_id, session_id))
        session_meta = runtime.registry.get_session(user_id, session_id) or {}
        title = session_meta.get("title", "session")
        safe_title = "".join(
            c if c.isalnum() or c in " _-" else "_" for c in title
        )[:60].strip() or "session"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{safe_title}_{stamp}{BUNDLE_EXTENSION}"

        ascii_title = "".join(
            c if (c.isascii() and c.isalnum()) or c in " _-" else "_" for c in title
        )[:60].strip() or "session"
        fallback_filename = f"{ascii_title}_{stamp}{BUNDLE_EXTENSION}"
        encoded_filename = quote(filename)

        return Response(
            content=bundle_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{fallback_filename}"; filename*=UTF-8\'\'{encoded_filename}',
            },
        )

    @app.post("/api/users/{user_id}/sessions/{session_id}/save")
    async def save_session_endpoint(user_id: str, session_id: str):
        """Save a session bundle to disk (no download)."""
        from CoScientist.web.session_bundle import export_session, save_to_disk
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        bundle_bytes = await export_session(runtime, (user_id, session_id))
        session_meta = runtime.registry.get_session(user_id, session_id) or {}
        title = session_meta.get("title", "session")
        filename = save_to_disk(bundle_bytes, title, user_id, session_id)
        return JSONResponse({"status": "success", "filename": filename})

    @app.post("/api/import-session/preview")
    async def preview_import_endpoint(request: Request):
        """Inspect a .cossession.zip bundle without importing it.

        Returns manifest info and whether MCP builds are included.
        """
        from CoScientist.web.session_bundle import preview_bundle

        content_type = (request.headers.get("content-type") or "").lower()
        if "multipart" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(status_code=400, detail="No file uploaded.")
            bundle_bytes = await upload.read()
        else:
            bundle_bytes = await request.body()
        if not bundle_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")
        try:
            result = preview_bundle(bundle_bytes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result)

    @app.post("/api/users/{user_id}/import-session")
    async def import_session_endpoint(user_id: str, request: Request):
        """Import a session from an uploaded .cossession.zip bundle.

        Accepts ``multipart/form-data`` with a ``file`` field (and an optional
        ``rebuild_mcp`` field), OR raw bytes with
        ``application/zip``/``application/octet-stream``.
        """
        from CoScientist.web.session_bundle import import_session

        rebuild_mcp = False
        content_type = (request.headers.get("content-type") or "").lower()
        if "multipart" in content_type:
            form = await request.form()
            upload = form.get("file")
            if upload is None:
                raise HTTPException(status_code=400, detail="No file uploaded.")
            bundle_bytes = await upload.read()
            rebuild_mcp = str(form.get("rebuild_mcp", "")).lower() in ("true", "1", "yes")
        else:
            bundle_bytes = await request.body()

        if not bundle_bytes:
            raise HTTPException(status_code=400, detail="Empty file.")

        # Always import into the ITMO_DEV user, ignoring the URL user_id
        target_nickname = "ITMO_DEV"
        try:
            result = await import_session(runtime, target_nickname, bundle_bytes,
                                          rebuild_mcp=rebuild_mcp)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result, status_code=201)

    @app.get("/api/saved-sessions")
    async def list_saved_sessions_endpoint():
        """List all session bundles saved to disk."""
        from CoScientist.web.session_bundle import list_saved_sessions
        return JSONResponse({"sessions": list_saved_sessions()})

    @app.post("/api/restore-session")
    async def restore_session_endpoint(data: dict):
        """Restore a previously saved session from disk."""
        from CoScientist.web.session_bundle import (
            import_session,
            read_saved_bundle,
        )
        filename = data.get("filename", "")
        if not filename:
            raise HTTPException(status_code=400, detail="filename is required.")
        try:
            bundle_bytes = read_saved_bundle(filename)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        rebuild_mcp = bool(data.get("rebuild_mcp", False))
        target_nickname = "ITMO_DEV"
        try:
            result = await import_session(runtime, target_nickname, bundle_bytes,
                                          rebuild_mcp=rebuild_mcp)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(result, status_code=201)

    @app.get("/api/saved-sessions/{filename}/events")
    async def saved_session_events_endpoint(filename: str):
        """The event log of a saved bundle, for replaying a run in the UI.

        Feeds the status indicator's `?demo=<filename>` mode: the frontend can
        drive its state machine off a real recorded run instead of costing a
        live one.
        """
        from CoScientist.web.session_bundle import read_saved_events
        try:
            events = read_saved_events(filename)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse({"events": events})

    @app.delete("/api/saved-sessions/{filename}")
    async def delete_saved_session_endpoint(filename: str):
        """Delete a saved session bundle from disk."""
        from CoScientist.web.session_bundle import delete_saved_session
        if delete_saved_session(filename):
            return JSONResponse({"status": "deleted"})
        raise HTTPException(status_code=404, detail=f"No saved session '{filename}'.")

    @app.get("/api/saved-sessions/{filename}/download")
    async def download_saved_session_endpoint(filename: str):
        """Download a previously saved bundle from disk."""
        from urllib.parse import quote
        from CoScientist.web.session_bundle import read_saved_bundle
        try:
            bundle_bytes = read_saved_bundle(filename)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        safe_name = Path(filename).name
        ascii_name = "".join(
            c if c.isascii() and (c.isalnum() or c in "._-") else "_" for c in safe_name
        ) or "session.cossession.zip"
        encoded_name = quote(safe_name)
        return Response(
            content=bundle_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded_name}',
            },
        )

    # --- HITL diagnostics ---
    @app.get("/api/hitl-status")
    async def hitl_status():
        """Why am I (not) being asked questions — one glance."""
        from CoScientist.config import get_settings

        agents = {
            name: getattr(agent, "hitl_handler", None) is not None
            for name, agent in agent_system.agents.items()
            if hasattr(agent, "hitl_handler")
        }
        return JSONResponse({
            "hitl_enabled": get_settings().web.hitl_enabled,
            "websocket_connections": runtime.hitl_handler.connection_count(),
            "session_agents_with_handler": agents,
            "pending_requests": runtime.hitl_handler.pending_summary(),
            "auto_approve_timeout_seconds": runtime.hitl_handler.HITL_TIMEOUT_SECONDS,
        })

    # --- Knowledge graph (live view) ---
    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page():
        # no-store: otherwise the browser heuristically caches graph.html and
        # silently serves an OLD viewer after an update.
        return HTMLResponse(
            (WEB_DIR / "templates" / "graph.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/trace", response_class=HTMLResponse)
    async def trace_page():
        """Sessions, newest first; pick one to see its requests in order."""
        return HTMLResponse(
            (WEB_DIR / "templates" / "trace.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/stats", response_class=HTMLResponse)
    async def stats_page():
        """The paper database statistics, as the paper-analysis server reports them."""
        return HTMLResponse(
            (WEB_DIR / "templates" / "stats.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/tools", response_class=HTMLResponse)
    async def tools_catalog_page():
        """Deployment-wide catalogue of configured MCP tools."""
        return HTMLResponse(
            _versioned_static_refs(
                (WEB_DIR / "templates" / "tools.html").read_text(encoding="utf-8"),
                _static_dir,
            ),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/agent-tree", response_class=HTMLResponse)
    async def agent_tree_page():
        """Visualize the effective multi-agent configuration of one session."""
        return HTMLResponse(
            _versioned_static_refs(
                (WEB_DIR / "templates" / "agent_tree.html").read_text(encoding="utf-8"),
                _static_dir,
            ),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/mcp-tools")
    async def mcp_tools_catalog_api():
        """Return the saved catalogue immediately and refresh it when stale."""
        from CoScientist.tools.mcp_catalog import get_catalog

        return JSONResponse(
            await get_catalog(refresh_if_stale=True),
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/mcp-tools/refresh", status_code=202)
    async def refresh_mcp_tools_catalog_api():
        """Start one bounded MCP discovery pass (or join the current one)."""
        from CoScientist.tools.mcp_catalog import get_catalog, request_refresh

        started = await request_refresh(force=True)
        payload = await get_catalog(refresh_if_stale=False)
        payload["refresh_started"] = started
        return JSONResponse(
            payload,
            status_code=202,
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/paper-statistics")
    async def paper_statistics_api():
        """Call get_papers_database_statistics on the paper-analysis MCP server.

        The tool answers from memory, so this is quick and the page can poll it.
        """
        from CoScientist.tools.paper_statistics_client import fetch_paper_statistics

        payload = await fetch_paper_statistics()
        status_code = {"not_configured": 503, "unavailable": 502}.get(payload["status"], 200)
        return JSONResponse(payload, status_code=status_code, headers={"Cache-Control": "no-store"})

    def graph_payload(user_id: str, session_id: str, view: str,
                      turn: str | None = None):
        """Return one session's graph: research, execution, or execution
        regrouped as a chronological trace (``view=trace``)."""
        runtime.registry.require_session(user_id, session_id)
        try:
            from CoScientist.graph.memory import get_knowledge_graph
            from CoScientist.graph.research.store import get_research_graph

            if view == "research":
                # One study at a time, and the session's others listed beside
                # it — the same shape the execution log uses for requests.
                payload = get_research_graph(
                    user_id=user_id,
                    session_id=session_id,
                ).view_of(turn)
                # The graph page is a self-contained monolith: it loads none of
                # the chat page's scripts and so cannot read the settings the
                # usual way. A setting it must honour rides along with the data
                # it already asks for.
                payload["auto_node_reports"] = get_settings().web.node_report_auto
                return payload
            execution = get_knowledge_graph(
                user_id=user_id,
                session_id=session_id,
            ).full()
            if view == "trace":
                # The same records grouped by the prompt that caused them and
                # ordered by time — what the call graph cannot show.
                from CoScientist.graph.projection import turns
                return turns(execution)
            if view in ("", "execution"):
                # One request at a time: roster and hub removed, placed on a
                # clock. The payload also lists the session's other requests.
                from CoScientist.graph.projection import execution_tree
                # The scope is what turns a stored reference into a link, and it
                # is the scope of whoever is *reading* — so an imported session
                # resolves its artifacts under its own id.
                return execution_tree(
                    execution, turn=turn, scope=(user_id, session_id)
                )
            if view not in ("", "execution"):
                # `knowledge` and `memory` were served here until the knowledge
                # memory was removed. Falling through to the execution graph
                # would answer a bookmark for one view with the contents of
                # another, so say plainly that the view is gone.
                raise HTTPException(
                    status_code=404,
                    detail=f"unknown graph view '{view}' — use research, "
                           f"execution or trace; the slide is at .../graph.svg",
                )
            return execution
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — never break the UI
            return {"nodes": [], "edges": [], "error": str(exc)}

    @app.get("/api/users/{user_id}/sessions/{session_id}/graph")
    async def api_session_graph(
        user_id: str,
        session_id: str,
        view: str = "execution",
        # Which request (execution log) or which study (research graph) to draw.
        turn: str | None = None,
    ):
        try:
            payload = graph_payload(user_id, session_id, view, turn)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload)

    @app.get("/api/users/{user_id}/sessions/{session_id}/graph/framing")
    async def api_research_framing(
        user_id: str,
        session_id: str,
        study_id: str = "active",
    ):
        """Structured technical specification for the selected research.

        The graph itself remains a compact polling payload.  The full field
        descriptions and long values are fetched only when the reader opens the
        Technical specification card.
        """
        try:
            runtime.registry.require_session(user_id, session_id)
            from CoScientist.graph.research.store import get_research_graph

            payload = get_research_graph(
                user_id=user_id,
                session_id=session_id,
            ).framing_view_of(study_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return JSONResponse(payload, headers={"Cache-Control": "no-store"})

    @app.post("/api/users/{user_id}/sessions/{session_id}/graph/agent_summary")
    async def api_agent_summary(user_id: str, session_id: str, request: Request):
        """A few lines from a small model on what one agent did in one request.

        Body: ``{"node_id", "turn", "lang"}``. Only on request — the button in
        the execution log's panel — and remembered per trace, so a click on an
        agent that has not changed since is free.
        """
        body = await request.json() if await request.body() else {}
        node_id, turn = body.get("node_id"), body.get("turn") or None
        if not node_id:
            raise HTTPException(status_code=422, detail="node_id is required")
        try:
            tree = graph_payload(user_id, session_id, "execution", turn)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        node = next((n for n in tree.get("nodes", []) if n.get("id") == node_id), None)
        if node is None or node.get("kind") not in ("agent", "agent_call"):
            raise HTTPException(status_code=404, detail=f"agent {node_id!r} is not in this request")
        from CoScientist.graph.agent_summary import summarize
        from CoScientist.graph.summary_store import for_session
        try:
            result = await summarize(node, lang=str(body.get("lang") or "ru"),
                                     scope=f"{user_id}/{session_id}",
                                     force=bool(body.get("again")),
                                     # Durable: the in-process cache dies with
                                     # the process, and re-opening a finished
                                     # study used to re-buy every account.
                                     store=for_session((user_id, session_id)))
        except Exception as exc:  # noqa: BLE001 — the model is an outside service
            raise HTTPException(status_code=502, detail=f"summary failed: {exc}") from exc
        return JSONResponse(result)

    @app.get("/api/users/{user_id}/sessions/{session_id}/graph/agent_summary")
    async def api_agent_summary_stored(user_id: str, session_id: str,
                                       node_id: str, lang: str = "ru"):
        """What has already been written about this agent's run, or 404.

        Reads, never writes. The node report cites these accounts, and one
        research node can have half a dozen contributors — generating them
        inside that request would turn opening a card into six model calls.
        """
        runtime.registry.require_session(user_id, session_id)
        from CoScientist.graph.summary_store import latest

        kept = latest((user_id, session_id), node_id, lang=lang)
        if not kept:
            raise HTTPException(status_code=404,
                                detail=f"no summary written for {node_id!r}")
        return JSONResponse(kept)

    @app.post("/api/users/{user_id}/sessions/{session_id}/graph/node_report")
    async def api_node_report(user_id: str, session_id: str, request: Request):
        """Write, or return, the account of one RESEARCH-graph node.

        A sibling of the agent summary and deliberately not the same route: the
        two graphs have separate id namespaces, and a route that accepted either
        would answer about whichever node happened to match.
        """
        runtime.registry.require_session(user_id, session_id)
        body = await request.json() if await request.body() else {}
        node_id = str(body.get("node_id") or "")
        if not node_id:
            raise HTTPException(status_code=422, detail="node_id is required")

        from CoScientist.graph.research.store import get_research_graph
        from CoScientist.graph.summary_store import for_session
        from CoScientist.reporting import node_report

        store = get_research_graph(user_id=user_id, session_id=session_id)
        view = store.view_of(body.get("turn") or None)
        if not node_report._reportable(view, node_id):
            raise HTTPException(
                status_code=404,
                detail=f"{node_id!r} is not a node a report is written for")
        # A node's write-up is a document of the STUDY, and a study is written in
        # one language. The panel sends its own interface language, and a node
        # holds ONE write-up: two readers set differently would take turns
        # overwriting each other's copy, each paying for a model call to do it,
        # and every other reader would meet the loser's language. An explicit
        # session choice therefore wins; absent one, the page's own is honoured.
        lang = (runtime.report_languages.get((user_id, session_id), "")
                or str(body.get("lang") or "") or "ru")
        try:
            written = await node_report.write_report(
                view, node_id, scope=(user_id, session_id), store=store,
                summaries=for_session((user_id, session_id)), lang=lang,
                force=bool(body.get("again")))
        except Exception as exc:  # noqa: BLE001 — the model is an outside service
            raise HTTPException(status_code=502,
                                detail=f"node report failed: {exc}") from exc
        if written is None:
            raise HTTPException(status_code=404, detail=f"no node {node_id!r}")
        return JSONResponse(written)

    @app.get("/api/users/{user_id}/sessions/{session_id}/graph.svg")
    async def api_session_graph_svg(user_id: str, session_id: str):
        """The research graph as a presentation slide (same renderer as the CLI).

        Served as SVG so the live view and an exported deck are the same artifact
        — what the operator watches during a run is what goes into the report.
        """
        runtime.registry.require_session(user_id, session_id)
        try:
            from CoScientist.graph.research.slide_render import render_slide
            from CoScientist.graph.research.store import get_research_graph

            data = get_research_graph(user_id=user_id, session_id=session_id).full()
            svg = render_slide(data)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 — never break the UI
            svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="80">'
                '<rect width="600" height="80" fill="#0e1117"/>'
                f'<text x="16" y="46" fill="#f0554d" font-family="Arial" font-size="13">'
                f'slide render failed: {_esc(str(exc)[:90])}</text></svg>'
            )
        return Response(content=svg, media_type="image/svg+xml",
                        headers={"Cache-Control": "no-store"})
    @app.delete("/api/users/{user_id}/sessions/{session_id}/graph")
    async def delete_session_graph(
        user_id: str,
        session_id: str,
        view: str = "all",
    ):
        """Drop stored graph data on the operator's explicit request.

        ``execution`` and ``research`` belong to this session alone; ``memory``
        is the installation-wide semantic memory, so it disappears for every
        session at once. Both the research graph and the semantic memory are
        archived next to their files first; the execution graph is re-seeded
        with the agent roster so the Graph view keeps rendering.
        """
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        targets = GRAPH_DELETE_TARGETS if view == "all" else (view,)
        unknown = [name for name in targets if name not in GRAPH_DELETE_TARGETS]
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown graph view {unknown[0]!r}; expected one of "
                    f"{', '.join((*GRAPH_DELETE_TARGETS, 'all'))}"
                ),
            )

        deleted: dict[str, Any] = {}
        failed = False
        for name in targets:
            try:
                if name == "execution":
                    from CoScientist.graph.memory import reset_knowledge_graph
                    reset_knowledge_graph(user_id=user_id, session_id=session_id)
                    deleted[name] = {"scope": "session", "cleared": True}
                elif name == "research":
                    from CoScientist.graph.research.store import get_research_graph
                    archived = get_research_graph(
                        user_id=user_id,
                        session_id=session_id,
                    ).reset(archive=True)
                    deleted[name] = {
                        "scope": "session",
                        "cleared": True,
                        "archived": archived,
                    }
            except Exception as exc:  # noqa: BLE001 — report every target's fate
                logging.getLogger("CoScientist.web").warning(
                    "Graph deletion failed for %s: %s", name, exc
                )
                deleted[name] = {"cleared": False, "error": str(exc)}
                failed = True

        return JSONResponse(
            {"status": "error" if failed else "success", "deleted": deleted},
            status_code=500 if failed else 200,
        )

    @app.delete("/api/users/{user_id}/sessions/{session_id}/graph")
    async def delete_session_graph(
        user_id: str,
        session_id: str,
        view: str = "all",
    ):
        """Drop stored graph data on the operator's explicit request."""
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        allowed_views = (*GRAPH_DELETE_TARGETS, "all", "session")
        if view not in allowed_views:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown graph view {view!r}; expected one of "
                    f"{', '.join(allowed_views)}"
                ),
            )

        deleted, failed = _clear_session_graphs(user_id, session_id, view=view)
        return JSONResponse(
            {"status": "error" if failed else "success", "deleted": deleted},
            status_code=500 if failed else 200,
        )

    @app.get("/api/graph")
    async def api_graph(
        user_id: str = "",
        session_id: str = "",
        view: str = "execution",
    ):
        """Compatibility endpoint; an explicit session scope is mandatory."""
        if not user_id or not session_id:
            raise HTTPException(
                status_code=400,
                detail="user_id and session_id are required",
            )
        return await api_session_graph(user_id, session_id, view)

    # --- MCP build dashboard ---
    # Standalone alembic app mounted as a sub-app: its `/`, `/ws`, `/builds`,
    # `/builds/{jid}`, `/api/builds*`, `/artifacts` all live under `/alembic`
    # and never collide with the chat `/ws`. Nothing at the CoScientist root:
    # the sidebar's MCPBuilder entry goes straight to `/alembic/`.
    from CoScientist.alembic.web.app import create_app as _create_alembic_app
    app.mount("/alembic", _create_alembic_app())


    @app.get("/fedot-trace", response_class=HTMLResponse)
    async def fedot_trace_page():
        return HTMLResponse(
            (WEB_DIR / "templates" / "fedot_trace.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store"},
        )

    from CoScientist.web.fedot_routes import register_fedot_routes
    register_fedot_routes(app, runtime, WEB_DIR, _RevalidatingStatic)

    # --- Roadmap endpoints ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/roadmap")
    async def get_roadmap(user_id: str, session_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is None:
            raise HTTPException(status_code=404, detail="ADK session not found.")
        tasks = adk_session.state.get("_master_active_tasks") or adk_session.state.get("active_tasks", [])
        return JSONResponse({
            "content": json.dumps(tasks, ensure_ascii=False, indent=2),
            "tasks": _json_safe(tasks),
        })

    @app.post("/api/users/{user_id}/sessions/{session_id}/roadmap")
    async def save_roadmap(user_id: str, session_id: str, data: dict):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        content = data.get("content", "")
        try:
            tasks = json.loads(content) if isinstance(content, str) and content.strip() else []
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid roadmap JSON: {exc.msg}") from exc
        if not isinstance(tasks, list):
            raise HTTPException(status_code=400, detail="Roadmap must be a JSON list of tasks.")

        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        if adk_session is None:
            raise HTTPException(status_code=404, detail="ADK session not found.")
        await runtime.session_service.append_event(
            adk_session,
            Event(
                invocation_id=f"roadmap_{uuid4().hex}",
                author="user",
                actions=EventActions(state_delta={"active_tasks": tasks, "_master_active_tasks": tasks}),
            ),
        )
        runtime.registry.touch_session(user_id, session_id)
        return JSONResponse({"status": "success", "tasks": _json_safe(tasks)})


    # --- ТЗ panel (filled by a profile that builds a ТЗ, e.g. microfluidics) ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/tz")
    async def get_tz(user_id: str, session_id: str):
        """The session's ТЗ as the ТЗ panel renders it: the latest live snapshot,
        or — once the ТЗ stage is over — the ТЗ stored in the session state."""
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        key = (user_id, session_id)
        if key in runtime.tz_snapshots:
            return JSONResponse(runtime.tz_snapshots[key])

        from CoScientist.hitl.tz_panel import stored_tz_view

        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
        )
        view = stored_tz_view(adk_session.state) if adk_session else None
        if view is None:
            return JSONResponse({"type": "tz_snapshot", "phase": "empty", "sections": []})
        return JSONResponse(_json_safe({"type": "tz_snapshot", "phase": "stored", **view}))

    # --- ТЗ document ---
    @app.get("/api/tz-document")
    async def get_tz_document(name: str = ""):
        """Serve a ТЗ document from tz_documents/ (the latest one by default).

        The framing stage announces the file in the chat and hangs it off the
        «Постановка» card of the research graph; both links land here.

        Markdown is returned as text so the browser shows it; a .docx is
        returned as a download, because a Word file rendered as text is a
        screenful of binary. The file name travels in the link, so the two
        formats of one document are two links, not a format switch.
        """
        from fastapi.responses import PlainTextResponse
        from urllib.parse import quote

        tz_dir = Path("tz_documents")
        if not tz_dir.is_dir():
            return JSONResponse({"error": "no ТЗ documents yet"}, status_code=404)
        if name:
            # Only bare file names inside tz_documents/ — no path traversal.
            candidate = tz_dir / Path(name).name
            if not candidate.is_file():
                return JSONResponse({"error": f"no such document: {name}"}, status_code=404)
        else:
            # Newest of either format. `TZ_*` is the microfluidics profile's
            # naming, `ТЗ_*` the research frame's; both live here.
            files = sorted((p for p in tz_dir.iterdir()
                            if p.suffix.lower() in (".md", ".docx")),
                           key=lambda p: p.stat().st_mtime)
            if not files:
                return JSONResponse({"error": "no ТЗ documents yet"}, status_code=404)
            candidate = files[-1]
        if candidate.suffix.lower() == ".docx":
            return FileResponse(
                str(candidate),
                media_type="application/vnd.openxmlformats-officedocument."
                           "wordprocessingml.document",
                # RFC 5987: the names are Russian, and a bare `filename=` would
                # reach the browser as mojibake.
                headers={"Content-Disposition":
                         "attachment; filename*=UTF-8''" + quote(candidate.name)},
            )
        return PlainTextResponse(
            candidate.read_text(encoding="utf-8"),
            media_type="text/markdown; charset=utf-8",
        )

    # --- Settings endpoints ---
    @app.get("/api/settings")
    async def get_settings_api():
        """Return current WebSettings."""
        return JSONResponse(_settings_payload())

    @app.post("/api/settings")
    async def save_settings_api(data: dict):
        """Update WebSettings from the frontend."""
        before = _current_settings()
        _apply_frontend_settings(data)
        changed = before != _current_settings()
        queued = runtime.invalidate_agent_trees() if changed else 0
        return JSONResponse({
            "status": "success",
            "agentTreesQueued": queued,
            **_settings_payload(),
        })

    @app.get("/api/users/{user_id}/sessions/{session_id}/settings")
    async def get_session_settings_api(user_id: str, session_id: str):
        """Return process settings with this session's agent overlay applied."""
        key = (user_id, session_id)
        try:
            runtime.registry.require_session(*key)
            _, snapshot = runtime.settings_snapshot(key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        with settings_scope(snapshot):
            payload = _settings_payload()
        payload["agentConfiguration"] = runtime.agent_configuration_state(key)
        return JSONResponse(_json_safe(payload))

    @app.post("/api/users/{user_id}/sessions/{session_id}/settings")
    async def save_session_settings_api(user_id: str, session_id: str, data: dict):
        """Save agent topology for one session and keep other settings global."""
        key = (user_id, session_id)
        try:
            runtime.registry.require_session(*key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        # Existing settings outside agent composition retain their historical
        # process-wide semantics. Only the fields that can change the topology
        # are isolated and persisted with the selected session.
        global_payload = _without_agent_configuration(data)
        _, current_snapshot = runtime.settings_snapshot(key)
        candidate = current_snapshot.model_copy(deep=True)
        requested_overlay = _agent_configuration_payload(data)
        from CoScientist.web.agent_configuration_policy import (
            validate_agent_configuration_transition,
            validate_raw_agent_configuration_request,
        )
        try:
            validate_raw_agent_configuration_request(current_snapshot, requested_overlay)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        with settings_scope(candidate):
            _apply_frontend_settings(requested_overlay)
            normalized_overlay = _agent_configuration_payload(_current_settings())
        # Validate the complete requested composition before applying even the
        # process-wide part of the same form submission.  A forbidden agent
        # change therefore cannot leave a partially saved settings form.
        try:
            validate_agent_configuration_transition(current_snapshot, candidate)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        before_global = _current_settings()
        if global_payload:
            _apply_frontend_settings(global_payload)
        global_changed = before_global != _current_settings()
        if global_changed:
            runtime.session_settings_cache.clear()
            runtime.invalidate_agent_trees()

        _, saved_overlay = runtime.agent_configuration(key)
        if normalized_overlay != saved_overlay:
            runtime.save_agent_configuration(key, normalized_overlay)

        _, effective = runtime.settings_snapshot(key)
        with settings_scope(effective):
            payload = _settings_payload()
        payload["agentConfiguration"] = runtime.agent_configuration_state(key)
        await runtime.send(key, {
            "type": "agent_configuration",
            **payload["agentConfiguration"],
        })
        return JSONResponse(_json_safe({"status": "success", **payload}))

    # --- Agent info ---
    @app.get("/api/agents/catalog")
    async def get_agents_catalog():
        """Every agent of the profile with its declared values, for Settings → Agents."""
        from CoScientist.web.agent_settings import agents_catalog
        return JSONResponse(_json_safe(agents_catalog()))

    @app.get("/api/users/{user_id}/sessions/{session_id}/agents/catalog")
    async def get_session_agents_catalog(user_id: str, session_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
            _, snapshot = runtime.settings_snapshot((user_id, session_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        from CoScientist.web.agent_settings import agents_catalog
        with settings_scope(snapshot):
            return JSONResponse(_json_safe(agents_catalog()))

    @app.post("/api/users/{user_id}/sessions/{session_id}/agents/{agent_name}/enabled")
    async def set_session_agent_enabled(user_id: str, session_id: str, agent_name: str, data: dict):
        """Atomically change one agent without replacing other session overrides."""
        key = (user_id, session_id)
        try:
            runtime.registry.require_session(*key)
            _, snapshot = runtime.settings_snapshot(key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if set(data) != {"enabled"} or not isinstance(data["enabled"], bool):
            raise HTTPException(status_code=422, detail="Ожидается логическое поле enabled.")
        from CoScientist.web.agent_settings import agent_enabled_patch, agents_catalog

        candidate = snapshot.model_copy(deep=True)
        with settings_scope(candidate):
            before = _agent_configuration_payload(_current_settings())
            try:
                patch = agent_enabled_patch(agent_name, data["enabled"])
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="Агент не найден.") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            _apply_frontend_settings(patch)
            overlay = _agent_configuration_payload(_current_settings())
            catalog = agents_catalog()
        # No await between reading, merging and saving: a concurrent switch
        # cannot overwrite an unrelated agent, model or reasoning setting.
        if overlay != before:
            runtime.save_agent_configuration(key, overlay)
        state = runtime.agent_configuration_state(key)
        await runtime.send(key, {"type": "agent_configuration", **state})
        return JSONResponse(_json_safe({**catalog, "agentConfiguration": state}))

    @app.get("/api/users/{user_id}/sessions/{session_id}/agent-tree")
    async def get_session_agent_tree(user_id: str, session_id: str):
        key = (user_id, session_id)
        try:
            runtime.registry.require_session(*key)
            revision, snapshot = runtime.settings_snapshot(key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        state = runtime.agent_configuration_state(key)
        from CoScientist.web.agent_tree import project_agent_tree

        payload = project_agent_tree(
            snapshot,
            desired_revision=revision,
            active_revision=state["activeRevision"],
            running=state["running"],
        )
        return JSONResponse(_json_safe(payload), headers={"Cache-Control": "no-store"})

    @app.get("/api/users/{user_id}/sessions/{session_id}/agents/{agent_name}/tools")
    async def get_session_agent_tools(user_id: str, session_id: str, agent_name: str):
        key = (user_id, session_id)
        try:
            runtime.registry.require_session(*key)
            _, snapshot = runtime.settings_snapshot(key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        from CoScientist.agents import config_for_mode
        from CoScientist.tools.mcp_catalog import get_catalog
        from CoScientist.web.agent_tree import agent_tools_payload

        with settings_scope(snapshot):
            config = config_for_mode()
        if agent_name not in config.agents:
            raise HTTPException(status_code=404, detail=f"Unknown agent '{agent_name}'.")
        session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        session_state = dict(getattr(session, "state", None) or {}) if session else {}
        catalog = await get_catalog(refresh_if_stale=False)
        with settings_scope(snapshot):
            payload = agent_tools_payload(config, agent_name, session_state, catalog)
        return JSONResponse(_json_safe(payload), headers={"Cache-Control": "no-store"})

    @app.get("/api/agents")
    async def get_agents():
        """Return list of registered agents and system hierarchy."""
        from CoScientist.assembly.schema import get_config
        from CoScientist.web.agent_tree import agent_structure, call_graph_titles
        cfg = get_config()
        hierarchy = cfg.agent_hierarchy_map()
        internal_names = cfg.internal_agent_names()
        agents_list = []
        for name in cfg.build_order():
            ac = cfg.agent(name)
            agents_list.append({
                "name": ac.name,
                "class": ac.cls,
                "role": ac.cls,
                "description": ac.description,
                "enabled": ac.is_enabled(),
                "tools": list(ac.tools),
                "subordinates": list(ac.subordinates),
                "children": list(ac.children),
                "is_root": bool(ac.root),
                "is_internal": name in internal_names,
            })
        return JSONResponse({
            "agents": agents_list,
            "hierarchy": hierarchy,
            "delegatable_names": list(cfg.delegatable_names()),
            "pipeline_stages": cfg.linear_stages(),
            "internal_agents": sorted(internal_names),
            "titles": call_graph_titles(),
            **agent_structure(),
        })

    @app.post("/api/fedot-debug-run")
    async def fedot_debug_run(data: dict):
        from CoScientist.capabilities import fedot_mas_enabled

        if not fedot_mas_enabled():
            raise HTTPException(
                status_code=409,
                detail=("FEDOT.MAS is disabled. Enable the experiment route or "
                        "executor fallback before starting a debug run."),
            )
        task_description = data.get(
            "task_description",
            "Ping test: say hello, do nothing else.",
        )
        from CoScientist.tools.fedotmas_tools import FedotMASToolset

        async def _run():
            toolset = FedotMASToolset()
            try:
                await toolset.fedot_tool(task_description=task_description, tool_context=None)
            except Exception as exc:  # noqa: BLE001 — this is a debug trigger, never crash the server
                logging.getLogger("CoScientist.web").warning("fedot-debug-run failed: %r", exc)

        asyncio.create_task(_run())
        return JSONResponse({"status": "started", "task_description": task_description})

    # --- Events log ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/events")
    async def get_events(user_id: str, session_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        key = (user_id, session_id)
        events = runtime.agent_events.get(key) or []
        if not events:
            # Reopening a session from an earlier process: replay its transcript
            # from disk and keep it in memory for subsequent requests.
            try:
                from CoScientist.web.session_store import load_events
                events = load_events(user_id, session_id)
                if events:
                    runtime.agent_events[key] = list(events)
            except Exception:  # noqa: BLE001
                events = []
        return JSONResponse({"events": events[-100:]})

    @app.get("/api/users/{user_id}/sessions/{session_id}/call-graph")
    async def get_call_graph_skeleton(user_id: str, session_id: str):
        """Where each agent can run in this session, for the side-nav graph."""
        key = (user_id, session_id)
        try:
            runtime.registry.require_session(*key)
            _, snapshot = runtime.settings_snapshot(key)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        from CoScientist.graph.memory import get_knowledge_graph
        from CoScientist.web.agent_tree import call_graph_skeleton, session_agent_names, session_profile
        events = runtime.agent_events.get(key) or []
        if not events:
            try:
                from CoScientist.web.session_store import load_events
                events = load_events(user_id, session_id)
            except Exception:  # noqa: BLE001
                events = []
        # Each profile has agents of its own: draw the one this session ran.
        execution = get_knowledge_graph(user_id=user_id, session_id=session_id).full()
        profile = session_profile(session_agent_names(events, execution))
        return JSONResponse(call_graph_skeleton(snapshot, profile), headers={"Cache-Control": "no-store"})

    @app.get("/api/users/{user_id}/sessions/{session_id}/agent-runs")
    async def get_agent_runs(user_id: str, session_id: str):
        """The session's agent runs from its execution graph, for the call
        graph of a session whose transcript predates agent-run records."""
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        from CoScientist.graph.memory import get_knowledge_graph
        from CoScientist.web.agent_tree import agent_run_events
        execution = get_knowledge_graph(user_id=user_id, session_id=session_id).full()
        return JSONResponse({"events": agent_run_events(execution)})

    # --- Usage and cost ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/metrics")
    async def get_metrics(user_id: str, session_id: str, report: bool = False):
        """What this session has spent, broken down by agent.

        Read straight from the ledger rather than from the last pushed snapshot,
        so the answer is current even when no tab is connected. ``?report=1``
        adds the console rendering, for a quick look from the terminal.
        """
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        from CoScientist.logging.metrics import format_report, snapshot

        data = snapshot(key=(user_id, session_id))
        if report:
            data = {**data, "report": format_report(data)}
        return JSONResponse(_json_safe(data))

    # --- Full (untruncated) tool args/results, for the ToolsViewer's "Show
    # full result" — the live socket stream only ever carries a preview.
    @app.get("/api/users/{user_id}/sessions/{session_id}/tool-activity/{call_id}")
    async def get_tool_activity_full(user_id: str, session_id: str, call_id: str):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        entry = runtime.tool_full_values.get((user_id, session_id), {}).get(call_id)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail="No stored full result for this call (it may have expired or was never truncated)",
            )
        return JSONResponse({"call_id": call_id, **entry})

    # --- Usage and cost ---
    @app.get("/api/users/{user_id}/sessions/{session_id}/metrics")
    async def get_metrics(user_id: str, session_id: str, report: bool = False):
        """What this session has spent, broken down by agent.

        Read straight from the ledger rather than from the last pushed snapshot,
        so the answer is current even when no tab is connected. ``?report=1``
        adds the console rendering, for a quick look from the terminal.
        """
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        from CoScientist.logging.metrics import format_report, snapshot

        data = snapshot(key=(user_id, session_id))
        if report:
            data = {**data, "report": format_report(data)}
        return JSONResponse(_json_safe(data))

    # --- WebSocket ---
    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        user_id = (ws.query_params.get("user_id") or "").strip()
        session_id = (ws.query_params.get("session_id") or "").strip()
        await ws.accept()
        try:
            user = runtime.registry.require_user(user_id)
            session_meta = runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            await ws.send_json({"type": "error", "message": str(exc)})
            await ws.close(code=4404, reason="Unknown user or session")
            return

        key = (user_id, session_id)
        runtime.registry.touch_session(user_id, session_id)
        if not runtime.agent_events.get(key):
            # A session from an earlier process: its chat and HITL cards live
            # only in the on-disk transcript until something reloads them.
            try:
                from CoScientist.web.session_store import load_events
                events = load_events(user_id, session_id)
                if events:
                    runtime.agent_events[key] = list(events)
            except Exception:  # noqa: BLE001
                pass
        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )
        await runtime.attach_with_snapshot(
            key,
            ws,
            user=user,
            session=session_meta,
            active_tasks=(
                (adk_session.state.get("_master_active_tasks") or adk_session.state.get("active_tasks", []))
                if adk_session else []
            ),
        )
        # Re-deliver only HITL requests belonging to this session.
        await runtime.hitl_handler.attach_websocket(ws, key)
        durable_handle = runtime.execution_handle(key)
        if durable_handle is not None:
            for cause, decision in durable_handle.status().pending_decisions.items():
                if cause.startswith("hitl:") and isinstance(decision.get("payload"), dict):
                    await runtime.send_socket(ws, decision["payload"], key)
        delivered_interrupts = set()
        for pending in runtime.pending_hitl.values():
            payload = pending.get("payload")
            wait_event = pending.get("event")
            unresolved = (
                pending.get("response") is None
                and (wait_event is None or not wait_event.is_set())
            )
            if (
                pending.get("session_key") == key
                and unresolved
                and payload
                and id(payload) not in delivered_interrupts
            ):
                await runtime.send_socket(ws, payload, key)
                delivered_interrupts.add(id(payload))

        try:
            while True:
                raw = await ws.receive_text()
                data = json.loads(raw)
                msg_type = data.get("type", "")

                if msg_type == "chat_message":
                    if await runtime.start_run(key, data):
                        await runtime.send_socket(ws, {
                            "type": "chat_accepted",
                            "message_text": data.get("message", ""),
                        }, key)
                    else:
                        await runtime.send_socket(ws, {
                            "type": "chat_rejected",
                            "message_text": data.get("message", ""),
                            "message": (
                                "This session is already processing a request. "
                                "Stop it or wait for completion before sending another."
                            ),
                        }, key)
                elif msg_type == "stop_chat":
                    await runtime.stop_run(key)
                elif msg_type == "set_dataset_url":
                    try:
                        url = _validated_dataset_url(data.get("dataset_url"))
                    except ValueError as exc:
                        await runtime.send_socket(ws, {
                            "type": "dataset_url_rejected",
                            "message": str(exc),
                        }, key)
                        continue
                    await runtime.apply_dataset_url(key, url)
                    # Broadcast: every tab on this session shows the same
                    # attachment, whichever one set it.
                    await runtime.send(key, {
                        "type": "dataset_url",
                        "dataset_url": url,
                    })
                elif msg_type == "set_report_language":
                    try:
                        lang = _validated_report_language(data.get("report_language"))
                    except ValueError as exc:
                        await runtime.send_socket(ws, {
                            "type": "report_language_rejected",
                            "message": str(exc),
                        }, key)
                        continue
                    current_run = runtime.active_runs.get(key)
                    if current_run is not None and not current_run.done():
                        # A run reads the language when it starts. A mid-run
                        # change would split the report between languages.
                        await runtime.send_socket(ws, {
                            "type": "report_language_rejected",
                            "message": (
                                "A run is in progress. "
                                "Wait for it to finish before changing the language."
                            ),
                            "reason": "run_active",
                        }, key)
                        continue
                    await runtime.apply_report_language(key, lang)
                    # Broadcast: the report is one document, so every tab on the
                    # session agrees on its language, whichever one picked it.
                    await runtime.send(key, {
                        "type": "report_language",
                        "report_language": lang,
                    })
                elif msg_type == "hitl_response":
                    _handle_hitl_response(runtime, key, data)
                elif msg_type == "hitl_hold":
                    # "Pause" on a Work Order veto window: stop the countdown.
                    request_id = data.get("request_id")
                    if request_id:
                        await runtime.hitl_handler.hold_request(request_id, key)
                elif msg_type == "ping":
                    await runtime.send_socket(ws, {"type": "pong"}, key)
                else:
                    await runtime.send_socket(ws, {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }, key)
        except WebSocketDisconnect:
            runtime.hitl_handler.detach_websocket(ws, key)
            runtime.detach_socket(key, ws)
            print("[WebSocket] Client disconnected")
        except Exception as exc:
            runtime.hitl_handler.detach_websocket(ws, key)
            runtime.detach_socket(key, ws)
            print(f"[WebSocket] Error: {exc}")

    return app


# ---------------------------------------------------------------------------
# HITL helpers
# ---------------------------------------------------------------------------
def _cancel_pending_hitl(
    runtime: WebRuntime,
    key: SessionKey | None = None,
) -> None:
    """Cancel ADK RequestInput waits for one session, or all on shutdown."""
    for interrupt_id, info in list(runtime.pending_hitl.items()):
        if key is not None and info.get("session_key") != key:
            continue
        info["event"].set()
        runtime.pending_hitl.pop(interrupt_id, None)


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------
async def _handle_chat(runtime: WebRuntime, key: SessionKey, data: dict):
    """Run user query through the agent pipeline, streaming events.

    Handles ADK RequestInput HITL: when the workflow pauses (interrupt event),
    this sends the HITL request to the browser, waits for the response, then
    resumes the workflow by calling run_async with a FunctionResponse message.
    """
    query = data.get("message", "").strip()
    internal_resume = bool(data.get("_checkpoint_resume") or data.get("_execution_resume"))
    run_status_version = int(
        data.get("_run_status_version", runtime.run_versions[key])
    )
    # A chat message may pin the report language (the settings-modal select on
    # the pre-#347 UI did this). Absent or invalid means "keep the per-session
    # choice" — defaulting here would clobber the set_report_language mirror.
    report_language = data.get("report_language")
    if report_language not in ("en", "ru"):
        report_language = None
    if not query:
        await runtime.send(key, {"type": "error", "message": "Empty query"})
        return

    user_id, session_id = key

    # A checkpoint continuation is a control instruction generated by the
    # server, not a new user utterance.  The restore marker already visible in
    # chat describes it; do not pretend the internal prompt came from the user.
    if not internal_resume:
        user_event = {
            "type": "user_message",
            "message": query,
            "timestamp": datetime.now().isoformat(),
        }
        runtime.record_event(key, user_event)
        await runtime.send(key, user_event)

    try:
        manager = await runtime.get_manager(
            user_id,
            session_id,
            settings_snapshot=data.get("_settings_snapshot"),
            agent_revision=data.get("_agent_revision"),
        )
        # The attachment may predate the ADK session (it is created with the
        # manager), so mirror it into state now that the session exists — this is
        # what puts the link in front of CoderAgent.
        await runtime.apply_dataset_url(key)
        # Same reason as the attachment above: the language may have been picked
        # before the ADK session existed.
        await runtime.apply_report_language(key)
        adk_session = await runtime.session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
        )
        runtime.run_contexts[key] = {
            "run_version": run_status_version,
            "adk_event_baseline": len(adk_session.events) if adk_session else 0,
            "root_query": data.get("_checkpoint_root_query") or query,
        }
        runtime.registry.touch_session(user_id, session_id, status="processing")
        execution_lock = runtime.execution_locks[key]

        async with execution_lock:
            with manager.settings_context():
                await _run_chat_invocation(
                    runtime,
                    key,
                    manager,
                    query,
                    run_status_version=run_status_version,
                    report_language=report_language,
                    continuation=bool(data.get("_execution_resume")),
                    continuation_instruction=data.get("_execution_resume_instruction"),
                )

    except asyncio.CancelledError:
        _cancel_pending_hitl(runtime, key)
        runtime.hitl_handler.reset(key)
        runtime.registry.touch_session(user_id, session_id, status="idle")
        raise
    except Exception as exc:
        _cancel_pending_hitl(runtime, key)
        runtime.hitl_handler.reset(key)
        runtime.registry.touch_session(user_id, session_id, status="idle")
        handle = runtime.execution_handle(key)
        if handle is not None:
            handle.controller.request_pause(handle.run_id, "execution_error",
                pending_decision={"error_type": type(exc).__name__, "message": str(exc)})

        # Whatever card the operator gets, the exception itself goes to the log.
        # Without this the proxy branch below was a dead end: it replaced the
        # error with advice about the proxy and recorded nothing else, so a run
        # that failed for some other reason left no trace to diagnose from.
        logging.getLogger("CoScientist.web").exception(
            "Run failed for %s/%s: %s", user_id, session_id, exc)

        if is_proxy_error(exc):
            msg = (
                f"**Error connecting to proxy server**\n\n"
                f"Failed to connect to the proxy server to execute the query to the language model. "
                f"Please ensure the proxy container is running, the corporate VPN is enabled (other - disabled), and "
                f"and the proxy is accessible.\n\n"
                f"`{type(exc).__name__}: {exc}`"
            )
            agent_msg = {
                "type": "agent_event",
                "author": "OrchestratorAgent",
                "content": msg,
                "is_final": True,
                "timestamp": datetime.now().isoformat(),
            }
            runtime.record_event(key, agent_msg)
            await runtime.send(key, agent_msg)
            await runtime.send(key, {
                "type": "final_response",
                "content": msg,
            })
        else:
            error_msg = f"Error processing query: {str(exc)}"
            error_event = {
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat(),
            }
            await runtime.send(key, error_event)
            runtime.record_event(key, error_event)



async def _run_chat_invocation(
    runtime: WebRuntime,
    key: SessionKey,
    manager: CoScientistManager,
    query: str,
    *,
    run_status_version: int,
    report_language: str | None = None,
    continuation: bool = False,
    continuation_instruction: str | None = None,
) -> None:
    """Execute one serialized ADK invocation for a session."""
    user_id, session_id = key
    current_message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    # The Result Aggregator runs as the terminal stage of the SAME run_async, so its
    # format_results reads report_config mid-invocation — set it before the run.
    # TODO(planning): thread a real ReportConfig (e.g. --latex mode) from the web layer.
    # The report LANGUAGE is not part of this — it travels as session state
    # (report_language) and reaches the prompt through inject_report_language.
    report_config = ReportConfig()
    await manager._set_state("report_config", report_config.to_state())
    # The question the user just asked is the root of the research graph, and
    # nothing else guarantees it exists — see seed_research_context.
    if not continuation:
        await manager.seed_research_context(query)
    else:
        current_message = types.Content(role="user", parts=[types.Part(text=(
            continuation_instruction
            or "Resume the saved research state. Preserve the approved plan and active attempt. "
               "Do not repeat completed tools or tasks. Process any saved pending result first."
        ))])
        from CoScientist.execution_control import current_run
        handle = current_run()
        restored_answer = handle.status().metadata.get("request_input_response") if handle else None
        if restored_answer:
            current_message = types.Content(role="user", parts=[create_request_input_response(
                restored_answer["interrupt_id"], restored_answer["response"],
            )])
            handle.controller.update_metadata(handle.run_id, {"request_input_response": None})
    # Prompt templates read {report_language?} from session state. An explicit
    # per-message choice overrides the per-session mirror for this run; a
    # message without one leaves the mirror (and the callback default) alone.
    if report_language:
        await manager._set_state("report_language", report_language)

    final_response = "No response"
    # The report is the LAST final-response text of the run — the terminal aggregator
    # stage's Markdown (the orchestrator's own answer is superseded by it).
    report_markdown = ""

    try:
        # Loop: run -> check for HITL interrupt -> wait for response -> resume
        while True:
            hitl_interrupt_event = None
            pending_wait_event = None

            async for event in manager.runner.run_async(
                user_id=manager.user_id,
                session_id=manager.session_id,
                new_message=current_message,
                # The shared durable budget gate owns the 100-call quota.
                run_config=RunConfig(
                    max_llm_calls=0
                ),
            ):
                # Stream each event to frontend
                event_data = {
                    "type": "agent_event",
                    "author": event.author or "system",
                    "is_final": event.is_final_response(),
                    "timestamp": datetime.now().isoformat(),
                }

                if event.content and event.content.parts:
                    # A model turn that only carries a function call often still
                    # ships a blank text part. Requiring real characters keeps
                    # it from surfacing as an empty message bubble.
                    # Skip thinking/reasoning parts and strip inline thinking tags.
                    text_parts = []
                    for p in event.content.parts:
                        if getattr(p, "thought", False):
                            continue
                        t = getattr(p, "text", None)
                        if not t or not t.strip():
                            continue
                        cleaned = strip_thinking(t)
                        if cleaned:
                            text_parts.append(cleaned)
                    if text_parts:
                        event_data["content"] = _readable_links(
                            "\n".join(text_parts), key
                        )

                    # Extract tool calls (function_call) and tool responses
                    # (function_response) so the frontend can show live
                    # tool activity for agents like ExperimentAgent.
                    tool_calls = []
                    tool_responses = []
                    for part in event.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            fc = part.function_call
                            tool_calls.append({
                                "name": fc.name,
                                "args": _json_safe(dict(fc.args) if fc.args else {}),
                            })
                        if hasattr(part, 'function_response') and part.function_response:
                            fr = part.function_response
                            # Deep JSON-safe: a tool may return a dict that NESTS a
                            # non-serializable object (e.g. a pydantic MCPServer under
                            # "result"). A shallow isinstance(dict) check passes such a
                            # payload straight through and then ws.send_json crashes the
                            # whole stream — so sanitise recursively (objects -> str).
                            tool_responses.append({
                                "name": fr.name,
                                "response": _json_safe(fr.response),
                            })
                    if tool_calls:
                        event_data["tool_calls"] = tool_calls
                    if tool_responses:
                        # Sandbox links are NOT lifted out of the response here:
                        # _wire_sandbox_links already delivered them when the
                        # container came up. The frontend still renders them from
                        # a response the push never covered, and skips duplicates.
                        event_data["tool_responses"] = tool_responses

                if event.actions and event.actions.escalate:
                    event_data["escalation"] = event.error_message or "Unknown error"

                # Check for HITL RequestInput interrupt
                if has_request_input_function_call(event):
                    hitl_interrupt_event = event
                    interrupt_ids = get_request_input_interrupt_ids(event)

                    # Extract the message and schema from the function call args
                    hitl_message = ""
                    hitl_schema = None
                    for part in event.content.parts:
                        if (part.function_call
                            and part.function_call.name == REQUEST_INPUT_FUNCTION_CALL_NAME):
                            args = part.function_call.args or {}
                            hitl_message = args.get("message", "")
                            hitl_schema = args.get("responseSchema") or args.get("response_schema")

                    # Send HITL request to browser
                    hitl_payload = {
                        "type": "hitl_request",
                        "request_id": interrupt_ids[0] if interrupt_ids else "",
                        "interrupt_id": interrupt_ids[0] if interrupt_ids else "",
                        "interrupt_ids": interrupt_ids,
                        "message": hitl_message,
                        "response_schema": hitl_schema,
                        "agent_name": event.author or "system",
                        "invoked_via": "request_input",
                        "timestamp": datetime.now().isoformat(),
                    }
                    event_data["hitl_request"] = hitl_payload
                    # Register before delivery so an immediate browser answer
                    # cannot race ahead of the pending-request table.
                    pending_wait_event = asyncio.Event()
                    for iid in interrupt_ids:
                        runtime.pending_hitl[iid] = {
                            "event": pending_wait_event,
                            "response": None,
                            "session_key": key,
                            "payload": hitl_payload,
                        }
                    from CoScientist.execution_control import current_run
                    execution_handle = current_run()
                    if execution_handle is not None:
                        for iid in interrupt_ids:
                            execution_handle.controller.request_pause(execution_handle.run_id, f"hitl:{iid}",
                                pending_decision={"payload": hitl_payload, "kind": "request_input"})
                            execution_handle.controller.journal(execution_handle.run_id, "hitl_requested",
                                                                action_id=iid, data=hitl_payload)
                    if runtime.checkpoint_hitl is not None:
                        await runtime.checkpoint_hitl(key, {
                            **hitl_payload,
                            "agent": event.author or "system",
                            "hitl_kind": "request_input",
                        })
                    runtime.record_event(key, hitl_payload)
                    await runtime.send(key, hitl_payload)

                runtime.record_event(key, event_data)
                await runtime.send(key, event_data)

                if not hitl_interrupt_event:
                    # Skip thinking parts; keep the LAST final-response text (the
                    # terminal aggregator stage produces the report).
                    text = CoScientistManager._final_text(event)
                    if text is not None:
                        final_response = text
                        report_markdown = text

            # If there was a HITL interrupt, wait for the browser response
            if hitl_interrupt_event:
                interrupt_ids = get_request_input_interrupt_ids(hitl_interrupt_event)

                wait_event = pending_wait_event or asyncio.Event()
                for iid in interrupt_ids:
                    runtime.pending_hitl.setdefault(iid, {
                        "event": wait_event,
                        "response": None,
                        "session_key": key,
                    })

                print(f"[HITL] Waiting for browser response for interrupts: {interrupt_ids}")

                # Wait for ALL interrupt responses (with timeout)
                try:
                    await asyncio.wait_for(wait_event.wait(), timeout=600)
                except asyncio.TimeoutError:
                    print("[HITL] Review window elapsed; waiting for an explicit decision")
                    timeout_event = {
                        "type": "hitl_timeout",
                        "request_id": interrupt_ids[0] if interrupt_ids else "",
                        "interrupt_ids": interrupt_ids,
                        "agent_name": hitl_interrupt_event.author or "system",
                        "timeout_seconds": 600,
                        "paused": True,
                        "timestamp": datetime.now().isoformat(),
                    }
                    runtime.record_event(key, timeout_event)
                    await runtime.send(key, timeout_event)
                    await wait_event.wait()

                # Build FunctionResponse message for resume
                response_parts = []
                for iid in interrupt_ids:
                    info = runtime.pending_hitl.pop(iid, None)
                    if not info or info.get("response") is None:
                        raise asyncio.CancelledError("HITL ended without an explicit decision")
                    response_data = info["response"]
                    response_parts.append(
                        create_request_input_response(iid, response_data)
                    )

                # Resume: send FunctionResponse back to run_async
                current_message = types.Content(
                    role="user",
                    parts=response_parts,
                )

                await runtime.send(key, runtime.status_payload(
                    key,
                    "processing",
                    "Resuming workflow after HITL response...",
                    version=run_status_version,
                ))

                # Continue the while loop to call run_async again with the FR message
                continue
            else:
                # No interrupt, we're done
                break

        # A normal runner return and a final-looking model message are not
        # scientific completion evidence.  Gate report finalization on the
        # freshest durable state before emitting `final_response`; the outer
        # execution controller will persist the same state and create the
        # durable pause.  This ordering prevents a transient false-completed UI
        # and avoids publishing a report for unfinished work.
        from CoScientist.experiments.outcome.reconciliation import (
            reconcile_scientific_outcome,
        )

        reconciled_session = await runtime.session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id,
        )
        reconciled_state = (
            reconciled_session.state if reconciled_session is not None else {}
        )
        from CoScientist.execution_control import current_run as current_execution_run

        execution_handle = current_execution_run()
        disposition = reconcile_scientific_outcome(
            reconciled_state,
            current_run_id=execution_handle.run_id if execution_handle else None,
        )
        if not disposition.report_allowed:
            logging.getLogger("CoScientist.web").info(
                "REPORT_FINALIZATION_DEFERRED disposition=%s reason=%s stage=%s",
                disposition.kind.value, disposition.reason, disposition.stage,
            )
            return

        # ── Package the deliverable ──────────────────────────────────────────────
        # The Result Aggregator already ran as the terminal stage of the single
        # run_async above (its events streamed like any other agent), so its report
        # is in `report_markdown`. Fall back to the orchestrator's own answer only if
        # the aggregator produced nothing.
        # The scope, not None. Everything downstream of finalize that writes
        # anywhere reads the user and session from here: passing None made
        # `_publish_to_research_graph` bail with "no user scope in state", so
        # the Report card was never published from the web path at all — only
        # from the CLI, which passes real state.
        from CoScientist.graph.session_scope import (
            GRAPH_SCOPE_SESSION_KEY,
            GRAPH_SCOPE_USER_KEY,
        )

        finalize_state = {
            GRAPH_SCOPE_USER_KEY: user_id,
            GRAPH_SCOPE_SESSION_KEY: session_id,
        }
        from CoScientist.execution_control import before_tool_action
        await before_tool_action("finalize_report")
        result = await asyncio.to_thread(
            finalize_report, manager.session_id,
            report_markdown or final_response, report_config, finalize_state,
        )

        runtime.registry.touch_session(user_id, session_id, status="idle")
        payload = {
            "type": "final_response",
            "content": _readable_links(result.markdown, key),
            "timestamp": datetime.now().isoformat(),
        }
        if result.report_dir:
            payload["report_dir"] = str(result.report_dir)
            payload["manifest"] = result.manifest
        if result.report_artifact_id:
            # `finalize` already mirrored report.md into this session, so the
            # panel opens that file rather than a second copy of the same text.
            from CoScientist.reporting.documents import summarise_markdown

            title, summary = summarise_markdown(result.markdown)
            payload["document"] = {
                "artifact_id": result.report_artifact_id,
                "title": title or "Отчёт по исследованию",
                "kind": "result",
            }
            payload["summary"] = summary
        await runtime.send(key, payload)

    finally:
        # A cancelled/failed runner must not leave RequestInput records that a
        # reconnecting tab could mistake for a live approval request.
        _cancel_pending_hitl(runtime, key)
        # Final cost of the run, past the rate limiter: a run that was stopped
        # or that crashed still spent money, and the tab should show the last
        # number rather than whichever one the limiter happened to let through.
        # The server log gets the per-agent breakdown the panel has no room for.
        try:
            from CoScientist.logging.metrics import log_report, publish
            publish(key, force=True)
            if os.getenv("LOG_USAGE_METRICS", "1") != "0":
                log_report(key, title=f"Usage & cost — run for {user_id}")
        except Exception:  # noqa: BLE001 - reporting never fails a run
            pass


def _handle_hitl_response(runtime: WebRuntime, key: SessionKey, data: dict):
    """Resolve a pending HITL request from the browser.

    Routes responses to either:
    1. WebHITLHandler (for SessionAgent's custom HITL, e.g. PlannerAgent)
    2. _pending_hitl dict (for ADK RequestInput workflow interrupts)

    The browser sends back:
        {
            "type": "hitl_response",
            "request_id": "<id>",   // for WebHITLHandler
            "interrupt_id": "<id>", // for ADK RequestInput
            "approved": true/false,
            "feedback": "..."
        }
    """
    request_id = data.get("request_id")
    interrupt_id = data.get("interrupt_id")

    # 1) Try WebHITLHandler (SessionAgent / PlannerAgent HITL)
    if request_id:
        resolved = runtime.hitl_handler.resolve_request(request_id, data, key)
        if resolved:
            return
        # If it was resolved there, no need to check _pending_hitl
        if request_id not in runtime.pending_hitl:
            if hasattr(runtime, "resolve_durable_hitl"):
                runtime.resolve_durable_hitl(key, request_id, data)
            return

    # 2) Try ADK RequestInput mechanism
    lookup_id = interrupt_id or request_id
    if not lookup_id:
        print("[HITL] No interrupt_id or request_id in hitl_response, ignoring")
        return

    info = runtime.pending_hitl.get(lookup_id)
    if not info:
        # Already handled by WebHITLHandler or unknown
        return
    if info.get("session_key") != key:
        logging.getLogger("CoScientist.web").warning(
            "Ignoring RequestInput response from the wrong session"
        )
        return
    if info.get("response") is not None:
        return  # A double click must not replace an already accepted decision.
    # Store the response data
    response = {
        "approved": data.get("approved", False),
        "feedback": data.get("feedback"),
        "instructions": data.get("instructions"),
        "free_input": data.get("free_input"),
    }
    info["response"] = response
    if hasattr(runtime, "execution_handle"):
        handle = runtime.execution_handle(key)
        if handle is not None:
            handle.controller.journal(handle.run_id, "hitl_decision", action_id=lookup_id, data=response)
            handle.controller.resume(handle.run_id, f"hitl:{lookup_id}")

    # Check if all interrupt IDs sharing this wait_event have responses
    wait_event = info["event"]
    for pending_id, pending in runtime.pending_hitl.items():
        if pending["event"] is wait_event and pending.get("session_key") == key:
            pending["response"] = response
            if hasattr(runtime, "execution_handle"):
                sibling_handle = runtime.execution_handle(key)
                if sibling_handle is not None:
                    sibling_handle.controller.resume(sibling_handle.run_id, f"hitl:{pending_id}")
    all_resolved = all(
        v["response"] is not None
        for v in runtime.pending_hitl.values()
        if v["event"] is wait_event and v.get("session_key") == key
    )
    if all_resolved:
        runtime.record_event(key, hitl_response_event(lookup_id, data))
        wait_event.set()  # Unblock the _handle_chat loop
