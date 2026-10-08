from CoScientist.agents.common import is_proxy_error
import asyncio
import json
import logging
import os
from collections import defaultdict, OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
from weakref import WeakKeyDictionary

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from CoScientist.agents.callbacks.tool_callbacks import DATASET_URL_STATE_KEY
from CoScientist.agents.callbacks.report_language import (
    REPORT_LANGUAGES,
    REPORT_LANGUAGE_STATE_KEY,
)
from CoScientist.main import CoScientistManager
from CoScientist.web.handler import WebHITLHandler
from CoScientist.web.session_registry import LocalSessionRegistry
from CoScientist.agents import agent_system, planner_agent
from CoScientist.config import ReportConfig
from CoScientist.reporting import finalize_report
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.config import get_settings
from CoScientist.tools.coder_tools.coder_tools import coder_toolset
from CoScientist.agents.common import sync_proxy_session
from CoScientist.utils.text import strip_thinking

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions import InMemorySessionService
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


# ---------------------------------------------------------------------------
# Runtime types and constants
# ---------------------------------------------------------------------------
WEB_DIR = Path(__file__).parent
TEMPLATE_PATH = WEB_DIR / "templates" / "index.html"
APP_NAME = "coscientist_app"
SessionKey = tuple[str, str]
SOCKET_SEND_TIMEOUT_SECONDS = 5.0
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


def _apply_frontend_settings(frontend: dict) -> None:
    """Map the frontend JS ``appSettings`` object to ``settings.web``.

    Called before every ``_get_manager()`` invocation so the config singleton
    is always up-to-date when the system is (re)built.
    """
    from CoScientist.config import get_settings
    web = get_settings().web

    general = frontend.get("general", {})
    if "startMode" in general:
        web.start_mode = general["startMode"]
    if "maxRetries" in general:
        web.max_retries = int(general["maxRetries"])
    if "hitlEnabled" in general:
        val = bool(general["hitlEnabled"])
        web.hitl_enabled = val
        get_settings().hitl.enabled = val
    if "hitlAutoApproveTimeout" in general:
        web.hitl_auto_approve_timeout = int(general["hitlAutoApproveTimeout"])
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

    hypotheses = frontend.get("hypothesesAgent", {})
    if "maxActiveHypotheses" in hypotheses:
        val = int(hypotheses["maxActiveHypotheses"])
        if 1 <= val <= 5:
            web.max_active_hypotheses = val

    coder = frontend.get("coderAgent", {})
    if "sandboxUrl" in coder:
        web.sandbox_url = coder["sandboxUrl"]
    if "workspaceId" in coder:
        val = coder["workspaceId"]
        web.coder_workspace_id = val if val else None
    if "mode" in coder:
        web.coder_mode = str(coder["mode"])


def _settings_payload() -> dict:
    """The frontend ``appSettings`` shape, read back off the config singleton.

    Both /api/settings endpoints answer with it, so a GET and the echo of a
    POST can never drift apart.
    """
    from CoScientist.config import get_settings
    settings = get_settings()
    web = settings.web
    return {
        "general": {
            "startMode": web.start_mode,
            "maxRetries": web.max_retries,
            "hitlEnabled": web.hitl_enabled,
            "hitlAutoApproveTimeout": web.hitl_auto_approve_timeout,
            "usePlanner": web.use_planner,
            "useProxy": web.use_proxy,
            "opikEnabled": web.opik_enabled,
            "autoNamingEnabled": web.auto_naming_enabled,
            "coscientistUsername": web.coscientist_username or "",
            "contextInitEnabled": settings.context_init.enabled,
            "knowledgeGraphEnabled": web.knowledge_graph_enabled,
            "autoClearGraphEnabled": web.auto_clear_graph_enabled,
            "researchGraphEnabled": settings.research_graph.enabled,
        },
        "plannerAgent": {
            "retrievalEnabled": web.planner_retrieval_enabled,
            "graphEnabled": web.planner_graph_enabled,
            "criticEnabled": web.planner_critic_enabled,
            "criticRounds": web.planner_critic_rounds,
            "mergeTasksEnabled": web.merge_tasks_enabled,
        },
        "researchAgent": {"maxSearches": web.max_searches},
        "hypothesesAgent": {
            "maxActiveHypotheses": web.max_active_hypotheses,
        },
        "taskExecutorAgent": {
            "keepScore": web.executor_tool_keep_score,
            "abstainScore": web.executor_tool_abstain_score,
        },
        "coderAgent": {
            "sandboxUrl": web.sandbox_url,
            "workspaceId": web.coder_workspace_id or "",
            "mode": web.coder_mode,
        },
    }



class WebRuntime:
    """Process-local users, ADK sessions, managers, sockets, and event logs."""

    def __init__(self) -> None:
        self.session_service = InMemorySessionService()
        self.registry = LocalSessionRegistry()
        self.managers: dict[SessionKey, CoScientistManager] = {}
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
        # Dataset archive attached to a session from the chat's "+" menu. Kept
        # here as well as in ADK state so a reconnecting tab and a session whose
        # manager has not been built yet both see the same link.
        self.dataset_urls: dict[SessionKey, str] = {}
        # Report language chosen for a session from the chat composer. Holds
        # ONLY explicit choices — an absent key means the browser has not spoken
        # yet, which is what lets the UI default follow its interface language.
        self.report_languages: dict[SessionKey, str] = {}
        self.pending_hitl: dict[str, dict[str, Any]] = {}
        self.hitl_handler = WebHITLHandler()
        self.hitl_handler.set_sender(self.send_socket)
        self.sockets: dict[SessionKey, list[WebSocket]] = defaultdict(list)
        self.active_runs: dict[SessionKey, asyncio.Task] = {}

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
        return {
            "type": "status",
            "status": status,
            "message": message,
            "run_status_version": (
                self.run_versions[key] if version is None else version
            ),
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
            task = asyncio.create_task(_handle_chat(self, key, run_data))
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
        await self.send(key, self.status_payload(
            key,
            "idle",
            "Session is ready for the next request.",
            version=version,
        ))
        return True

    async def stop_run(self, key: SessionKey) -> bool:
        """Cancel and remove the exact run currently owning this session."""
        async with self.control_lock(key):
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

    async def get_manager(self, user_id: str, session_id: str) -> CoScientistManager:
        self.registry.require_session(user_id, session_id)
        key = (user_id, session_id)
        manager = self.managers.get(key)
        if manager is not None:
            return manager
        async with self.manager_lock:
            manager = self.managers.get(key)
            if manager is None:
                if get_settings().web.auto_clear_graph_enabled:
                    _clear_session_graphs(user_id, session_id, view="all")
                manager = CoScientistManager(
                    app_name=APP_NAME,
                    user_id=user_id,
                    session_id=session_id,
                    session_service=self.session_service,
                )
                await manager.initialize()
                self.managers[key] = manager
                self.execution_locks[key] = asyncio.Lock()
        return manager

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
                    "run_status_version": version,
                    "metrics": self.metrics.get(key),
                    "dataset_url": self.dataset_urls.get(key, ""),
                    "report_language": self.report_languages.get(key, ""),
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

    def trim(events: list[dict[str, Any]]) -> None:
        """Bound replay history: drop the oldest tool records, keep the chat."""
        if len(events) <= MAX_TOOL_ACTIVITY_EVENTS + TOOL_ACTIVITY_TRIM_SLACK:
            return
        indexes = [
            index for index, event in enumerate(events)
            if event.get("type") == "tool_activity"
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
            for field, key_name in (("args", "args_full"), ("result", "result_full"), ("error", "error_full"))
            if key_name in event
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
        events = runtime.agent_events[key]
        events.append(event)
        trim(events)
        await runtime.send(key, event)

    set_tool_activity_sink(deliver)


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
            payload["content"] = strip_thinking(payload["content"])
            if not payload["content"].strip():
                return
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
    yield
    print("[CoScientist Web] Shutting down …")
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
    runtime = WebRuntime()
    _wire_hitl(runtime)
    _wire_sandbox_links(runtime)
    _wire_sandbox_plan(runtime)
    _wire_tool_activity(runtime)
    _wire_agent_output(runtime)
    _wire_metrics(runtime)
    app = FastAPI(
        title="CoScientist Web UI",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    # Vendored JS/CSS (e.g. vis-network for the live graph) so the UI works
    # offline / behind a VPN without any CDN.
    _static_dir = WEB_DIR / "static"
    if _static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

    # --- HTML endpoint ---
    @app.get("/", response_class=HTMLResponse)
    async def index():
        # no-store: a cached index.html silently serves an OLD frontend — HITL
        # review cards then render without controls/content.
        return HTMLResponse(
            TEMPLATE_PATH.read_text(encoding="utf-8"),
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
                return get_research_graph(
                    user_id=user_id,
                    session_id=session_id,
                ).view_of(turn)
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
                return execution_tree(execution, turn=turn)
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


    # --- ТЗ document (microfluidics profile) ---
    @app.get("/api/tz-document")
    async def get_tz_document(name: str = ""):
        """Serve a ТЗ document from tz_documents/ (the latest one by default).

        The TZSpecAgent announces the file in the chat; this endpoint lets the
        user open it in the browser.
        """
        from fastapi.responses import PlainTextResponse

        tz_dir = Path("tz_documents")
        if not tz_dir.is_dir():
            return JSONResponse({"error": "no ТЗ documents yet"}, status_code=404)
        if name:
            # Only bare file names inside tz_documents/ — no path traversal.
            candidate = tz_dir / Path(name).name
            if not candidate.is_file():
                return JSONResponse({"error": f"no such document: {name}"}, status_code=404)
        else:
            files = sorted(tz_dir.glob("TZ_*.md"))
            if not files:
                return JSONResponse({"error": "no ТЗ documents yet"}, status_code=404)
            candidate = files[-1]
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
        _apply_frontend_settings(data)
        return JSONResponse({"status": "success", **_settings_payload()})

    # --- Agent info ---
    @app.get("/api/agents")
    async def get_agents():
        """Return list of registered agents and system hierarchy."""
        from CoScientist.assembly.schema import get_config
        cfg = get_config()
        hierarchy = cfg.agent_hierarchy_map()
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
            })
        return JSONResponse({
            "agents": agents_list,
            "hierarchy": hierarchy,
            "delegatable_names": list(cfg.delegatable_names()),
        })

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
                    await runtime.apply_report_language(key, lang)
                    # Broadcast: the report is one document, so every tab on the
                    # session agrees on its language, whichever one picked it.
                    await runtime.send(key, {
                        "type": "report_language",
                        "report_language": lang,
                    })
                elif msg_type == "hitl_response":
                    _handle_hitl_response(runtime, key, data)
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

    # Echo user message
    user_event = {
        "type": "user_message",
        "message": query,
        "timestamp": datetime.now().isoformat(),
    }
    runtime.record_event(key, user_event)
    await runtime.send(key, user_event)

    try:
        manager = await runtime.get_manager(user_id, session_id)
        # The attachment may predate the ADK session (it is created with the
        # manager), so mirror it into state now that the session exists — this is
        # what puts the link in front of CoderAgent.
        await runtime.apply_dataset_url(key)
        # Same reason as the attachment above: the language may have been picked
        # before the ADK session existed.
        await runtime.apply_report_language(key)
        runtime.registry.touch_session(user_id, session_id, status="processing")
        execution_lock = runtime.execution_locks[key]

        async with execution_lock:
            await _run_chat_invocation(
                runtime,
                key,
                manager,
                query,
                run_status_version=run_status_version,
                report_language=report_language,
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

        if is_proxy_error(exc):
            msg = (
                f"**Error connecting to proxy server**\n\n"
                f"Failed to connect to the proxy server to execute the query to the language model. "
                f"Please ensure the proxy container is running, the corporate VPN is enabled (other - disabled), and "
                f"and the proxy is accessible."
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
                # Lift ADK's 500-LLM-call default so a long autonomous run driven
                # by a single prompt isn't cut off mid-work.
                run_config=RunConfig(
                    max_llm_calls=get_settings().orchestrator.max_llm_calls
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
                        event_data["content"] = "\n".join(text_parts)

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
                    print(f"[HITL] Timeout waiting for response, auto-approving")
                    for iid in interrupt_ids:
                        if iid in runtime.pending_hitl and runtime.pending_hitl[iid]["response"] is None:
                            runtime.pending_hitl[iid]["response"] = {"approved": True}
                    await runtime.send(key, {
                        "type": "hitl_timeout",
                        "request_id": interrupt_ids[0] if interrupt_ids else "",
                        "interrupt_ids": interrupt_ids,
                        "agent_name": hitl_interrupt_event.author or "system",
                        "timeout_seconds": 600,
                    })

                # Build FunctionResponse message for resume
                response_parts = []
                for iid in interrupt_ids:
                    info = runtime.pending_hitl.pop(iid, None)
                    response_data = (info["response"] if info and info["response"] else {"approved": True})
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

        # ── Package the deliverable ──────────────────────────────────────────────
        # The Result Aggregator already ran as the terminal stage of the single
        # run_async above (its events streamed like any other agent), so its report
        # is in `report_markdown`. Fall back to the orchestrator's own answer only if
        # the aggregator produced nothing.
        result = await asyncio.to_thread(
            finalize_report, manager.session_id,
            report_markdown or final_response, report_config, None,
        )

        runtime.registry.touch_session(user_id, session_id, status="idle")
        payload = {
            "type": "final_response",
            "content": result.markdown,
            "timestamp": datetime.now().isoformat(),
        }
        if result.report_dir:
            payload["report_dir"] = str(result.report_dir)
            payload["manifest"] = result.manifest
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
    
    # Store the response data
    response = {
        "approved": data.get("approved", False),
        "feedback": data.get("feedback"),
        "instructions": data.get("instructions"),
        "free_input": data.get("free_input"),
    }
    info["response"] = response
    
    # Check if all interrupt IDs sharing this wait_event have responses
    wait_event = info["event"]
    for pending in runtime.pending_hitl.values():
        if pending["event"] is wait_event and pending.get("session_key") == key:
            pending["response"] = response
    all_resolved = all(
        v["response"] is not None
        for v in runtime.pending_hitl.values()
        if v["event"] is wait_event and v.get("session_key") == key
    )
    if all_resolved:
        wait_event.set()  # Unblock the _handle_chat loop
