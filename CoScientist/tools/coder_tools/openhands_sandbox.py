"""Sandbox client for external (multi-)agent systems.

This module exposes a *session-scoped* tool that external agent frameworks can
hand to an LLM.  The session semantics are the whole point of the file:

* **Repeated calls** — an agent may call the tool many times to finish one
  experiment.  Every call after the first is delivered to the *same* container
  as a follow-up message, so the agent-side conversation memory and the whole
  ``/workspace`` are preserved.
* **Reuse by default** — no extra arguments are needed to continue an
  experiment; reuse is what happens unless something is explicitly asked for.
* **Explicit fresh sandbox** — ``new_sandbox=True`` provisions a clean
  container inside the same session (independent experiments, poisoned
  workspace, "start over").
* **Session isolation** — a session that has no binding yet always provisions
  its own container.  Bindings are never shared between sessions.

The binding (``session -> sandbox id``) is kept in two places: the ADK session
state when a ``tool_context`` is available (survives ``AgentTool`` sub-sessions
and user messages), and a process-local registry as a fallback for frameworks
that do not carry a context object.

Server-side contract used here (see ``api/routes.py``):

* ``POST /api/v1/run`` / ``/run-external`` accept ``session_id`` (the
  ``task_id`` of the first run of the session) and ``new_container``.
* When ``session_id`` points at a container that is still alive the task is
  routed to it and the response carries ``reused=True`` with the *same*
  ``task_id``.  When the container is gone the server transparently creates a
  new one and returns a **new** ``task_id`` with ``reused=False`` — which is
  why the returned id is always re-bound to the session here.
* Terminal statuses are ``cooldown``/``completed`` (success), ``error`` and
  ``cancelled``.  ``cooldown`` means "finished, container still alive" — that
  is the state a follow-up can be sent into.
* ``GET /api/v1/status?task_id=<id>`` carries the run's uploaded artifacts as
  a structured ``s3_uploads`` list (``filename``/``url``/``size``/``key``).
  That list is the ONLY carrier of those links — the sandbox agent keeps them
  out of its ``summary`` prose — so every layer between here and the model
  passes the field through untouched instead of re-reading it out of text.
* ``GET /api/v1/metrics?task_id=<id>`` returns what the run cost: wall clock,
  CPU/GPU work, energy, LLM tokens and money.

Metrics travel on their own channel — the journal behind
:func:`get_sandbox_metrics` and the optional ``metrics_sink`` callback — and are
deliberately kept OUT of the dict every run function returns.  That dict becomes
a tool result, i.e. prompt text; a model that can see its own bill starts
optimising the bill instead of the task (cutting steps short, abandoning
experiments).  What cannot be interpolated by accident will not be.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_POLL_INTERVAL = 10.0
DEFAULT_SUBMIT_TIMEOUT = 3600
DEFAULT_STATUS_TIMEOUT = 15.0
DEFAULT_METRICS_TIMEOUT = 15.0
#: Trajectories are recorded event-by-event and can run to gigabytes of raw
#: tool output, so they get a much longer budget than status/metrics polls.
DEFAULT_TRAJECTORY_TIMEOUT = 120.0

#: Statuses at which a task stops progressing.
TERMINAL_STATUSES = frozenset({"completed", "cooldown", "error", "cancelled"})
#: Terminal statuses that mean the task ran to the end without failing.
SUCCESS_STATUSES = frozenset({"completed", "cooldown"})

#: Key under which the sandbox id is pinned in ADK session state.
SESSION_STATE_KEY = "sandbox_session_id"

#: Key under which the metrics journal is mirrored into ADK session state.
#: NEVER interpolate this key into an agent instruction template — that would
#: put the run's own price in front of the model (see the module docstring of
#: ``CoScientist.logging.metrics`` for why that is harmful).
METRICS_STATE_KEY = "sandbox_metrics"

#: Session key used when the caller provides no session identity at all.
PROCESS_SESSION_KEY = "__process_default__"

#: Give up polling after this many consecutive status-request failures.
_MAX_STATUS_FAILURES = 10


class SandboxConfigError(RuntimeError):
    """Raised when the sandbox base URL cannot be resolved."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def resolve_sandbox_url(explicit: Optional[str] = None) -> str:
    """Resolve the sandbox base URL.

    Priority: explicit argument, ``SANDBOX_URL`` env var, CoScientist settings
    (``settings.web.sandbox_url``) when that package happens to be importable.
    """
    if explicit:
        return explicit.rstrip("/")

    env_url = os.getenv("SANDBOX_URL")
    if env_url:
        return env_url.rstrip("/")

    try:  # optional dependency — only present inside CoScientist
        from CoScientist.config import get_settings

        url = get_settings().web.sandbox_url
        if url:
            return str(url).rstrip("/")
    except Exception:  # noqa: BLE001 - absence of the package is normal here
        pass

    raise SandboxConfigError(
        "Sandbox URL is not configured. Set the SANDBOX_URL environment "
        "variable or pass sandbox_url=... explicitly."
    )


def _api(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/v1"


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------

class _SessionRegistry:
    """Thread-safe ``session key -> sandbox id`` map, mirrored to disk.

    Each task runs in its OWN container with its OWN /workspace, so the binding
    is the only thing that lets a follow-up call land where the previous one left
    its files. Keeping it in memory alone meant a server restart silently
    provisioned a fresh container: the work was still in the old one, the new
    task saw an empty workspace, and nothing reported that the two were
    different machines.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._bindings: Dict[str, str] = {}
        self._loaded_from: Optional[Path] = None

    @staticmethod
    def _path() -> Path:
        # Resolved per call, not pinned at import: this object is a module-level
        # singleton, so a pinned path would ignore any later configuration — and
        # would send every test's fake container id into the file the running
        # system reads to decide which sandbox to continue in.
        return Path(os.getenv(
            "SANDBOX_BINDINGS_FILE",
            os.path.join(os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs"),
                         "sandbox_bindings.json")))

    def _sync(self) -> Path:
        """Load the bindings for the currently configured path, once per path."""
        path = self._path()
        if self._loaded_from == path:
            return path
        self._bindings = {}
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._bindings = {str(k): str(v) for k, v in data.items() if v}
                    logger.info("Restored %d sandbox binding(s) from %s",
                                len(self._bindings), path)
        except Exception:  # noqa: BLE001 — a bad file must not stop the process
            logger.warning("Could not read sandbox bindings from %s", path,
                           exc_info=True)
        self._loaded_from = path
        return path

    def _save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._bindings, indent=1), encoding="utf-8")
            os.replace(tmp, path)
        except Exception:  # noqa: BLE001
            logger.warning("Could not persist sandbox bindings", exc_info=True)

    def get(self, session: str) -> Optional[str]:
        with self._lock:
            self._sync()
            return self._bindings.get(session)

    def set(self, session: str, sandbox_id: str) -> None:
        with self._lock:
            path = self._sync()
            if self._bindings.get(session) == sandbox_id:
                return
            self._bindings[session] = sandbox_id
            self._save(path)

    def drop(self, session: str) -> Optional[str]:
        with self._lock:
            path = self._sync()
            previous = self._bindings.pop(session, None)
            if previous is not None:
                self._save(path)
            return previous

    def snapshot(self) -> Dict[str, str]:
        with self._lock:
            self._sync()
            return dict(self._bindings)


_REGISTRY = _SessionRegistry()
_WARNED_NO_SESSION = False


# ---------------------------------------------------------------------------
# Per-session concurrency guard
# ---------------------------------------------------------------------------
#
# ``arun_sandbox_task`` / ``await_sandbox_task`` read the session's current
# binding, then (much later, after a network round-trip and — for a run — a
# possibly long inline wait) write the new one. If the same session's agent
# fires two calls a few seconds apart — a coder-agent slip, e.g. calling the
# tool again before the first reply lands — both would read the SAME stale
# binding and each provision (or attach to) a sandbox independently; whichever
# write lands last silently orphans the other, splitting the session across
# two containers instead of one.
#
# Rather than queue the second call behind the first (which could block a
# tool call for the run's ENTIRE duration — up to the hour-plus jobs this
# module is built for), a session with a call already in flight rejects the
# second one immediately with the same ``status="busy"`` the server itself
# already uses for this — the coder agent's tools already know to read that
# as "try again shortly" (see ``sandbox_tools._shape``).

_ACTIVE_SESSIONS: set = set()
_ACTIVE_SESSIONS_GUARD = threading.Lock()


def _try_acquire_session(session: str) -> bool:
    """Claim ``session`` for the duration of one call; False if already claimed."""
    with _ACTIVE_SESSIONS_GUARD:
        if session in _ACTIVE_SESSIONS:
            return False
        _ACTIVE_SESSIONS.add(session)
        return True


def _release_session(session: str) -> None:
    with _ACTIVE_SESSIONS_GUARD:
        _ACTIVE_SESSIONS.discard(session)


def _concurrent_busy_result(session: str, target_id: Optional[str]) -> Dict[str, Any]:
    """The result a second concurrent call for ``session`` gets, unsubmitted."""
    return _normalize({
        "status": "busy",
        "error": (
            "Sandbox busy: another call for this session is already in "
            "flight — wait for it to finish before calling again."
        ),
        "session": session,
        "sandbox_id": target_id,
        "reused": bool(target_id),
        "sandbox_expired": False,
    })


def _scope_from_context(tool_context: Any) -> Optional[str]:
    """Best-effort session identity from a framework context object."""
    if tool_context is None:
        return None

    # CoScientist / ADK: a scope pinned in session state, stable across the
    # ephemeral sub-sessions that AgentTool delegations run in.
    try:
        from CoScientist.graph.session_scope import DEFAULT_SESSION_KEY, session_key

        scope = session_key(tool_context)
        if scope != DEFAULT_SESSION_KEY:
            return f"{scope[0]}/{scope[1]}"
    except Exception:  # noqa: BLE001 - not running under CoScientist
        pass

    # Generic fallback: any context exposing a session with an id.
    invocation = (
        getattr(tool_context, "_invocation_context", None)
        or getattr(tool_context, "invocation_context", None)
    )
    session = getattr(tool_context, "session", None) or getattr(
        invocation, "session", None
    )
    session_id = getattr(session, "id", None)
    if session_id:
        user_id = getattr(session, "user_id", None) or "user"
        return f"{user_id}/{session_id}"

    return None


def resolve_session_key(
    session_id: Optional[str] = None,
    tool_context: Any = None,
) -> str:
    """Resolve the key identifying the *caller's* session.

    Priority: explicit ``session_id``, framework context, ``AGENT_SESSION_ID``
    env var, and finally a single process-wide key.  The last fallback means
    "one process == one session"; a long-lived multi-tenant host that does not
    pass any session identity would otherwise share one sandbox across users,
    so a warning is emitted once.
    """
    global _WARNED_NO_SESSION

    if session_id:
        return str(session_id)

    scope = _scope_from_context(tool_context)
    if scope:
        return scope

    env_session = os.getenv("AGENT_SESSION_ID")
    if env_session:
        return env_session

    if not _WARNED_NO_SESSION:
        _WARNED_NO_SESSION = True
        logger.warning(
            "No session identity available (no session_id, no tool_context, no "
            "AGENT_SESSION_ID). Falling back to a single process-wide sandbox "
            "session — concurrent user sessions in this process would share it."
        )
    return PROCESS_SESSION_KEY


# ---------------------------------------------------------------------------
# Binding storage
# ---------------------------------------------------------------------------

def _state_of(tool_context: Any) -> Any:
    state = getattr(tool_context, "state", None)
    if state is None:
        return None
    # Reading must not explode on exotic/read-only state implementations.
    try:
        state.get(SESSION_STATE_KEY)
    except Exception:  # noqa: BLE001
        return None
    return state


def read_binding(session: str, tool_context: Any = None) -> Optional[str]:
    """Return the sandbox id currently bound to ``session``, if any."""
    state = _state_of(tool_context)
    if state is not None:
        bound = state.get(SESSION_STATE_KEY)
        if bound:
            return str(bound)
    return _REGISTRY.get(session)


def write_binding(session: str, sandbox_id: str, tool_context: Any = None) -> None:
    """Bind ``sandbox_id`` to ``session`` in every available store."""
    _REGISTRY.set(session, sandbox_id)
    state = _state_of(tool_context)
    if state is not None:
        try:
            state[SESSION_STATE_KEY] = sandbox_id
        except Exception:  # noqa: BLE001 - read-only state still works via registry
            logger.debug("Could not pin sandbox id in session state.", exc_info=True)


def _write_state(tool_context: Any, key: str, value: Any) -> None:
    """Best-effort write into framework session state (read-only state is fine)."""
    state = _state_of(tool_context)
    if state is None:
        return
    try:
        state[key] = value
    except Exception:  # noqa: BLE001 - read-only state still works via the registry
        logger.debug("Could not write %s into session state.", key, exc_info=True)


def clear_binding(session: str, tool_context: Any = None) -> Optional[str]:
    """Forget the binding for ``session``; the next call provisions a new sandbox."""
    previous = read_binding(session, tool_context)
    _REGISTRY.drop(session)
    state = _state_of(tool_context)
    if state is not None:
        try:
            state[SESSION_STATE_KEY] = None
        except Exception:  # noqa: BLE001
            logger.debug("Could not clear sandbox id in session state.", exc_info=True)
    return previous


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001
        return response.text[:500]
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload)
    return str(payload)


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": "error", "error": message}
    result.update(extra)
    return result


def _parse_task(response: httpx.Response) -> Optional[Dict[str, Any]]:
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json().get("current_task") or {}


def _fetch_task(api_url: str, sandbox_id: str) -> Optional[Dict[str, Any]]:
    """Return the server-side task entry for ``sandbox_id`` (``None`` if gone)."""
    return _parse_task(httpx.get(
        f"{api_url}/status",
        params={"task_id": sandbox_id},
        timeout=DEFAULT_STATUS_TIMEOUT,
    ))


async def _afetch_task(api_url: str, sandbox_id: str) -> Optional[Dict[str, Any]]:
    """Async twin of :func:`_fetch_task`."""
    async with httpx.AsyncClient(timeout=DEFAULT_STATUS_TIMEOUT) as client:
        return _parse_task(await client.get(
            f"{api_url}/status", params={"task_id": sandbox_id},
        ))


# ---------------------------------------------------------------------------
# Metrics: what a run cost (a channel of its own, never the tool result)
# ---------------------------------------------------------------------------

#: Session totals, summed over every sandbox of the session.
_TOTAL_FIELDS = (
    "runs", "wall_seconds", "agent_seconds", "queue_seconds",
    "cpu_core_seconds", "gpu_seconds", "energy_wh",
    "llm_calls", "total_tokens",
    "api_cost_usd", "energy_cost_usd", "total_cost_usd",
)


def _section(record: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = record.get(name)
    return value if isinstance(value, dict) else {}


def _number(source: Dict[str, Any], key: str) -> float:
    value = source.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def sandbox_run_digest(record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce one metrics record to the flat numbers worth carrying around.

    The full record is a nested document with seven sections; everything that
    reporting and "why was this so slow / so expensive" actually need is these
    twelve numbers, in the same vocabulary the session totals use. Missing
    sections read as zero, so a partial record still digests.
    """
    record = record or {}
    wall = _section(record, "wall_clock")
    compute = _section(record, "compute")
    energy = _section(compute, "energy")
    api = _section(record, "api")
    cost = _section(record, "cost")

    api_cost = _number(cost, "api_cost_usd") or _number(api, "cost_usd")
    energy_cost = _number(cost, "energy_cost_usd")
    return {
        "runs": 1,
        "wall_seconds": _number(wall, "total_seconds"),
        "agent_seconds": _number(wall, "agent_seconds"),
        "queue_seconds": _number(wall, "queue_seconds"),
        "cpu_core_seconds": _number(compute, "cpu_seconds"),
        "gpu_seconds": _number(compute, "gpu_seconds"),
        "energy_wh": _number(energy, "total_energy_wh"),
        "llm_calls": _number(api, "llm_calls"),
        "total_tokens": _number(api, "total_tokens"),
        "api_cost_usd": api_cost,
        "energy_cost_usd": energy_cost,
        "total_cost_usd": _number(cost, "total_cost_usd") or (api_cost + energy_cost),
    }


def _sum_digests(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Session totals over per-sandbox records (``runs`` counts sandboxes)."""
    totals = {field: 0.0 for field in _TOTAL_FIELDS}
    for record in records:
        digest = sandbox_run_digest(record)
        for field in _TOTAL_FIELDS:
            totals[field] += digest[field]
    totals["runs"] = int(totals["runs"])
    totals["llm_calls"] = int(totals["llm_calls"])
    totals["total_tokens"] = int(totals["total_tokens"])
    return totals


class _MetricsJournal:
    """``session -> {sandbox id: last record}``, process-local and thread-safe.

    Keyed by sandbox rather than by call on purpose: the server reports
    *cumulative* figures per container, so a follow-up into the same sandbox
    replaces its entry instead of adding a second one — otherwise every
    follow-up would count the whole run again.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def put(self, session: str, sandbox_id: str, record: Dict[str, Any]) -> None:
        with self._lock:
            runs = self._records.setdefault(session, {})
            runs.pop(sandbox_id, None)  # re-insert so ordering stays chronological
            runs[sandbox_id] = record

    def get(self, session: str, sandbox_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            runs = self._records.get(session) or {}
            if sandbox_id:
                return runs.get(sandbox_id)
            return next(reversed(runs.values()), None) if runs else None

    def runs(self, session: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list((self._records.get(session) or {}).values())

    def totals(self, session: str) -> Dict[str, Any]:
        return _sum_digests(self.runs(session))

    def clear(self, session: str) -> Dict[str, Any]:
        with self._lock:
            runs = list((self._records.pop(session, None) or {}).values())
        return _sum_digests(runs)


_METRICS = _MetricsJournal()


def _parse_metrics(response: httpx.Response) -> Optional[Dict[str, Any]]:
    """Return the metrics record from a ``/metrics`` reply, or ``None``.

    404 (no such task / metrics disabled) and 409 (no task id and nothing
    running) are ordinary answers here, not failures.
    """
    if response.status_code in (404, 409):
        return None
    response.raise_for_status()
    payload = response.json() or {}
    record = payload.get("metrics")
    if not isinstance(record, dict):
        return None
    # The envelope carries the identity/finality of the record; older servers
    # only put them there, so fold them in rather than lose them.
    for key in ("task_id", "status", "final"):
        if key in payload:
            record.setdefault(key, payload[key])
    return record


def _fetch_metrics(api_url: str, sandbox_id: str) -> Optional[Dict[str, Any]]:
    return _parse_metrics(httpx.get(
        f"{api_url}/metrics",
        params={"task_id": sandbox_id},
        timeout=DEFAULT_METRICS_TIMEOUT,
    ))


async def _afetch_metrics(api_url: str, sandbox_id: str) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=DEFAULT_METRICS_TIMEOUT) as client:
        return _parse_metrics(await client.get(
            f"{api_url}/metrics", params={"task_id": sandbox_id},
        ))


# ---------------------------------------------------------------------------
# Trajectory: the full recorded trace of one sandbox run
# ---------------------------------------------------------------------------

def _parse_trajectory(response: httpx.Response) -> Optional[Dict[str, Any]]:
    """Return the trajectory payload from a ``/trajectory`` reply, or ``None``.

    404 (no trace was ever recorded, or it was already deleted with its
    container) and 409 (the task — or a follow-up — hasn't finished, so the
    trace file is still being written) are ordinary answers here, not
    failures: there is simply nothing to attach yet.
    """
    if response.status_code in (404, 409):
        return None
    response.raise_for_status()
    return response.json()


def _fetch_trajectory(
    api_url: str,
    sandbox_id: str,
    *,
    include_raw_output: bool = True,
    timeout: float = DEFAULT_TRAJECTORY_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    return _parse_trajectory(httpx.get(
        f"{api_url}/trajectory",
        params={"task_id": sandbox_id, "include_raw_output": include_raw_output},
        timeout=timeout,
    ))


async def _afetch_trajectory(
    api_url: str,
    sandbox_id: str,
    *,
    include_raw_output: bool = True,
    timeout: float = DEFAULT_TRAJECTORY_TIMEOUT,
) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        return _parse_trajectory(await client.get(
            f"{api_url}/trajectory",
            params={"task_id": sandbox_id, "include_raw_output": include_raw_output},
        ))


def get_sandbox_trajectory(
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    include_raw_output: bool = True,
) -> Optional[Dict[str, Any]]:
    """Fetch the full agent trajectory (trace) recorded for a sandbox run.

    Unlike metrics there is no local journal to fall back on — the trace lives
    only on the sandbox server, for as long as its container is up. Returns
    ``None`` when there is nothing to fetch: no sandbox bound to this session,
    no trace recorded for it, or the run hasn't finished yet. Transport and
    server errors (a timeout on a huge trace, a 500 reading a corrupt file)
    are raised — a caller folding this into a larger export should treat a
    failure here as "skip this section", not "abort the whole export".
    """
    target = sandbox_id or read_binding(resolve_session_key(session_id, tool_context), tool_context)
    if not target:
        return None
    return _fetch_trajectory(
        _api(resolve_sandbox_url(sandbox_url)), str(target),
        include_raw_output=include_raw_output,
    )


async def aget_sandbox_trajectory(
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    include_raw_output: bool = True,
) -> Optional[Dict[str, Any]]:
    """Async twin of :func:`get_sandbox_trajectory`."""
    target = sandbox_id or read_binding(resolve_session_key(session_id, tool_context), tool_context)
    if not target:
        return None
    return await _afetch_trajectory(
        _api(resolve_sandbox_url(sandbox_url)), str(target),
        include_raw_output=include_raw_output,
    )


def _publish_metrics(
    session: str,
    sandbox_id: str,
    record: Optional[Dict[str, Any]],
    tool_context: Any,
    sink: Optional[Callable[[Dict[str, Any]], Any]],
) -> None:
    """File a fetched record: journal, session state, subscriber."""
    if not record:
        return
    _METRICS.put(session, sandbox_id, record)
    _write_state(tool_context, METRICS_STATE_KEY, {
        "runs": _METRICS.runs(session),
        "totals": _METRICS.totals(session),
    })
    if sink is None:
        return
    try:
        sink(dict(record))
    except Exception:  # noqa: BLE001 - broken telemetry must not fail a run
        logger.warning("Sandbox metrics sink failed.", exc_info=True)


def _should_collect(collect: bool, status: Optional[str], sandbox_id: Optional[str]) -> bool:
    """Only ask for metrics once the run is over and there is a run to ask about.

    A non-terminal status means the task is still going, so the server would
    answer with a live snapshot — useful for a progress bar, wrong for a
    journal of what runs cost. ``error``/``cancelled`` ARE collected: a task
    that burned an hour and then failed is exactly what one wants to see.
    """
    return bool(collect and sandbox_id and status in TERMINAL_STATUSES)


def _collect_metrics(
    *,
    api_url: str,
    session: str,
    sandbox_id: Optional[str],
    status: Optional[str],
    tool_context: Any,
    sink: Optional[Callable[[Dict[str, Any]], Any]],
    collect: bool,
) -> None:
    """Fetch and file the metrics of a finished run (never raises)."""
    if not _should_collect(collect, status, sandbox_id):
        return
    try:
        record = _fetch_metrics(api_url, str(sandbox_id))
    except Exception as exc:  # noqa: BLE001 - metrics are never worth a failure
        logger.debug("Sandbox metrics unavailable for %s: %s", sandbox_id, exc)
        return
    _publish_metrics(session, str(sandbox_id), record, tool_context, sink)


async def _acollect_metrics(
    *,
    api_url: str,
    session: str,
    sandbox_id: Optional[str],
    status: Optional[str],
    tool_context: Any,
    sink: Optional[Callable[[Dict[str, Any]], Any]],
    collect: bool,
) -> None:
    """Async twin of :func:`_collect_metrics`."""
    if not _should_collect(collect, status, sandbox_id):
        return
    try:
        record = await _afetch_metrics(api_url, str(sandbox_id))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Sandbox metrics unavailable for %s: %s", sandbox_id, exc)
        return
    _publish_metrics(session, str(sandbox_id), record, tool_context, sink)


def get_sandbox_metrics(
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    live: bool = False,
) -> Dict[str, Any]:
    """What the session's sandbox runs cost.

    Args:
        session_id / tool_context: the caller's session, resolved exactly as in
            :func:`run_sandbox_task` — pass whichever the framework gives you.
        sandbox_id: read one specific run instead of the session's latest.
        live: ask the server instead of reading the journal. Use it *while* a
            task runs to get a non-final snapshot; a finished run is already in
            the journal and needs no request.

    Returns:
        ``{"session", "metrics", "runs", "totals"}``. ``metrics`` is the full
        record of one run and may be ``None`` — the server did not answer,
        collection was off, or the journal was cleared. The key always exists.
    """
    session = resolve_session_key(session_id, tool_context)
    target = sandbox_id or read_binding(session, tool_context)

    record: Optional[Dict[str, Any]] = None
    if live and target:
        try:
            record = _fetch_metrics(_api(resolve_sandbox_url(sandbox_url)), str(target))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Live metrics request failed for %s: %s", target, exc)
    if record is None:
        record = _METRICS.get(session, sandbox_id)

    return {
        "session": session,
        "metrics": record,
        "runs": _METRICS.runs(session),
        "totals": _METRICS.totals(session),
    }


def clear_sandbox_metrics(
    *,
    session_id: Optional[str] = None,
    tool_context: Any = None,
) -> Dict[str, Any]:
    """Drop the journal of one session, returning its totals one last time.

    The journal lives in process memory and grows with the number of sessions,
    so a long-lived host should call this when a user session ends — after
    taking the totals it hands back.
    """
    session = resolve_session_key(session_id, tool_context)
    totals = _METRICS.clear(session)
    _write_state(tool_context, METRICS_STATE_KEY, {"runs": [], "totals": {}})
    return {
        "session": session,
        "discarded_runs": int(totals.get("runs", 0)),
        "discarded_totals": totals,
    }


# ---------------------------------------------------------------------------
# Submission (shared by the sync and async entry points)
# ---------------------------------------------------------------------------

class _Submission(NamedTuple):
    """Everything the session logic decided, before the HTTP call happens."""

    base_url: str
    api_url: str
    session: str
    target_id: Optional[str]
    url: str
    body: Dict[str, Any]


def _prepare(
    *,
    task: str,
    dataset_url: Optional[str],
    new_sandbox: bool,
    session_id: Optional[str],
    sandbox_id: Optional[str],
    tool_context: Any,
    sandbox_url: Optional[str],
    wait_in_queue: bool,
    verbose: bool,
) -> _Submission:
    """Resolve the session, pick the target sandbox and build the request."""
    base_url = resolve_sandbox_url(sandbox_url)
    api_url = _api(base_url)
    session = resolve_session_key(session_id, tool_context)

    if new_sandbox:
        previous = clear_binding(session, tool_context)
        target_id: Optional[str] = None
        if previous:
            logger.info("Session %s: discarding sandbox %s.", session, previous)
            if verbose:
                print(f"[*] Discarding sandbox {previous}; a clean one will be created.")
    else:
        target_id = sandbox_id or read_binding(session, tool_context)

    body: Dict[str, Any] = {"task": task, "wait": wait_in_queue}
    if target_id:
        body["session_id"] = target_id
    if new_sandbox:
        body["new_container"] = True
    if dataset_url:
        body["dataset_url"] = dataset_url

    endpoint = "run-external" if dataset_url else "run"
    return _Submission(
        base_url=base_url,
        api_url=api_url,
        session=session,
        target_id=target_id,
        url=f"{api_url}/{endpoint}",
        body=body,
    )


def _busy_result(sub: _Submission, response: httpx.Response, verbose: bool) -> Dict[str, Any]:
    detail = _detail(response)
    if verbose:
        print(f"[-] Sandbox busy: {detail}", file=sys.stderr)
    return _normalize({
        "status": "busy",
        "error": f"Sandbox busy: {detail}",
        "session": sub.session,
        "sandbox_id": sub.target_id,
        "reused": bool(sub.target_id),
        "sandbox_expired": False,
    })


def _submit_failure(sub: _Submission, exc: Exception) -> Dict[str, Any]:
    if isinstance(exc, httpx.HTTPStatusError):
        message = (
            f"Submission failed: HTTP {exc.response.status_code} — "
            f"{_detail(exc.response)}"
        )
    else:
        message = f"Submission failed: {exc}"
    return _normalize(_error(
        message, session=sub.session, sandbox_id=sub.target_id, reused=False,
        sandbox_expired=False,
    ))


def _interpret(
    sub: _Submission,
    data: Dict[str, Any],
    *,
    new_sandbox: bool,
    tool_context: Any,
    verbose: bool,
) -> Dict[str, Any]:
    """Re-bind the session to whatever the server returned and describe the run.

    The server is authoritative: when the bound container has expired it creates
    a fresh one and answers with a **new** ``task_id``, so the binding is always
    rewritten from the response rather than assumed.
    """
    new_id = str(data["task_id"])
    reused = bool(data.get("reused", False))
    expired = bool(sub.target_id) and not reused and not new_sandbox
    write_binding(sub.session, new_id, tool_context)

    watch_url = data.get("watch_url") or f"{sub.base_url}/?task_id={new_id}"

    if reused:
        headline = f"Follow-up delivered to existing sandbox: {new_id}"
    elif expired:
        headline = f"Previous sandbox expired — new sandbox created: {new_id}"
    else:
        headline = f"New sandbox created: {new_id}"
    logger.info("Session %s: %s | watch: %s", sub.session, headline, watch_url)

    if verbose:
        print("=" * 80)
        print(f"[+] {headline}")
        print("[+] WATCH LIVE IN WEB CONSOLE:")
        print(f"    👉 {watch_url} 👈")
        print("=" * 80)
        sys.stdout.flush()

    return {
        "sandbox_id": new_id,
        "session": sub.session,
        "reused": reused,
        "sandbox_expired": expired,
        "watch_url": watch_url,
        "vscode_url": data.get("vscode_url", ""),
        "message": data.get("message", ""),
    }


def _announce(on_start: Optional[Callable[[Dict[str, Any]], Any]],
              base_result: Dict[str, Any]) -> Any:
    """Hand the just-provisioned sandbox to the caller, before the wait starts.

    The live console and VS Code URLs are known the moment the server accepts
    the task, but the call itself keeps running for as long as the job does —
    so anything that wants to *watch* the run (a web UI, a log line) has to be
    told here; the return value arrives when there is nothing left to watch.
    A broken callback must never take the run down with it.
    """
    if on_start is None:
        return None
    try:
        return on_start(dict(base_result))
    except Exception:  # noqa: BLE001 - the observer is not worth failing a run
        logger.warning("Sandbox start callback failed.", exc_info=True)
        return None


def _announce_sync(on_start: Optional[Callable[[Dict[str, Any]], Any]],
                   base_result: Dict[str, Any]) -> None:
    """:func:`_announce` for the blocking API, where nothing can be awaited."""
    pending = _announce(on_start, base_result)
    if inspect.isawaitable(pending):
        logger.warning(
            "on_start returned an awaitable in the blocking API; use "
            "arun_sandbox_task for coroutine callbacks."
        )
        getattr(pending, "close", lambda: None)()


async def _aannounce(on_start: Optional[Callable[[Dict[str, Any]], Any]],
                     base_result: Dict[str, Any]) -> None:
    """Async twin of :func:`_announce`; awaits a coroutine callback."""
    pending = _announce(on_start, base_result)
    if inspect.isawaitable(pending):
        try:
            await pending
        except Exception:  # noqa: BLE001
            logger.warning("Sandbox start callback failed.", exc_info=True)


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

def run_sandbox_task(
    task: str,
    dataset_url: Optional[str] = None,
    new_sandbox: bool = False,
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    wait_for_result: bool = True,
    wait_in_queue: bool = True,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: Optional[float] = None,
    on_start: Optional[Callable[[Dict[str, Any]], Any]] = None,
    on_plan: Optional[Callable[[Dict[str, Any]], Any]] = None,
    collect_metrics: bool = True,
    metrics_sink: Optional[Callable[[Dict[str, Any]], Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run ``task`` in the sandbox bound to the caller's session.

    Args:
        task: Instructions for the agent inside the sandbox.
        dataset_url: Optional public URL of a ``.zip`` unpacked into
            ``/workspace``.  Works for a fresh sandbox and for a reused one.
        new_sandbox: Provision a clean container for this session, discarding
            the previous binding.  Use it for an independent experiment.
        session_id: Explicit session identity.  Omit when ``tool_context`` or
            ``AGENT_SESSION_ID`` already identifies the session.
        sandbox_id: Attach to a specific existing sandbox (e.g. handed over by
            another agent) instead of the session's own binding.
        tool_context: Framework context (ADK ``ToolContext``); used both to
            derive the session identity and to pin the binding in session state.
        sandbox_url: Override the sandbox base URL.
        wait_for_result: Block until the task reaches a terminal status.  When
            ``False`` the call returns as soon as the task is accepted, and the
            caller polls with :func:`get_sandbox_status`.
        wait_in_queue: Queue the task when the sandbox is busy instead of
            failing with a "busy" result.
        poll_interval: Seconds between status polls.
        timeout: Give up waiting after this many seconds (the task keeps
            running server-side; the returned ``sandbox_id`` stays valid).
        on_plan: Called with the sandbox agent's own task list each time the
            agent edits it (``{revision, current, progress, items}``), so a
            host can show which step is in progress while the run is still
            going. Never called twice for the same ``revision``, and never at
            all for a task the agent did not plan.
        on_start: Called with ``sandbox_id``/``watch_url``/``vscode_url``/
            ``reused`` as soon as the sandbox is up — i.e. while the task is
            still running, which is the only time the live URLs are useful.
        collect_metrics: Fetch what the run cost once it finishes. ``False``
            skips the request entirely.
        metrics_sink: Called with the metrics record right after the run ends,
            for hosts that route it into their own telemetry. The record is
            journalled either way — see :func:`get_sandbox_metrics`.
        verbose: Print the live-monitoring URL and progress to stdout.

    Returns:
        A dict with ``status``, ``sandbox_id``, ``session``, ``reused``,
        ``summary``, ``s3_uploads``, ``watch_url``, ``vscode_url`` and
        ``error``.
        ``status`` is one of the server statuses plus ``submitted`` (when
        ``wait_for_result=False``), ``busy``, ``timeout`` or ``error``.
        Metrics are deliberately NOT among these keys; they are metadata for
        the host, not context for the model that reads this result.
    """
    try:
        sub = _prepare(
            task=task, dataset_url=dataset_url, new_sandbox=new_sandbox,
            session_id=session_id, sandbox_id=sandbox_id, tool_context=tool_context,
            sandbox_url=sandbox_url, wait_in_queue=wait_in_queue, verbose=verbose,
        )
    except SandboxConfigError as exc:
        return _normalize(_error(str(exc)))

    try:
        response = httpx.post(sub.url, json=sub.body, timeout=DEFAULT_SUBMIT_TIMEOUT)
        if response.status_code == 429:
            return _busy_result(sub, response, verbose)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:  # noqa: BLE001 - network/transport failures
        return _submit_failure(sub, exc)

    base_result = _interpret(
        sub, data, new_sandbox=new_sandbox, tool_context=tool_context, verbose=verbose,
    )
    _announce_sync(on_start, base_result)

    if not wait_for_result:
        return _normalize({**base_result, "status": "submitted"})

    comp = _wait_for_completion(
        api_url=sub.api_url,
        sandbox_id=base_result["sandbox_id"],
        poll_interval=poll_interval,
        timeout=timeout,
        verbose=verbose,
        on_plan=on_plan,
    )
    _collect_metrics(
        api_url=sub.api_url,
        session=sub.session,
        sandbox_id=base_result["sandbox_id"],
        status=comp.get("status"),
        tool_context=tool_context,
        sink=metrics_sink,
        collect=collect_metrics,
    )
    _collect_metrics(
        api_url=sub.api_url,
        session=sub.session,
        sandbox_id=base_result["sandbox_id"],
        status=comp.get("status"),
        tool_context=tool_context,
        sink=metrics_sink,
        collect=collect_metrics,
    )
    merged = {**base_result, **comp}
    if not merged.get("watch_url") and base_result.get("watch_url"):
        merged["watch_url"] = base_result["watch_url"]
    if not merged.get("vscode_url") and base_result.get("vscode_url"):
        merged["vscode_url"] = base_result["vscode_url"]
    return _normalize(merged)


async def arun_sandbox_task(
    task: str,
    dataset_url: Optional[str] = None,
    new_sandbox: bool = False,
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    wait_for_result: bool = True,
    wait_in_queue: bool = True,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    timeout: Optional[float] = None,
    on_start: Optional[Callable[[Dict[str, Any]], Any]] = None,
    on_plan: Optional[Callable[[Dict[str, Any]], Any]] = None,
    collect_metrics: bool = True,
    metrics_sink: Optional[Callable[[Dict[str, Any]], Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Async twin of :func:`run_sandbox_task` — never blocks the event loop.

    Use this from any asyncio runtime (ADK, LangGraph, FastAPI).  Session
    semantics, arguments and the returned shape are identical; only the I/O is
    non-blocking, so a long wait costs a coroutine rather than a thread.
    ``on_start`` and ``on_plan`` may be coroutine functions here and are
    awaited.

    Calls for the SAME session are exclusive (see :func:`_try_acquire_session`):
    if another call for this session is still in flight, this one is refused
    at once with ``status="busy"`` instead of racing it for the binding — a
    caller that fires the tool twice in quick succession gets a clean signal
    to retry instead of silently splitting the session across two sandboxes.
    """
    session = resolve_session_key(session_id, tool_context)
    if not _try_acquire_session(session):
        return _concurrent_busy_result(
            session, sandbox_id or read_binding(session, tool_context),
        )
    try:
        try:
            sub = _prepare(
                task=task, dataset_url=dataset_url, new_sandbox=new_sandbox,
                session_id=session_id, sandbox_id=sandbox_id, tool_context=tool_context,
                sandbox_url=sandbox_url, wait_in_queue=wait_in_queue, verbose=verbose,
            )
        except SandboxConfigError as exc:
            return _normalize(_error(str(exc)))

        try:
            async with httpx.AsyncClient(timeout=DEFAULT_SUBMIT_TIMEOUT) as client:
                response = await client.post(sub.url, json=sub.body)
            if response.status_code == 429:
                return _busy_result(sub, response, verbose)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001 - network/transport failures
            return _submit_failure(sub, exc)

        base_result = _interpret(
            sub, data, new_sandbox=new_sandbox, tool_context=tool_context, verbose=verbose,
        )
        await _aannounce(on_start, base_result)

        if not wait_for_result:
            return _normalize({**base_result, "status": "submitted"})

        comp = await _await_completion(
            api_url=sub.api_url,
            sandbox_id=base_result["sandbox_id"],
            poll_interval=poll_interval,
            timeout=timeout,
            on_plan=on_plan,
        )
        await _acollect_metrics(
            api_url=sub.api_url,
            session=sub.session,
            sandbox_id=base_result["sandbox_id"],
            status=comp.get("status"),
            tool_context=tool_context,
            sink=metrics_sink,
            collect=collect_metrics,
        )
        merged = {**base_result, **comp}
        if not merged.get("watch_url") and base_result.get("watch_url"):
            merged["watch_url"] = base_result["watch_url"]
        if not merged.get("vscode_url") and base_result.get("vscode_url"):
            merged["vscode_url"] = base_result["vscode_url"]
        return _normalize(merged)
    finally:
        _release_session(session)


async def await_sandbox_task(
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    timeout: Optional[float] = None,
    poll_interval: float = DEFAULT_POLL_INTERVAL,
    collect_metrics: bool = True,
    metrics_sink: Optional[Callable[[Dict[str, Any]], Any]] = None,
    on_plan: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Wait (without blocking the event loop) for the session's sandbox task.

    Returns the same shape as :func:`run_sandbox_task`.  When ``timeout``
    elapses the status is ``"timeout"`` and the task keeps running server-side,
    so the call can simply be repeated.

    Exclusive per session, same as :func:`arun_sandbox_task` — see
    :func:`_try_acquire_session`. In practice this only ever contends with
    another ``await_sandbox_task``/``arun_sandbox_task`` call for the SAME
    session, since a running :func:`arun_sandbox_task` holds the claim for
    its own inline wait and only releases it once that wait is over.
    """
    try:
        api_url = _api(resolve_sandbox_url(sandbox_url))
    except SandboxConfigError as exc:
        return _normalize(_error(str(exc)))

    session = resolve_session_key(session_id, tool_context)
    if not _try_acquire_session(session):
        return _concurrent_busy_result(
            session, sandbox_id or read_binding(session, tool_context),
        )
    try:
        target = sandbox_id or read_binding(session, tool_context)
        if not target:
            return _normalize({
                "status": "none",
                "session": session,
                "sandbox_id": None,
                "message": "No sandbox is bound to this session yet.",
            })

        result = await _await_completion(
            api_url=api_url, sandbox_id=target,
            poll_interval=poll_interval, timeout=timeout,
            on_plan=on_plan,
        )
        await _acollect_metrics(
            api_url=api_url,
            session=session,
            sandbox_id=target,
            status=result.get("status"),
            tool_context=tool_context,
            sink=metrics_sink,
            collect=collect_metrics,
        )
        return _normalize({**result, "session": session, "sandbox_id": target})
    finally:
        _release_session(session)


def _normalize(result: Dict[str, Any]) -> Dict[str, Any]:
    """Guarantee the same key set on every return path of :func:`run_sandbox_task`."""
    result.setdefault("summary", "")
    result.setdefault("s3_uploads", [])
    result.setdefault("watch_url", "")
    result.setdefault("vscode_url", "")
    result.setdefault("error", None)
    result.setdefault("succeeded", result.get("status") in SUCCESS_STATUSES)
    return result


class _PollState:
    """Shared bookkeeping for the sync and async wait loops."""

    def __init__(self, timeout: Optional[float], verbose: bool) -> None:
        self.timeout = timeout
        self.verbose = verbose
        self.started = time.monotonic()
        self.last_status: Optional[str] = None
        self.failures = 0
        if verbose:
            print("[*] Waiting for execution to complete...", end="", flush=True)

    def on_poll(self, task_details: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Fold one successful poll in; return a final result once there is one."""
        self.failures = 0

        if task_details is None:
            # The entry was pruned server-side; the container is long gone.
            if self.verbose:
                print("\n[!] Task record no longer available on the server.")
            return {
                "status": self.last_status or "unknown",
                "error": "Task record expired on the server before completion.",
            }

        status = task_details.get("status", "unknown")
        if self.verbose:
            if status != self.last_status:
                print(f"\n[*] Status: {self.last_status or 'submitted'} -> {status}",
                      end="", flush=True)
            else:
                print(".", end="", flush=True)
        self.last_status = status

        if status in TERMINAL_STATUSES:
            if self.verbose:
                print("\n[+] Task execution finished.")
            return {
                "status": status,
                "succeeded": status in SUCCESS_STATUSES,
                "summary": task_details.get("summary", ""),
                # The artifacts the run uploaded, as the server structured
                # them. Carried through verbatim and NOT re-derived from the
                # summary: the sandbox agent no longer writes its links into
                # that text at all, and a presigned URL is the one string a
                # model cannot be trusted to reproduce anyway.
                "s3_uploads": task_details.get("s3_uploads") or [],
                "watch_url": task_details.get("watch_url", ""),
                "vscode_url": task_details.get("vscode_url", ""),
                "error": task_details.get("error"),
            }
        return None

    def on_failure(self, exc: Exception) -> Optional[Dict[str, Any]]:
        """Fold a failed poll in; return a final result once retries run out."""
        self.failures += 1
        if self.verbose:
            print(f"\n[!] Status check failed ({exc}); retrying...", end="", flush=True)
        if self.failures >= _MAX_STATUS_FAILURES:
            return _error(
                f"Lost contact with the sandbox server after "
                f"{self.failures} failed status checks: {exc}"
            )
        return None

    def expired(self) -> Optional[Dict[str, Any]]:
        """Return a timeout result once the caller's deadline has passed."""
        if self.timeout is None or (time.monotonic() - self.started) < self.timeout:
            return None
        if self.verbose:
            print(f"\n[!] Stopped waiting after {self.timeout:.0f}s "
                  f"(the task keeps running server-side).")
        return {
            "status": "timeout",
            "error": (
                f"Still running after {self.timeout:.0f}s. The task continues in "
                f"the sandbox — check it again with this sandbox_id."
            ),
        }


class _PlanRelay:
    """Forward the sandbox agent's own task list to a watcher, once per edit.

    The agent keeps a plan of its own (the SDK's ``task_tracker`` tool) and
    rewrites it as the work goes; ``/status`` returns the current snapshot on
    every poll, unchanged for minutes at a time. ``revision`` rises only on a
    real edit, so it is the only correct "this changed" signal — everything
    else is dropped here rather than re-announced to the watcher.

    A broken watcher must never take a run down: every dispatch is guarded,
    exactly like :func:`_announce`.
    """

    def __init__(self, on_plan: Optional[Callable[[Dict[str, Any]], Any]]) -> None:
        self._on_plan = on_plan
        self._revision: Any = None

    def _next(self, task_details: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self._on_plan is None or not isinstance(task_details, dict):
            return None
        plan = task_details.get("plan")
        if not isinstance(plan, dict):
            # ``plan: null`` is the normal state of a task the agent never
            # planned — a short job can finish without one.
            return None
        revision = plan.get("revision")
        if revision is not None and revision == self._revision:
            return None
        self._revision = revision
        return plan

    def _dispatch(self, plan: Dict[str, Any]) -> Any:
        try:
            return self._on_plan(dict(plan))
        except Exception:  # noqa: BLE001 - the observer is not worth failing a run
            logger.warning("Sandbox plan callback failed.", exc_info=True)
            return None

    def offer(self, task_details: Optional[Dict[str, Any]]) -> None:
        """Blocking API: nothing can be awaited here."""
        plan = self._next(task_details)
        if plan is None:
            return
        pending = self._dispatch(plan)
        if inspect.isawaitable(pending):
            logger.warning(
                "on_plan returned an awaitable in the blocking API; use "
                "arun_sandbox_task for coroutine callbacks."
            )
            getattr(pending, "close", lambda: None)()

    async def aoffer(self, task_details: Optional[Dict[str, Any]]) -> None:
        """Async twin of :meth:`offer`; awaits a coroutine callback."""
        plan = self._next(task_details)
        if plan is None:
            return
        pending = self._dispatch(plan)
        if inspect.isawaitable(pending):
            try:
                await pending
            except Exception:  # noqa: BLE001
                logger.warning("Sandbox plan callback failed.", exc_info=True)


def _wait_for_completion(
    *,
    api_url: str,
    sandbox_id: str,
    poll_interval: float,
    timeout: Optional[float],
    verbose: bool,
    on_plan: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Poll ``/status`` until the task is terminal, times out, or vanishes."""
    state = _PollState(timeout, verbose)
    plans = _PlanRelay(on_plan)
    while True:
        try:
            task_details = _fetch_task(api_url, sandbox_id)
            plans.offer(task_details)
            done = state.on_poll(task_details)
        except Exception as exc:  # noqa: BLE001 - transient network errors
            done = state.on_failure(exc)
        if done is not None:
            return done
        expired = state.expired()
        if expired is not None:
            return expired
        time.sleep(poll_interval)


async def _await_completion(
    *,
    api_url: str,
    sandbox_id: str,
    poll_interval: float,
    timeout: Optional[float],
    verbose: bool = False,
    on_plan: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    """Async twin of :func:`_wait_for_completion`."""
    state = _PollState(timeout, verbose)
    plans = _PlanRelay(on_plan)
    while True:
        try:
            task_details = await _afetch_task(api_url, sandbox_id)
            await plans.aoffer(task_details)
            done = state.on_poll(task_details)
        except Exception as exc:  # noqa: BLE001 - transient network errors
            done = state.on_failure(exc)
        if done is not None:
            return done
        expired = state.expired()
        if expired is not None:
            return expired
        await asyncio.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Session inspection & control
# ---------------------------------------------------------------------------

def get_sandbox_status(
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the current state of the session's sandbox without blocking."""
    try:
        api_url = _api(resolve_sandbox_url(sandbox_url))
    except SandboxConfigError as exc:
        return _error(str(exc))

    session = resolve_session_key(session_id, tool_context)
    target = sandbox_id or read_binding(session, tool_context)
    if not target:
        return {
            "status": "none",
            "session": session,
            "sandbox_id": None,
            "message": "No sandbox is bound to this session yet.",
        }

    try:
        task_details = _fetch_task(api_url, target)
    except Exception as exc:  # noqa: BLE001
        return _error(f"Status request failed: {exc}", session=session, sandbox_id=target)

    if task_details is None:
        return {
            "status": "expired",
            "session": session,
            "sandbox_id": target,
            "message": "The sandbox is gone; the next task will create a new one.",
        }

    status = task_details.get("status", "unknown")
    return {
        "status": status,
        "session": session,
        "sandbox_id": target,
        "busy": status in ("queued", "running"),
        "accepts_followup": status == "cooldown",
        "summary": task_details.get("summary", ""),
        "s3_uploads": task_details.get("s3_uploads") or [],
        "watch_url": task_details.get("watch_url", ""),
        "vscode_url": task_details.get("vscode_url", ""),
        "error": task_details.get("error"),
    }


def reset_sandbox_session(
    *,
    session_id: Optional[str] = None,
    tool_context: Any = None,
) -> Dict[str, Any]:
    """Detach the session from its sandbox without stopping the container.

    The next :func:`run_sandbox_task` call for this session provisions a clean
    sandbox.  The old container finishes its cooldown on its own; use
    :func:`stop_sandbox_session` to free it immediately.
    """
    session = resolve_session_key(session_id, tool_context)
    previous = clear_binding(session, tool_context)
    return {
        "status": "ok",
        "session": session,
        "released_sandbox_id": previous,
        "message": (
            f"Session detached from sandbox {previous}."
            if previous
            else "Session had no sandbox bound."
        ),
    }


def stop_sandbox_session(
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Force-stop the session's container and drop the binding."""
    try:
        api_url = _api(resolve_sandbox_url(sandbox_url))
    except SandboxConfigError as exc:
        return _error(str(exc))

    session = resolve_session_key(session_id, tool_context)
    target = sandbox_id or read_binding(session, tool_context)
    if not target:
        return {"status": "none", "session": session,
                "message": "No sandbox is bound to this session."}

    try:
        response = httpx.post(
            f"{api_url}/stop",
            params={"task_id": target},
            timeout=DEFAULT_STATUS_TIMEOUT,
        )
        response.raise_for_status()
        message = response.json().get("message", "Stopped.")
    except Exception as exc:  # noqa: BLE001
        clear_binding(session, tool_context)
        return _error(
            f"Stop request failed ({exc}); the binding was dropped anyway.",
            session=session,
            sandbox_id=target,
        )

    clear_binding(session, tool_context)
    return {"status": "ok", "session": session, "sandbox_id": target, "message": message}


# ---------------------------------------------------------------------------
# Workspace access
# ---------------------------------------------------------------------------

def list_sandbox_files(
    path: str = "/workspace",
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
) -> Dict[str, Any]:
    """List files produced by the agent inside the session's sandbox."""
    try:
        api_url = _api(resolve_sandbox_url(sandbox_url))
    except SandboxConfigError as exc:
        return _error(str(exc))

    session = resolve_session_key(session_id, tool_context)
    target = sandbox_id or read_binding(session, tool_context)
    if not target:
        return _error("No sandbox is bound to this session.", session=session)

    try:
        response = httpx.get(
            f"{api_url}/files",
            params={"path": path, "task_id": target},
            timeout=DEFAULT_STATUS_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        return _error(
            f"Listing failed: HTTP {exc.response.status_code} — {_detail(exc.response)}",
            session=session,
            sandbox_id=target,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(f"Listing failed: {exc}", session=session, sandbox_id=target)

    entries: List[Dict[str, Any]] = payload.get("entries", [])
    return {
        "status": "ok",
        "session": session,
        "sandbox_id": target,
        "path": payload.get("path", path),
        "entries": entries,
    }


def download_sandbox_file(
    remote_path: str,
    local_path: str,
    *,
    session_id: Optional[str] = None,
    sandbox_id: Optional[str] = None,
    tool_context: Any = None,
    sandbox_url: Optional[str] = None,
    timeout: float = 600.0,
) -> Dict[str, Any]:
    """Download a file (or a directory, as a ZIP) from the sandbox workspace."""
    try:
        api_url = _api(resolve_sandbox_url(sandbox_url))
    except SandboxConfigError as exc:
        return _error(str(exc))

    session = resolve_session_key(session_id, tool_context)
    target = sandbox_id or read_binding(session, tool_context)
    if not target:
        return _error("No sandbox is bound to this session.", session=session)

    try:
        with httpx.stream(
            "GET",
            f"{api_url}/files/download",
            params={"path": remote_path, "task_id": target},
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            with open(local_path, "wb") as fh:
                for chunk in response.iter_bytes():
                    fh.write(chunk)
    except httpx.HTTPStatusError as exc:
        return _error(
            f"Download failed: HTTP {exc.response.status_code}",
            session=session,
            sandbox_id=target,
        )
    except Exception as exc:  # noqa: BLE001
        return _error(f"Download failed: {exc}", session=session, sandbox_id=target)

    return {
        "status": "ok",
        "session": session,
        "sandbox_id": target,
        "remote_path": remote_path,
        "local_path": local_path,
        "size_bytes": os.path.getsize(local_path),
    }


# ---------------------------------------------------------------------------
# LLM-facing tool
# ---------------------------------------------------------------------------

def run_sandbox_task_tool(
    task: str,
    dataset_url: Optional[str] = None,
    new_sandbox: bool = False,
    tool_context: Any = None,
) -> Dict[str, Any]:
    """Run a heavy or long-running task in an isolated sandbox.

    The sandbox is bound to the current session: the first call creates it, and
    every later call continues in the SAME sandbox, keeping the workspace and
    the agent's memory of previous steps. Call it repeatedly to carry one
    experiment forward step by step.

    Args:
        task: Detailed instructions for the AI agent inside the sandbox.
        dataset_url: Optional direct URL of a .zip archive to unpack into the
            workspace.
        new_sandbox: Set to true ONLY to start an independent experiment from a
            clean workspace. Everything from the previous sandbox is lost.

    Returns:
        A dict with the execution status, the result summary and the sandbox id.
    """
    return run_sandbox_task(
        task,
        dataset_url=dataset_url,
        new_sandbox=new_sandbox,
        tool_context=tool_context,
    )


def make_sandbox_tool(
    session_id: str,
    *,
    sandbox_url: Optional[str] = None,
    verbose: bool = True,
    **defaults: Any,
) -> Callable[..., Dict[str, Any]]:
    """Build a tool bound to ``session_id`` for frameworks without a context.

    The returned callable has the same LLM-facing signature as
    :func:`run_sandbox_task_tool` minus ``tool_context``, so the session
    identity is never exposed to (or invented by) the model.
    """

    def sandbox_tool(
        task: str,
        dataset_url: Optional[str] = None,
        new_sandbox: bool = False,
    ) -> Dict[str, Any]:
        """Run a heavy or long-running task in an isolated sandbox.

        The sandbox is bound to the current session: the first call creates it,
        and every later call continues in the SAME sandbox, keeping the
        workspace and the agent's memory of previous steps.

        Args:
            task: Detailed instructions for the AI agent inside the sandbox.
            dataset_url: Optional direct URL of a .zip archive to unpack into
                the workspace.
            new_sandbox: Set to true ONLY to start an independent experiment
                from a clean workspace.
        """
        return run_sandbox_task(
            task,
            dataset_url=dataset_url,
            new_sandbox=new_sandbox,
            session_id=session_id,
            sandbox_url=sandbox_url,
            verbose=verbose,
            **defaults,
        )

    sandbox_tool.__name__ = "run_sandbox_task_tool"
    return sandbox_tool


__all__ = [
    "SandboxConfigError",
    "resolve_sandbox_url",
    "resolve_session_key",
    "read_binding",
    "write_binding",
    "clear_binding",
    "run_sandbox_task",
    "arun_sandbox_task",
    "await_sandbox_task",
    "run_sandbox_task_tool",
    "make_sandbox_tool",
    "get_sandbox_status",
    "get_sandbox_metrics",
    "clear_sandbox_metrics",
    "sandbox_run_digest",
    "reset_sandbox_session",
    "stop_sandbox_session",
    "list_sandbox_files",
    "download_sandbox_file",
]


if __name__ == "__main__":
    # Smoke test: two calls in one session must land in the same sandbox,
    # and a third with new_sandbox=True must land in a different one.
    logging.basicConfig(level=logging.INFO)

    demo_session = "smoke-test-session"

    first = run_sandbox_task(
        "Create a file notes.txt containing the current UTC time.",
        session_id=demo_session,
    )
    print("\n[1] first call:", first.get("status"), first.get("sandbox_id"))

    second = run_sandbox_task(
        "Append the line 'second step' to notes.txt and print the whole file.",
        session_id=demo_session,
    )
    print("[2] follow-up:", second.get("status"), second.get("sandbox_id"),
          "reused =", second.get("reused"))

    third = run_sandbox_task(
        "List the files in /workspace.",
        session_id=demo_session,
        new_sandbox=True,
    )
    print("[3] clean sandbox:", third.get("status"), third.get("sandbox_id"))

    assert first.get("sandbox_id") == second.get("sandbox_id"), "session reuse broken"
    assert third.get("sandbox_id") != first.get("sandbox_id"), "new_sandbox broken"
    print("\nSmoke test passed.")
