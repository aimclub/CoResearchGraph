"""ADK tools exposing the OpenHands sandbox to the CoderAgent.

The portable client lives in :mod:`openhands_sandbox` and knows nothing about
ADK. This module is the thin adapter that makes it usable as agent tools:

* **Async.** ADK invokes a *sync* function tool directly on the event loop, so a
  blocking wait would freeze the whole system (web UI, SSE log streaming, every
  other agent) for as long as the sandbox runs — up to hours. Every tool here is
  a coroutine using the client's async API.
* **Bounded inline wait.** Mirrors the ``execute_bash`` / ``check_job`` idiom the
  CoderAgent already knows: the call waits inline for the result, and only hands
  back ``status="running"`` when the task outlives that wait — so the model
  usually gets its answer in ONE call instead of burning turns on polling.
* **One session identity.** The sandbox session is keyed by the SAME id as the
  coder's own workspace (:meth:`CoderToolset._workspace_id`), so an explicit
  ``CODER_WORKSPACE_ID`` pin (A2A), the ADK state pin and the session id all
  behave identically for both environments.
* **Live links out of band.** Because the call only returns when the job is
  over, the ``watch_url`` / ``vscode_url`` in its result reach a UI too late to
  watch anything. So a host (the Web app) registers a sink with
  :func:`set_sandbox_start_sink` and gets them the moment the container is up.
* **Cost out of band too.** What the run cost (its agent's tokens, GPU seconds,
  electricity) travels on the client's metrics channel into the session ledger,
  never through the tool result — see :func:`_metrics_sink`.

Note that the sandbox is a *separate* machine from the coder's ``execute_bash``
workspace: files do not cross between them. Data goes in through ``dataset_url``
and results come back as the sandbox agent's summary (plus the workspace listing
and the live console / VS Code URLs).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Union

from google.adk.tools import ToolContext

from CoScientist.config import get_settings
from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
from CoScientist.tools.coder_tools.coder_tools import CoderToolset

logger = logging.getLogger(__name__)

_settings = get_settings()

#: Seconds ``run_sandbox_task`` waits inline before handing back "running".
RUN_WAIT = int(os.getenv("SANDBOX_RUN_WAIT", "600"))
#: Seconds one ``check_sandbox_task`` call waits inline before returning.
CHECK_WAIT = int(os.getenv("SANDBOX_CHECK_WAIT", "2400"))
#: Seconds between status polls (the sandbox reports coarse-grained progress).
POLL_INTERVAL = float(os.getenv("SANDBOX_POLL_INTERVAL", "10"))


def sandbox_configured() -> bool:
    """True when a sandbox URL is configured for this deployment."""
    try:
        sandbox.resolve_sandbox_url()
        return True
    except sandbox.SandboxConfigError:
        return False


def _session(tool_context: Optional[ToolContext]) -> str:
    """Session key for the sandbox — the coder's own workspace id.

    Keeping the two environments on one identity means the pins that already
    govern the coder workspace (``CODER_WORKSPACE_ID`` for A2A, the ADK session
    state, the session id) govern the sandbox too, so "the same session" means
    the same thing everywhere.
    """
    return CoderToolset._workspace_id(tool_context)


#: ``(session_key, info) -> None | Awaitable``, where ``session_key`` is the
#: host's ``(user_id, session_id)`` or ``None`` when the caller has no host
#: session. Set by the host process; ``None`` outside one (CLI, A2A, tests).
StartSink = Callable[
    [Optional[Tuple[str, str]], Dict[str, Any]],
    Union[None, Awaitable[None]],
]

_start_sink: Optional[StartSink] = None


def set_sandbox_start_sink(sink: Optional[StartSink]) -> None:
    """Register where "the sandbox is up" notices go; ``None`` unregisters.

    The Web runtime wires itself in here so a browser tab gets the live console
    and VS Code links while the sandbox is still working, instead of when
    ``run_sandbox_task`` finally returns.
    """
    global _start_sink
    _start_sink = sink


#: ``(session_key, info) -> None | Awaitable`` for the sandbox agent's own plan.
#: Same shape as :data:`StartSink`; the info carries ``plan`` and the agent
#: that started the run.
PlanSink = StartSink

_plan_sink: Optional[PlanSink] = None


def set_sandbox_plan_sink(sink: Optional[PlanSink]) -> None:
    """Register where the sandbox agent's plan updates go; ``None`` unregisters.

    The agent inside the container keeps a task list of its own and rewrites it
    as the work goes. It is the only account of what is happening in there
    between "task submitted" and the final summary — the tool call itself
    returns nothing until the job is over — so the Web runtime wires itself in
    here and relays each new revision to the tabs watching the session.
    """
    global _plan_sink
    _plan_sink = sink


def _host_session(tool_context: Optional[ToolContext]) -> Optional[Tuple[str, str]]:
    """The host's ``(user_id, session_id)`` for this call, if there is one.

    This is the Web session the user is watching — a different identity from
    :func:`_session`, which keys the sandbox itself by coder workspace.
    """
    try:
        from CoScientist.graph.session_scope import DEFAULT_SESSION_KEY, session_key

        key = session_key(tool_context)
        return None if key == DEFAULT_SESSION_KEY else key
    except Exception:  # noqa: BLE001 - no ADK/graph context available
        return None


def _metrics_sink(tool_context: Optional[ToolContext]):
    """Route what a sandbox run cost into the session's usage ledger.

    The sandbox bills separately from this process — its own agent's LLM calls,
    GPU seconds and electricity — and none of it passes through the ADK model
    callbacks that price everything else. Folding the record in here is what
    makes the session total cover the whole system.

    Charged to the agent that started the run: the sandbox is the CoderAgent's
    tool, and reporting it as a nameless line item hides who spent the money.
    The record itself never goes back to the model — the client keeps it out of
    the tool result, and nothing here puts it back.
    """
    from CoScientist.graph.session_scope import session_key
    from CoScientist.logging.metrics import record_sandbox_run

    key = session_key(tool_context)
    agent = getattr(tool_context, "agent_name", None) or "CoderAgent"

    def sink(record: Dict[str, Any]) -> None:
        sandbox_id = str(record.get("task_id") or "")
        if not sandbox_id:
            return
        record_sandbox_run(key, agent, sandbox_id, sandbox.sandbox_run_digest(record))

    return sink


def _start_notifier(tool_context: Optional[ToolContext]):
    """Build the ``on_start`` callback that pushes the live URLs to the host."""
    host_key = _host_session(tool_context)

    async def announce(info: Dict[str, Any]) -> None:
        sink = _start_sink
        if sink is None:
            return
        result = sink(host_key, {
            "sandbox_id": info.get("sandbox_id"),
            "watch_url": info.get("watch_url", ""),
            "vscode_url": info.get("vscode_url", ""),
            "reused": bool(info.get("reused")),
        })
        if inspect.isawaitable(result):
            await result

    return announce


def _plan_notifier(tool_context: Optional[ToolContext]):
    """Build the ``on_plan`` callback that relays the agent's plan to the host."""
    host_key = _host_session(tool_context)
    agent = getattr(tool_context, "agent_name", None) or "CoderAgent"

    async def relay(plan: Dict[str, Any]) -> None:
        sink = _plan_sink
        if sink is None:
            return
        result = sink(host_key, {"agent": agent, "plan": plan})
        if inspect.isawaitable(result):
            await result

    return relay


def _shape(result: Dict[str, Any], *, waited: int) -> Dict[str, Any]:
    """Map the client result onto the coder toolset's response vocabulary."""
    status = result.get("status")

    if status in ("cooldown", "completed"):
        norm, hint = "success", ""
    elif status in ("timeout", "submitted", "queued", "running"):
        norm = "running"
        hint = (
            f"Still running after {waited}s of inline waiting. Call "
            "check_sandbox_task() ONCE later to pick up the result — do not "
            "poll in a tight loop."
        )
    elif status == "busy":
        norm = "busy"
        hint = "Another task occupies the sandbox. Try again shortly."
    elif status == "none":
        norm, hint = "error", "No sandbox has been started in this session yet."
    else:
        norm, hint = "error", ""

    shaped = {
        "status": norm,
        "summary": result.get("summary", ""),
        "s3_uploads": result.get("s3_uploads") or [],
        "sandbox_id": result.get("sandbox_id"),
        "reused": result.get("reused", False),
        "watch_url": result.get("watch_url", ""),
        "vscode_url": result.get("vscode_url", ""),
        "error": result.get("error"),
    }
    if hint:
        shaped["next_step"] = hint
    if result.get("sandbox_expired"):
        shaped["note"] = (
            "The previous sandbox had expired, so this ran in a NEW empty one — "
            "files from earlier steps are gone."
        )
    return shaped


async def run_sandbox_task(
    task: str,
    dataset_url: str = None,
    new_sandbox: bool = False,
    sandbox_id: str = None,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """
    Delegate a heavy, long-running or GPU-bound job to the OpenHands sandbox —
    a separate, isolated machine where an autonomous coding agent carries the
    task out end to end (training runs, large data processing, long experiments).

    The sandbox is bound to your session: the FIRST call creates it, and every
    later call continues in the SAME sandbox, keeping its `/workspace` files and
    its memory of what it already did. Drive one experiment forward with several
    successive calls (set up, then train, then evaluate) — each call sees the
    results of the previous one.

    This waits inline for the result. Only if the job outlives that wait does it
    return status "running"; then call `check_sandbox_task()` ONCE later — never
    poll in a tight loop.

    IMPORTANT — this is NOT your `execute_bash` workspace. It is a different
    machine, and files do not move between the two. Send data in via
    `dataset_url`; get results back through the returned summary and the
    download links in `s3_uploads`, and verify them with `list_sandbox_files`.

    Use it for work that is too heavy or too long for `execute_bash`. For
    ordinary code, shell and git work, keep using `execute_bash`.

    Args:
        task: Self-contained instructions for the sandbox agent. Be explicit
            about the deliverable and where to write it (e.g. "save the
            checkpoint to /workspace/results/model.pt"), because you cannot see
            its screen — you only get its report back.
        dataset_url: Optional direct URL of a .zip archive to unpack into the
            sandbox `/workspace`.
        new_sandbox: Set true ONLY to start an independent experiment from a
            clean machine. Everything the previous sandbox produced is lost.
        sandbox_id: Continue inside a SPECIFIC sandbox instead of the one bound
            to your session. Use it when a previous task's files live in a
            container this session is no longer bound to — the id is in that
            task's result and in its watch_url. Without it a new container is
            created, with an EMPTY /workspace, and the earlier work is not there.

    Returns:
        Dict with status ("success" | "running" | "busy" | "error"), the
        sandbox agent's summary, s3_uploads (the files the run uploaded, each
        with filename, url, size and key — the summary text does NOT repeat
        these links, so pass them on from here), sandbox_id, watch_url (live
        console), vscode_url, and next_step when a follow-up call is needed.
    """
    result = await sandbox.arun_sandbox_task(
        task,
        dataset_url=dataset_url,
        new_sandbox=new_sandbox,
        sandbox_id=sandbox_id,
        session_id=_session(tool_context),
        tool_context=tool_context,
        timeout=RUN_WAIT,
        poll_interval=POLL_INTERVAL,
        # Announced from inside the client: this call returns only when the job
        # is over, so waiting for its result to carry the links would show them
        # when there is nothing left to watch.
        on_start=_start_notifier(tool_context),
        # The sandbox agent's own plan, relayed as it changes: without it the
        # UI has nothing to say for the minutes (or hours) this call takes.
        on_plan=_plan_notifier(tool_context),
        metrics_sink=_metrics_sink(tool_context),
    )
    logger.info(
        "run_sandbox_task -> %s (sandbox=%s, reused=%s)",
        result.get("status"), result.get("sandbox_id"), result.get("reused"),
    )
    result = await _handle_cancellation_hitl(result, tool_context)
    return _shape(result, waited=RUN_WAIT)


async def _handle_cancellation_hitl(result: Dict[str, Any], tool_context: Optional[ToolContext]) -> Dict[str, Any]:
    """If sandbox task/download was cancelled, pause the agent with a HITL request."""
    if not isinstance(result, dict):
        return result
    err_str = str(result.get("error") or "").lower()
    status_str = str(result.get("status") or "").lower()
    is_cancelled = "cancelled" in err_str or "499" in err_str or status_str == "cancelled"

    if is_cancelled and _settings.web.hitl_enabled:
        from CoScientist.agents.common import hitl_handler
        from CoScientist.hitl.models import HITLAction, HITLRequest

        host_key = _host_session(tool_context)
        user_id, session_id = host_key if host_key else (None, None)

        hitl_req = HITLRequest(
            agent_name="DatasetCollectorAgent",
            action_type=HITLAction.PROVIDE_INPUT,
            message="Dataset download was cancelled. What should I do next?",
            options=[],
            context={
                "download_id": result.get("download_id"),
                "reason": "Download cancelled by user",
                "_session": {
                    "user_id": user_id,
                    "session_id": session_id,
                } if user_id and session_id else None,
            },
            invoked_via="sandbox_download_cancel",
        )
        try:
            response = await hitl_handler.handle_request(hitl_req)
            user_input = (
                response.instructions
                or response.free_input
                or response.selected_option
                or ""
            )
            result["status"] = "cancelled"
            result["summary"] = f"Dataset download was cancelled by user. User instructions: {user_input}"
            result["user_instructions"] = user_input
            result["next_step"] = f"Follow user instructions: {user_input}"
        except Exception as exc:
            logger.warning("HITL request for cancelled download failed: %s", exc)

    return result


async def check_sandbox_task(tool_context: ToolContext = None) -> Dict[str, Any]:
    """
    Pick up the result of a sandbox task that was still "running".

    You normally do NOT need this: `run_sandbox_task` already waits and returns
    the result directly. Use it only when a call came back with status
    "running". It waits inline for the task to finish; if it is STILL running
    you get status "running" again — do other useful work and check once more
    later, never in a tight loop.

    Returns:
        Dict with status ("success" | "running" | "error"), the sandbox agent's
        summary, s3_uploads (the files the run uploaded, each with filename,
        url, size and key), sandbox_id, watch_url and vscode_url.
    """
    result = await sandbox.await_sandbox_task(
        session_id=_session(tool_context),
        tool_context=tool_context,
        timeout=CHECK_WAIT,
        poll_interval=POLL_INTERVAL,
        metrics_sink=_metrics_sink(tool_context),
        on_plan=_plan_notifier(tool_context),
    )
    return _shape(result, waited=CHECK_WAIT)


async def list_sandbox_files(
    path: str = "/workspace",
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """
    List the files inside the sandbox of this session — use it to VERIFY that
    the artifacts the sandbox agent reported (datasets, checkpoints, plots)
    really exist and have a plausible size, before you rely on them.

    This reads the sandbox machine, not your `execute_bash` workspace.

    Args:
        path: Absolute path inside the sandbox, under `/workspace`.

    Returns:
        Dict with status, path, and entries (name, type, size, path).
    """
    result = await asyncio.to_thread(
        sandbox.list_sandbox_files, path, session_id=_session(tool_context),
    )
    return {
        "status": "success" if result.get("status") == "ok" else "error",
        "path": result.get("path", path),
        "entries": result.get("entries", []),
        "error": result.get("error"),
    }


def get_sandbox_tools() -> list:
    """The sandbox tools, or an empty list when no sandbox URL is configured."""
    if not sandbox_configured():
        logger.info("Sandbox URL not configured — sandbox tools not attached.")
        return []
    return [run_sandbox_task, check_sandbox_task, list_sandbox_files]
