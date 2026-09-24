"""AgentTool / control-tool callbacks: one route + mandatory record_result."""
from __future__ import annotations

import asyncio

import copy
import json
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from google.adk.models import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .state_machine import (
    ROUTE_AGENT_BY_ROUTE,
    ExperimentRuntimeError,
    active_attempt,
    alembic_route_available,
    fedot_route_available,
    mark_route_returned,
)
from .shared import GATE_ROUTED_STATE_KEY, audit, schema_offers_s3_upload, session_inventory_rows

logger = logging.getLogger(__name__)
ROUTE_AGENT_NAMES = frozenset(ROUTE_AGENT_BY_ROUTE.values())
ROUTE_ALREADY_RETURNED_MESSAGE = (
    "Route already returned for this attempt. Call record_result, retry_task, "
    "fallback_task, skip_task, or amend_task."
)
NO_MATCHING_TOOL_STATE_KEY = "experiment_no_matching_tool"
_NO_MATCHING_TOOL_TOKEN = "NO_MATCHING_TOOL"
_PENDING_RECORD_ALLOWED = frozenset(
    {"record_result", "skip_task", "amend_task", "get_experiment_plan"}
)
RECORD_REQUIRED_MESSAGE = (
    "Route already returned for this attempt but record_result was not called. "
    "Call record_result with the same task_id and attempt_id from start_task "
    "(or skip_task / amend_task). Do not call retry_task/fallback_task until "
    "after record_result closes this attempt. Do not start another task or "
    "finish in prose."
)


def _stringify_agent_tool_request(args: dict[str, Any]) -> None:
    """ADK AgentTool requires ``request`` as a string (``Part.text``)."""
    if "request" not in args or isinstance(args["request"], str):
        return
    val = args["request"]
    args["request"] = (
        json.dumps(val, ensure_ascii=False) if isinstance(val, (dict, list))
        else str(val) if val is not None else val
    )


_EM_ALEMBIC_PIN: dict[str, Any] = {}


def _set_em_alembic_pin(
    *,
    repo_url: str,
    run_id: str | None = None,
    task_id: str | None = None,
    attempt_id: str | None = None,
    runtime: dict[str, Any] | None = None,
    attempt: dict[str, Any] | None = None,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pin: dict[str, Any] = {
        "repo_url": repo_url,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
    }
    if snapshot is not None:
        pin["snapshot"] = copy.deepcopy(snapshot)
    if isinstance(runtime, dict):
        stored = dict(runtime.get("alembic_pin") or {})
        stored.update({k: v for k, v in pin.items() if v is not None})
        if snapshot is not None:
            stored["snapshot"] = copy.deepcopy(snapshot)
        runtime["alembic_pin"] = stored
        pin = stored
    if isinstance(attempt, dict):
        attempt["alembic_pin"] = dict(pin)
        if snapshot is not None:
            attempt["alembic_snapshot"] = copy.deepcopy(snapshot)
    _EM_ALEMBIC_PIN.clear()
    _EM_ALEMBIC_PIN.update(pin)
    return pin


def _read_em_alembic_pin(
    state: dict[str, Any] | None,
    *,
    runtime: dict[str, Any] | None = None,
    attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime is None and isinstance(state, dict):
        raw = state.get("experiment_runtime")
        runtime = raw if isinstance(raw, dict) else None
    for candidate in (attempt, runtime):
        if isinstance(candidate, dict):
            pin = candidate.get("alembic_pin")
            if isinstance(pin, dict) and pin.get("repo_url"):
                return dict(pin)
    return dict(_EM_ALEMBIC_PIN)


def _alembic_snapshot(
    *,
    runtime: dict[str, Any] | None = None,
    attempt: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    for candidate in (attempt, runtime):
        if isinstance(candidate, dict):
            snap = candidate.get("alembic_snapshot")
            if isinstance(snap, dict):
                return snap
            pin = candidate.get("alembic_pin")
            if isinstance(pin, dict) and isinstance(pin.get("snapshot"), dict):
                return pin["snapshot"]
    snap = _EM_ALEMBIC_PIN.get("snapshot")
    return snap if isinstance(snap, dict) else None


def _em_alembic_attempt(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Active alembic_build attempt, or None."""
    try:
        runtime, task_runtime, attempt = active_attempt(state)
    except ExperimentRuntimeError:
        return None
    if str(attempt.get("route") or "") != "alembic_build":
        return None
    return runtime, task_runtime, attempt


def _inject_alembic_repo_url(
    args: dict[str, Any],
    task: dict[str, Any],
    *,
    runtime: dict[str, Any] | None = None,
    attempt: dict[str, Any] | None = None,
) -> None:
    """Force ``task.repo_url`` onto the McpBuilder request."""
    repo_url = task.get("repo_url")
    if not repo_url:
        return
    raw = args.get("request")
    payload: dict[str, Any] = {}
    if isinstance(raw, dict):
        payload = dict(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    payload["repo_url"] = repo_url
    args["request"] = payload
    _set_em_alembic_pin(
        repo_url=str(repo_url),
        run_id=str((runtime or {}).get("run_id") or "") or None,
        task_id=str((runtime or {}).get("active_task_id") or "") or None,
        attempt_id=str((attempt or {}).get("attempt_id") or "") or None,
        runtime=runtime if isinstance(runtime, dict) else None,
        attempt=attempt if isinstance(attempt, dict) else None,
    )


def pin_alembic_build_args(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """before_tool on McpBuilder: pin ``repo_url`` from the EM task."""
    if getattr(tool, "name", "") != "build_mcp_server":
        return None
    state = tool_context.state
    ctx = _em_alembic_attempt(state)
    runtime = task_runtime = attempt = None
    repo_url = ""
    run_id = task_id = attempt_id = None
    if ctx is not None:
        runtime, task_runtime, attempt = ctx
        repo_url = str((task_runtime.get("task") or {}).get("repo_url") or "")
        run_id = runtime.get("run_id")
        task_id = runtime.get("active_task_id")
        attempt_id = attempt.get("attempt_id")
    pin = _read_em_alembic_pin(state, runtime=runtime, attempt=attempt)
    if not repo_url:
        repo_url = str(pin.get("repo_url") or "")
        run_id = run_id or pin.get("run_id")
        task_id = task_id or pin.get("task_id")
        attempt_id = attempt_id or pin.get("attempt_id")
    if not repo_url:
        return None
    args["repo_url"] = repo_url
    args["force_rebuild"] = False
    if run_id:
        args["run_id"] = str(run_id)
    if task_id:
        args["task_id"] = str(task_id)
        args["idempotency_key"] = f"{run_id or ''}:{task_id}:{repo_url.rstrip('/').lower()}"
    if attempt_id:
        args["attempt_id"] = str(attempt_id)
    if runtime is None and isinstance(state.get("experiment_runtime"), dict):
        runtime = state["experiment_runtime"]
    _set_em_alembic_pin(
        repo_url=str(repo_url),
        run_id=str(run_id) if run_id else None,
        task_id=str(task_id) if task_id else None,
        attempt_id=str(attempt_id) if attempt_id else None,
        runtime=runtime if isinstance(runtime, dict) else None,
        attempt=attempt if isinstance(attempt, dict) else None,
    )
    audit(logger, f"EXPERIMENT_ALEMBIC_PIN repo_url={repo_url} task_id={task_id}")
    return None


async def await_alembic_job_if_experiment(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any,
) -> Any:
    """after_tool on McpBuilder: block until the EM build is done or failed.

    Fires on ``build_mcp_server`` and on ``check_mcp_build``: a build that
    another agent started earlier in the session (the orchestrator, the
    builds page) comes back to the module as a running job the builder only
    polls, and polling is what the repeat-call guard cuts off. Waiting here,
    inside the one call, is what keeps the executor from recording a failure
    while the build is still going (2026-09-23, Informer2020 run).
    """
    name = getattr(tool, "name", "")
    if name not in ("build_mcp_server", "check_mcp_build") or not isinstance(tool_response, dict):
        return None
    state = tool_context.state
    ctx = _em_alembic_attempt(state)
    pin = _read_em_alembic_pin(state)
    if ctx is None and not pin.get("repo_url"):
        return None
    job_id = str(tool_response.get("job_id") or (args or {}).get("job_id") or "").strip()
    if not job_id:
        return None

    from CoScientist.tools.alembic_tools import enrich_snapshot_with_tools, wait_mcp_build
    from CoScientist.config import get_settings

    runtime = attempt = None
    if ctx is not None:
        runtime, _, attempt = ctx
        attempt["alembic_job_id"] = job_id

    if tool_response.get("status") in {"done", "failed", "error"}:
        snap = enrich_snapshot_with_tools(dict(tool_response))
        _set_em_alembic_pin(
            repo_url=str(pin.get("repo_url") or (attempt or {}).get("alembic_pin", {}).get("repo_url") or ""),
            run_id=pin.get("run_id"),
            task_id=pin.get("task_id"),
            attempt_id=pin.get("attempt_id"),
            runtime=runtime,
            attempt=attempt,
            snapshot=snap,
        )
        audit(
            logger,
            f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
            f"mcp_url={snap.get('mcp_url') or ''} reused=1",
            stdout=(
                f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
                f"mcp_url={snap.get('mcp_url') or ''}"
            ),
        )
        return snap

    cfg = get_settings().experiments
    audit(logger, f"EXPERIMENT_ALEMBIC_WAIT job_id={job_id} timeout_s={cfg.alembic_timeout_s}")
    # The wait polls with time.sleep; run it in a thread so the web server's
    # event loop (the pages, the API, other sessions) keeps serving meanwhile.
    snap = await asyncio.to_thread(
        wait_mcp_build, job_id, timeout_s=cfg.alembic_timeout_s, poll_s=cfg.alembic_poll_s,
    )
    while snap.get("status") == "running":
        audit(
            logger,
            f"EXPERIMENT_ALEMBIC_WAIT_EXTEND job_id={job_id} "
            f"timeout_s={cfg.alembic_timeout_s}",
        )
        snap = await asyncio.to_thread(
            wait_mcp_build, job_id, timeout_s=cfg.alembic_timeout_s, poll_s=cfg.alembic_poll_s,
        )

    snap = enrich_snapshot_with_tools(snap if isinstance(snap, dict) else {})
    _set_em_alembic_pin(
        repo_url=str(pin.get("repo_url") or ""),
        run_id=pin.get("run_id"),
        task_id=pin.get("task_id"),
        attempt_id=pin.get("attempt_id"),
        runtime=runtime,
        attempt=attempt,
        snapshot=snap,
    )
    if ctx is not None:
        ctx[2]["alembic_job_id"] = job_id
    audit(
        logger,
        f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
        f"mcp_url={snap.get('mcp_url') or ''} timed_out={bool(snap.get('wait_timed_out'))}",
        stdout=(
            f"EXPERIMENT_ALEMBIC_WAIT_DONE job_id={job_id} status={snap.get('status')} "
            f"mcp_url={snap.get('mcp_url') or ''}"
        ),
    )
    return snap


def _pending_record_attempt(
    state: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Active attempt with route returned and no stored result yet."""
    try:
        runtime, task_runtime, attempt = active_attempt(state)
    except ExperimentRuntimeError:
        return None
    if (
        not attempt.get("route_returned")
        or attempt.get("result_id")
        or attempt.get("status") not in {None, "running"}
    ):
        return None
    return runtime, task_runtime, attempt


def guard_route_agent_tool(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Refuse second/mismatched AgentTool, or control calls before record."""
    tool_name = getattr(tool, "name", "") or ""
    state = tool_context.state
    pending = _pending_record_attempt(state)
    if tool_name in ROUTE_AGENT_NAMES:
        if tool_name == "FedotAgent" and not fedot_route_available():
            return {
                "status": "refused", "error_code": "route_disabled",
                "message": "FEDOT.MAS route is disabled by current settings.",
            }
        if tool_name == "McpBuilderAgent" and not alembic_route_available():
            return {
                "status": "refused", "error_code": "route_disabled",
                "message": "Alembic route is disabled by current settings.",
            }
        try:
            _, task_runtime, attempt = active_attempt(state)
        except ExperimentRuntimeError as exc:
            return exc.as_dict()
        if tool_name != (expected := ROUTE_AGENT_BY_ROUTE.get(attempt["route"])):
            return {
                "status": "refused", "error_code": "route_mismatch",
                "message": f"Active attempt requires {expected}, not {tool_name}.",
            }
        if attempt.get("route_returned"):
            return {
                "status": "refused", "error_code": "route_already_returned",
                "message": ROUTE_ALREADY_RETURNED_MESSAGE,
            }
        if tool_name == "McpBuilderAgent":
            _inject_alembic_repo_url(
                args, task_runtime.get("task") or {},
                runtime=tool_context.state.get("experiment_runtime") or {},
                attempt=attempt,
            )
        elif tool_name in {"FedotAgent", "ExperimentAgent"}:
            from CoScientist.experiments.runtime.alembic_bridge import (
                pin_alembic_post_build_request,
            )

            if pin_alembic_post_build_request(
                args, task_runtime,
                runtime=tool_context.state.get("experiment_runtime") or {},
                state=state,
            ):
                audit(
                    logger,
                    "EXPERIMENT_ALEMBIC_POST_BUILD_PIN "
                    f"agent={tool_name} task_id={task_runtime.get('task', {}).get('id')}",
                )
        _stringify_agent_tool_request(args)
        return None
    if pending is not None and tool_name and tool_name not in _PENDING_RECORD_ALLOWED:
        runtime, _, attempt = pending
        return {
            "status": "refused", "error_code": "record_result_required",
            "message": RECORD_REQUIRED_MESSAGE,
            "must_record_task_id": runtime.get("active_task_id"),
            "must_record_attempt_id": attempt.get("attempt_id"),
            "next_action": "record_result",
        }
    return None


def _schema_from_tool(tool: BaseTool, tool_context: ToolContext) -> dict[str, Any]:
    for attr in ("input_schema", "schema"):
        val = getattr(tool, attr, None)
        if isinstance(val, dict):
            return val
    name = getattr(tool, "name", "")
    for item in tool_context.state.get("filtered_tools") or []:
        if isinstance(item, dict) and item.get("tool") == name and isinstance(item.get("input_schema"), dict):
            return item["input_schema"]
    return {}


def force_schema_s3_upload(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """If the tool schema offers upload_results_to_s3, force it on during EM runs."""
    if not tool_context.state.get("experiment_runtime"):
        return None
    if schema_offers_s3_upload(_schema_from_tool(tool, tool_context)):
        args["upload_results_to_s3"] = True
        args.setdefault("output_s3_prefix", "generated")
    return None


def force_molecule_generator_s3_upload(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """Backward-compatible alias — schema-driven, not a named-tool list."""
    return force_schema_s3_upload(tool, args, tool_context)


_TOOL_RESULT_ROUTES = frozenset({"react_tools", "fedot_mas"})


def capture_experiment_tool_results(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any,
) -> None:
    """after_tool on route agents: keep each structured tool result for the attempt."""
    tool_name = getattr(tool, "name", "")
    if not tool_name or tool_name in ROUTE_AGENT_NAMES:
        return
    try:
        _, _, attempt = active_attempt(tool_context.state)
    except ExperimentRuntimeError:
        return
    if str(attempt.get("route") or "") not in _TOOL_RESULT_ROUTES:
        return
    from CoScientist.experiments.runtime.inline_artifacts import record_tool_result

    try:
        record_tool_result(
            tool_context.state, attempt_id=str(attempt["attempt_id"]),
            tool=tool_name, args=args, response=tool_response,
        )
    except Exception as exc:  # noqa: BLE001 — a capture must never break a tool call
        logger.warning("capture_experiment_tool_results failed: %s", exc)


def _materialize_route_tool_results(state: Any, task_runtime: dict[str, Any], attempt: dict[str, Any]) -> str:
    """Write the attempt's tool results under the planner's names; the note for the executor."""
    from CoScientist.experiments.runtime.artifacts import captured_delta
    from CoScientist.experiments.runtime.inline_artifacts import materialize_tool_results

    task = task_runtime.get("task") if isinstance(task_runtime.get("task"), dict) else {}
    existing = {
        str(raw.get("name") or Path(str(raw.get("workspace_path") or raw.get("s3_key") or "")).name)
        for raw in captured_delta(state, attempt)
    }
    created = materialize_tool_results(
        state,
        task_id=str(task.get("id") or ""),
        attempt_id=str(attempt["attempt_id"]),
        expected_artifacts=list(task.get("expected_artifacts") or []),
        existing_names=existing,
    )
    if not created:
        return ""
    listed = ", ".join(f"{c['name']} ({c['rows']} rows, {c['workspace_path']})" for c in created)
    audit(logger, f"EXPERIMENT_TOOL_RESULTS_MATERIALIZED task_id={task.get('id')} attempt_id={attempt['attempt_id']} names={[c['name'] for c in created]}")
    return (
        "\n\n[Experiment module] Materialized from this route's tool results: "
        f"{listed}. They are captured artifacts of this attempt: cite them in "
        "record_result and judge the criteria they satisfy as passed."
    )


def on_route_agent_returned(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: Any,
) -> Optional[dict[str, Any]]:
    """Close the route slot after a successful or failed agent response."""
    tool_name = getattr(tool, "name", "")
    if tool_name not in ROUTE_AGENT_NAMES:
        return None
    try:
        runtime, task_runtime, attempt = active_attempt(tool_context.state)
        if tool_name != ROUTE_AGENT_BY_ROUTE.get(attempt["route"]) or attempt.get("route_returned"):
            return None
        if tool_name == "CoderAgent":
            from CoScientist.experiments.runtime.coder_artifacts import promote_coder_workspace_artifacts
            promote_coder_workspace_artifacts(tool_context.state)
        note = ""
        if tool_name in {"ExperimentAgent", "FedotAgent"}:
            try:
                note = _materialize_route_tool_results(tool_context.state, task_runtime, attempt)
            except Exception as exc:  # noqa: BLE001 — materialization must not block the return
                logger.warning("materialize route tool results failed: %s", exc)
                note = ""
        stored = tool_response
        if note and isinstance(tool_response, dict) and isinstance(tool_response.get("result"), str):
            stored = {**tool_response, "result": tool_response["result"] + note}
        snap = _alembic_snapshot(runtime=runtime, attempt=attempt)
        if tool_name == "McpBuilderAgent" and isinstance(snap, dict):
            stored = copy.deepcopy(snap)
            if not attempt.get("alembic_snapshot"):
                attempt["alembic_snapshot"] = stored
            if snap.get("job_id"):
                attempt["alembic_job_id"] = snap["job_id"]
        mark_route_returned(tool_context.state, tool_name)
        tool_context.state["experiment_last_route_response"] = copy.deepcopy(stored)
        return stored if stored is not tool_response else None
    except ExperimentRuntimeError:
        return None


def _force_call(name: str, args: dict[str, Any], role: str = "model") -> LlmResponse:
    """LlmResponse that replaces the model turn with one forced function call."""
    return LlmResponse(content=types.Content(
        role=role,
        parts=[types.Part.from_function_call(name=name, args=args)],
    ))


def _llm_has_pending_close_call(llm_response: LlmResponse) -> bool:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    return any(
        getattr(getattr(p, "function_call", None), "name", None) in _PENDING_RECORD_ALLOWED
        for p in (parts or [])
    )


def _summary_from_last_route(state: dict[str, Any]) -> str:
    fallback = "Route returned; executor omitted record_result — auto-closing attempt."
    last = state.get("experiment_last_route_response")
    if last is None:
        return fallback
    if isinstance(last, dict):
        for key in ("summary", "message", "status"):
            if last.get(key):
                return str(last.get(key))[:1500]
        return str(last)[:1500]
    text = str(last).strip()
    return text[:1500] if text else fallback


def _make_criteria_checks(criteria: Any, passed: bool, details: str) -> list[dict[str, Any]]:
    if not isinstance(criteria, list):
        return []
    return [
        {"criterion_id": cid, "passed": passed, "details": details}
        for item in criteria
        if isinstance(item, dict) and (cid := str(item.get("criterion_id") or "").strip())
    ]


def _auto_record_result_payload(
    state: dict[str, Any], task_runtime: dict[str, Any], attempt: dict[str, Any],
) -> dict[str, Any]:
    """Best-effort TaskResult so the control loop cannot skip record_result."""
    from CoScientist.experiments.runtime.artifacts import captured_delta

    criteria = (task_runtime.get("task") or {}).get("success_criteria") or []
    summary = _summary_from_last_route(state)
    last = state.get("experiment_last_route_response")
    snap = last if isinstance(last, dict) else {}
    if not snap and isinstance(attempt.get("alembic_snapshot"), dict):
        snap = attempt["alembic_snapshot"]

    route = str(attempt.get("route") or "")
    if route == "alembic_build":
        from CoScientist.experiments.runtime.alembic_bridge import harvest_alembic_mcp_url

        task = task_runtime.get("task") if isinstance(task_runtime.get("task"), dict) else {}
        mcp_url = harvest_alembic_mcp_url(
            snap, last, summary, repo_url=str(task.get("repo_url") or "").strip() or None,
        )
        if mcp_url.startswith("http"):
            return {
                "status": "success",
                "summary": f"Alembic MCP ready at {mcp_url}",
                "outputs": {
                    "mcp_url": mcp_url,
                    "mcp_endpoint": mcp_url,
                    "tools": snap.get("tools") or [],
                    "image": snap.get("image"),
                    "container": snap.get("container"),
                    "job_id": snap.get("job_id") or attempt.get("alembic_job_id"),
                },
                "criteria_checks": _make_criteria_checks(criteria, True, f"mcp_url={mcp_url}"),
                "retryable": False,
                "warnings": ["auto_recorded_alembic_success"],
            }

    has_artifacts = bool(captured_delta(state, attempt)) and route != "alembic_build"
    detail = (
        "Auto-recorded: route returned and executor omitted record_result; evidence taken from route capture."
        if has_artifacts
        else "Auto-recorded failure: route returned with no captured artifacts and executor omitted record_result."
    )
    base: dict[str, Any] = {
        "summary": summary,
        "criteria_checks": _make_criteria_checks(criteria, has_artifacts, detail),
        "outputs": {},
        "warnings": ["auto_recorded_omitted_record_result"],
    }
    if has_artifacts:
        return {**base, "status": "success", "retryable": False}
    return {
        **base,
        "status": "failure",
        "error_code": "route_failed_or_empty",
        "error_message": "Auto-recorded: route returned without captured artifacts; executor omitted record_result.",
        "retryable": True,
    }


def _alembic_job_still_running(attempt: dict[str, Any]) -> bool:
    if str(attempt.get("route") or "") != "alembic_build":
        return False
    job_id = str(attempt.get("alembic_job_id") or "").strip()
    if not job_id:
        return False
    from CoScientist.tools.alembic_tools import peek_mcp_build

    return peek_mcp_build(job_id).get("status") == "running"


def enforce_pending_record_result(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: force record_result when route returned but model skips close."""
    state = callback_context.state
    pending = _pending_record_attempt(state)
    if pending is None or _llm_has_pending_close_call(llm_response):
        return None
    _, task_runtime, attempt = pending
    if _alembic_job_still_running(attempt):
        return None
    runtime = state.get("experiment_runtime") or {}
    task_id = str(runtime.get("active_task_id") or "")
    attempt_id = str(attempt.get("attempt_id") or runtime.get("active_attempt_id") or "")
    if not task_id or not attempt_id:
        return None
    payload = _auto_record_result_payload(state, task_runtime, attempt)
    audit(
        logger,
        f"EXPERIMENT_FORCE_RECORD_RESULT task_id={task_id} attempt_id={attempt_id} "
        f"status={payload.get('status')} retryable={payload.get('retryable')}",
    )
    return _force_call(
        "record_result",
        {"task_id": task_id, "attempt_id": attempt_id, "result": payload},
    )


def _llm_has_any_function_call(llm_response: LlmResponse) -> bool:
    return bool(_llm_function_names(llm_response))


def _llm_function_names(llm_response: LlmResponse) -> list[str]:
    content = getattr(llm_response, "content", None)
    names: list[str] = []
    for part in getattr(content, "parts", None) or []:
        name = getattr(getattr(part, "function_call", None), "name", None)
        if name:
            names.append(str(name))
    return names


def _pending_route_agent(state: Any) -> tuple[str, dict[str, Any]] | None:
    """Active attempt waiting for its route AgentTool — not a control tool."""
    try:
        runtime, _, attempt = active_attempt(state)
    except ExperimentRuntimeError:
        return None
    if attempt.get("route_returned") or attempt.get("result_id"):
        return None
    name = ROUTE_AGENT_BY_ROUTE.get(str(attempt.get("route") or ""))
    if not name:
        return None
    envelope = state.get("experiment_active_envelope") if hasattr(state, "get") else None
    if isinstance(envelope, dict) and envelope:
        request = json.dumps(envelope, ensure_ascii=False)
    else:
        request = json.dumps(
            {
                "task_id": runtime.get("active_task_id"),
                "attempt_id": runtime.get("active_attempt_id"),
            },
            ensure_ascii=False,
        )
    return name, {"request": request}


def _iter_task_runtimes(runtime: dict[str, Any]):
    tasks = runtime.get("tasks") or {}
    for tid in runtime.get("task_order") or []:
        tr = tasks.get(tid)
        if isinstance(tr, dict):
            yield str(tid), tr


def _running_task_id(runtime: dict[str, Any]) -> str | None:
    for tid, tr in _iter_task_runtimes(runtime):
        if str(tr.get("status") or "") == "running":
            return tid
    return None


def _next_control_action(runtime: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if _running_task_id(runtime) is not None:
        return None
    for tid, tr in _iter_task_runtimes(runtime):
        status = str(tr.get("status") or "")
        if status == "ready":
            return "start_task", {"task_id": tid}
        if status == "retry_pending":
            return "retry_task", {"task_id": tid}
        if status == "fallback_pending":
            return "fallback_task", {"task_id": tid}
    return None


def enforce_continue_until_reporting(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: block prose-only exit while ready/retry/fallback work remains."""
    state = callback_context.state
    runtime = state.get("experiment_runtime") or {}
    if runtime.get("phase") != "execution":
        return None
    if pending_route := _pending_route_agent(state):
        name, args = pending_route
        if name in _llm_function_names(llm_response):
            return None
        audit(logger, f"EXPERIMENT_FORCE_ROUTE_AGENT action={name}")
        return _force_call(name, args)
    if _pending_record_attempt(state) is not None or _llm_has_any_function_call(llm_response):
        return None
    action = _next_control_action(runtime)
    if action is None:
        return None
    name, args = action
    audit(logger, f"EXPERIMENT_FORCE_CONTINUE action={name} args={args}")
    return _force_call(name, args)


_CONTROL_TRANSITION_TOOLS = frozenset(
    {"retry_task", "fallback_task", "start_task", "skip_task", "amend_task"}
)


def rewrite_mismatched_control_action(
    callback_context: Any, llm_response: LlmResponse,
) -> LlmResponse | None:
    """after_model: rewrite wrong retry/fallback/start to the next control action."""
    state = callback_context.state
    runtime = state.get("experiment_runtime") or {}
    phase = str(runtime.get("phase") or "")
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    control_fcs: list[tuple[int, str, dict[str, Any]]] = [
        (i, str(getattr(getattr(p, "function_call", None), "name", "")), dict(getattr(getattr(p, "function_call", None), "args", None) or {}))
        for i, p in enumerate(parts)
        if getattr(getattr(p, "function_call", None), "name", None) in _CONTROL_TRANSITION_TOOLS
    ]

    if phase == "execution" and _pending_record_attempt(state) is None:
        if pending_route := _pending_route_agent(state):
            name, args = pending_route
            if name not in _llm_function_names(llm_response):
                called = {n for _, n, _ in control_fcs} | {
                    n for n in _llm_function_names(llm_response)
                    if n in ROUTE_AGENT_NAMES or n in _CONTROL_TRANSITION_TOOLS or n == "get_experiment_plan"
                }
                if called:
                    audit(
                        logger,
                        f"EXPERIMENT_REWRITE_CONTROL from={sorted(called)} to={name} reason=pending_route_agent",
                        stdout=f"EXPERIMENT_REWRITE_CONTROL to={name} reason=pending_route_agent",
                    )
                    return _force_call(name, args, role=getattr(content, "role", None) or "model")

    if not control_fcs:
        return None

    def _suppress(reason: str) -> LlmResponse:
        audit(
            logger,
            f"EXPERIMENT_REWRITE_CONTROL suppress reason={reason} from={[n for _, n, _ in control_fcs]} phase={phase}",
            stdout=f"EXPERIMENT_REWRITE_CONTROL suppress reason={reason} phase={phase}",
        )
        return _force_call("get_experiment_plan", {}, role=getattr(content, "role", None) or "model")

    if phase != "execution":
        return _suppress(f"phase_{phase or 'none'}")

    if _pending_record_attempt(state) is not None:
        return None

    running = _running_task_id(runtime)
    if running is not None:
        if any(n in {"start_task", "retry_task", "fallback_task", "skip_task"} for _, n, _ in control_fcs):
            return _suppress(f"while_running:{running}")
        return None

    expected = _next_control_action(runtime)
    if expected is None:
        return _suppress("no_pending_transition")

    exp_name, exp_args = expected
    # Allow skip_task on the same task that would otherwise start
    for _, name, args in control_fcs:
        if name == "skip_task" and exp_name == "start_task":
            if str(args.get("task_id") or "") == str(exp_args.get("task_id") or ""):
                return None
        if name == exp_name and str(args.get("task_id") or "") == str(exp_args.get("task_id") or ""):
            return None

    fixed_args = dict(exp_args)
    if exp_name == "fallback_task" and not str(fixed_args.get("reason") or "").strip():
        wrong = ",".join(sorted({n for _, n, _ in control_fcs}))
        fixed_args["reason"] = f"Auto-corrected control action (model called {wrong}; runtime requires {exp_name})."

    audit(
        logger,
        f"EXPERIMENT_REWRITE_CONTROL from={[n for _, n, _ in control_fcs]} to={exp_name} args={fixed_args}",
        stdout=f"EXPERIMENT_REWRITE_CONTROL to={exp_name} task_id={fixed_args.get('task_id')}",
    )
    return _force_call(exp_name, fixed_args, role=getattr(content, "role", None) or "model")


def assess_experiment_inventory_feasibility(callback_context: Any) -> None:
    """Record inventory coverage without blocking the always-available Coder lane."""
    from CoScientist.experiments.capabilities.inventory import (
        index_inventory_tools,
        inventory_covers_capabilities,
    )
    state = callback_context.state
    gate_routed = bool(state.get(GATE_ROUTED_STATE_KEY))
    state[GATE_ROUTED_STATE_KEY] = None

    by_tool = index_inventory_tools(session_inventory_rows(state))
    covered = inventory_covers_capabilities(by_tool)
    # An empty/mismatched MCP inventory is not proof that the experiment is
    # infeasible: ExperimentModuleAgent owns CoderAgent and can implement or
    # run repository code directly.  Keep the observation for diagnostics, but
    # never short-circuit the planner/executor here.
    state[NO_MATCHING_TOOL_STATE_KEY] = None
    audit(
        logger,
        f"EXPERIMENT_FEASIBILITY_OK gate_routed={gate_routed} "
        f"inventory={len(by_tool)} covered={covered} coder_fallback=true",
        stdout=(
            f"EXPERIMENT_FEASIBILITY_OK gate_routed={gate_routed} "
            f"inventory={len(by_tool)} covered={covered} coder_fallback=true"
        ),
    )


def skip_when_experiment_not_feasible(callback_context: Any) -> Optional[types.Content]:
    """before_agent: short-circuit EM children after an early NO_MATCHING_TOOL."""
    state = callback_context.state
    # ADK State does not inherit from Mapping (MRO: State -> object), so
    # isinstance(state, Mapping) is always False here and the guard stayed silent.
    getter = getattr(state, "get", None)
    message = getter(NO_MATCHING_TOOL_STATE_KEY) if callable(getter) else None
    if not isinstance(message, str) or not message.strip():
        return None
    state["experiment_execution_summary"] = message
    state["experiment_summary"] = message
    state["hypotheses"] = message
    audit(logger, "EXPERIMENT_SKIP_NOT_FEASIBLE")
    return types.Content(role="model", parts=[types.Part(text=message)])


def skip_when_experiment_stage_complete(callback_context: Any) -> Optional[types.Content]:
    """before_agent: skip completed EM hops, rediscovery on replan, or exhausted replans."""
    state = callback_context.state
    # As above: duck-typed check instead of isinstance(state, Mapping), which
    # never holds for an ADK State and made this whole gate unreachable.
    getter = getattr(state, "get", None)
    if not callable(getter):
        return None
    runtime = getter("experiment_runtime")
    if not isinstance(runtime, dict):
        # An open gate on a finished stage means replanning the whole experiment,
        # so say what it saw. runtime=NoneType means completion never reached the
        # caller across the AgentTool boundary (see _REVIEW_OWNED_STATE_KEYS in
        # experiments/review.py).
        audit(
            logger,
            "EXPERIMENT_STAGE_GATE_OPEN runtime=%s" % type(runtime).__name__,
        )
        return None
    phase = runtime.get("phase")
    agent = str(getattr(callback_context, "agent_name", None) or "")
    from CoScientist.experiments.runtime.state_machine import REPLAN_ROUNDS_KEY
    try:
        replan_count = int(
            getter(REPLAN_ROUNDS_KEY) or runtime.get("replan_count") or 0
        )
    except (TypeError, ValueError):
        replan_count = 0

    if phase == "completed":
        from CoScientist.experiments.review import result_tasks_ok
        if not result_tasks_ok(runtime):
            return None
        summary = state.get("experiment_summary") or state.get("experiment_execution_summary")
        if not isinstance(summary, str) or not summary.strip():
            summary = (
                "Experiment stage already completed for this session; "
                "not starting a second plan on the same ask."
            )
        audit(logger, "EXPERIMENT_SKIP_STAGE_COMPLETE")
        return types.Content(role="model", parts=[types.Part(text=summary)])

    if state.get("experiment_plan_review_paused"):
        # Say WHICH pause and how wide it is. A rejected plan stops the module
        # for THIS ask — a new, different request clears the flag on its first
        # turn — and an operator told only "paused for this session" reasonably
        # concludes the session is dead and starts over from nothing.
        why = str(state.get("experiment_review_pause_reason") or "")
        message = (
            "Experiment plan review is paused"
            + (f" ({why})" if why else "")
            + ": not starting another plan for this request. Send a new request "
              "to plan the experiment again."
        )
        audit(logger, f"EXPERIMENT_SKIP_PLAN_PAUSED reason={why or 'unrecorded'}")
        return types.Content(role="model", parts=[types.Part(text=message)])

    if agent == "ToolPreparerAgent":
        has_inventory = bool(
            state.get("experiment_retrieved_capabilities")
            or state.get("experiment_discovered_capabilities")
        )
        if has_inventory and (phase == "replan_requested" or replan_count > 0):
            message = "Reusing session inventory; skipping tool discovery on replan."
            audit(logger, "EXPERIMENT_SKIP_DISCOVERY_REPLAN")
            return types.Content(role="model", parts=[types.Part(text=message)])

    if agent == "ExperimentPlannerAgent":
        from CoScientist.config import get_settings
        max_replans = get_settings().experiments.max_replans
        if replan_count >= max_replans:
            state["experiment_plan_review_paused"] = True
            message = (
                f"Experiment replan budget exhausted ({replan_count}/{max_replans}); "
                "not starting another plan."
            )
            audit(logger, f"EXPERIMENT_SKIP_REPLAN_BUDGET count={replan_count}")
            return types.Content(role="model", parts=[types.Part(text=message)])
    return None


def pin_fedot_alembic_task(
    tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
) -> dict[str, Any] | None:
    """before_tool on FedotAgent: replace scripty fedot_tool briefs after Alembic."""
    if getattr(tool, "name", "") != "fedot_tool":
        return None
    from CoScientist.experiments.runtime.alembic_bridge import (
        alembic_post_build_context,
        compose_alembic_fedot_task,
    )

    ctx = alembic_post_build_context(tool_context.state)
    if not ctx:
        return None
    original = str(args.get("task_description") or "")
    args["task_description"] = compose_alembic_fedot_task(ctx, original)
    audit(logger, f"EXPERIMENT_ALEMBIC_FEDOT_PIN mcp_url={ctx.get('mcp_url')}")
    return None


__all__ = [
    "ROUTE_ALREADY_RETURNED_MESSAGE",
    "RECORD_REQUIRED_MESSAGE",
    "NO_MATCHING_TOOL_STATE_KEY",
    "assess_experiment_inventory_feasibility",
    "await_alembic_job_if_experiment",
    "capture_experiment_tool_results",
    "force_molecule_generator_s3_upload",
    "force_schema_s3_upload",
    "guard_route_agent_tool",
    "on_route_agent_returned",
    "pin_alembic_build_args",
    "pin_fedot_alembic_task",
    "enforce_pending_record_result",
    "enforce_continue_until_reporting",
    "rewrite_mismatched_control_action",
    "skip_when_experiment_not_feasible",
    "skip_when_experiment_stage_complete",
]
