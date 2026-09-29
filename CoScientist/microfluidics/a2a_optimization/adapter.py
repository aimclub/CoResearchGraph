"""One external A2A task owns optimization, CFD and equipment execution.

Local tools validate the hand-off, retain raw task responses, and resume the
same task. They never interpret a remote plan as local equipment commands.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from google.adk.tools import ToolContext

from .a2a_test_client import A2AClient, A2ARequestError
from .contracts import INPUT_KEYS, EconomicsRankingError, prepare_inputs
from .operator_ranking import DECLINED, RANKING_KEY, request_operator_ranking

ACTIVE_KEY = "optimization_a2a_task"
HISTORY_KEY = "optimization_a2a_runs"
RESULT_KEY = "optimization_result"
TERMINAL = {"completed", "failed", "rejected", "canceled"}


def _client() -> A2AClient:
    url = os.getenv("OPTIMIZATION_A2A_URL") or os.getenv(
        "A2A_ROUTER_PUBLIC_URL", "https://ailab.se.ifmo.ru/"
    )
    return A2AClient(url, timeout=30.0)


def _publish_result(context: ToolContext, record: dict) -> None:
    # Separate service data from the LLM's narrative. Preserve every response,
    # including earlier plan/artifact snapshots and partial results on failure.
    context.state[RESULT_KEY] = {
        key: record.get(key) for key in (
            "experiment_id", "task_id", "state", "phase", "planning_only",
            "task", "responses", "error", "error_response",
        )
    }


@dataclass(frozen=True)
class Channel:
    """One external A2A system: the agent that tends it and its state keys.

    ``client`` and ``publish`` are looked up at call time, so tests can patch
    the module-level ``_client``. ``publish`` writes the channel's view of the
    service result next to the task record. With ``require_human`` a plan is
    approved only through the web interface card (hitl/human_gate.py); with no
    one to ask, the approval is refused instead of sent unreviewed.
    """
    agent_name: str
    active_key: str
    history_key: str
    invalid_key: str
    client: Callable[[], A2AClient]
    publish: Callable[[ToolContext, dict], None]
    require_human: bool = False


OPTIMIZATION = Channel(
    agent_name="ReactorAgent", active_key=ACTIVE_KEY, history_key=HISTORY_KEY,
    invalid_key=RESULT_KEY, client=lambda: _client(),
    publish=lambda context, record: _publish_result(context, record),
)


def _save(context: ToolContext, record: dict, channel: Channel = OPTIMIZATION) -> dict:
    context.state[channel.active_key] = record
    history = dict(context.state.get(channel.history_key) or {})
    history[record["experiment_id"]] = record
    context.state[channel.history_key] = history
    channel.publish(context, record)
    return record


def _parts(message: Any) -> list:
    if not isinstance(message, dict) or not isinstance(message.get("parts", []), list):
        raise ValueError("Malformed A2A message")
    parts = message.get("parts", [])
    if any(not isinstance(part, dict) for part in parts):
        raise ValueError("Malformed A2A message part")
    return parts


def _receive(context: ToolContext, record: dict, response: dict,
             channel: Channel = OPTIMIZATION) -> dict:
    try:
        if not isinstance(response, dict) or response.get("error"):
            raise ValueError("A2A error or invalid response")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ValueError("A2A response contains no result object")
        task = result.get("task", result)
        if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"]:
            raise ValueError("A2A response contains no task id")
        if record.get("task_id") and task["id"] != record["task_id"]:
            raise ValueError("A2A returned a different task id")
        if task.get("contextId") and task["contextId"] != record["context_id"]:
            raise ValueError("A2A returned a different context id")
        status = task.get("status")
        if not isinstance(status, dict) or not isinstance(status.get("state"), str):
            raise ValueError("A2A task contains no status state")
        state = status["state"].removeprefix("TASK_STATE_").lower().replace("-", "_")
        if state == "cancelled":
            state = "canceled"
        if state not in TERMINAL | {"submitted", "working", "input_required", "auth_required"}:
            raise ValueError(f"Unknown A2A task state: {state}")
        parts = _parts(status.get("message") or {})
        phase = next((str(p["data"]["phase"]) for p in parts
                      if isinstance(p.get("data"), dict) and p["data"].get("phase")), "")
        artifacts = task.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ValueError("Malformed A2A artifacts")
        for artifact in artifacts:
            _parts(artifact)
        messages = task.get("history", [])
        if not isinstance(messages, list):
            raise ValueError("Malformed A2A task history")
        for message in messages:
            _parts(message)
    except ValueError as exc:
        raise A2ARequestError(str(exc), response if isinstance(response, dict) else None) from exc
    return _save(context, {
        **record, "task_id": task["id"], "state": state, "phase": phase,
        "task": task, "response": response,
        "responses": [*record.get("responses", []), response],
        "error": None, "error_response": None,
    }, channel)


def _completed_steps(record: dict) -> int:
    """Steps the external system reports as executed; no counter means none."""
    message = ((record.get("task") or {}).get("status") or {}).get("message") or {}
    return next((part["data"]["completed_steps"] for part in message.get("parts", [])
                 if isinstance(part.get("data"), dict)
                 and isinstance(part["data"].get("completed_steps"), int)), 0)


def _failure(context, record, state, exc, channel: Channel = OPTIMIZATION):
    return _save(context, {
        **record, "state": state, "error": str(exc),
        "error_response": getattr(exc, "response", None),
    }, channel)


def _operator_reachable(tool_context: ToolContext) -> bool:
    """HITL is on and this is a real ADK context a human can answer on.

    Bare unit/CLI contexts have no invocation/session channel on which a
    human could answer; real ADK ToolContext objects do.
    """
    from CoScientist.config import get_settings

    interactive_context = bool(
        getattr(tool_context, "_invocation_context", None)
        or getattr(tool_context, "invocation_context", None)
        or getattr(tool_context, "session", None)
    )
    return bool(get_settings().web.hitl_enabled and interactive_context)


async def _inputs_with_operator_ranking(
    tool_context: ToolContext, exc: EconomicsRankingError,
) -> tuple[dict | None, dict]:
    """Close a missing/unusable economics_ranking with the operator's costs.

    Returns ``(inputs, {})`` once the operator's ranking is stored and the
    hand-off validates, else ``(None, invalid_input result)``. The result is
    flagged ``economics_ranking_required`` so the session does not loop on a
    gap only new data can close.
    """
    failure = {"state": "invalid_input", "error": str(exc), "economics_ranking_required": True}
    if not _operator_reachable(tool_context):
        return None, {**failure, "operator_available": False}
    ranking, reason = await request_operator_ranking(tool_context, exc.route_ids, str(exc))
    if ranking is None:
        return None, {
            **failure, "operator_available": True,
            "operator_declined": reason == DECLINED,
            **({} if reason == DECLINED else {"operator_error": reason}),
        }
    tool_context.state[RANKING_KEY] = ranking
    try:
        return prepare_inputs(tool_context.state), {}
    except (ValueError, TypeError) as retry_exc:
        return None, {**failure, "error": str(retry_exc), "operator_available": True}


async def start_task(
    tool_context: ToolContext, channel: Channel, *, instruction: str,
    id_prefix: str, planning_only: bool = False,
) -> dict[str, Any]:
    """Validate the shared hand-off and send it as one new A2A task.

    A channel owns one task per session: once a task is recorded, repeated
    calls return it instead of sending again.
    """
    previous = tool_context.state.get(channel.active_key)
    if previous:
        return previous
    try:
        inputs = prepare_inputs(tool_context.state, planning_only=planning_only)
    except EconomicsRankingError as exc:
        # Only the cost ranking is missing: the operator can supply it here,
        # in the tool, instead of the model improvising a ranking it cannot store.
        inputs, result = await _inputs_with_operator_ranking(tool_context, exc)
        if inputs is None:
            tool_context.state[channel.invalid_key] = result
            return result
    except (ValueError, TypeError) as exc:
        result = {"state": "invalid_input", "error": str(exc)}
        tool_context.state[channel.invalid_key] = result
        return result
    suffix = uuid.uuid4().hex
    record = {
        "experiment_id": f"{id_prefix}-{suffix}", "context_id": f"ctx-{suffix}",
        "message_id": f"msg-{suffix}", "state": "submitting",
        "planning_only": planning_only, "inputs": inputs, "responses": [],
    }
    _save(tool_context, record, channel)
    text = instruction + "\n\n" + json.dumps(inputs, ensure_ascii=False, allow_nan=False)
    try:
        response = await asyncio.to_thread(
            channel.client().send_message, text=text, experiment_id=record["experiment_id"],
            context_id=record["context_id"], message_id=record["message_id"],
        )
        return _receive(tool_context, record, response, channel)
    except (A2ARequestError, OSError, ValueError) as exc:
        return _failure(tool_context, record, "submission_unknown", exc, channel)


async def optimization_start(tool_context: ToolContext, planning_only: bool = False) -> dict[str, Any]:
    """Delegate the complete experiment/optimization workflow to one A2A task.

    planning_only is for explicitly requested planning and smoke tests. Repeated
    calls reuse the same task, including completed tasks; never rerun hardware
    automatically. A different investigation requires a new CoScientist session.
    """
    instruction = (
        "Ты — внешняя система оптимизации и выполнения экспериментов. "
        "Получаешь ТЗ (tz: исходный запрос и заданные требования), выбранные "
        "маршруты синтеза со стадиями и условиями (routes) и ранжирование по стоимости. "
        "Самостоятельно управляй полным циклом: планирование, необходимые CFD "
        "через свой MCP, оборудование, сбор фактических результатов и оптимизация "
        "по выполненным опытам. Учти результаты последнего опыта перед завершением. "
        "CoScientist не исполняет план локально. Не выбирай invalid/unpriceable "
        "маршруты; partial означает неполную стоимость, сохрани оговорки. "
        "economics_ranking.source=\"operator\" означает, что стоимости задал оператор, "
        "а не прайс поставщика (cost_source у маршрута). "
        "Верни исходные результаты: план и его изменения, выполненные опыты "
        "с параметрами, единицами и измерениями, CFD со статусами и идентификаторами, "
        "итог оптимизации с причиной остановки и незакрытыми критериями. "
        "Используй сообщения/artifacts A2A; не выдавай симуляции за измерения. "
        "Недостающие данные и необходимые подтверждения запрашивай в этой задаче. "
        "Сначала верни план на подтверждение. Не запускай CFD, физическое "
        "оборудование или эксперименты до отдельного сообщения с подтверждением плана. "
    )
    if planning_only:
        instruction += (
            "РЕЖИМ ТОЛЬКО ПЛАНИРОВАНИЯ/СКРИНИНГА: не запускай CFD, оборудование "
            "или эксперименты. Построй план верификации маршрута, явно свяжи каждое "
            "измерение с незакрытым ограничением и запроси подтверждение."
        )
    else:
        instruction += "Выполни задачу в пределах заданных требований и ограничений."
    return await start_task(tool_context, OPTIMIZATION, instruction=instruction,
                            id_prefix="optimization", planning_only=planning_only)


async def get_task_status(tool_context: ToolContext, channel: Channel) -> dict[str, Any]:
    """Poll the channel's task; never resubmit, including after message timeouts."""
    record = tool_context.state.get(channel.active_key)
    if not record or not record.get("task_id"):
        return {"state": "error", "error": "No known task id; do not resubmit an uncertain task"}
    if record["state"] in TERMINAL:
        return record
    try:
        response = await asyncio.to_thread(channel.client().get_task, record["task_id"])
        return _receive(tool_context, record, response, channel)
    except (A2ARequestError, OSError, ValueError) as exc:
        return _failure(tool_context, record, "status_error", exc, channel)


async def optimization_get_status(tool_context: ToolContext) -> dict[str, Any]:
    """Poll the existing task; no resubmission, including after message timeouts."""
    return await get_task_status(tool_context, OPTIMIZATION)


async def _continue(context: ToolContext, record: dict, text: str,
                    channel: Channel = OPTIMIZATION) -> dict:
    message_id = f"msg-{uuid.uuid4().hex}"
    record = {**record, "state": "sending_input", "last_message_id": message_id,
              "sent_messages": [*record.get("sent_messages", []), {"message_id": message_id, "text": text}]}
    _save(context, record, channel)  # Blocks duplicate confirmations during an in-flight send.
    try:
        response = await asyncio.to_thread(
            channel.client().send_message, text=text, experiment_id=record["experiment_id"],
            context_id=record["context_id"], message_id=message_id, task_id=record["task_id"],
        )
        return _receive(context, record, response, channel)
    except (A2ARequestError, OSError, ValueError) as exc:
        return _failure(context, record, "followup_unknown", exc, channel)


async def provide_task_input(details: str, tool_context: ToolContext, channel: Channel) -> dict[str, Any]:
    """Supply known/user-provided missing data to the channel's waiting_input task."""
    record = tool_context.state.get(channel.active_key) or {}
    if not record.get("task_id") or record.get("state") != "input_required" or record.get("phase") != "waiting_input":
        return {"state": "error", "error": "Task is not waiting for missing data"}
    if not details.strip():
        return {"state": "error", "error": "Nonempty clarification required"}
    text = details.strip()
    if record.get("planning_only"):
        text += "\nРежим только планирования сохраняется. Не запускай CFD и оборудование."
    return await _continue(tool_context, record, text, channel)


async def optimization_provide_input(details: str, tool_context: ToolContext) -> dict[str, Any]:
    """Supply known/user-provided missing data to the same waiting_input task."""
    return await provide_task_input(details, tool_context, OPTIMIZATION)


async def approve_task_plan(
    tool_context: ToolContext, channel: Channel, *, operator_message: str, trigger: str,
    action: str,
) -> dict[str, Any]:
    """Approve the channel's current external plan after the operator agrees.

    May start physical equipment remotely. Only valid for input_required/approval;
    planning-only tasks cannot be approved.
    """
    record = tool_context.state.get(channel.active_key) or {}
    if record.get("planning_only"):
        return {"state": "error", "error": "Planning-only task cannot execute"}
    if not record.get("task_id") or record.get("state") != "input_required" or record.get("phase") != "approval":
        return {"state": "error", "error": "Task is not waiting for plan approval"}
    task = record.get("task") or {}
    plan = {"message": (task.get("status") or {}).get("message"), "artifacts": task.get("artifacts")}
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    approvals = record.get("approved_plans", [])
    if fingerprint in approvals:
        return {"state": "error", "error": "This plan was already approved; poll instead of confirming again"}
    # A rebuilt plan carries a new fingerprint, so only executed steps show the
    # remote is running the plan instead of re-planning on every confirmation.
    # Without this, one confirmation per new design loops the operator forever.
    steps = _completed_steps(record)
    if approvals and steps <= record.get("approved_at_step", 0):
        return {"state": "error", "error": (
            f"The external system asks for approval again with {steps} steps executed: it is "
            "re-planning instead of running the approved plan. Do not confirm again; stop and "
            "report this to the operator.")}
    if channel.require_human or _operator_reachable(tool_context):
        from CoScientist.agents.common import hitl_handler
        from CoScientist.graph.session_scope import session_key
        from CoScientist.hitl.human_gate import unavailable_reason
        from CoScientist.hitl.models import HITLAction, HITLRequest

        user_id, session_id = session_key(tool_context)
        request = HITLRequest(
            agent_name=channel.agent_name,
            action_type=HITLAction.APPROVE,
            message=operator_message,
            context={
                # The card prints `output` as text: one sentence on what the
                # approval sets off, not the raw plan dict ("[object Object]").
                "output": f"{action} (задача {record.get('task_id')}).",
                "plan": plan,
                "task_id": record.get("task_id"),
                "_session": {"user_id": user_id, "session_id": session_id},
            },
            invoked_via="tool",
            trigger=trigger,
        )
        if channel.require_human:
            unavailable = unavailable_reason(hitl_handler)
            if unavailable is not None:
                # Nobody can be asked: the plan stays waiting, nothing is sent.
                return {"state": "error", "error": (
                    f"The plan needs approval in the web interface, but {unavailable}; "
                    "nothing was sent to the external system.")}
        decision = await hitl_handler.handle_request(request)
        if not decision.approved:
            return _save(tool_context, {
                **record,
                "state": "rejected",
                "phase": "operator_rejected",
                "error": decision.instructions or decision.free_input or "План отклонён оператором.",
            }, channel)
    record = {**record, "approved_plans": [*approvals, fingerprint], "approved_at_step": steps}
    # The literal token the remote agent executes on, as sent by the vendor's
    # own A2A test script. Any prose — including the Russian
    # sentence of its testing guide — is taken for a clarification, so the
    # remote rebuilds the plan and asks for approval again, forever.
    return await _continue(tool_context, record, "Approve", channel)


async def optimization_approve(tool_context: ToolContext) -> dict[str, Any]:
    """Approve the external system's current plan within the authorized work order.

    May start physical equipment remotely. Only valid for input_required/approval;
    planning-only tasks cannot be approved through this adapter.
    """
    return await approve_task_plan(
        tool_context, OPTIMIZATION,
        operator_message=(
            "Внешняя A2A-система подготовила план, подтверждение может запустить "
            "CFD и физическое оборудование. Разрешить выполнение этого плана?"
        ),
        trigger="optimization_plan_approval",
        action="Подтверждение запустит во внешней A2A-системе оптимизации CFD и опыты на оборудовании по её плану",
    )
