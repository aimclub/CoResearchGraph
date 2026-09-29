"""The flow-synthesis condition-optimization block: a campaign on the rig over A2A.

It runs before the ReactorAgent's A2A task, from exactly the same hand-off
(``contracts.prepare_inputs``), with the same lifecycle (one task per session,
poll, clarify, approve). The difference is what is kept: the raw
``campaign_result`` the service returns is normalized into
``state["optimization"]``::

    {"current": {experiment_id, task_id, context_id, route_id, a2a_state,
                 status, stop_reason, best, recipe, flags, received_at, raw},
     "history": [the same entries without raw],
     "has_blockers": bool}   # a flag of severity "blocker" in current
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

from google.adk.tools import ToolContext

from .a2a_test_client import A2AClient
from .adapter import (
    TERMINAL,
    Channel,
    approve_task_plan,
    get_task_status,
    provide_task_input,
    start_task,
)

ACTIVE_KEY = "campaign_a2a_task"
HISTORY_KEY = "campaign_a2a_runs"
# Local validation failures; the task itself is never started with them.
INPUT_ERROR_KEY = "campaign_input_error"
RESULT_KEY = "optimization"
# The JSON-RPC endpoint from the agent card (its ``url``), e.g.
# https://mcp2.rzhevskyrobotics.com/a2a — set in .env, no built-in default.
URL_ENV = "CAMPAIGN_A2A_URL"
# Top-level fields of a campaign_result; any one of them marks the object.
RESULT_FIELDS = ("best", "recipe", "flags", "stop_reason")


def _client() -> A2AClient:
    """The JSON-RPC endpoint from the agent card (``url``), v1.0 binding.

    Method names and the protocol header are overridable until the service is
    verified against them.
    """
    url = urlsplit(os.environ[URL_ENV].strip())
    version = os.getenv("CAMPAIGN_A2A_VERSION", "1.0")
    return A2AClient(
        f"{url.scheme}://{url.netloc}", timeout=30.0, rpc_path=url.path.lstrip("/"),
        send_method=os.getenv("CAMPAIGN_A2A_SEND_METHOD", "SendMessage"),
        get_method=os.getenv("CAMPAIGN_A2A_GET_METHOD", "GetTask"),
        headers={"A2A-Version": version} if version else {},
    )


def _is_result(value: Any) -> bool:
    return isinstance(value, dict) and any(key in value for key in RESULT_FIELDS)


def _result_in(data: Any) -> dict | None:
    """A campaign_result carried by one part: bare, wrapped or pre-normalized."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError:
            return None
    if not isinstance(data, dict):
        return None
    if _is_result(data.get("campaign_result")):
        return data["campaign_result"]
    optimization = data.get("optimization")
    current = optimization.get("current") if isinstance(optimization, dict) else None
    if isinstance(current, dict):
        return current["raw"] if _is_result(current.get("raw")) else current
    return data if _is_result(data) else None


def campaign_result(task: dict) -> dict | None:
    """The last campaign_result in the task: status message, then artifacts."""
    messages = [(task.get("status") or {}).get("message") or {}, *(task.get("artifacts") or [])]
    found = None
    for message in messages:
        for part in message.get("parts") or []:
            result = _result_in(part.get("data", part.get("text")))
            if result is not None:
                found = result
    return found


def _route_id(result: dict, record: dict) -> str | None:
    if result.get("route_id"):
        return result["route_id"]
    routes = (record.get("inputs") or {}).get("routes") or []
    return routes[0].get("route_id") if len(routes) == 1 else None


def _publish(context: ToolContext, record: dict) -> None:
    task = record.get("task") or {}
    raw = campaign_result(task)
    if raw is None and record.get("state") not in TERMINAL:
        return  # Nothing to report yet: the task record carries the progress.
    raw = raw or {}
    flags = raw.get("flags") if isinstance(raw.get("flags"), list) else []
    current = {
        "experiment_id": record.get("experiment_id"),
        "task_id": record.get("task_id"),
        "context_id": record.get("context_id"),
        "route_id": _route_id(raw, record),
        "a2a_state": (task.get("status") or {}).get("state"),
        "status": record.get("state"),
        "stop_reason": raw.get("stop_reason"),
        "best": raw.get("best"),
        "recipe": raw.get("recipe"),
        "flags": flags,
        "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw": raw or None,
    }
    previous = context.state.get(RESULT_KEY) or {}
    history = list(previous.get("history") or [])
    entry = {key: value for key, value in current.items() if key != "raw"}
    last = history[-1] if history else {}
    # A poll that brings nothing new does not add a history entry.
    if {k: v for k, v in last.items() if k != "received_at"} != {
            k: v for k, v in entry.items() if k != "received_at"}:
        history.append(entry)
    context.state[RESULT_KEY] = {
        "current": current,
        "history": history,
        "has_blockers": any(isinstance(flag, dict) and flag.get("severity") == "blocker"
                            for flag in flags),
    }


CAMPAIGN = Channel(
    agent_name="OptimizationAgent", active_key=ACTIVE_KEY, history_key=HISTORY_KEY,
    invalid_key=INPUT_ERROR_KEY, client=lambda: _client(),
    publish=lambda context, record: _publish(context, record),
    # campaign_approve may start the physical rig: only the web card approves it.
    require_human=True,
)

INSTRUCTION = (
    "Ты — блок оптимизации условий проточного синтеза. Получаешь ТЗ (tz: исходный "
    "запрос и заданные требования), выбранные маршруты синтеза со стадиями и "
    "условиями (routes) и ранжирование по стоимости (economics_ranking). "
    "Поставь кампанию оптимизации условий на установке по выбранному маршруту и "
    "верни её результат (campaign_result): лучшие параметры с целевой функцией, "
    "рецептуру для установки, причину остановки и флаги ограничений результата. "
    "CoScientist не исполняет план локально. Не выбирай invalid/unpriceable "
    "маршруты; partial означает неполную стоимость, сохрани оговорки. "
    "economics_ranking.source=\"operator\" означает, что стоимости задал оператор. "
    "Не выдавай симуляции за измерения. Недостающие данные и необходимые "
    "подтверждения запрашивай в этой задаче. Сначала верни план кампании на "
    "подтверждение и не запускай установку до отдельного сообщения с подтверждением."
)


async def campaign_start(tool_context: ToolContext) -> dict[str, Any]:
    """Start the rig campaign with the validated TZ, routes and cost ranking.

    Repeated calls return the same task, including a completed one; the
    campaign is never rerun automatically.
    """
    if not tool_context.state.get(ACTIVE_KEY) and not os.getenv(URL_ENV, "").strip():
        # Nothing was sent: once the address is set the campaign can start.
        return {"state": "not_configured", "error": f"{URL_ENV} is not set; the campaign was not sent"}
    return await start_task(tool_context, CAMPAIGN, instruction=INSTRUCTION, id_prefix="campaign")


async def campaign_get_status(tool_context: ToolContext) -> dict[str, Any]:
    """Poll the campaign task; no resubmission, including after message timeouts."""
    return await get_task_status(tool_context, CAMPAIGN)


async def campaign_provide_input(details: str, tool_context: ToolContext) -> dict[str, Any]:
    """Supply known/user-provided missing data to the waiting_input campaign task."""
    return await provide_task_input(details, tool_context, CAMPAIGN)


async def campaign_approve(tool_context: ToolContext) -> dict[str, Any]:
    """Approve the campaign plan at input_required/approval; the operator decides.

    May start the physical rig remotely.
    """
    return await approve_task_plan(
        tool_context, CAMPAIGN,
        operator_message=(
            "Блок оптимизации условий подготовил план кампании; подтверждение "
            "запустит физическую установку. Разрешить выполнение этого плана?"
        ),
        trigger="campaign_plan_approval",
        action="Подтверждение запустит кампанию оптимизации условий на физической установке по плану блока оптимизации",
    )
