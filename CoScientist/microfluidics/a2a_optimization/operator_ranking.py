"""Operator-supplied economics ranking when the costing gave no usable one.

The production hand-off needs a costed ranking of the qualified routes. When
the economics stage could not produce one — ``rank_routes_by_cost`` never
answered, or answered but priced no route (the supplier lists may not index a
reagent by structure, so every route comes back ``unpriceable``) —
``optimization_start`` asks the operator for the route costs in a form, builds
the ranking from the answers and checks it with the same contract. The costs
are the operator's, and the ranking says so (``source="operator"``); the
server's statuses, missing reagents and warnings are kept next to them.
"""
from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .contracts import RANK_BY, TARGET_UNITS, validate_economics_ranking

RANKING_KEY = "economics_ranking"
SOURCE = "operator"
PARAMS_BLOCK = "Параметры рейтинга"
MAX_ATTEMPTS = 3
DECLINED = "declined"

_NUMBER = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_PRICED = {"ok", "partial"}
_UNPRICED = {"invalid", "unpriceable"}


def route_block_title(route_id: str) -> str:
    return f"Маршрут {route_id}"


def _existing(state: Any) -> dict:
    """The ranking already in the state (the server's, possibly unusable)."""
    value = state.get(RANKING_KEY)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _server_rows(existing: dict) -> dict:
    rows = existing.get("routes")
    return rows if isinstance(rows, dict) else {}


def parse_amount(value: Any, name: str) -> Decimal | None:
    """A nonnegative finite number from a form answer; None when left empty.

    Accepts ``42``, ``42.5``, ``42,5``, ``1 000,50`` and a single number with
    a unit around it (``42 руб``); anything else is a ValueError naming the field.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name}: требуется число, получено {value!r}")
    if isinstance(value, (int, float, Decimal)):
        number = Decimal(str(value))
    else:
        text = str(value).replace(" ", "").replace(" ", "").strip()
        if not text:
            return None
        try:
            number = Decimal(text.replace(",", "."))
        except InvalidOperation:
            found = _NUMBER.findall(text)
            if len(found) != 1:
                raise ValueError(f"{name}: требуется одно число, получено {value!r}") from None
            number = Decimal(found[0].replace(",", "."))
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name}: требуется неотрицательное число, получено {value!r}")
    return number


def _json_number(number: Decimal) -> int | float:
    return int(number) if number == number.to_integral_value() else float(number)


def _same(a: Any, b: Decimal) -> bool:
    try:
        return Decimal(str(a)) == b
    except (InvalidOperation, ValueError):
        return False


def _field(name: str, label: str, value: Any, placeholder: str, status: str = "server") -> dict:
    known = value not in (None, "")
    return {
        "name": name,
        "value": str(value) if known else "",
        "status": status if known else "missing",
        "open": not known,
        "label": label,
        "placeholder": placeholder,
    }


def _route_usage(row: dict) -> str:
    if not row:
        return "Сервер цен этот маршрут не посчитал. Укажите стоимость или оставьте пустым — маршрут не будет ранжирован."
    parts = [f"Сервер: статус {row.get('status') or '—'}"]
    if row.get("status") not in _PRICED and row.get("cost_per_unit") not in (None, ""):
        parts.append(f"посчитанная часть {row.get('cost_per_unit')} {row.get('currency') or ''}".rstrip())
    missing = [
        str(item.get("smiles") or item.get("name") or "?") + (f" ({item['reason']})" if item.get("reason") else "")
        for item in row.get("missing") or [] if isinstance(item, dict)
    ]
    if missing:
        parts.append("нет цены: " + ", ".join(missing))
    return "; ".join(parts) + ". Пустые поля — маршрут не будет ранжирован."


def ranking_form(state: Any, route_ids: list[str], error: str) -> dict:
    """HITL form: ranking parameters + one block of costs per qualified route,
    prefilled with whatever the server already answered."""
    existing = _existing(state)
    rows = _server_rows(existing)
    currency = existing.get("preferred_currency") or "RUB"
    blocks = [{
        "title": PARAMS_BLOCK,
        "usage": "Общие параметры расчёта; все стоимости маршрутов — в одной валюте.",
        "fields": [
            _field("target_qty", "Целевое количество продукта", existing.get("target_qty"), "например, 1"),
            _field("target_unit", "Единица: g, kg, mol или mmol", existing.get("target_unit") or "g", "g"),
            _field("preferred_currency", "Валюта стоимостей", currency, "RUB"),
            _field("rank_by", "Ранжировать по: per_unit (себестоимость) или packs (чек за упаковки)",
                   existing.get("rank_by") or "per_unit", "per_unit"),
        ],
    }]
    for route_id in route_ids:
        row = rows.get(route_id) if isinstance(rows.get(route_id), dict) else {}
        priced = row.get("status") in _PRICED
        blocks.append({
            "title": route_block_title(route_id),
            "usage": _route_usage(row),
            "fields": [
                _field("cost_per_unit", "Себестоимость на целевое количество (cost_per_unit)",
                       row.get("cost_per_unit") if priced else None, "например, 120.50"),
                _field("cost_packs", "Чек за целые упаковки (cost_packs); пусто — как себестоимость",
                       row.get("cost_packs") if priced else None, "например, 150"),
            ],
        })
    return {
        "kind": "economics_ranking",
        "title": "Экономический рейтинг для запуска эксперимента",
        "intro": (
            "Внешней системе нужен рейтинг маршрутов по стоимости, но расчёт экономики не дал "
            f"пригодного рейтинга: {error}. Укажите стоимость хотя бы одного маршрута — "
            "ранги будут посчитаны по возрастанию стоимости. Значения будут помечены как "
            "заданные оператором, а не как цены поставщиков."
        ),
        # Costs come only from the operator: no "agent decides" way out.
        "skippable": False,
        "blocks": blocks,
    }


def ranking_from_form(form_values: dict, route_ids: list[str], state: Any, error: str = "") -> dict:
    """Build the ranking from the operator's answers (ValueError on a bad answer).

    A route with a cost is priced (``ok``, or the server's ``partial`` when the
    operator kept the server's numbers); a route without one stays unranked.
    Ranks follow the chosen cost, cheapest first, ties in route order.
    """
    existing = _existing(state)
    rows = _server_rows(existing)
    answered = form_values.get(PARAMS_BLOCK) or {}

    def param(name: str, default: Any = None) -> Any:
        # An empty answer keeps what the server's ranking already said.
        value = answered.get(name)
        if value in (None, ""):
            value = existing.get(name)
        return default if value in (None, "") else value

    target_qty = parse_amount(param("target_qty"), "target_qty")
    if target_qty is None or target_qty == 0:
        raise ValueError("target_qty: требуется положительное целевое количество продукта")
    target_unit = str(param("target_unit", "g")).strip()
    if target_unit not in TARGET_UNITS:
        raise ValueError(f"target_unit: допустимы {', '.join(TARGET_UNITS)}, получено {target_unit!r}")
    currency = str(param("preferred_currency", "RUB")).strip().upper()
    rank_by = str(param("rank_by", "per_unit")).strip()
    if rank_by not in RANK_BY:
        raise ValueError(f"rank_by: допустимы {', '.join(RANK_BY)}, получено {rank_by!r}")

    routes: dict[str, dict] = {}
    priced: list[tuple[Decimal, int, str]] = []
    for index, route_id in enumerate(route_ids):
        server = rows.get(route_id) if isinstance(rows.get(route_id), dict) else {}
        answers = form_values.get(route_block_title(route_id)) or {}
        per_unit = parse_amount(answers.get("cost_per_unit"), f"{route_id}.cost_per_unit")
        packs = parse_amount(answers.get("cost_packs"), f"{route_id}.cost_packs")
        row = {
            "starting_materials": list(server.get("starting_materials") or []),
            "missing": list(server.get("missing") or []),
            "warnings": list(server.get("warnings") or []),
            "server_status": server.get("status"),
        }
        if per_unit is None and packs is None:
            status = server.get("status") if server.get("status") in _UNPRICED else "unpriceable"
            row.update(status=status, rank=None, currency=server.get("currency"),
                       cost_per_unit=None, cost_packs=None, cost_source=None)
            routes[route_id] = row
            continue
        per_unit = per_unit if per_unit is not None else packs
        packs = packs if packs is not None else per_unit
        kept = (server.get("status") in _PRICED and server.get("currency") == currency
                and _same(server.get("cost_per_unit"), per_unit) and _same(server.get("cost_packs"), packs))
        row.update(
            status=server["status"] if kept else "ok",
            currency=currency,
            cost_per_unit=str(per_unit),
            cost_packs=str(packs),
            cost_source="server" if kept else SOURCE,
        )
        if not kept:
            row["warnings"].append("Стоимость задана оператором, не по прайсу поставщика.")
        routes[route_id] = row
        priced.append((per_unit if rank_by == "per_unit" else packs, index, route_id))

    if not priced:
        raise ValueError("укажите стоимость хотя бы одного маршрута")
    for rank, (_, _, route_id) in enumerate(sorted(priced), 1):
        routes[route_id]["rank"] = rank

    ranking = {
        "target_qty": _json_number(target_qty),
        "target_unit": target_unit,
        "preferred_currency": currency,
        "rank_by": rank_by,
        "source": SOURCE,
        "operator_confirmed": True,
        "server_ranking_error": error or None,
        "preliminary": bool(existing.get("preliminary")),
        "default_yield": existing.get("default_yield"),
        "yield_fallback_steps": existing.get("yield_fallback_steps") or {},
        "routes": routes,
    }
    return validate_economics_ranking(ranking, route_ids)


async def request_operator_ranking(tool_context: Any, route_ids: list[str], error: str) -> tuple[dict | None, str]:
    """Ask the operator for the route costs; ``(ranking, "")`` once the answer
    passes the contract, ``(None, DECLINED)`` when the form is skipped, or
    ``(None, reason)`` after MAX_ATTEMPTS unusable answers."""
    from CoScientist.agents.common import hitl_handler
    from CoScientist.graph.session_scope import session_key
    from CoScientist.hitl.models import HITLAction, HITLRequest

    user_id, session_id = session_key(tool_context)
    problem = error
    for _ in range(MAX_ATTEMPTS):
        response = await hitl_handler.handle_request(HITLRequest(
            agent_name="ReactorAgent",
            action_type=HITLAction.EDIT,
            message="Нет пригодного экономического рейтинга маршрутов. Укажите стоимости, чтобы запустить эксперимент.",
            context={
                "output": (
                    f"Стоимости маршрутов {', '.join(route_ids)} войдут в экономический "
                    "рейтинг, по которому внешняя A2A-система запустит эксперимент."
                ),
                "error": problem,
                "route_ids": route_ids,
                "_session": {"user_id": user_id, "session_id": session_id},
            },
            form=ranking_form(tool_context.state, route_ids, problem),
            invoked_via="tool",
            trigger="economics_ranking_intake",
        ))
        values = getattr(response, "form_values", None)
        if not isinstance(values, dict):
            return None, DECLINED
        try:
            return ranking_from_form(values, route_ids, tool_context.state, error), ""
        except ValueError as exc:
            problem = str(exc)
    return None, f"рейтинг оператора не прошёл проверку ({MAX_ATTEMPTS} попытки): {problem}"


__all__ = [
    "DECLINED",
    "MAX_ATTEMPTS",
    "PARAMS_BLOCK",
    "parse_amount",
    "ranking_form",
    "ranking_from_form",
    "request_operator_ranking",
    "route_block_title",
]
