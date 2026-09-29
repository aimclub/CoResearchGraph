"""Что известно про инструменты к моменту, когда выдвигаются гипотезы.

Генератор гипотез просят судить, «насколько гипотеза проверяема имеющимися
инструментами», но инвентаря он не видит: тулов поиска у него нет, а срез графа
показывает узлы Tool одной строкой без описания. Между тем к этому моменту
инвентарь обычно уже собран — планировщик и ретривер MCP отработали раньше и
положили его в состояние сессии.

Этот колбэк берёт то, что там лежит, и превращает в одну короткую вводную:

* инструменты под задачу есть — формулируй гипотезу так, чтобы именно они её и
  подтверждали: измеримый исход, который эти инструменты дают напрямую;
* инструментов нет — проверка будет стоить перебора и разработки, поэтому
  вложись в ОЦЕНКУ: сравни кандидатов по разрешающей силе и объясни выбор.

Вводная детерминирована: она пересказывает инвентарь, а не додумывает его. Если
инвентаря нет вовсе, колбэк молчит — пустая строка лучше, чем утверждение о
средствах, которых никто не считал.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

#: Ключ состояния, который читает промпт через ADK-подстановку {hypothesis_brief?}.
BRIEF_STATE_KEY = "hypothesis_brief"

#: Сколько инструментов показать. Список нужен, чтобы гипотеза формулировалась
#: под конкретный измеряемый исход, а не «под инструменты вообще»; но полтора
#: десятка карточек — это уже не вводная, а шум.
_SHOW = 8

#: Сколько символов описания инструмента доходит до вводной. Описание отвечает
#: на единственный вопрос, ради которого оно здесь: ЧТО этот инструмент меряет.
_DESC = 160

#: Сколько шагов плана показать и насколько подробно. План — это то, чем
#: гипотезу собираются проверять, поэтому описание шага режется вчетверо мягче,
#: чем описание инструмента: последовательность вызовов важна целиком.
_PLAN_STEPS = 6
_PLAN_DESC = 700


def _rows(state: Any) -> List[Dict[str, Any]]:
    """Инвентарь инструментов сессии, если модуль экспериментов его собрал."""
    try:
        from CoScientist.experiments.runtime.shared import session_inventory_rows

        rows = session_inventory_rows(state)
    except Exception as exc:  # noqa: BLE001 — профиль без EM это норма
        logger.debug("гипотезы: инвентарь недоступен (%s)", exc)
        return []
    return [r for r in (rows or []) if isinstance(r, dict)]


def _clean(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _signature(row: Dict[str, Any]) -> str:
    """``tool(arg, arg*)`` — аргументы из JSON-схемы, обязательные со звёздочкой.

    Аргументы здесь не украшение. Инструмент БЕЗ них нельзя навести на
    подмножество, которое выберет гипотеза: он вернёт то, что заложено в него
    самого. Именно на этом в реальном прогоне сломалась гипотеза, требовавшая
    сравнения кластеров через инструмент, который всегда отдаёт один кластер.
    """
    name = str(row.get("tool") or row.get("name") or "").strip()
    schema = row.get("input_schema")
    if not isinstance(schema, dict) or not schema:
        return f"{name}(?)"
    props = schema.get("properties")
    if not isinstance(props, dict):
        return f"{name}(?)"
    if not props:
        return f"{name}()"
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    args = [f"{key}*" if key in required else str(key) for key in props]
    return f"{name}({', '.join(args)})"


def _tools(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Инструменты инвентаря: имя, сигнатура и что он меряет — без повторов."""
    out: List[Dict[str, str]] = []
    seen: set = set()
    for row in rows:
        name = str(row.get("tool") or row.get("name") or "").strip()
        server = _clean(row.get("server_id") or row.get("server_name"), 40)
        identity = (server, name)
        if not name or identity in seen:
            continue
        seen.add(identity)
        out.append({
            "name": name,
            "signature": _signature(row),
            "description": _clean(row.get("description"), _DESC),
            "server": server,
            "no_args": _signature(row).endswith("()"),
        })
    return out


def _names(rows: List[Dict[str, Any]]) -> List[str]:
    return [t["name"] for t in _tools(rows)]


def _plan(state: Any) -> List[Dict[str, str]]:
    """Шаги плана ЦЕЛИКОМ — из мастер-списка, а не из почищенного среза.

    `clean_tasks_for_agent` снимает `description` со всех задач, кроме
    назначенных читателю, поэтому через `{active_tasks}` генератор гипотез видит
    шаг исполнения одним заголовком: последовательность вызовов, ради которой
    план и писали, до него не доходит. Мастер-список эту чистку не проходил.
    """
    getter = getattr(state, "get", None)
    if not callable(getter):
        return []
    try:
        tasks = getter("_master_active_tasks") or getter("active_tasks") or []
    except Exception:  # noqa: BLE001
        return []
    candidates: List[tuple[int, Dict[str, str]]] = []
    for task in tasks if isinstance(tasks, list) else []:
        if not isinstance(task, dict):
            continue
        title = _clean(task.get("title") or task.get("name"), 120)
        body = _clean(task.get("description"), _PLAN_DESC)
        if not (title or body):
            continue
        assignee = _clean(task.get("assignee"), 40)
        haystack = f"{title} {body} {assignee}".lower()
        # Reporting/self-management steps describe delivery, not a measurement
        # that can make a hypothesis true or false.
        if any(word in haystack for word in (
            "resultaggregator", "final report", "итогов", "отчёт", "отчет",
            "hypothesesagent", "сформулировать гипотез",
        )):
            continue
        score = 1 if any(word in haystack for word in (
            "experiment", "эксперимент", "measure", "измер", "analysis", "анализ",
            "model", "модел", "simulation", "симуляц", "research", "исслед",
        )) else 0
        candidates.append((score, {
            "id": _clean(task.get("id"), 24),
            "assignee": assignee,
            "title": title,
            "description": body,
        }))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in candidates[:_PLAN_STEPS]]


#: По этим ключам видно, что поиск инструментов УЖЕ состоялся. Разница важна:
#: «искали и не нашли» и «ещё не искали» — это разные указания агенту, а пустой
#: инвентарь выглядит одинаково в обоих случаях.
_SEARCH_MARKERS = (
    "retrieval_queries",                    # retrieve_tools отработал
    "executor_tool_match",                  # реранкер вынес вердикт
    "experiment_retrieved_capabilities",
    "experiment_discovered_capabilities",
    "filtered_tools",
)


def _rejected_all(state: Any) -> bool:
    """Реранкер судил кандидатов и не оставил ни одного.

    Важно именно это, а не пустота инвентаря: `accumulated_tools` и
    `experiment_retrieved_capabilities` переживают отбор и продолжают
    перечислять кандидатов, которых реранкер только что отверг. Прочитанные как
    «вот чем это можно измерить», они уводят гипотезу к инструменту, который
    никто не собирается запускать.
    """
    getter = getattr(state, "get", None)
    if not callable(getter):
        return False
    try:
        from CoScientist.agents.callbacks.tool_callbacks import UNJUDGED_REASONS

        verdict = getter("executor_tool_match") or {}
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(verdict, dict) or not verdict:
        return False
    # Вердикта не было (разбор упал, ранжирование пустое) — это не отказ.
    if verdict.get("reason") in UNJUDGED_REASONS:
        return False
    return not verdict.get("matched")


def _searched(state: Any) -> bool:
    getter = getattr(state, "get", None)
    if not callable(getter):
        return False
    for key in _SEARCH_MARKERS:
        try:
            if getter(key):
                return True
        except Exception:  # noqa: BLE001
            return False
    return False


def _plan_section(plan: List[Dict[str, str]] | None) -> str:
    """Шаги, которыми гипотезу собираются проверять — если план уже есть.

    Планировщик отрабатывает раньше и видит инвентарь; его шаги — это готовый
    ответ на вопрос «чем это будет меряться». Гипотеза, написанная мимо них,
    отправляет прогон делать не то, что запланировано.
    """
    if not plan:
        return ""
    lines = []
    for step in plan:
        who = f" [{step['assignee']}]" if step["assignee"] else ""
        head = f"- **{step['id']}**{who} {step['title']}".rstrip()
        lines.append(head if not step["description"]
                     else f"{head}\n  {step['description']}")
    return (
        "\n\n### AVAILABLE MEANS OF TESTING THE STUDY\n"
        "These are the planned observations and analyses available for this "
        "study; the planner saw the inventory when it selected them:\n\n"
        + "\n".join(lines)
        + "\n\nState a scientific claim about the material, system, mechanism or "
        "effect under study — never a claim about what this plan will show. "
        "Use these steps as the VerificationMethod: say which observation or "
        "analysis tests the claim and which result would support it. If the "
        "available methods cannot distinguish the claim, narrow the scientific "
        "claim to what they can measure; do not invent an unplanned method."
    )


def _catalogue(tools: List[Dict[str, str]]) -> str:
    """Карточки инструментов: сигнатура и что инструмент меряет."""
    lines = []
    for tool in tools[:_SHOW]:
        what = f" — {tool['description']}" if tool["description"] else ""
        owner = f"{tool['server']}:" if tool["server"] else ""
        lines.append(f"- `{owner}{tool['signature']}`{what}")
    if len(tools) > _SHOW:
        lines.append(f"- (+{len(tools) - _SHOW} more in the inventory)")
    return "\n".join(lines)


#: Что следует из инструмента без аргументов. Отдельным абзацем, потому что это
#: не придирка к оформлению, а граница проверяемости: гипотезу, требующую среза,
#: которого инструмент не делает, никто не проверит — её просто тихо ослабят на
#: этапе выводов.
_NO_ARGS_RULE = (
    "\nA tool whose signature is `name()` takes NO arguments: it cannot be "
    "aimed at a subset you choose, and returns only the scope it was built "
    "for. Before you write a claim that compares groups — clusters, cohorts, "
    "arms — check that some tool here actually SPLITS by that grouping. If "
    "none does, the comparison cannot be measured in this run: make the claim "
    "about the scope the tools do return, or the verdict will quietly be "
    "rewritten to fit what was measurable."
)

_ARTEFACTS_RULE = (
    "\nPrefer the claim whose verification LEAVES THE MOST BEHIND. These tools "
    "return figures, tables and files as well as numbers, and those artefacts "
    "are the report. Between two claims of equal sharpness, commit the one "
    "whose VerificationMethod puts more of this inventory to work — and name "
    "those tools in the method, so the run actually calls them."
)


def _compose(names: List[str], searched: bool = True,
             tools: List[Dict[str, str]] | None = None,
             plan: List[Dict[str, str]] | None = None) -> Tuple[str, str]:
    """Текст вводной и короткая метка режима для лога."""
    if names:
        tools = tools or [{"name": n, "signature": f"{n}(?)", "description": "",
                           "server": "", "no_args": False} for n in names]
        # «the inventory holds», а не «retrieved for this task»: инвентарь живёт
        # в сессии и мог быть собран под предыдущий запрос. Указать на него
        # честно можно, выдать за поиск именно под эту задачу — нет.
        text = (
            "### WHAT THIS RUN CAN ALREADY MEASURE\n"
            f"The tool inventory of this session holds {len(names)} ready "
            "tools. Signature first, then what it measures:\n\n"
            f"{_catalogue(tools)}\n\n"
            "The study can therefore be settled by RUNNING them — so formulate "
            "the hypothesis as the claim these tools come closest to settling: "
            "its outcome must be a number (or a named check) that one of them "
            "produces directly, and its ConfirmationCriteria must threshold "
            "exactly that output. Do not propose a claim whose measurement "
            "none of them yields; do not invent a tool when one of these "
            "already answers the question."
        )
        if any(t.get("no_args") for t in tools[:_SHOW]):
            text += _NO_ARGS_RULE
        text += _ARTEFACTS_RULE
        text += _plan_section(plan)
        return text, f"tools:{len(names)}"
    judging = (
        "So spend the effort on JUDGING the candidates before you commit: write "
        "out the alternatives you considered, compare them by how sharply a "
        "single run could separate them and by how much the answer would change "
        "what happens next, drop those that no single measurement can decide, "
        "and state in the rationale why the one you commit survived. A claim "
        "that cannot be measured at all is not a hypothesis here, however "
        "interesting."
    )
    if searched:
        return (
            "### NO READY TOOL HAS BEEN FOUND FOR THIS TASK\n"
            "The inventory was searched and nothing in it measures this outcome, "
            "so verifying whatever you propose will cost a search, an adaptation "
            "or new code — an expensive branch, and the wrong claim is expensive "
            "twice.\n" + judging + _plan_section(plan),
            "tools:none",
        )
    return (
        "### THE TOOL INVENTORY HAS NOT BEEN TAKEN YET\n"
        "Nothing has looked for instruments for this task, so assume neither "
        "that they exist nor that they do not. What follows from that: name the "
        "MEASUREMENT your hypothesis needs — the quantity, its unit, the data it "
        "is taken on — because that name is what the planner will search for, "
        "and a claim whose measurement is unnamed cannot be equipped at all.\n"
        + judging + _plan_section(plan),
        "tools:unknown",
    )


def brief_hypotheses_regime(callback_context) -> None:
    """before_agent: положить вводную о режиме туда, откуда её читает промпт.

    Промпт рендерится один раз при сборке дерева, а инвентарь свой у каждой
    сессии — поэтому вводная едет через состояние и подстановку
    `{hypothesis_brief?}`.
    """
    try:
        state = callback_context.state
        tools = [] if _rejected_all(state) else _tools(_rows(state))
        names = [t["name"] for t in tools]
        plan = _plan(state)
        text, mode = _compose(names, _searched(state), tools, plan)
        state[BRIEF_STATE_KEY] = text
        logger.info("гипотезы: режим %s, шагов плана %d", mode, len(plan))
    except Exception as exc:  # noqa: BLE001 — без вводной агент просто работает как раньше
        logger.warning("гипотезы: вводная не собрана (%s)", exc)
        try:
            callback_context.state[BRIEF_STATE_KEY] = ""
        except Exception:  # noqa: BLE001
            pass
    return None


__all__ = ["BRIEF_STATE_KEY", "brief_hypotheses_regime"]
