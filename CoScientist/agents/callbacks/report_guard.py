"""Отчёт не пишется о работе, которой не было.

`ResultAggregatorAgent` — терминальная стадия конвейера (`pipeline.post`), и
запускается она независимо от того, выполнилось ли что-нибудь. Пока модуль
экспериментов доходил до конца, это было незаметно. Но у него есть ветка, в
которой он останавливается тихо: обзор плана не подтверждён, `initialize_runtime`
не вызван, `skip_executor_without_runtime` пропускает исполнителя — и дальше
агрегатор как ни в чём не бывало пишет полный научный отчёт.

По сохранённым сессиям это не редкость: **5 из 8 «завершённых» прогонов на
профиле experiments не имеют ни одного узла ExperimentTask**, а три из них —
`0a31978b`, `196dd0f7`, `236c0325` — прямо показывают связку «обзор плана истёк →
задач не возникло → написан отчёт на 8–12 тысяч знаков». Для читателя такой
прогон неотличим от успешного. Это хуже зависания: зависание видно, а это нет.

Решение детерминированное, без участия модели — просить агрегатор «не выдумывать»
значило бы лечить то же тем же:

* модуль не задействовали вовсе (чисто литературное исследование) — молчим,
  отчёт законен;
* модуль задействовали, и ничего не вышло — вместо отчёта короткая справка: что
  планировалось, почему не пошло, что нажать;
* модуль задействовали, часть прошла — отчёт пишется, но в промпт кладётся
  врезка о непройденном, и промпт обязан её воспроизвести.

«Что-то вышло» считается не по задачам модуля, а шире — по свидетельствам в
графе: оркестратор мог собрать литературу сам, и терять её описание из-за того,
что встал модуль экспериментов, неправильно.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from google.genai import types

from CoScientist.experiments.outcome.reconciliation import (
    DispositionKind,
    reconcile_scientific_outcome,
)

logger = logging.getLogger(__name__)

#: Ключ состояния, который промпт агрегатора читает через ADK-подстановку
#: `{report_unexecuted_note?}`.
UNEXECUTED_NOTE_KEY = "report_unexecuted_note"

#: Исходы задачи, которые считаются сделанной работой. `partial` здесь наравне с
#: `success`: частичный результат — это всё-таки измерение, о котором есть что
#: написать. Совпадает с тем, что `result_tasks_ok` считает приемлемым.
_CARRIED_OUT = frozenset({"success", "partial"})


def _module_was_engaged(state: Any) -> bool:
    """Запускался ли модуль экспериментов в этом прогоне.

    Три следа, любой из которых означает «модуль трогали»: рантайм (даже
    неутверждённый), карточка плана для обзора и сообщение о пропуске
    исполнения. Ни одного — значит исследование шло без модуля, и отчёт по нему
    ничем не скомпрометирован.
    """
    runtime = state.get("experiment_runtime")
    if isinstance(runtime, dict) and runtime:
        return True
    return bool(state.get("experiment_plan_view")
                or state.get("experiment_execution_summary"))


def _carried_out(state: Any) -> int:
    """Сколько задач эксперимента реально дали результат."""
    done = 0
    for row in state.get("experiment_task_results") or []:
        if isinstance(row, dict) and str(row.get("status") or "") in _CARRIED_OUT:
            done += 1
    if not done:
        tasks = state.get("_master_active_tasks") or state.get("active_tasks") or []
        done = sum(
            1 for row in tasks
            if isinstance(row, dict) and str(row.get("status") or "").upper() == "DONE"
        )
    return done


def _planned(state: Any) -> int:
    """Сколько задач было в утверждённом плане."""
    runtime = state.get("experiment_runtime")
    if isinstance(runtime, dict):
        order = runtime.get("task_order")
        if isinstance(order, (list, tuple)):
            return len(order)
    view = state.get("experiment_plan_view")
    if isinstance(view, dict):
        tasks = view.get("tasks")
        if isinstance(tasks, (list, tuple)):
            return len(tasks)
    tasks = state.get("_master_active_tasks") or state.get("active_tasks") or []
    if isinstance(tasks, (list, tuple)):
        return len(tasks)
    return 0


def _evidence_in_graph(callback_context: Any) -> int:
    """Свидетельства, записанные в граф кем угодно.

    Это и есть страховка от того, чтобы привратник не съел литературную часть
    прогона: `ResearchAgent` пишет свидетельства напрямую, без модуля.
    """
    try:
        from CoScientist.config import get_settings

        if not get_settings().research_graph.enabled:
            return 0
        from CoScientist.graph.research.store import get_research_graph

        nodes = get_research_graph(callback_context).overview().get("nodes") or []
        return sum(1 for n in nodes
                   if isinstance(n, dict) and n.get("type") == "Evidence")
    except Exception as exc:  # noqa: BLE001 — привратник не ломает прогон
        logger.warning("отчёт: граф недоступен, свидетельства не сочтены: %s", exc)
        return 0


def _why(state: Any, lang: str) -> str:
    """Почему исполнение не состоялось, словами самой системы.

    Пустая строка, когда причины нет: фаза «execution» ничего не останавливала,
    и называть её причиной остановки значило бы придумать событие. Прогон, где
    часть задач просто не дошла очередь, — это не прогон, который встал.
    """
    pause = str(state.get("experiment_review_pause_reason") or "").strip()
    runtime = state.get("experiment_runtime")
    phase = str(runtime.get("phase") or "") if isinstance(runtime, dict) else ""
    # The pause reason FIRST, and translated. `experiment_execution_summary` is
    # `skip_executor_without_runtime`'s own English line — "Experiment execution
    # skipped: no approved experiment runtime is active (plan_review_timeout
    # (phase=awaiting_review))." — written for a log, and preferring it put that
    # sentence, state keys and all, in the middle of a Russian note to the
    # operator. It stays as the fallback for the cases the table has no word for.
    if pause:
        said = _PAUSE_WORDS[lang].get(pause, pause)
        diagnosis = state.get("experiment_plan_stop_diagnosis") if hasattr(state, "get") else None
        if isinstance(diagnosis, dict) and diagnosis.get("primary_issue"):
            # Name the blocker that was actually left. Do not substitute
            # "the plan is too complex" when the critique said something else.
            if not (diagnosis.get("blamed_complexity") is False and "complex" in said.lower()):
                said = f"{said}: {diagnosis['primary_issue']}"
        return f"{said} (phase={phase})" if phase else said
    said = str(state.get("experiment_execution_summary") or "").strip()
    if said:
        return said
    return f"phase={phase}" if phase and phase != "execution" else ""


#: Причины паузы словами, а не ключами состояния: читатель справки — оператор,
#: и `plan_rejected_by_operator` посреди русского абзаца ему ничего не говорит.
_PAUSE_WORDS = {
    "ru": {
        "plan_rejected_by_operator": "план эксперимента отклонён оператором",
        "plan_review_timeout": "обзор плана не подтверждён в отведённое время",
        "result_review_timeout": "результат не подтверждён в отведённое время",
        "max_plan_revisions": "исчерпан лимит переработок плана",
        "inventory_blocker_repeated": "план дважды упёрся в отсутствующий инструмент",
    },
    "en": {
        "plan_rejected_by_operator": "the experiment plan was rejected by the operator",
        "plan_review_timeout": "the plan review was not confirmed in time",
        "result_review_timeout": "the result review was not confirmed in time",
        "max_plan_revisions": "the plan revision budget was exhausted",
        "inventory_blocker_repeated": "the plan twice required an unavailable tool",
    },
}

#: Что делать дальше. «Подтвердите план на карточке обзора и запустите прогон
#: заново» было невыполнимо с двух сторон: карточка к этому моменту закрыта
#: (`disableHitlControls` срабатывает и на тайм-ауте), а повторный прогон того
#: же запроса отклоняет привратник модуля. Работает только новый запрос.
_WHAT_NEXT = {
    "ru": {
        "plan_rejected_by_operator":
            "Вы отклонили план. Чтобы спланировать эксперимент заново, "
            "отправьте новый запрос — с тем, что в отклонённом плане было не так.",
        None:
            "Чтобы продолжить, отправьте новый запрос: этот план подтвердить "
            "уже нельзя, его карточка обзора закрыта.",
    },
    "en": {
        "plan_rejected_by_operator":
            "You rejected the plan. To plan the experiment again, send a new "
            "request saying what was wrong with the rejected one.",
        None:
            "To continue, send a new request: this plan can no longer be "
            "approved, its review card is closed.",
    },
}


def _what_next(lang: str, state: Any) -> str:
    pause = str(state.get("experiment_review_pause_reason") or "").strip()
    if pause == "max_plan_revisions":
        return (
            "Возобновите этот прогон: сохранённый исполнимый план будет повторно "
            "проверен и показан для явного решения без нового планирования."
            if lang == "ru"
            else "Resume this run to revalidate and explicitly review the saved "
                 "executable plan without starting planning again."
        )
    table = _WHAT_NEXT[lang]
    text = table.get(pause) or table[None]
    if state.get("experiment_plan_record_id"):
        text += (
            " Сам план сохранён в графе выполнения."
            if lang == "ru"
            else " The plan itself is retained in the execution graph."
        )
    return text


_INSTEAD_OF_A_REPORT = {
    "ru": (
        "## Отчёт не составлен: эксперимент не выполнялся\n\n"
        "План был подготовлен, но исполнение не состоялось, поэтому писать "
        "научный отчёт не о чем — любой текст здесь был бы описанием работы, "
        "которой не было.\n\n"
        "- **Задач в плане:** {planned}\n"
        "- **Выполнено:** 0\n"
        "- **Свидетельств в графе:** {evidence}\n"
        "- **Причина остановки:** {why}\n\n"
        "{next}"
    ),
    "en": (
        "## No report: the experiment was never executed\n\n"
        "A plan was prepared but execution never happened, so there is nothing "
        "to write a scientific report about — any text here would describe work "
        "that was not done.\n\n"
        "- **Tasks planned:** {planned}\n"
        "- **Carried out:** 0\n"
        "- **Evidence in the graph:** {evidence}\n"
        "- **Why it stopped:** {why}\n\n"
        "{next}"
    ),
}

_UNACCEPTED_PARTIAL_REPORT = {
    "ru": (
        "## Итоговый отчёт отложен: научная работа не принята как завершённая\n\n"
        "Прогон остановлен до принятого итогового решения. Ни литература в графе, "
        "ни отдельный частичный результат сами по себе не подтверждают выполнение "
        "всего запрошенного исследования.\n\n"
        "- **Задач в плане:** {planned}\n"
        "- **Выполнено с результатом:** {done}\n"
        "- **Свидетельств в графе:** {evidence}\n"
        "- **Причина остановки:** {why}\n\n"
        "{next}"
    ),
    "en": (
        "## Final report deferred: scientific work was not accepted as complete\n\n"
        "The run stopped before an accepted final disposition. Literature evidence "
        "or an isolated partial result does not by itself prove that the requested "
        "study was completed.\n\n"
        "- **Tasks planned:** {planned}\n"
        "- **Carried out with a result:** {done}\n"
        "- **Evidence in the graph:** {evidence}\n"
        "- **Why it stopped:** {why}\n\n"
        "{next}"
    ),
}

_PARTIAL_NOTE = {
    "ru": (
        "ВНИМАНИЕ — ПЛАН ВЫПОЛНЕН НЕ ПОЛНОСТЬЮ. Отчёт обязан открываться "
        "предупреждением об этом и не должен описывать результаты задач, "
        "которые не выполнялись.\n"
        "Задач в плане: {planned}. Выполнено: {done}.{why}"
    ),
    "en": (
        "WARNING — THE PLAN WAS NOT CARRIED OUT IN FULL. The report must open "
        "with a statement saying so, and must not describe results of tasks "
        "that never ran.\n"
        "Tasks planned: {planned}. Carried out: {done}.{why}"
    ),
}

#: Причина дописывается только когда она есть — см. `_why`. Прогон, где до части
#: задач просто не дошла очередь, ничем не «останавливался».
_WHY_TAIL = {"ru": " Причина остановки: {why}.", "en": " Why it stopped: {why}."}


def _language(state: Any) -> str:
    try:
        from CoScientist.agents.callbacks.report_language import (
            session_report_language,
        )

        return "ru" if str(session_report_language(state)).lower().startswith("ru") else "en"
    except Exception:  # noqa: BLE001
        return "ru"


def guard_report_without_execution(callback_context: Any) -> types.Content | None:
    """before_agent на агрегаторе: не дать написать отчёт о невыполненной работе.

    Возврат `Content` замыкает агента — тот же приём, которым
    `skip_executor_without_runtime` останавливает исполнителя без утверждённого
    плана. Ошибка внутри самого привратника не должна стоить прогону отчёта,
    поэтому всё обёрнуто: не смогли посчитать — пропускаем.
    """
    try:
        state = callback_context.state
        from CoScientist.execution_control import current_run

        execution_handle = current_run()
        disposition = reconcile_scientific_outcome(
            state,
            current_run_id=execution_handle.run_id if execution_handle else None,
        )
        if (not _module_was_engaged(state)
                and disposition.kind is not DispositionKind.COMPLETED_LIMITED):
            return None

        done, planned = _carried_out(state), _planned(state)
        evidence = _evidence_in_graph(callback_context)
        lang = _language(state)
        why = _why(state, lang)
        typed_completion_required = bool(
            state.get("experiment_module_outcome")
            or state.get("experiment_review_pause_reason")
            or state.get("experiment_plan_review_paused")
            or state.get("_master_active_tasks")
            or state.get("pipeline_scope")
        )

        # Evidence can support a literature section, but cannot prove that a
        # requested experiment or unfinished roadmap completed.  A partial
        # scientific report requires a typed, explicitly accepted disposition.
        blocked_typed_outcome = (
            typed_completion_required
            and disposition.module_engaged
            and not disposition.report_allowed
        )
        if (not blocked_typed_outcome
                and (done or evidence
                     or disposition.kind is DispositionKind.COMPLETED_LIMITED)):
            # Что-то сделано. Отчёт законен, но обязан сказать, чего в нём нет.
            if (planned and done < planned) or disposition.kind is DispositionKind.COMPLETED_LIMITED:
                tail = _WHY_TAIL[lang].format(why=why) if why else ""
                state[UNEXECUTED_NOTE_KEY] = _PARTIAL_NOTE[lang].format(
                    planned=planned, done=done, why=tail)
            else:
                state[UNEXECUTED_NOTE_KEY] = ""
            return None

        # Ничего. Ни результата задачи, ни свидетельства — писать не о чем.
        state[UNEXECUTED_NOTE_KEY] = ""
        logger.warning(
            "REPORT_WITHOUT_EXECUTION_REFUSED planned=%d done=%d evidence=%d why=%s",
            planned, done, evidence, why)
        unknown = "не записано" if lang == "ru" else "not recorded"
        template = (
            _UNACCEPTED_PARTIAL_REPORT[lang]
            if done or evidence else _INSTEAD_OF_A_REPORT[lang]
        )
        return types.Content(
            role="model",
            parts=[types.Part(text=template.format(
                planned=planned or unknown, done=done, evidence=evidence,
                why=why or unknown,
                next=_what_next(lang, state)))],
        )
    except Exception as exc:  # noqa: BLE001 — привратник не ломает прогон
        logger.warning("проверка отчёта на пустое исполнение не выполнена: %s", exc)
        return None


__all__ = ["UNEXECUTED_NOTE_KEY", "guard_report_without_execution"]
