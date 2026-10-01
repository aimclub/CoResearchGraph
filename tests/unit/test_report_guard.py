"""Отчёт не пишется о работе, которой не было, и окно обзора слушает оператора.

Две стороны одного дефекта, найденного разбором сохранённых сессий. Модуль
экспериментов умеет останавливаться тихо: обзор плана не подтверждён — рантайм не
создан — исполнитель пропущен. А `ResultAggregatorAgent` стоит в `pipeline.post`
и срабатывает независимо, поэтому прогон выдавал полный научный отчёт, не выполнив
ни одной задачи. По записям так закончились три сессии (`0a31978b`, `196dd0f7`,
`236c0325`), и шире — 5 из 8 «завершённых» прогонов профиля experiments не имеют
ни одного узла ExperimentTask.

Run from the repo root:  pytest tests/unit/test_report_guard.py -q
"""
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.agents.callbacks.report_guard import (  # noqa: E402
    UNEXECUTED_NOTE_KEY,
    guard_report_without_execution,
)


def _run(state):
    """Возвращает (замыкание агента или None, врезка в промпт)."""
    blocked = guard_report_without_execution(SimpleNamespace(state=state))
    text = blocked.parts[0].text if blocked is not None else None
    return text, state.get(UNEXECUTED_NOTE_KEY)


def _skipped(tasks=3):
    """Состояние ровно такое, каким его оставляет `skip_executor_without_runtime`."""
    return {
        "experiment_plan_view": {"tasks": list(range(tasks))},
        "experiment_review_pause_reason": "plan_review_timeout",
        "experiment_execution_summary": (
            "Experiment execution skipped: no approved experiment runtime is "
            "active (plan_review_timeout (phase=awaiting_review))."),
    }


# ── что привратник не трогает ────────────────────────────────────────────────

def test_a_study_that_never_used_the_module_reports_as_before():
    """Чисто литературное исследование модуль экспериментов не запускает, и
    отчёт по нему ничем не скомпрометирован."""
    assert _run({}) == (None, None)
    assert _run({"final_report": "…", "research_context": "…"}) == (None, None)


def test_a_plan_carried_out_in_full_reports_without_a_warning():
    text, note = _run({
        "experiment_runtime": {"task_order": ["EXP-1", "EXP-2"], "phase": "completed"},
        "experiment_task_results": [{"status": "success"}, {"status": "partial"}],
    })
    assert text is None, "отчёт по выполненному плану не должен блокироваться"
    assert note == "", "и предупреждать не о чем"


# ── что он останавливает ─────────────────────────────────────────────────────

def test_no_report_is_written_over_a_plan_that_never_ran():
    """Главный случай. Вместо отчёта — справка, и она называет причину."""
    text, _ = _run(_skipped())
    assert text is not None, "агрегатор должен быть замкнут"
    assert "не выполнялся" in text
    assert "обзор плана не подтверждён" in text, (
        "причина берётся из состояния, не выдумывается — и словами, а не ключом")
    assert "plan_review_timeout" not in text, "ключ состояния читателю ничего не говорит"
    assert "**Задач в плане:** 3" in text
    # И сказано, что делать дальше: справка без выхода — это тупик.
    assert "новый запрос" in text.lower(), (
        "«подтвердите план на карточке» было невыполнимо: карточка уже закрыта")


def test_the_refusal_says_plainly_that_nothing_was_measured():
    """Справка не должна читаться как отчёт: ни выводов, ни результатов."""
    text, _ = _run(_skipped())
    assert "**Выполнено:** 0" in text
    assert "**Свидетельств в графе:** 0" in text


def test_a_run_with_a_plan_and_no_results_at_all_is_stopped_even_without_a_reason():
    """Причина могла не записаться — это не повод писать отчёт."""
    text, _ = _run({"experiment_runtime": {"task_order": ["EXP-1"], "phase": "awaiting_review"}})
    assert text is not None


# ── что он пропускает, но с оговоркой ────────────────────────────────────────

def test_a_partly_executed_plan_reports_with_a_mandatory_warning():
    text, note = _run({
        "experiment_runtime": {"task_order": ["a", "b", "c"], "phase": "execution"},
        "experiment_task_results": [{"status": "success"}],
    })
    assert text is None, "то, что сделано, должно быть описано"
    assert "НЕ ПОЛНОСТЬЮ" in note
    assert "Задач в плане: 3. Выполнено: 1." in note


def test_a_partial_run_that_never_stopped_is_not_told_it_stopped():
    """Фаза «execution» ничего не останавливала. Назвать её причиной остановки
    значило бы придумать событие — ровно тот сорт неправды, против которого
    этот привратник и написан."""
    _, note = _run({
        "experiment_runtime": {"task_order": ["a", "b"], "phase": "execution"},
        "experiment_task_results": [{"status": "success"}],
    })
    assert "Причина остановки" not in note


def test_a_partial_run_that_did_stop_says_why():
    text, note = _run({
        "experiment_runtime": {"task_order": ["a", "b"], "phase": "awaiting_review"},
        "experiment_review_pause_reason": "plan_review_timeout",
        "experiment_task_results": [{"status": "partial"}],
    })
    assert text is not None
    assert "обзор плана не подтверждён" in text
    assert note == ""


def test_a_failed_task_is_not_carried_out_work(tmp_path, monkeypatch):
    """`failed` — это не результат, о котором есть что написать."""
    monkeypatch.setenv("RESEARCH_GRAPH__DIR", str(tmp_path))
    text, _ = _run({
        "experiment_runtime": {"task_order": ["EXP-1"], "phase": "completed"},
        "experiment_task_results": [{"status": "failure"}],
    })
    assert text is not None


# ── страховка: литературная часть не должна теряться ─────────────────────────

def test_stray_evidence_does_not_complete_a_paused_experiment(monkeypatch):
    """Literature evidence cannot stand in for requested experiment results."""
    import CoScientist.agents.callbacks.report_guard as mod

    monkeypatch.setattr(mod, "_evidence_in_graph", lambda _ctx: 4)
    text, note = _run(_skipped(tasks=2))
    assert text is not None
    assert "**Свидетельств в графе:** 4" in text
    assert note == ""


def test_an_unreadable_graph_never_costs_the_run_its_report(monkeypatch):
    """Привратник не имеет права ронять прогон: не смог посчитать — пропускает."""
    import CoScientist.agents.callbacks.report_guard as mod

    def explode(_ctx):
        raise RuntimeError("граф недоступен")

    monkeypatch.setattr(mod, "_carried_out", explode)
    assert guard_report_without_execution(SimpleNamespace(state=_skipped())) is None


# ── подключение ──────────────────────────────────────────────────────────────

def test_the_guard_runs_first_in_both_profiles():
    """Колбэк, возвращающий ответ, останавливает цепочку — значит привратник
    должен стоять перед дайджестом графа и вопросом про НИР, чтобы не тратить
    их впустую. И `user_links` обязан остаться последним."""
    from CoScientist.assembly.schema import load_config, resolve_config_path

    for profile in ("system", "experiments"):
        hooks = load_config(resolve_config_path(profile)).agent(
            "ResultAggregatorAgent").callbacks.before_agent
        assert hooks[0] == "guard_report_without_execution", profile
        assert hooks[-1] == "user_links", profile


@pytest.mark.parametrize("prompt_name", ["result_aggregator", "experiment_result_aggregator"])
def test_the_prompt_has_a_place_for_the_warning(prompt_name):
    """Врезка без слота в промпте — это состояние, которого никто не читает."""
    from CoScientist.assembly import load_config
    from CoScientist.assembly.prompting import PromptContext
    from CoScientist.assembly.registry import REGISTRY

    cfg = load_config()
    text = REGISTRY.prompt(prompt_name)(
        PromptContext(config=cfg.agent("ResultAggregatorAgent"), system=cfg))
    assert "{" + UNEXECUTED_NOTE_KEY + "?}" in text
