"""The frame around an agent's words speaks the session's language too.

The agent writes its goal, its steps and its findings in the language the
session chose. The labels around them — "Goal", "Done when", "Steps" — were
English whatever the session said, so a Russian study opened its work order on
"Work Order: HypothesesAgent" and read "Goal: Сформировать и записать…". It was
invisible while those renderings only went to a console; the chat now writes
them into documents a person opens.
"""
from __future__ import annotations

import pytest

from CoScientist.agents.callbacks.report_language import (
    DEFAULT_REPORT_LANGUAGE,
    session_report_language,
)
from CoScientist.hitl.work_order import (
    Artifact,
    Assumption,
    Finding,
    WorkOrder,
    WorkReport,
    WorkStep,
    render_work_order,
    render_work_report,
)


def _order() -> WorkOrder:
    return WorkOrder(
        agent="HypothesesAgent",
        goal="Сформировать единственную проверяемую гипотезу H1.",
        done_criteria="В графе созданы узлы H1 и критерии приёмки.",
        assumptions=[Assumption(id="A1", text="Метаболиты содержат фуранокумарины.")],
        steps=[WorkStep(id="S1", title="Сформулировать H1",
                        expected_outcome="H1 записана как активная", tools=["research_commit"])],
        planned_tools=["research_commit"],
        expected_outcome="Активная H1 про фуранокумариновый кластер.",
        fallback="Сформулировать критерии текстом и уведомить оркестратора.",
    )


def _reported() -> WorkOrder:
    order = _order()
    order.report = WorkReport(
        summary="Гипотеза записана, альтернативы отложены.",
        done_verdict="met",
        actual_outcome="H1 активна, две альтернативы отложены.",
        findings=[Finding(id="F1", text="Фуранокумарины образуют отдельный кластер.",
                          evidence="E1", confidence="high")],
        artifacts=[Artifact(kind="dataset", ref="s3://b/h1.json", description="Узлы гипотезы")],
    )
    order.tool_calls = {"research_commit": 2}
    return order


# ── where the language comes from ───────────────────────────────────────────
def test_a_session_that_said_nothing_gets_the_default():
    assert session_report_language(None) == DEFAULT_REPORT_LANGUAGE
    assert session_report_language({}) == DEFAULT_REPORT_LANGUAGE
    assert session_report_language({"report_language": ""}) == DEFAULT_REPORT_LANGUAGE
    # The default is Russian, which is what made the English frame wrong.
    assert DEFAULT_REPORT_LANGUAGE == "ru"


def test_the_language_is_read_off_a_plain_state_or_a_context():
    assert session_report_language({"report_language": "en"}) == "en"

    class _Ctx:
        state = {"report_language": "en"}

    assert session_report_language(_Ctx()) == "en"


def test_an_unreadable_state_does_not_raise():
    class _Hostile:
        @property
        def state(self):
            raise RuntimeError("no")

    assert session_report_language(_Hostile()) == DEFAULT_REPORT_LANGUAGE


# ── the work order ──────────────────────────────────────────────────────────
def test_a_russian_work_order_has_no_english_frame():
    text = render_work_order(_order(), "ru")
    assert "План работы агента" in text
    assert "## Цель" in text and "## Шаги" in text
    assert "## Условия и ограничения" in text
    for english in ("Work Order:", "Goal:", "Done when:", "Assumptions:", "Steps:",
                    "Expected outcome:", "If it fails:"):
        assert english not in text, english


def test_the_agent_s_own_words_are_never_translated():
    text = render_work_order(_order(), "ru")
    assert "Сформировать единственную проверяемую гипотезу H1." in text
    assert "Метаболиты содержат фуранокумарины." in text
    # Identifiers stay identifiers.
    assert "**A1**" in text and "**S1**" in text and "`research_commit`" in text


def test_english_is_still_available_for_a_session_that_asked_for_it():
    text = render_work_order(_order(), "en")
    assert "# Work Order — HypothesesAgent" in text
    assert "## Goal" in text and "## Steps" in text
    assert "План работы" not in text


def test_the_order_is_markdown_now():
    """It is written into a document and rendered in a panel."""
    text = render_work_order(_order(), "ru")
    assert text.startswith("# ")
    assert "\n## " in text


# ── the work report ─────────────────────────────────────────────────────────
def test_a_russian_work_report_has_no_english_frame():
    text = render_work_report(_reported(), "ru")
    assert "Отчёт о работе" in text
    assert "## Итог" in text and "## Находки" in text and "## Артефакты" in text
    for english in ("Work Report:", "Summary:", "Findings:", "Artifacts:",
                    "Tool calls:", "Expected:", "Actual:"):
        assert english not in text, english


def test_the_outcome_table_survives_a_pipe_in_the_text():
    order = _reported()
    order.expected_outcome = "A | B"
    text = render_work_report(order, "ru")
    row = next(line for line in text.splitlines() if line.startswith("| Ожидалось"))
    # The pipe inside the value is escaped, so it is text and not a new cell.
    assert r"A \| B" in row
    assert row.count("|") - row.count(r"\|") == 3


def test_an_empty_report_still_renders():
    order = _order()
    order.report = WorkReport()
    for lang in ("ru", "en"):
        assert render_work_report(order, lang).startswith("# ")


# ── the experiment plan ─────────────────────────────────────────────────────
@pytest.fixture()
def plan():
    from CoScientist.experiments.schemas import ExperimentPlan

    return ExperimentPlan.model_validate({
        "schema_version": "experiment-plan/1.0", "created_at": "2026-09-23T12:00:00Z",
        "plan_id": "PLAN-1", "experiment_run_id": "EXRUN-1", "revision": 2,
        "goal": "Построить токсикологический профиль метаболитов.",
        "source_request": "профиль токсичности",
        "methods": ["ECFP4", "Butina"],
        "hypotheses": [{"hypothesis_id": "H1", "statement": "Фуранокумарины токсичнее."}],
        "risks": ["Литературная выборка может быть неполной."],
        "tasks": [{
            "id": "EXP-1", "name": "Кластеризация метаболитов", "route": "coder",
            "description": "Кластеризовать по структурному сходству.",
            "est_duration_min": 30,
            "design": {"hypothesis_ref": "H1",
                       "experiment_question": "Какие кластеры образуют метаболиты?",
                       "dataset": {"name": "Перечень метаболитов"}},
        }],
    })


def test_a_russian_experiment_plan_has_no_english_frame(plan):
    from CoScientist.experiments.review import render_experiment_plan

    text = render_experiment_plan(plan, "ru")
    assert text.startswith("# План эксперимента · ревизия 2")
    assert "## Матрица плана" in text and "## Риски" in text
    assert "| Задача | Гипотеза |" in text
    for english in ("# Experiment plan", "Goal:", "Methods:", "Total duration:",
                    "## Design matrix", "| Task | Hypothesis |", "## Risks",
                    "Success criteria:", "Expected artifacts:", "Warnings:"):
        assert english not in text, english


def test_the_plan_keeps_its_identifiers_and_its_own_words(plan):
    from CoScientist.experiments.review import render_experiment_plan

    text = render_experiment_plan(plan, "ru")
    assert "Построить токсикологический профиль метаболитов." in text
    assert "`H1`" in text and "EXP-1" in text and "`coder`" in text


def test_the_plan_still_renders_in_english(plan):
    from CoScientist.experiments.review import render_experiment_plan

    text = render_experiment_plan(plan, "en")
    assert text.startswith("# Experiment plan · revision 2")
    assert "## Design matrix" in text and "## Risks" in text
    assert "План" not in text


def test_an_unknown_language_falls_back_rather_than_raising(plan):
    from CoScientist.experiments.review import render_experiment_plan

    text = render_experiment_plan(plan, "klingon")
    assert text.startswith(f"# {'План эксперимента' if DEFAULT_REPORT_LANGUAGE == 'ru' else 'Experiment plan'}")
    assert render_work_order(_order(), None).startswith("# ")


# ── the results ─────────────────────────────────────────────────────────────
def test_the_results_read_their_language_off_the_state_they_are_given():
    from CoScientist.experiments.review import render_experiment_results

    state = {"experiment_task_results": [], "report_language": "ru"}
    text = render_experiment_results(state)
    assert text.startswith("# Результаты эксперимента")
    assert "# Experiment results" not in text

    state["report_language"] = "en"
    assert render_experiment_results(state).startswith("# Experiment results")
