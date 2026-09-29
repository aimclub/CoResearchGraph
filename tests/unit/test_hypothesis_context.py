"""Контекст, из которого генератор гипотез строит проверяемую гипотезу.

Гипотезу просят сделать «проверяемой имеющимися инструментами», но до неё
доезжали только ИМЕНА инструментов, а план, который их видел, приходил без
описаний: `clean_tasks_for_agent` снимает описание со всех задач, кроме
назначенных читателю, — и шаг исполнения выглядел одним заголовком.

Чего это стоило, видно на живом прогоне по борщевику: гипотеза требовала
сравнить долю токсичных соединений МЕЖДУ кластерами через инструмент, который
не принимает аргументов и всегда возвращает один кластер. Сравнение не провели
ни разу, а на этапе выводов формулировку тихо ослабили до той, которую
измерить было можно.

Здесь проверяется, что к моменту формулировки агент знает: чем инструмент
меряет, что он принимает на вход, и каким планом его гипотезу будут проверять.

Run from the repo root:  pytest tests/unit/test_hypothesis_context.py -q
"""
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

from CoScientist.agents.callbacks.hypothesis_brief import (  # noqa: E402
    BRIEF_STATE_KEY,
    brief_hypotheses_regime,
)


def _brief(state):
    brief_hypotheses_regime(SimpleNamespace(state=state))
    return state[BRIEF_STATE_KEY]


def _tox_inventory():
    """Инвентарь того прогона, дословно по ответу retrieve_tools."""
    return [
        {"tool": "predict_ld50", "server_id": "bfc62a28",
         "description": "Predict acute LD50 (mouse) by cluster and route.",
         "input_schema": {"properties": {}, "type": "object"}},
        {"tool": "predict_general_toxicity", "server_id": "bfc62a28",
         "description": "Hepatotoxicity / DILI / cardiotoxicity for cluster E.",
         "input_schema": {"properties": {}, "type": "object"}},
        {"tool": "estimate_synthesis_cost", "server_id": "bfc62a28",
         "description": "Estimate synthesis cost (USD/g).",
         "input_schema": {"properties": {"name_or_smiles": {"type": "string"}},
                          "required": ["name_or_smiles"], "type": "object"}},
    ]


# ── Чем инструмент меряет, а не только как называется ────────────────────────


def test_the_brief_says_what_each_tool_measures_not_just_its_name():
    """Критерий подтверждения обязан пороговать ВЫХОД инструмента — значит из
    вводной должно быть видно, что этот инструмент вообще выдаёт."""
    said = _brief({"accumulated_tools": _tox_inventory()})
    assert "Predict acute LD50 (mouse) by cluster and route." in said
    assert "Estimate synthesis cost (USD/g)." in said


def test_arguments_are_shown_because_they_bound_what_can_be_asked():
    """Обязательные аргументы — со звёздочкой, отсутствие аргументов — пустыми
    скобками: по сигнатуре видно, можно ли навести инструмент на срез."""
    said = _brief({"accumulated_tools": _tox_inventory()})
    assert "`bfc62a28:predict_ld50()`" in said
    assert "`bfc62a28:estimate_synthesis_cost(name_or_smiles*)`" in said


def test_a_tool_without_arguments_warns_against_a_comparison_it_cannot_make():
    """Ровно это сломало гипотезу живого прогона: сравнение кластеров через
    инструмент, который аргументов не принимает и отдаёт один кластер."""
    said = _brief({"accumulated_tools": _tox_inventory()})
    assert "takes NO arguments" in said
    assert "cannot be aimed at a subset" in said
    assert "SPLITS by that grouping" in said


def test_a_parametrised_inventory_is_not_warned_about():
    """Предупреждение стоит денег во внимании — там, где все инструменты
    параметризуются, предупреждать не о чем."""
    said = _brief({"accumulated_tools": [
        {"tool": "predict_molecule_profile",
         "description": "Full profile for any molecule.",
         "input_schema": {"properties": {"name_or_smiles": {"type": "string"}}}},
    ]})
    assert "takes NO arguments" not in said


def test_the_brief_asks_for_the_claim_that_leaves_the_most_behind():
    """Артефакты прогона и есть отчёт: при равной остроте выигрывает гипотеза,
    проверка которой задействует больше инвентаря."""
    said = _brief({"accumulated_tools": _tox_inventory()})
    assert "LEAVES THE MOST BEHIND" in said
    assert "figures, tables and files" in said


# ── План, который видел инструменты, доходит до гипотезы ─────────────────────


def _plan_state(**extra):
    state = {"_master_active_tasks": [
        {"id": "TASK-1", "assignee": "HypothesesAgent",
         "title": "Сформулировать гипотезу", "description": "по Q1, EB1, T1"},
        {"id": "TASK-2", "assignee": "TaskExecutorAgent",
         "title": "Профиль через MCP heracleum-tox",
         "description": "chemical_space_clustering(n_clusters=5); predict_ld50; "
                        "predict_general_toxicity для самого токсичного кластера"},
    ]}
    state.update(extra)
    return state


def test_the_plan_reaches_the_generator_with_the_steps_it_does_not_own():
    """Описание ЧУЖОЙ задачи — то самое, что теряется в почищенном срезе, и
    именно оно называет инструменты, которыми гипотезу будут проверять."""
    said = _brief(_plan_state(accumulated_tools=_tox_inventory()))
    assert "AVAILABLE MEANS OF TESTING THE STUDY" in said
    assert "TASK-2" in said
    assert "chemical_space_clustering(n_clusters=5)" in said
    assert "never a claim about what this plan will show" in said
    assert "Use these steps as the VerificationMethod" in said


def test_the_plan_is_a_method_not_the_subject_of_the_hypothesis():
    """Набор доступных вызовов обосновывает проверку, но научное утверждение
    относится к метаболитам и их эффекту, а не к ожидаемому выводу конвейера."""
    said = _brief(_plan_state(accumulated_tools=_tox_inventory()))
    assert "scientific claim about the material, system, mechanism or effect" in said
    assert "Aim the hypothesis at THIS pipeline" not in said
    assert "outcome should be what these steps produce" not in said


def test_the_plan_is_shown_even_when_no_tool_was_found():
    """Без готовых инструментов план тем более нужен: иначе агент придумает
    метод, которого никто не запустит."""
    said = _brief(_plan_state(retrieval_queries=["ld50"], accumulated_tools=[]))
    assert "NO READY TOOL" in said
    assert "AVAILABLE MEANS OF TESTING THE STUDY" in said
    assert "TASK-2" in said


def test_without_a_plan_the_brief_stays_silent_about_one():
    """Пустая секция хуже отсутствующей: она утверждает, что план есть."""
    said = _brief({"accumulated_tools": _tox_inventory()})
    assert "AVAILABLE MEANS OF TESTING THE STUDY" not in said


def test_a_broken_plan_costs_the_section_not_the_run():
    """Мусор в мастер-списке не должен ронять генерацию гипотез."""
    said = _brief({"accumulated_tools": _tox_inventory(),
                   "_master_active_tasks": ["не словарь", {}, None]})
    assert "WHAT THIS RUN CAN ALREADY MEASURE" in said


# ── Цепочка гипотез вместо списка независимых идей ───────────────────────────


def _hypotheses_prompt(max_active: int):
    from CoScientist.assembly import load_config
    from CoScientist.assembly.prompting import PromptContext
    from CoScientist.assembly.registry import REGISTRY
    from CoScientist.config import get_settings

    web = get_settings().web
    before = web.max_active_hypotheses
    web.max_active_hypotheses = max_active
    try:
        cfg = load_config()
        return REGISTRY.prompt("hypotheses")(
            PromptContext(config=cfg.agent("HypothesesAgent"), system=cfg))
    finally:
        web.max_active_hypotheses = before


def test_a_later_hypothesis_is_written_for_the_world_where_the_first_failed():
    """Вторая гипотеза — не независимая идея и не следующий шаг конвейера, а
    продолжение линии на случай, когда первая опровергнута."""
    prompt = _hypotheses_prompt(2)
    assert "CHAIN, NOT A LIST" in prompt
    assert "WRITTEN AGAINST THE REFUTATION" in prompt
    assert "previous one was REFUTED" in prompt


def test_a_later_hypothesis_must_salvage_the_run_rather_than_repeat_it():
    """Измерения, которые первая проверка уже дала, идут второй на вход —
    иначе это не продолжение, а второе исследование."""
    prompt = _hypotheses_prompt(2)
    assert "IT REUSES THE RUN" in prompt
    assert "PLUS A DELTA" in prompt
    assert "not a continuation but a second study" in prompt


def test_the_ban_on_committing_a_method_step_survives_the_rewrite():
    """Прежний запрет никуда не делся: шаг метода — не гипотеза."""
    prompt = _hypotheses_prompt(2)
    assert "these are steps of H1's VerificationMethod" in prompt


def test_one_hypothesis_run_is_not_told_about_chains():
    """При потолке в одну гипотезу правило про цепочку читается как
    приглашение выйти за потолок."""
    assert "CHAIN, NOT A LIST" not in _hypotheses_prompt(1)


def test_every_claim_must_be_equipped_from_the_inventory():
    """Критерий пороговает то, что инструмент возвращает, — иначе прогон
    переинтерпретирует гипотезу вместо того, чтобы её проверить."""
    prompt = _hypotheses_prompt(1)
    assert "IF THE INVENTORY BELOW LISTS TOOLS, equip each claim" in prompt
    assert "ConfirmationCriteria against a value one of them RETURNS" in prompt
    # Граница проверяемости названа прямо: инструмент без аргументов не наводится
    # на подгруппу, значит сравнения подгрупп через него не бывает.
    assert "a tool that takes no arguments" in prompt


def test_the_prompt_requires_a_claim_about_the_study_not_about_its_pipeline():
    """Даже при готовых средствах исследования постановка начинается с
    предмета и эффекта; инструменты попадают только в метод проверки."""
    prompt = _hypotheses_prompt(1)
    assert "Available methods do not define the hypothesis" in prompt
    assert "scientific question" in prompt
    assert "automated pipeline will show" not in prompt
    assert "EXPECTED OUTCOME of the pipeline" not in prompt


# ── Стенд ────────────────────────────────────────────────────────────────────


def test_the_bench_runs_the_study_the_hypothesis_steers():
    """Стенд меняет только генератор. Модуль экспериментов остаётся: гипотезу
    судят по тому, какие эксперименты она вызвала и какой вердикт они дали, —
    без исполнения это видно только на бумаге."""
    from CoScientist.assembly import build_system, load_config
    from CoScientist.assembly.schema import resolve_config_path

    cfg = load_config(resolve_config_path("hypotheses"))
    assert cfg.agent("ExperimentModuleAgent").is_enabled()
    assert cfg.agent("HypothesesAgent").reasoning == "high"

    system = build_system(cfg)
    roster = {getattr(t, "name", "") for t in system.agent("OrchestratorAgent").tools}
    assert {"HypothesesAgent", "ExperimentModuleAgent"} <= roster
    assert "{hypothesis_brief?}" in system.agent("HypothesesAgent").instruction


def test_the_bench_keeps_the_brief_callback_last_of_the_context_hooks():
    """Вводная читает инвентарь и план, которые обновляют предыдущие хуки, —
    поэтому она должна идти после них."""
    from CoScientist.assembly.schema import load_config, resolve_config_path

    hooks = load_config(
        resolve_config_path("hypotheses")).agent("HypothesesAgent").callbacks.before_agent
    assert "brief_hypotheses_regime" in hooks
    assert hooks.index("brief_hypotheses_regime") > hooks.index("inject_research_context")
