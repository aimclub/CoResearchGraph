"""Вводная о режиме, которую генератор гипотез получает перед работой.

Агента просят судить, «насколько гипотеза проверяема имеющимися инструментами»,
но инвентаря он не видит: тулов поиска у него нет, а срез графа показывает узлы
Tool одной строкой. К этому моменту инвентарь обычно уже собран — планировщик и
ретривер MCP отработали раньше, — и здесь проверяется, что он доходит до промпта
и что из него следует.

Run from the repo root:  pytest tests/unit/test_hypothesis_brief.py -q
"""
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

from CoScientist.agents.callbacks.hypothesis_brief import (  # noqa: E402
    BRIEF_STATE_KEY,
    brief_hypotheses_regime,
)


def _ctx(state):
    return SimpleNamespace(state=state)


def _brief(state):
    brief_hypotheses_regime(_ctx(state))
    return state[BRIEF_STATE_KEY]


def test_with_tools_in_hand_the_claim_is_aimed_at_them():
    """Задача, которая закрывается готовыми инструментами, требует гипотезы,
    которую ИМЕННО ОНИ и подтверждают — иначе прогон измеряет не то."""
    said = _brief({"accumulated_tools": [
        {"tool": "predict_ld50", "server_id": "tox"},
        {"tool": "chemical_space_clustering", "server_id": "chem"},
    ]})
    assert "predict_ld50" in said and "chemical_space_clustering" in said
    assert "2 ready tools" in said
    # И что с ними делать: измеримый исход, который они дают напрямую.
    assert "ConfirmationCriteria must threshold exactly that output" in said


def test_when_the_search_found_nothing_the_effort_goes_into_judging():
    """Проверка без готовых инструментов стоит перебора и разработки — значит
    цена ошибки в выборе гипотезы выше, и выбор надо обосновать."""
    said = _brief({"retrieval_queries": ["ld50 prediction"], "accumulated_tools": []})
    assert "NO READY TOOL" in said
    assert "JUDGING the candidates" in said
    # Ни одного выдуманного инструмента в вводной про их отсутствие.
    assert "predict_ld50" not in said


def test_an_inventory_nobody_took_is_not_reported_as_an_empty_one():
    """«Искали и не нашли» и «ещё не искали» — разные указания. Агент гипотез
    может отработать раньше ретривера, и объявить тогда, что инструментов нет,
    значит сказать неправду о том, чего никто не проверял."""
    said = _brief({})
    assert "HAS NOT BEEN TAKEN YET" in said
    assert "NO READY TOOL" not in said
    # Из этого следует конкретное требование: назвать измерение.
    assert "name the MEASUREMENT" in said


def test_candidates_the_reranker_threw_out_are_not_offered_as_measuring_devices():
    """`accumulated_tools` survives the rerank: it keeps listing the candidates
    that were just judged and rejected. Read as «this is what you can measure
    with», they aim the hypothesis at a tool nobody is going to run."""
    said = _brief({
        "accumulated_tools": [{"tool": "unrelated_docking"}],
        "executor_tool_match": {"matched": False, "reason": "scored",
                                "kept": 0, "candidates": 1},
    })
    assert "NO READY TOOL" in said
    assert "unrelated_docking" not in said

    # A verdict that judged nothing (a parse failure) is not a rejection.
    said = _brief({
        "accumulated_tools": [{"tool": "predict_ld50"}],
        "executor_tool_match": {"matched": False, "reason": "parse_failed"},
    })
    assert "predict_ld50" in said


def test_the_brief_does_not_claim_the_inventory_was_taken_for_this_task():
    """Инвентарь живёт в сессии и мог быть собран под предыдущий запрос.
    Указать на него честно можно, выдать за поиск под эту задачу — нет."""
    said = _brief({"accumulated_tools": [{"tool": "predict_ld50"}]})
    assert "inventory of this session" in said
    assert "for this task" not in said


def test_the_inventory_is_reported_not_invented():
    """Вводная пересказывает то, что собрали, и не добавляет от себя: иначе
    гипотеза будет нацелена на средство, которого нет."""
    said = _brief({"filtered_tools": [{"tool": "name2smiles"}],
                   "accumulated_tools": [{"tool": "name2smiles"}]})
    # Один и тот же инструмент из двух ключей — это один инструмент.
    assert "1 ready tools" in said
    assert said.count("name2smiles") == 1


def test_a_broken_inventory_costs_the_brief_not_the_run():
    """Генерация гипотез не должна падать из-за того, что не собралась
    вводная: без неё агент работает как раньше."""
    class Exploding(dict):
        def get(self, *_args, **_kwargs):
            raise RuntimeError("состояние недоступно")

    state = Exploding()
    brief_hypotheses_regime(_ctx(state))   # не поднимает исключение
    said = dict.get(state, BRIEF_STATE_KEY)
    # Что бы ни случилось, вводная не утверждает, что инструментов нет:
    # непрочитанный инвентарь — это незнание, а не пустота.
    assert said is not None
    assert "NO READY TOOL" not in (said or "")


def test_the_prompt_has_a_place_for_it():
    """Подстановка `{hypothesis_brief?}` — единственный канал, который переживёт
    затравку модуля экспериментов: она затирает contents, но не инструкцию."""
    from CoScientist.assembly import load_config
    from CoScientist.assembly.prompting import PromptContext
    from CoScientist.assembly.registry import REGISTRY

    cfg = load_config()
    prompt = REGISTRY.prompt("hypotheses")(
        PromptContext(config=cfg.agent("HypothesesAgent"), system=cfg))
    assert "{hypothesis_brief?}" in prompt


def test_the_callback_is_wired_to_the_agent_in_both_profiles():
    """Вводная бесполезна, если колбэк не навешен: в профиле экспериментов
    у агента свой список before_agent, и он перекрывает основной."""
    from CoScientist.assembly.schema import load_config, resolve_config_path

    for profile in ("system", "experiments"):
        cfg = load_config(resolve_config_path(profile))
        hooks = cfg.agent("HypothesesAgent").callbacks.before_agent
        assert "brief_hypotheses_regime" in hooks, profile
