"""A revised plan must revise the plan column, not grow it.

Read off session_d3ce3a45bdb24272b28efd3f976ec16b: the operator asked the
planner to change the tasks, the planner returned six steps again with two of
them reworded, and the graph ended up with eight —

    PS3  «Кластеризация метаболитов по структурному сходству + дендрограмма»   не начат
    PS7  «Поиск метаболитов (search_similar) + кластеризация … + дендрограмма» не начат
    PS4  «Прогноз LD50 (мышь, все пути введения) + домен применимости …»       не начат
    PS8  «LD50 (мышь, все пути) + домен применимости + тепловая карта …»       не начат

Two cards for each piece of work, the obsolete one still reading as work
ahead. The mirror recognised a step by its title alone, so rewording one made
a stranger of it; nothing ever retired what the plan had dropped.

Run from the repo root:  pytest tests/unit/experiment/test_replan_mirror.py -q
"""
from __future__ import annotations

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.agents.callbacks.tool_callbacks import (  # noqa: E402
    sync_plan_to_research_graph,
)
from CoScientist.graph.research.store import ResearchGraphStore  # noqa: E402

_EM = "ExperimentModuleAgent"

#: The plan of that session as `create_plan` registered it, titles verbatim.
PLAN_1 = [
    {"id": "TASK-1", "assignee": "HypothesesAgent", "status": "TODO",
     "title": "Сформулировать проверяемую гипотезу конвейера",
     "description": "Сформулировать одну проверяемую гипотезу о том, что "
                    "покажет in silico конвейер профилирования метаболитов"},
    {"id": "TASK-2", "assignee": "ResearchAgent", "status": "TODO",
     "title": "Собрать литературные данные о метаболитах и их SMILES",
     "description": "Собрать структуры метаболитов Heracleum sosnowskyi и их "
                    "SMILES из литературы"},
    {"id": "TASK-3", "assignee": _EM, "status": "TODO",
     "title": "Кластеризация метаболитов по структурному сходству + дендрограмма",
     "description": "Кластеризовать метаболиты по структурному сходству ECFP4 "
                    "и построить дендрограмму"},
    {"id": "TASK-4", "assignee": _EM, "status": "TODO",
     "title": "Прогноз LD50 (мышь, все пути введения) + домен применимости + "
              "тепловая карта",
     "description": "Предсказать LD50 для мыши по всем путям введения, оценить "
                    "домен применимости, построить тепловую карту"},
]

#: The revision. TASK-3 and TASK-4 are the same work, reworded; the rest is
#: untouched.
PLAN_2 = [
    dict(PLAN_1[0]),
    dict(PLAN_1[1]),
    {"id": "TASK-3", "assignee": _EM, "status": "TODO",
     "title": "Поиск метаболитов (search_similar) + кластеризация "
              "(chemical_space_clustering) + дендрограмма",
     "description": "Найти метаболиты через search_similar, кластеризовать "
                    "через chemical_space_clustering, построить дендрограмму"},
    {"id": "TASK-4", "assignee": _EM, "status": "TODO",
     "title": "LD50 (мышь, все пути) + домен применимости + тепловая карта "
              "токсичности",
     "description": "Предсказать LD50 для мыши по всем путям, оценить домен "
                    "применимости, построить тепловую карту токсичности"},
]


@pytest.fixture
def graph(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    store.ensure_root("Токсикологическое профилирование метаболитов борщевика")
    return store


def _column(graph):
    """step id -> (status, title), in the order the graph holds them."""
    return {n["id"]: (n["status"], (n["attrs"] or {}).get("title", ""))
            for n in graph.full()["nodes"] if n["type"] == "PlanStep"}


def _card_of(graph, step_id):
    """The attrs of one step as the graph holds them."""
    return next(n["attrs"] or {} for n in graph.full()["nodes"]
                if n["id"] == step_id)


def _mirror(graph, plan, state):
    state["_master_active_tasks"] = [dict(t) for t in plan]
    sync_plan_to_research_graph(state["_master_active_tasks"], graph, state,
                               "борщевик")
    return _column(graph)


# ── the live case ───────────────────────────────────────────────────────────

def test_a_reworded_step_keeps_its_card(graph):
    state: dict = {}
    first = _mirror(graph, PLAN_1, state)
    assert len(first) == 4, first

    second = _mirror(graph, PLAN_2, state)
    assert len(second) == 4, second                      # eight in the live run
    assert set(second) == set(first)                     # the same four nodes
    # And the card now reads what the plan says.
    assert second["PS3"][1] == PLAN_2[2]["title"]
    assert second["PS4"][1] == PLAN_2[3]["title"]
    assert all(status == "todo" for status, _ in second.values()), second


def test_mirroring_the_same_plan_twice_changes_nothing(graph):
    state: dict = {}
    once = _mirror(graph, PLAN_1, state)
    assert _mirror(graph, PLAN_1, state) == once


def test_a_step_the_plan_dropped_and_nobody_started_is_retired(graph):
    """Left at «не начат» it reads as work still ahead, and the operator has
    no way to tell it from a step that is genuinely waiting."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    after = _mirror(graph, PLAN_1[:3], state)
    assert after["PS4"][0] == "blocked", after
    assert after["PS3"][0] == "todo", after
    history = next(n["status_history"] for n in graph.full()["nodes"]
                   if n["id"] == "PS4")
    assert "убран при пересмотре плана" in history[-1]["reason"]


def test_a_step_that_already_ran_is_left_alone_when_the_plan_drops_it(graph):
    """`blocked` is a verdict on work ahead. A step that finished happened —
    the plan changing afterwards does not unhappen it."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    done = [dict(t) for t in PLAN_1]
    done[3]["status"] = "DONE"
    _mirror(graph, done, state)
    assert _column(graph)["PS4"][0] == "done"

    after = _mirror(graph, PLAN_1[:3], state)
    assert after["PS4"][0] == "done", after


# ── the limits of the match ─────────────────────────────────────────────────

def test_a_step_with_nothing_in_common_is_recorded_as_a_new_one(graph):
    """The plan's ids are positional — `create_plan` hands out TASK-1…n after
    ordering — so holding an id is not on its own a reason to rewrite a card.
    When the words disagree too, the honest answer is a new step."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    swapped = [dict(t) for t in PLAN_1]
    swapped[3] = {"id": "TASK-4", "assignee": _EM, "status": "TODO",
                  "title": "Синтезировать три соединения в лаборатории",
                  "description": "Заказать реагенты и провести синтез"}
    after = _mirror(graph, swapped, state)
    assert len(after) == 5, after
    assert after["PS4"][0] == "blocked", after           # the one it replaced


def test_a_step_handed_to_a_different_agent_is_a_different_step(graph):
    """Same slot, same words, another performer: that is a re-assignment of
    the work, not a rewording of the card."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    moved = [dict(t) for t in PLAN_1]
    moved[3] = dict(PLAN_1[3], assignee="CoderAgent",
                    title=PLAN_1[3]["title"] + " (в песочнице)")
    after = _mirror(graph, moved, state)
    assert len(after) == 5, after


def test_two_tasks_cannot_be_mirrored_onto_one_step(graph):
    """Exclusive claims, or a plan that says the same thing twice would leave
    one card doing the work of two."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    # Same plan slot and same assignee as PLAN_1[3], so both tasks reach the
    # slot pass and compete for PS4; only the exclusive claim keeps the second
    # from landing on it too.
    twice = [dict(t) for t in PLAN_1] + [
        dict(PLAN_1[3], id="TASK-4",
             title=PLAN_1[3]["title"] + " — повторный прогон")]
    after = _mirror(graph, twice, state)
    assert len(after) == 5, after
    assert after["PS4"][1] == PLAN_1[3]["title"]


def test_the_column_survives_a_plan_that_lost_its_ids(graph):
    """An operator can put a roadmap straight into state from the web UI, and
    it need not carry ids at all. Then the title is all there is — which is
    what the mirror used to rely on for everything."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    before = _column(graph)
    anonymous = [{k: v for k, v in t.items() if k != "id"} for t in PLAN_1]
    after = _mirror(graph, anonymous, state)
    assert after == before, after


def test_an_empty_task_list_retires_nothing(graph):
    """The mirror runs on every orchestrator turn, and the task list can be
    empty for reasons that say nothing about the plan — a restarted process
    reading an existing graph, a call before the tracker is filled. Reading
    that as "the plan dropped everything" would blank the whole column."""
    state: dict = {}
    before = _mirror(graph, PLAN_1, state)
    sync_plan_to_research_graph([], graph, state, "борщевик")
    assert _column(graph) == before


# ── what the re-review found ────────────────────────────────────────────────

def test_a_replan_after_a_step_is_done_still_reaches_the_graph(graph):
    """`create_plan` re-issues EVERY task as TODO when a plan is revised, so a
    finished step is asked to go back to «не начат». PlanStep has no such
    transition and a commit is all-or-nothing: the store refused the whole
    thing, and the retitles, the new steps and the retirements went with it —
    for the rest of the run, since every later mirror re-sent the same update.
    """
    state: dict = {}
    done = [dict(t) for t in PLAN_1]
    done[0]["status"] = "DONE"
    _mirror(graph, done, state)
    assert _column(graph)["PS1"][0] == "done"

    # The revision: same six steps, two reworded, everything back to TODO.
    revised = [dict(t) for t in PLAN_2]
    after = _mirror(graph, revised, state)
    assert after["PS1"][0] == "done", after            # not talked out of it
    assert after["PS3"][1] == PLAN_2[2]["title"], after  # and the retitle landed
    assert len(after) == 4, after


def test_a_description_the_planner_deleted_leaves_the_card(graph):
    """`_step_attrs` drops empty values, which is right for a creation and
    wrong for a rewrite: the removed sentence stayed on the card, and the
    comparison could not even see that it had been removed."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    assert _card_of(graph, "PS3")["description"]

    stripped = [dict(t) for t in PLAN_1]
    stripped[2] = dict(PLAN_1[2], description="")
    _mirror(graph, stripped, state)
    assert _card_of(graph, "PS3").get("description", "") == ""


def test_a_retitle_does_not_renumber_the_step(graph):
    """The plan's ids are positional. A method that realises a step carries the
    id the step had when it was written, so following the plan's renumbering
    would leave the step holding an id that, on those methods, belongs to a
    different step — and `_realises_edges` would draw the link onto it."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    assert _card_of(graph, "PS3")["plan_task_id"] == "TASK-3"

    # The operator drops the first step; create_plan renumbers positionally.
    shifted = [dict(t, id="TASK-%d" % i) for i, t in enumerate(PLAN_1[1:], start=1)]
    _mirror(graph, shifted, state)
    assert _card_of(graph, "PS3")["plan_task_id"] == "TASK-3"


def test_the_memo_forgets_the_wording_a_retitle_replaced(graph):
    """A memo that keeps the superseded title goes on asserting a wording the
    plan no longer uses; the next plan to contain that wording then takes the
    step away from the task whose slot it actually is."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    _mirror(graph, PLAN_2, state)          # PS3 is reworded
    memo = (state.get("_research_vm_by_task") or {}).get("ids") or {}
    assert PLAN_1[2]["title"].lower() not in memo, memo
    assert memo.get(PLAN_2[2]["title"].lower()) == "PS3", memo


def test_the_memo_is_read_off_the_echo_by_ref(graph):
    """One commit now carries creations AND retitles, and the store may answer
    a creation with a node it already held — so the echo is no longer one entry
    per new step in the order they were sent, and reading it positionally would
    map a task to someone else's step."""
    state: dict = {}
    _mirror(graph, PLAN_1, state)
    # A revision that retitles one step and adds another: the echo holds a
    # merge and a creation together.
    revised = [dict(t) for t in PLAN_2] + [
        {"id": "TASK-5", "assignee": _EM, "status": "TODO",
         "title": "Оценка стоимости синтеза трёх соединений",
         "description": "Посчитать стоимость синтеза через ASKCOS"}]
    _mirror(graph, revised, state)
    memo = (state.get("_research_vm_by_task") or {}).get("ids") or {}
    assert memo["оценка стоимости синтеза трёх соединений"] == "PS5", memo
    assert memo[PLAN_2[2]["title"].lower()] == "PS3", memo
