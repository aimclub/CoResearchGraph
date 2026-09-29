"""Which plan step an experiment task belongs to, and how far along that step is.

Both defects were read off one live graph — session_892e756a of 2026-09-21,
the richest study in the operator's history:

    XT1..XT7  -elaborates->  PS3        (all seven, on one step)
    PS3  «Кластеризация метаболитов…»            выполнен
    PS4  «Предсказание LD50 (мышь, все пути)…»   НЕ НАЧАТ
    XT3  «Предсказание LD50 (мышь, все пути)…»   выполнен

So the step that reads "not started" has its work finished under a different
step, and its own card repeats the text of a task hanging off its neighbour —
which is exactly what the operator noticed.

Cause 1: `_outer_step` resolved ONE step for the whole detailed plan (the
tracker step in progress when the plan was approved) and every task got it.
True only while one dispatch covered one step; this dispatch covered four.

Cause 2: a step's status came from the tracker alone, and the tracker is moved
by the work order, which the experiment module does not use — it records
through its own runtime. Nothing carried "my tasks are done" back to the step.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from CoScientist.agents.callbacks.tool_callbacks import (
    _status_from_tasks,
    sync_plan_to_research_graph,
)
from CoScientist.experiments.runtime.graph_bridge import (
    _plan_step_texts,
    _step_for_task,
    _task_attrs,
    _task_match_text,
    _words,
)

#: The plan of session_989a309886da43a4a612f1e23c55fa74 (2026-09-22) as the
#: tracker held it — three steps, descriptions trimmed to the part that carries
#: the vocabulary. The assignees are real: only an executor's step can host an
#: experiment task, and TASK-1's description is the distractor that made text
#: matching fail before the filter — it recaps the whole pipeline.
STEPS = [
    {"id": "TASK-1", "assignee": "HypothesesAgent", "status": "DONE",
     "title": "Сформулировать гипотезу для пайплайна",
     "description": "Зафиксировать одну проверяемую гипотезу о том, что покажет "
                    "пайплайн токсикологического профилирования метаболитов "
                    "Heracleum sosnowskyi: кластер фуранокумаринов окажется "
                    "наиболее токсичным по предсказанным LD50, ADMET-флаги "
                    "кардиотоксичности преобладают"},
    {"id": "TASK-2", "assignee": "ExperimentModuleAgent", "status": "TODO",
     "title": "Собрать литературные данные и SMILES метаболитов",
     "description": "Составить стандартизированный корпус метаболитов Heracleum "
                    "sosnowskyi и их SMILES: собрать из научной литературы "
                    "(EB1, PubMed/PubChem/ChEMBL) структурные данные по "
                    "фуранокумаринам (бергаптен, псорален, ксантотоксин, "
                    "бергамоттин, императорин, умбеллиферон, скополетин), "
                    "кумаринам, терпеноидам и жирным кислотам растения. Выход: "
                    "валидированный список SMILES метаболитов с источниками "
                    "(уникальность, канонизация через RDKit/PubChem). Успех: "
                    "готовый машинно-читаемый список SMILES, пригодный для "
                    "передачи в шаг профилирования."},
    {"id": "TASK-3", "assignee": "ExperimentModuleAgent", "status": "TODO",
     "title": "Провести структурную кластеризацию и полное in-silico "
              "токсикологическое профилирование",
     "description": "На сервере heracleum-tox по списку SMILES из TASK-2 "
                    "выполнить: (1) chemical_space_clustering — кластеризацию по "
                    "структурному сходству (ECFP4 + агломеративная "
                    "кластеризация) с построением дендрограммы; (2) для каждого "
                    "метаболита predict_molecule_profile — предсказание LD50 для "
                    "мыши по всем путям введения там, где нет экспериментальных "
                    "данных, и оценку домена применимости модели (kNN(5)+Gaussian "
                    "AD). По результатам определить наиболее токсичный кластер; "
                    "для его соединений использовать predict_molecule_profile для "
                    "предсказания гепатотоксичности, кардиотоксичности (hERG), "
                    "DILI и канцерогенности; для 3 наиболее перспективных "
                    "соединений получить оценку стоимости синтеза (ASKCOS). "
                    "Успех: получены дендрограмма кластеризации, LD50 всех "
                    "метаболитов с AD, предсказания 4 эндпоинтов для наиболее "
                    "токсичного кластера и стоимости синтеза 3 соединений."},
]

#: Its experiment plan, in the shape a PLAN has: `name`, and the question under
#: `design`. Written out because that is the whole point of these tests.
TASKS = [
    ("EXP-1", "Сбор литературных данных о метаболитах Heracleum sosnowskyi и их SMILES",
     "Какие метаболиты Heracleum sosnowskyi и их SMILES присутствуют в "
     "реконструированном литературном корпусе?", "TASK-2"),
    # Declines: 0.45 on the clustering step against 0.36 on the corpus step,
    # whose description also names SMILES and the metabolites. 0.09 of lead is
    # under the bar, and lowering the bar to catch it put a WRONG line on the
    # 2026-09-21 study, so it stays where it was dispatched.
    ("EXP-2", "Кластеризация метаболитов по структурному сходству",
     "Какие структурные кластеры образуются среди метаболитов корпуса и какие "
     "соединения входят в фуранокумариновый кластер?", None),
    ("EXP-3", "Предсказание LD50 (мышь) для всех путей введения",
     "Каковы предсказанные значения LD50 (мышь, мг/кг, lg-шкала) для метаболитов "
     "корпуса по всем путям введения?", "TASK-3"),
    ("EXP-4", "Оценка домена применимости QSAR-моделей",
     "Какая доля предсказаний LD50 фуранокумаринового кластера попадает в домен "
     "применимости моделей (kNN(5)+Gaussian AD)?", "TASK-3"),
    ("EXP-5", "Предсказание гепато-/кардиотоксичности, DILI и канцерогенности "
     "наиболее токсичного кластера",
     "Каковы предсказанные значения гепатотоксичности, кардиотоксичности, DILI и "
     "канцерогенности для наиболее токсичного (фуранокумаринового) кластера?", "TASK-3"),
    ("EXP-6", "Оценка стоимости синтеза 3 перспективных соединений",
     "Какова оценка стоимости синтеза (USD/g) трёх перспективных соединений из "
     "наиболее токсичного кластера?", "TASK-3"),
]


def _plan_task(name, question):
    """A task as the plan writes it — NOT as the graph stores it."""
    return {"id": "EXP-x", "name": name,
            "design": {"experiment_question": question}}


# ── which step ──────────────────────────────────────────────────────────────

def test_the_matcher_and_the_card_read_the_same_words():
    """The defect that made the whole matcher a no-op.

    `_task_attrs` writes the card's `title` from the plan's `name` and its
    `question` from `design.experiment_question`. The matcher used to read
    `task["title"]` and `task["question"]` — the names those fields have AFTER
    the bridge has written them — so it scored two empty strings and every task
    fell back to the dispatched step. Nothing failed: the graph just hung six
    tasks off one step.
    """
    task = _plan_task("Предсказание LD50 (мышь)", "Каковы значения LD50?")
    text = _task_match_text(task)
    assert "Предсказание LD50" in text and "Каковы значения" in text
    # And the card the reader sees is built from the same two fields.
    attrs = _task_attrs(task, {"plan_id": "PLAN-1"}, "TASK-3")
    assert attrs["title"] in text
    assert attrs["question"] in text
    # A graph-shaped task carries no words for this matcher, which is exactly
    # how the bug hid: both the code and its test used that shape.
    assert not _words(_task_match_text({"title": "x", "question": "y"}))


def test_each_task_lands_on_the_step_it_carries_out():
    """Five of the six, measured on the live run that hung all six off the
    literature step. EXP-2 is the honest gap: «Кластеризация метаболитов по
    структурному сходству» scores 0.45 on the clustering step against 0.36 on
    the corpus step — the plan's own description mentions the SMILES corpus —
    and 0.09 of lead is under the bar, so it stays where it was dispatched
    rather than being moved on a coin toss."""
    steps = _plan_step_texts({"_master_active_tasks": STEPS})
    for task_id, name, question, expected in TASKS:
        got, best, runner = _step_for_task(_plan_task(name, question), steps, "TASK-2")
        assert got == (expected or "TASK-2"), f"{task_id}: {got} at {best:.2f}/{runner:.2f}"


def test_a_step_an_experiment_task_cannot_carry_out_is_never_offered():
    """A hypothesis step's description recaps the pipeline the tasks implement,
    so it shares vocabulary with every one of them — on the 2026-09-21 study
    «ADMET-профилирование наиболее токсичного кластера» scored a confident 1.00
    against «Сформулировать гипотезу исследования». No threshold removes that;
    the assignee does."""
    steps = _plan_step_texts({"_master_active_tasks": STEPS})
    assert [s for s, _ in steps] == ["TASK-2", "TASK-3"]
    got, _, _ = _step_for_task(
        _plan_task("ADMET-профилирование наиболее токсичного кластера фуранокумаринов",
                   "Каковы ADMET-флаги кардиотоксичности?"), steps, "TASK-2")
    assert got == "TASK-3", got


def test_one_shared_word_does_not_decide():
    """«Предсказание LD50 (мышь)» shares exactly «метабол» with a corpus step
    and nothing with the step it belongs to. A single incidental word must not
    settle where a task hangs."""
    steps = _plan_step_texts({"_master_active_tasks": [
        {"id": "TASK-2", "assignee": "ExperimentModuleAgent",
         "title": "Собрать метаболиты", "description": ""},
        {"id": "TASK-3", "assignee": "ExperimentModuleAgent",
         "title": "Построить графики", "description": ""},
    ]})
    got, best, _ = _step_for_task(
        _plan_task("Предсказание LD50 для метаболитов", ""), steps, "TASK-9")
    assert got == "TASK-9" and best == 0.0


def test_two_steps_that_read_alike_leave_the_link_where_it_was():
    """A lead over the runner-up, not just a floor: when two steps describe the
    same work, the text cannot say which one, and inventing a winner puts a
    confident wrong line on the graph."""
    steps = _plan_step_texts({"_master_active_tasks": [
        {"id": "TASK-1", "assignee": "TaskExecutorAgent",
         "title": "Предсказание LD50 по кластерам метаболитов", "description": ""},
        {"id": "TASK-2", "assignee": "TaskExecutorAgent",
         "title": "Предсказание LD50 по кластерам метаболитов", "description": ""},
    ]})
    got, _, _ = _step_for_task(
        _plan_task("Предсказание LD50 по кластерам метаболитов", ""), steps, "TASK-9")
    assert got == "TASK-9"


def test_no_outer_plan_means_no_invented_step():
    got, _, _ = _step_for_task(_plan_task("что угодно", ""), [], "")
    assert got == ""


#: The status half needs a plan with two executor steps to move, and the
#: hogweed study's five-step plan is what those assertions were written
#: against. Kept separate from STEPS so a change to the matcher's fixture
#: cannot silently rewrite what "a step follows its tasks" is tested on.
STATUS_STEPS = [
    {"id": "TASK-1", "title": "Сформулировать гипотезу исследования",
     "assignee": "HypothesesAgent", "status": "DONE"},
    {"id": "TASK-2", "title": "Собрать литературные данные о метаболитах и их SMILES",
     "assignee": "TaskExecutorAgent", "status": "DONE"},
    {"id": "TASK-3", "title": "Кластеризация метаболитов по структурному сходству",
     "assignee": "TaskExecutorAgent", "status": "DONE"},
    {"id": "TASK-4", "title": "Предсказание LD50 (мышь, все пути) и домена применимости",
     "assignee": "TaskExecutorAgent", "status": "TODO"},
    {"id": "TASK-5", "title": "ADMET-эндпоинты для токсичного кластера и стоимость синтеза",
     "assignee": "TaskExecutorAgent", "status": "TODO"},
]


# ── how far along ───────────────────────────────────────────────────────────

class _Graph:
    """Just enough store to answer `full()`."""

    def __init__(self, nodes, edges):
        self._full = {"nodes": nodes, "edges": edges}

    def full(self):
        return self._full

    def root_id(self):
        return "Q1"


def _graph_of(*task_statuses: tuple[str, str, str]):
    """(task id, status, step id) triples into a graph of steps and tasks."""
    nodes = [{"id": "PS4", "type": "PlanStep", "status": "todo",
              "attrs": {"plan_task_id": "TASK-4"}},
             {"id": "PS5", "type": "PlanStep", "status": "todo",
              "attrs": {"plan_task_id": "TASK-5"}}]
    edges = []
    for task_id, status, step in task_statuses:
        nodes.append({"id": task_id, "type": "ExperimentTask", "status": status})
        edges.append({"type": "elaborates", "from": task_id, "to": step})
    return _Graph(nodes, edges)


def test_a_step_whose_every_task_is_done_is_done():
    derived = _status_from_tasks(_graph_of(("XT3", "done", "PS4"),
                                          ("XT4", "done", "PS4")))
    assert derived == {"PS4": "done"}


def test_a_step_with_work_under_way_is_under_way():
    derived = _status_from_tasks(_graph_of(("XT3", "done", "PS4"),
                                           ("XT4", "running", "PS4")))
    assert derived == {"PS4": "in_progress"}


def test_a_plan_that_has_only_been_written_starts_nothing():
    """`planned` is the plan, not the work: a step with nothing but planned
    tasks under it has not begun."""
    assert _status_from_tasks(_graph_of(("XT3", "planned", "PS4"),
                                        ("XT4", "planned", "PS4"))) == {}


def test_a_failed_task_stops_the_step_instead_of_leaving_it_running():
    derived = _status_from_tasks(_graph_of(("XT3", "failed", "PS4")))
    assert derived == {"PS4": "blocked"}


def test_a_failed_task_does_not_stop_a_sibling_that_is_still_running():
    derived = _status_from_tasks(_graph_of(("XT3", "failed", "PS4"),
                                           ("XT4", "running", "PS4")))
    assert derived == {"PS4": "in_progress"}


def test_a_skipped_optional_task_does_not_hold_its_step_back():
    derived = _status_from_tasks(_graph_of(("XT3", "done", "PS4"),
                                           ("XT4", "skipped", "PS4")))
    assert derived == {"PS4": "done"}


def test_a_step_with_only_skipped_tasks_is_not_left_running():
    derived = _status_from_tasks(_graph_of(("XT3", "skipped", "PS4"),
                                           ("XT4", "skipped", "PS4")))
    assert derived == {"PS4": "blocked"}


def test_the_step_of_a_finished_task_stops_saying_not_started(monkeypatch):
    """The defect end to end: the tracker still says TODO, the graph says the
    tasks are done, and the plan column has to show the work."""
    graph = _graph_of(("XT3", "done", "PS4"), ("XT4", "done", "PS4"),
                      ("XT5", "running", "PS5"))
    updates = []
    graph.commit = lambda **kw: updates.append(kw) or SimpleNamespace(
        ok=True, errors=[], committed={"nodes": []})

    state = {"_master_active_tasks": [dict(s) for s in STATUS_STEPS],
             # the mirror has already created these two steps
             "_vm_by_task": {"gen": "", "ids": {}}}
    # Pretend the steps were mirrored earlier: key them the way the mirror does.
    from CoScientist.agents.callbacks.tool_callbacks import (
        _study_generation, _task_key, _VM_BY_TASK_KEY,
    )
    state[_VM_BY_TASK_KEY] = {
        "gen": _study_generation(graph),
        "ids": {_task_key(STATUS_STEPS[3]): "PS4",
                _task_key(STATUS_STEPS[4]): "PS5"},
    }

    sync_plan_to_research_graph(state["_master_active_tasks"], graph, state)

    moved = {u["id"]: (u["status"], u["reason"])
             for kw in updates for u in (kw.get("status_updates") or [])}
    assert moved.get("PS4", ("",))[0] == "done", moved
    assert moved.get("PS5", ("",))[0] == "in_progress", moved
    # The reason is what the card shows on its «почему» line, so it is
    # written in the language of the person reading the graph.
    assert "задачи эксперимента" in moved["PS4"][1], moved["PS4"][1]

    # And the roadmap the operator reads agrees with the graph.
    tracker = {t["id"]: t["status"] for t in state["_master_active_tasks"]}
    assert tracker["TASK-4"] == "DONE" and tracker["TASK-5"] == "IN_PROGRESS", tracker


def test_a_blocked_step_is_not_talked_out_of_it(monkeypatch):
    """`blocked` is a verdict on the step — a rejected report — not a measure
    of how much of it ran, so finished tasks must not clear it."""
    graph = _graph_of(("XT3", "done", "PS4"))
    updates = []
    graph.commit = lambda **kw: updates.append(kw) or SimpleNamespace(
        ok=True, errors=[], committed={"nodes": []})

    from CoScientist.agents.callbacks.tool_callbacks import (
        _study_generation, _task_key, _VM_BY_TASK_KEY,
    )
    steps = [dict(s) for s in STATUS_STEPS]
    steps[3]["status"] = "FAILED"          # the tracker's word for blocked
    state = {"_master_active_tasks": steps,
             _VM_BY_TASK_KEY: {"gen": _study_generation(graph),
                               "ids": {_task_key(steps[3]): "PS4"}}}

    sync_plan_to_research_graph(steps, graph, state)

    moved = {u["id"]: u["status"]
             for kw in updates for u in (kw.get("status_updates") or [])}
    assert moved.get("PS4", "blocked") == "blocked", moved
    assert state["_master_active_tasks"][3]["status"] == "FAILED"
