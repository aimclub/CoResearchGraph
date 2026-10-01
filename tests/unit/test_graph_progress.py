"""Что граф говорит о происходящем прямо сейчас, и что такое «метод проверки».

Две жалобы оператора, обе про одно: по графу нельзя понять, какой этап идёт.
Первая — отметки «начат / не начат / выполнен» врут и не обновляются. Вторая —
«Метод проверки» жил в словаре ЗАДАЧИ («запланирован», «выполнен»), хотя метод
не работа, а перечень средств: он предложен, использован или не использован.

Здесь проверяется серверная сторона обоих ответов: словарь статусов метода,
чтение старых графов в новых словах, заголовок карточки метода и перечень
инструментов на ней. Мигание — на канве, оно в tests/unit/test_graph_palette.py.

Run from the repo root:  pytest tests/unit/test_graph_progress.py -q
"""
import json

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.graph.research import schema  # noqa: E402
from CoScientist.graph.research.store import (  # noqa: E402
    _STATUS_WORDS,
    ResearchGraphStore,
)


@pytest.fixture
def store(tmp_path):
    return ResearchGraphStore(directory=str(tmp_path))


def _study(store):
    store.init_research(
        source="OrchestratorAgent",
        question="Does compound X inhibit target Y?",
        tools=[{"name": "AutoDock", "tool_type": "computational"}],
    )
    store.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "ref": "h",
                "attrs": {"formulation": "X binds Y"}},
               {"type": "VerificationMethod", "ref": "vm",
                "attrs": {"method_type": "computational"}}],
        edges=[{"type": "motivates", "from": "Q1", "to": "#h"},
               {"type": "tested_by", "from": "#h", "to": "#vm"}],
    )


def _node(store, node_id):
    return next(n for n in store.full()["nodes"] if n["id"] == node_id)


def _card(store, node_id):
    return next(n for n in store.to_view()["nodes"] if n["id"] == node_id)


# ── the vocabulary a method speaks ───────────────────────────────────────────

def test_a_method_is_offered_not_scheduled(store):
    """«Запланирован» — состояние работы. У метода его быть не может: метод —
    это средство, и единственное, что о нём можно сказать при создании, что он
    предложен."""
    _study(store)
    assert _node(store, "VM1")["status"] == "proposed"
    assert schema.NODE_TYPES["VerificationMethod"].creatable == ("proposed",)


def test_a_method_cannot_be_executed_or_completed(store):
    """Ровно та ошибка, на которую жаловался оператор: карточка средства
    отвечала в словаре задачи. Отказ здесь — это то, что не даст словарю
    вернуться обратно по частям."""
    _study(store)
    for forbidden in ("planned", "running", "done", "failed"):
        r = store.commit(source="ExperimentAgent",
                         status_updates=[{"id": "VM1", "status": forbidden}])
        assert not r.ok, f"метод не может быть «{forbidden}»"


def test_a_method_the_study_leaned_on_and_one_it_left_aside(store):
    """Два исхода, которые вообще возможны, и возврат из второго в первый:
    метод, от которого отказались, может оказаться тем, что решит вопрос."""
    _study(store)
    assert store.commit(source="ExperimentAgent",
                        status_updates=[{"id": "VM1", "status": "not_used",
                                         "reason": "не дошли"}]).ok
    assert store.commit(source="ExperimentAgent",
                        status_updates=[{"id": "VM1", "status": "used"}]).ok
    assert _node(store, "VM1")["status"] == "used"
    # …и обратно нельзя: метод, давший свидетельство, остался использованным.
    assert not store.commit(
        source="ExperimentAgent",
        status_updates=[{"id": "VM1", "status": "not_used"}]).ok


def test_the_words_the_reader_sees(store):
    """Слова карточки. «Выполняется» — одно и то же событие для шага плана и
    для задачи эксперимента, и пишется оно одинаково: читатель ищет, что идёт
    сейчас, а не разницу между двумя формулировками одного факта."""
    assert _STATUS_WORDS["proposed"] == "предложен"
    assert _STATUS_WORDS["used"] == "использован"
    assert _STATUS_WORDS["not_used"] == "не использован"
    assert _STATUS_WORDS["in_progress"] == _STATUS_WORDS["running"] == "выполняется"
    # Свидетельство появляется только после получения результата. Его
    # достоверность оценивается отдельно и не меняет этот факт.
    assert _STATUS_WORDS["obtained"] == "получено"
    assert _STATUS_WORDS["under_verification"] == "проверяется"


def test_a_method_left_aside_is_asked_why(store):
    """«Не использован» без причины — дыра в записи, и граф называет её так же,
    как называл «не удался» без причины."""
    _study(store)
    store.commit(source="ExperimentAgent",
                 status_updates=[{"id": "VM1", "status": "not_used"}])
    gaps = {g["code"] for g in store.to_view()["gaps"]}
    assert "unreasoned_failures" in gaps
    assert _card(store, "VM1")["why_missing"] is True


# ── исследования, записанные до смены словаря ────────────────────────────────

_OLD_GRAPH = {
    "research_id": "research",
    "created_at": 1.0,
    "root_id": "Q1",
    "nodes": [
        {"id": "Q1", "type": "ResearchQuestion", "status": "open",
         "attrs": {"formulation": "старый вопрос"}, "source": "OrchestratorAgent"},
        {"id": "VM1", "type": "VerificationMethod", "status": "done",
         "attrs": {"method_type": "computational"}, "source": "ExperimentModule",
         "status_history": [{"from": None, "to": "planned", "source": "ExperimentModule"},
                            {"from": "planned", "to": "running", "source": "ExperimentAgent"},
                            {"from": "running", "to": "done", "source": "ExperimentModule"}]},
        {"id": "VM2", "type": "VerificationMethod", "status": "planned",
         "attrs": {"method_type": "literature_review"}, "source": "ResearchAgent",
         "status_history": [{"from": None, "to": "planned", "source": "ResearchAgent"}]},
        {"id": "VM3", "type": "VerificationMethod", "status": "failed",
         "attrs": {"method_type": "computational"}, "source": "ExperimentModule",
         "status_history": [{"from": None, "to": "planned", "source": "ExperimentModule"},
                            {"from": "planned", "to": "failed", "source": "ExperimentModule"}]},
    ],
    "edges": [],
}


def _reopen(tmp_path, data=None):
    (tmp_path / "research_active.json").write_text(
        json.dumps(data or _OLD_GRAPH, ensure_ascii=False), encoding="utf-8")
    return ResearchGraphStore(directory=str(tmp_path))


def test_an_old_study_is_re_read_not_left_speechless(tmp_path):
    """Сохранённых методов в старых словах на диске сотня с лишним. Ни один из
    них не «неизвестный статус» — это те же факты, записанные словарём задачи."""
    store = _reopen(tmp_path)
    got = {n["id"]: n["status"] for n in store.full()["nodes"]
           if n["type"] == "VerificationMethod"}
    assert got == {"VM1": "used", "VM2": "proposed", "VM3": "not_used"}


def test_an_old_study_is_not_frozen_by_the_new_words(tmp_path):
    """Главная опасность смены словаря: `validate_transition` читает `from` с
    самого узла, и пара, начинающаяся с «planned», больше не существует. Без
    перечитывания каждый записанный метод стал бы неподвижен навсегда."""
    store = _reopen(tmp_path)
    assert store.commit(source="ExperimentAgent",
                        status_updates=[{"id": "VM2", "status": "used"}]).ok


def test_no_recorded_move_is_lost_in_the_re_reading(tmp_path):
    """`planned → running` в новых словах читается как «предложен → предложен»,
    и выбросить такую запись было бы опрятнее на вид и неверно: в ней свой
    источник и своя причина, `_save` тут же пишет укороченный след поверх
    файла, и второй копии нет. По сохранённым исследованиям это 46 переходов,
    41 из них с написанной причиной."""
    store = _reopen(tmp_path)
    trail = _card(store, "VM1")["status_history"]
    assert [(h["from"], h["to"]) for h in trail] == [(None, "proposed"),
                                                     ("proposed", "proposed"),
                                                     ("proposed", "used")]
    assert [h["source"] for h in trail] == ["ExperimentModule", "ExperimentAgent",
                                            "ExperimentModule"]
    assert trail[-1]["to_word"] == "использован"


def test_the_re_reading_does_not_shorten_the_file_it_read(tmp_path):
    """Перечитывание происходит в памяти, но первый же коммит записывает
    результат поверх исходного файла — то есть любая потеря здесь окончательна."""
    store = _reopen(tmp_path)
    store.commit(source="ExperimentAgent",
                 status_updates=[{"id": "VM2", "status": "used"}])
    on_disk = json.loads(
        (tmp_path / "research_active.json").read_text(encoding="utf-8"))
    vm1 = next(n for n in on_disk["nodes"] if n["id"] == "VM1")
    assert len(vm1["status_history"]) == 3
    assert any(h.get("source") == "ExperimentAgent" for h in vm1["status_history"])


def test_re_reading_touches_only_methods(tmp_path):
    """«planned» — это ещё и статус задачи эксперимента, и он остаётся собой."""
    data = json.loads(json.dumps(_OLD_GRAPH))
    data["nodes"].append({"id": "XT1", "type": "ExperimentTask",
                          "status": "planned", "attrs": {"title": "EXP-1"},
                          "source": "experiment-plan-mirror"})
    store = _reopen(tmp_path, data)
    assert _node(store, "XT1")["status"] == "planned"


# ── что на карточке метода ───────────────────────────────────────────────────

def test_a_method_is_not_called_computational(store):
    """Модуль экспериментов пишет `name` и не пишет `description`, а заголовок
    брал `method_type` вторым — и все методы одного прогона выходили на канву
    с одним и тем же заголовком «computational»."""
    _study(store)
    store.commit(source="ExperimentModule",
                 nodes=[{"id": "VM1", "attrs": {
                     "name": "Кластеризация метаболитов по ECFP4"}}],
                 enforce_permissions=False)
    assert _card(store, "VM1")["label"] == "Кластеризация метаболитов по ECFP4"


def test_the_card_says_what_the_method_is_run_with(store):
    """Метод — это перечень средств, и до сих пор карточка не умела прочитать
    тот перечень, который утверждённый план уже на неё положил: инструменты MCP
    лежали в `mcp_servers` списком словарей, а агент — в `assignee`."""
    _study(store)
    store.commit(source="ExperimentModule", nodes=[{"id": "VM1", "attrs": {
        "mcp_servers": [{"name": "heracleum-tox", "url": "http://x/mcp",
                         "tools": ["predict_ld50", "chemical_space_clustering"]}],
        "assignee": "ExperimentAgent",
    }}], enforce_permissions=False)
    named = _card(store, "VM1")["instruments"]
    assert "heracleum-tox:predict_ld50" in named
    assert "heracleum-tox:chemical_space_clustering" in named
    assert "ExperimentAgent" in named


def test_the_instruments_are_reported_not_invented(store):
    """Метод, за которым не стоит ни одного названного средства, не получает
    выдуманного: пусто честнее, чем «инструмент не указан», выданное за факт."""
    _study(store)
    assert "instruments" not in _card(store, "VM1")


def test_one_instrument_named_twice_is_one_instrument(store):
    """Одно и то же средство приходит с ребра `uses` и из `mcp_servers`."""
    _study(store)
    store.commit(source="ExperimentModule", nodes=[{"id": "VM1", "attrs": {
        "tools": "AutoDock, autodock", "instruments": "AutoDock",
    }}], edges=[{"type": "uses", "from": "VM1", "to": "T1"}],
        enforce_permissions=False)
    assert _card(store, "VM1")["instruments"].lower().count("autodock") == 1


def test_a_tool_named_bare_and_named_by_its_server_is_one_tool(store):
    """Ребро `uses` даёт голое имя инструмента, утверждённый план — имя с
    сервером. Карточка, перечислившая оба, говорит, что метод работал двумя
    средствами; остаётся то из двух имён, которое читатель не восстановит сам."""
    _study(store)
    store.commit(source="ExperimentModule", nodes=[
        {"type": "Tool", "ref": "t", "attrs": {"name": "predict_ld50"}},
        {"id": "VM1", "attrs": {"mcp_servers": [
            {"name": "heracleum-tox", "tools": ["predict_ld50"]}]}}],
        edges=[{"type": "uses", "from": "VM1", "to": "#t"}],
        enforce_permissions=False)
    assert _card(store, "VM1")["instruments"] == "heracleum-tox:predict_ld50"


def test_two_servers_offering_the_same_tool_stay_two(store):
    """Схлопывается только голое имя. Два сервера с одноимённым инструментом —
    это два разных средства, и ни одно из них не безымянное."""
    from CoScientist.graph.research.store import _one_name_each

    assert _one_name_each(["a:run", "b:run"]) == ["a:run", "b:run"]
    assert _one_name_each(["run", "a:run"]) == ["a:run"]
    assert _one_name_each(["a:run", "run"]) == ["a:run"]


def test_a_method_named_only_in_label_is_not_headlined_by_its_kind(store):
    """`label` пишет агент-исследователь, и заголовок его не читал — карточка
    литературного обзора выходила на канву со словом «literature_review»."""
    _study(store)
    store.commit(source="ResearchAgent", nodes=[{"id": "VM1", "attrs": {
        "method_type": "literature_review",
        "label": "Литературный обзор метаболитов"}}],
        enforce_permissions=False)
    assert _card(store, "VM1")["label"] == "Литературный обзор метаболитов"


def test_an_instrument_list_survives_the_shape_it_was_written_in(store):
    """Три автора — три формы записи. Метод, назвавший средства не тем
    разделителем, — это не метод, не назвавший ничего."""
    from CoScientist.graph.research.store import _named_instruments

    assert _named_instruments("a; b") == ["a", "b"]
    assert _named_instruments("a, b") == ["a", "b"]
    assert _named_instruments(["a", "b"]) == ["a", "b"]
    assert _named_instruments(None) == []
    assert _named_instruments("") == []


def test_a_replayed_study_speaks_the_new_words_too(tmp_path):
    """Запись прогона — файл того же возраста, что сохранённое исследование, и
    этот путь не проходит через загрузчик хранилища. Без перечитывания карточка
    «Метода проверки» в демо-повторе читалась «запланирован»/«выполнен» — ровно
    то, что оператор и просил убрать."""
    import networkx as nx

    from CoScientist.web.replay import ReplaySession

    graph = nx.MultiDiGraph()
    for node in _OLD_GRAPH["nodes"]:
        ReplaySession._write(graph, "node", node)
    assert graph.nodes["VM1"]["status"] == "used"
    assert graph.nodes["VM3"]["status"] == "not_used"
    # …и то же для отдельного события смены статуса.
    ReplaySession._write(graph, "status", {"id": "VM2", "status": "done"})
    assert graph.nodes["VM2"]["status"] == "used"


def test_the_slide_cli_reads_an_old_study_in_the_new_words(tmp_path):
    """`main()` читает JSON с диска напрямую, минуя хранилище, и таблица фраз
    «свидетельств нет» ключуется статусом метода."""
    import json as _json

    from CoScientist.graph.research import slide_render

    src = tmp_path / "study.json"
    src.write_text(_json.dumps(_OLD_GRAPH, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "slide.svg"
    assert slide_render.main([str(src), str(out)]) == 0
    # Узлы были переписаны на месте перед отрисовкой.
    data = _json.loads(src.read_text(encoding="utf-8"))
    assert {n["id"]: n["status"] for n in data["nodes"]
            if n["type"] == "VerificationMethod"} == {
        "VM1": "done", "VM2": "planned", "VM3": "failed"}, "файл не тронут"
    assert out.exists()


def test_a_claimed_plan_step_is_drawn_as_running_at_once(tmp_path, monkeypatch):
    """Шаг, не подкреплённый модулем экспериментов, до сих пор проходил
    «не начат» → «выполнен»: трекер идёт IN_PROGRESS → DONE внутри одного хода
    субагента, а зеркало плана работает только на тиках оркестратора. Момент,
    когда шаг действительно начинается, — это утверждение наряда на работу."""
    from CoScientist.hitl import work_order_tools as wo

    store = ResearchGraphStore(directory=str(tmp_path))
    store.init_research(source="OrchestratorAgent", question="Q?")

    state = {"_master_active_tasks": [
        {"id": "TASK-1", "title": "Собрать литературу",
         "assignee": "ResearchAgent", "status": "TODO"}]}
    from CoScientist.graph.research import store as store_mod
    monkeypatch.setattr(store_mod, "get_research_graph", lambda *a, **k: store)

    order = type("Order", (), {"plan_task_id": "TASK-1"})()
    wo._claim_plan_step(state, "ResearchAgent", order)

    steps = [n for n in store.full()["nodes"] if n["type"] == "PlanStep"]
    assert steps, "шаг плана должен появиться в графе"
    assert steps[0]["status"] == "in_progress"
    card = next(n for n in store.to_view()["nodes"] if n["id"] == steps[0]["id"])
    assert card["status_word"] == "выполняется"
