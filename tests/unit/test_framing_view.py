import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from CoScientist.context_init.commit import seed_frame
from CoScientist.context_init.models import ResearchFrame
from CoScientist.graph.research.framing_view import build_framing_details
from CoScientist.graph.research.store import ResearchGraphStore


def _set(frame: ResearchFrame, block: str, name: str, value: str,
         status: str = "предложено агентом") -> None:
    target = frame.block(block)
    assert target is not None
    field = next(item for item in target.fields if item.name == name)
    field.value = value
    field.status = status


def _frame(question: str, approach: str) -> ResearchFrame:
    frame = ResearchFrame.blank(original_request=f"Исследовать: {question}")
    _set(frame, "Вопрос исследования", "formulation", question,
         "задано заказчиком")
    _set(frame, "Основание и приёмка", "topic_name", question,
         "уточнено оператором")
    _set(frame, "Профиль исследования", "modality", approach)
    _set(frame, "Методологические нормы", "norms",
         "Фиксировать параметры и сохранять воспроизводимость")
    _set(frame, "Теоретические рамки", "frameworks",
         "Байесовская модель причинности")
    return frame


def _section(details, section_id):
    return next(section for section in details["sections"]
                if section["id"] == section_id)


def _shown(section, block, field):
    return next(item for item in section["fields"]
                if item["block_key"] == block and item["field_key"] == field)


def test_framing_view_uses_reader_facing_russian_labels():
    frame = _frame("Оценить устойчивость модели", "вычислительный")
    details = build_framing_details(
        {"research_id": "r1", "root_id": "Q1", "nodes": []},
        {"frame": frame.model_dump()},
    )

    method = _section(details, "method")
    assert _shown(method, "Профиль исследования", "modality")["label"]["ru"] \
        == "Подход к исследованию"
    assert _shown(method, "Методологические нормы", "norms")["label"]["ru"] \
        == "Правила проведения исследования"
    framework = _shown(method, "Теоретические рамки", "frameworks")
    assert framework["label"]["ru"] == "Теоретические модели и подходы"
    assert "не обязательно библиотеки ПО" in framework["help"]["ru"]
    assert details["source_mode"] == "research_frame"


def test_empty_fields_and_sections_are_omitted_from_the_sidebar():
    frame = _frame("Исследование без реквизитов", "вычислительный")
    customer = frame.block("Основание и приёмка")
    next(field for field in customer.fields if field.name == "customer").status = \
        "уточнено оператором"
    resources = frame.block("Ресурсы и бюджеты")
    gpu = next(field for field in resources.fields if field.name == "gpu_hours")
    gpu.value = "Не задано / Не задано"
    gpu.status = "уточнено оператором"

    details = build_framing_details(
        {"research_id": "r1", "root_id": "Q1", "nodes": []},
        {"frame": frame.model_dump()},
    )
    shown = [field for section in details["sections"] for field in section["fields"]]
    assert all(not field["empty"] for field in shown)
    assert all(field["field_key"] != "customer" for field in shown)
    assert all(field["field_key"] != "gpu_hours" for field in shown)
    assert all(section["id"] != "plan" for section in details["sections"])


def test_old_packed_constraints_are_recovered_without_splitting_value_semicolons():
    details = build_framing_details({
        "research_id": "legacy",
        "root_id": "Q1",
        "nodes": [
            {"id": "Q1", "type": "ResearchQuestion", "source": "human",
             "attrs": {"formulation": "Старый вопрос"}},
            {"id": "C1", "type": "Constraint", "source": "ContextInitAgent",
             "attrs": {
                 "subtype": "profile",
                 "content": "modality: вычислительный; с ручной проверкой; "
                            "target_setting: сравнительная оценка",
             }},
        ],
    })

    method = _section(details, "method")
    approach = _shown(method, "Профиль исследования", "modality")
    setting = _shown(method, "Профиль исследования", "target_setting")
    assert approach["value"] == "вычислительный; с ручной проверкой"
    assert setting["value"] == "сравнительная оценка"
    assert approach["status"] == "восстановлено из графа"
    assert details["source_mode"] == "graph_recovery"


def test_framing_snapshot_follows_its_archived_study(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    first = _frame("Первое исследование", "экспериментальный")
    assert seed_frame(store, first)["ok"]

    second = _frame("Второе исследование", "теоретический")
    result = seed_frame(store, second)
    assert result["ok"]
    archived_id = Path(result["archived"]).name

    active = store.framing_view_of("active")
    archived = store.framing_view_of(archived_id)
    assert active["title"] == "Второе исследование"
    assert archived["title"] == "Первое исследование"
    assert _shown(_section(active, "method"), "Профиль исследования", "modality")["value"] \
        == "теоретический"
    assert _shown(_section(archived, "method"), "Профиль исследования", "modality")["value"] \
        == "экспериментальный"


def test_legacy_frame_sidecar_is_not_treated_as_a_study_archive(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    frame = _frame("Старое исследование", "аналитический")
    created = store.init_research(
        source="ContextInitAgent",
        question="Старое исследование",
    )
    assert created["ok"]
    (tmp_path / "research_frame.json").write_text(
        json.dumps({
            "root_id": created["root_id"],
            "frame": frame.model_dump(),
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    details = store.framing_view_of()
    approach = _shown(
        _section(details, "method"), "Профиль исследования", "modality",
    )
    assert details["source_mode"] == "research_frame"
    assert approach["value"] == "аналитический"
    assert [study["study_id"] for study in store.studies()] == ["active"]


def test_documents_are_kept_separate_from_requirement_fields(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    assert seed_frame(store, _frame("Исследование с документом", "смешанный"))["ok"]
    committed = store.commit(
        source="ContextInitAgent",
        nodes=[{"type": "Spec", "ref": "tz", "attrs": {
            "content": "# Техническое задание",
            "name": "TZ.docx",
            "path": "https://example.test/TZ.docx",
        }}],
        edges=[{"type": "derived_from", "from": "#tz", "to": "Q1"}],
        partial_edges=True,
    )
    assert committed.ok

    details = store.framing_view_of()
    documents = _section(details, "documents")["documents"]
    assert len(documents) == 1
    assert documents[0]["title"]["ru"] == "Предварительный текст ТЗ"
    assert documents[0]["href"] == "https://example.test/TZ.docx"
    assert all(field["field_key"] != "Spec"
               for section in details["sections"] for field in section["fields"])


def test_a_result_specification_is_not_misreported_as_the_research_brief(tmp_path):
    store = ResearchGraphStore(directory=str(tmp_path))
    assert seed_frame(store, _frame("Исследование", "вычислительный"))["ok"]
    committed = store.commit(
        source="OrchestratorAgent",
        enforce_permissions=False,
        nodes=[
            {"type": "Conclusion", "ref": "conclusion",
             "attrs": {"synthesis": "Вывод"}},
            {"type": "Spec", "ref": "result_spec",
             "attrs": {"content": "Спецификация результата"}},
        ],
        edges=[{"type": "derived_from", "from": "#result_spec", "to": "#conclusion"}],
    )
    assert committed.ok
    assert _section(store.framing_view_of(), "documents")["documents"] == []


def test_framing_api_returns_the_selected_session_specification(tmp_path, monkeypatch):
    from CoScientist.graph.research import store as research_store

    monkeypatch.setattr(research_store, "_default_dir", lambda: str(tmp_path / "research"))
    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        user = client.post("/api/users", json={"nickname": f"frame-{uuid4().hex}"})
        assert user.status_code == 201
        user_id = user.json()["user"]["id"]
        session = client.post(
            f"/api/users/{user_id}/sessions", json={"title": "Specification"},
        )
        assert session.status_code == 201
        session_id = session.json()["session"]["id"]

        graph = research_store.get_research_graph(
            user_id=user_id, session_id=session_id,
        )
        assert seed_frame(graph, _frame("API-исследование", "вычислительный"))["ok"]
        response = client.get(
            f"/api/users/{user_id}/sessions/{session_id}/graph/framing",
            params={"study_id": "active"},
        )
        assert response.status_code == 200
        assert response.json()["title"] == "API-исследование"

        assert client.get(
            f"/api/users/{user_id}/sessions/{session_id}/graph/framing",
            params={"study_id": "missing.json"},
        ).status_code == 404
