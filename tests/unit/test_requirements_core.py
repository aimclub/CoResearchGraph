"""Acceptance cases for the normalized statement. A01–A20 and the checks
that do not need a live model or a live service."""
from __future__ import annotations

from CoScientist.graph.research.store import ResearchGraphStore
from CoScientist.requirements.completion import evaluate_fulfillment
from CoScientist.requirements.execution import (
    is_infrastructure_failure,
    service_block_reason,
    step_contract,
)
from CoScientist.requirements.graph_io import commit_statement
from CoScientist.requirements.routing import (
    add_working_hypothesis,
    merge_statements,
    normalize_statement,
)

S_LIMIT = (
    "Сгенерируй молекулы, которые могут связываться с белком GSK-3beta. "
    "Если количество таких молекул больше 30, то останови генерацию и "
    "выполни ровно 15 симуляций молекулярного докинга на лучших кандидатах, "
    "ранжированных по скору связывания, и представь результаты."
)


def _part(kind, formulation, quote, *, role="user_obligation", source="user_request",
          criteria=None, volume=None, **semantics):
    row = {
        "kind": kind,
        "formulation": formulation,
        "role": role,
        "source": source,
        "quote": quote,
    }
    if criteria:
        row["criteria"] = criteria
    if volume:
        row["volume"] = volume
    row.update(semantics)
    return row


def test_a01_arithmetic_is_one_question():
    source = "Сколько будет 2+2?"
    statement = normalize_statement(source, {"parts": [
        _part("question", "Сколько будет 2+2?", "Сколько будет 2+2?"),
    ]})
    assert [p.kind for p in statement.parts] == ["question"]
    assert statement.root_mode == "question"
    assert statement.uncertain is False


def test_a02_format_is_a_criterion_not_an_obligation():
    source = "Сколько будет 2+2? Ответ выведи в виде таблицы CSV."
    statement = normalize_statement(source, {"parts": [
        _part("question", "Сколько будет 2+2?", "Сколько будет 2+2?",
              criteria=[{"text": "CSV", "origin": "user", "quote": "Ответ выведи в виде таблицы CSV."}]),
    ]})
    assert len(statement.parts) == 1
    assert any("CSV" in c.text for c in statement.parts[0].criteria)


def test_a03_failed_parse_is_not_fulfilled():
    statement = normalize_statement("Сколько будет 2+2?", {"parts": []})
    assert statement.uncertain
    done = evaluate_fulfillment(statement, [])
    assert done.fulfillment == "unfulfilled"
    assert done.report_ready is True


def test_a04_refuted_check_is_fulfilled():
    source = "Проверь гипотезу, что молекула X связывается с белком Y сильнее, чем Z."
    statement = normalize_statement(source, {"parts": [
        _part("hypothesis", "Молекула X связывается с белком Y сильнее, чем Z.", source, hypothesis_mode="check"),
    ]})
    assert statement.parts[0].hypothesis_mode == "check"
    done = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "executed": True,
        "verdict": "refuted",
        "grounded": True,
    }])
    assert done.fulfillment == "fulfilled"


def test_a05_inconclusive_does_not_close_a_definite_answer():
    source = "Сколько будет 2+2?"
    statement = normalize_statement(source, {"parts": [
        _part("question", source, source),
    ]})
    done = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "executed": True,
        "verdict": "inconclusive",
        "grounded": True,
        "answer": "не удалось решить",
    }])
    assert done.fulfillment == "unfulfilled"


def test_a06_service_error_is_not_a_refutation():
    assert is_infrastructure_failure({"status": "failure", "error_code": "service_unavailable"})
    source = "Проверь гипотезу, что X связывается с Y."
    statement = normalize_statement(source, {"parts": [_part("hypothesis", source, source, hypothesis_mode="check")]})
    done = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "executed": True,
        "verdict": "refuted",
        "grounded": True,
        "infrastructure_error": True,
    }])
    assert done.fulfillment == "unfulfilled"
    assert "service failed" in done.remaining[0]["reason"]


def test_graph_infrastructure_status_uses_the_explicit_flag():
    from CoScientist.requirements.completion import evaluate_nodes

    node = {"id": "H-1", "type": "Hypothesis", "status": "refuted",
            "attrs": {"obligation": True, "formulation": "X binds Y",
                      "failure_reason": "service comparison supports the refutation"}}
    assert evaluate_nodes([node]).fulfillment == "fulfilled"
    node["attrs"]["infrastructure_error"] = True
    assert evaluate_nodes([node]).fulfillment == "unfulfilled"


def test_a07_and_a08_a_file_is_not_a_table_and_a_missing_file_fails():
    source = "Собери таблицу кандидатов в CSV."
    statement = normalize_statement(source, {"parts": [
        _part("deliverable", "таблица кандидатов в CSV", source),
    ]})
    pointer = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "status": "delivered",
        "grounded": True,
        "artifacts": [{"kind": "pointer"}],
    }])
    assert pointer.fulfillment == "unfulfilled"
    missing = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "status": "delivered",
        "grounded": True,
        "artifacts": [{"kind": "table", "path": "/no/such/table.csv", "inline": False}],
    }])
    assert missing.fulfillment == "unfulfilled"


def test_a09_formulations_do_not_require_confirmation():
    source = "Предложи 3 гипотезы о причине эффекта."
    statement = normalize_statement(source, {"parts": [
        _part("hypothesis", "Гипотеза о причине эффекта.", "Предложи 3 гипотезы о причине эффекта.", hypothesis_mode="formulate"),
    ]})
    assert statement.parts[0].hypothesis_mode == "formulate"
    done = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "executed": True,
        "answer": "Гипотеза о причине эффекта.",
        "grounded": True,
    }])
    assert done.fulfillment == "fulfilled"


def test_a10_threshold_must_be_in_the_quote():
    source = "Generate molecules which can bind to GSK-3beta protein."
    statement = normalize_statement(source, {"parts": [
        _part("deliverable", "molecules that can bind to GSK-3beta", source),
        _part(
            "deliverable", "molecules with binding score below -7",
            "bind to GSK-3beta",
            criteria=[{"text": "score < -7", "origin": "user", "threshold": -7}],
        ),
    ]})
    user_thresholds = [
        c.threshold for p in statement.parts for c in p.criteria
        if c.origin == "user" and c.threshold == "-7"
    ]
    assert user_thresholds == []


def test_a12_a13_a14_volume_and_condition_are_separate():
    generated = normalize_statement(
        "Generate molecules which can bind to GSK-3beta protein.",
        {"parts": [
            _part("deliverable", "molecules that can bind to GSK-3beta",
                  "Generate molecules which can bind to GSK-3beta protein."),
        ]},
    )
    assert [p.kind for p in generated.parts] == ["deliverable"]
    assert generated.volume.requested is None

    limited = normalize_statement(S_LIMIT, {"parts": [
        _part("deliverable", "молекулы, которые могут связываться с GSK-3beta",
              "Сгенерируй молекулы, которые могут связываться с белком GSK-3beta."),
    ], "conditions": [{"text": S_LIMIT.split("Если")[1], "quote": S_LIMIT}],
        "volume": {"run_cap": 15, "run_cap_quote": "выполни ровно 15", "limit_scope": "generation"}})
    assert len(limited.conditions) == 1
    assert limited.volume.run_cap == 15
    assert limited.volume.requested is None
    assert all("больше 30" not in p.formulation for p in limited.parts)

    counted = normalize_statement(
        "Generate 100 candidate molecules. If there are more than 30, run exactly 15 dockings.",
        {"parts": [_part("deliverable", "100 candidate molecules", "Generate 100 candidate molecules.")],
         "volume": {"requested": 100, "requested_quote": "Generate 100 candidate molecules.",
                    "run_cap": 15, "run_cap_quote": "run exactly 15 dockings."}},
    )
    assert counted.volume.requested == 100
    assert counted.volume.run_cap == 15


def test_a15_a_provided_set_is_not_shrunk_by_a_run_cap():
    source = (
        "Обработай предоставленный набор из 40 объектов. "
        "Если их больше 30, выполни ровно 15."
    )
    statement = normalize_statement(source, {"parts": [
        _part("deliverable", "обработанный набор из 40 объектов",
              "Обработай предоставленный набор из 40 объектов."),
    ], "volume": {"input_count": 40, "input_count_quote": "набор из 40 объектов",
                   "run_cap": 15, "run_cap_quote": "выполни ровно 15", "limit_scope": "generation"}})
    assert statement.volume.input_count == 40
    assert statement.volume.limit_scope == "generation"
    assert statement.volume.run_cap == 15


def test_a16_and_a17_and_a20_kind_of_result():
    physical = normalize_statement(
        "Synthesize a novel inhibitor and measure its cellular specificity.",
        {"parts": [_part(
            "deliverable", "a novel inhibitor and its cellular specificity",
            "Synthesize a novel inhibitor and measure its cellular specificity.",
            physical_sample=True,
        )]},
    )
    assert physical.parts[0].physical_sample is True
    smiles = evaluate_fulfillment(physical, [{
        "part_id": physical.parts[0].id,
        "executed": True,
        "grounded": True,
        "artifacts": [{"kind": "table", "inline": True, "content": {"rows": [1]}, "text": "CCO"}],
    }])
    assert smiles.fulfillment == "unfulfilled"

    table = normalize_statement(
        "Собери таблицу кандидатов: CSV, минимум на 100 строк, колонки id и score.",
        {"parts": [_part(
            "deliverable", "таблица кандидатов",
            "Собери таблицу кандидатов: CSV, минимум на 100 строк, колонки id и score.",
            criteria=[{"text": "минимум на 100 строк", "origin": "user", "threshold": 100}],
        )], "volume": {"requested": 100, "requested_quote": "минимум на 100 строк"}},
    )
    assert table.volume.requested == 100
    assert any(c.origin == "user" and c.threshold == "100" for c in table.parts[0].criteria)

    protocol = normalize_statement(
        "Напиши протокол проверки клеточной специфичности.",
        {"parts": [_part(
            "deliverable", "протокол проверки клеточной специфичности",
            "Напиши протокол проверки клеточной специфичности.",
            protocol_only=True,
        )]},
    )
    assert protocol.parts[0].protocol_only is True
    done = evaluate_fulfillment(protocol, [{
        "part_id": protocol.parts[0].id,
        "executed": True,
        "grounded": True,
        "artifacts": [{"kind": "document", "inline": True, "content": {"rows": [1]}, "text": "protocol"}],
    }])
    assert done.fulfillment == "fulfilled"


def test_a18_an_agent_number_is_not_a_user_obligation():
    source = "Generate molecules which can bind to X."
    statement = normalize_statement(source, {"parts": [
        _part("deliverable", "molecules which can bind to X", source),
        _part(
            "deliverable", "molecules with score below -7",
            "bind to X",
            criteria=[{"text": "score < -7", "origin": "user", "threshold": -7}],
        ),
    ]})
    assert all(c.origin != "user" or c.threshold != "-7" for p in statement.parts for c in p.criteria)


def test_a19_docking_score_does_not_prove_cellular_specificity():
    source = "Измерь клеточную специфичность ингибитора."
    statement = normalize_statement(source, {"parts": [
        _part("deliverable", "клеточная специфичность ингибитора", source, criteria=[
            {"id": "specificity", "text": "клеточная специфичность", "origin": "user",
             "quote": source, "check": "substantive", "evidence": "lab_trace"},
        ]),
    ]})
    done = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "executed": True,
        "grounded": True,
        "property_verified": False,
        "artifacts": [{"kind": "docking_score", "inline": True, "content": {"rows": [1]}, "summary": "docking score"}],
    }])
    assert done.fulfillment != "fulfilled"


def test_merge_keeps_obligations_and_working_hypotheses_are_explicit():
    source = "Сколько будет 2+2?"
    first = normalize_statement(source, {"parts": [_part("question", source, source)]})
    silent = normalize_statement(source, {"parts": [_part("question", source, source)]})
    merged = merge_statements(first, silent)
    assert merged.parts[0].id == first.parts[0].id
    again = normalize_statement(source, {"parts": [_part("question", source, source)]})
    assert again.parts[0].id == first.parts[0].id
    with_h = add_working_hypothesis(
        first, "Промежуточное предположение для следующего шага.",
        because="нужно, чтобы выбрать следующий шаг",
    )
    assert [p.kind for p in with_h.parts] == ["question", "hypothesis"]
    assert with_h.parts[1].obligation is False


def test_graph_commit_is_idempotent_and_optional_when_the_graph_is_off():
    statement = normalize_statement("Сколько будет 2+2?", {"parts": [
        _part("question", "Сколько будет 2+2?", "Сколько будет 2+2?"),
    ]})
    missing = commit_statement(None, statement)
    assert missing["persisted"] is False
    assert missing["node_ids"] == {}


def test_graph_roundtrip(tmp_path):
    statement = normalize_statement(
        "Сколько будет 2+2? И собери таблицу ответа в CSV.",
        {"parts": [
            _part("question", "Сколько будет 2+2?", "Сколько будет 2+2?"),
            _part("deliverable", "таблица ответа в CSV", "собери таблицу ответа в CSV"),
        ]},
    )
    store = ResearchGraphStore(directory=str(tmp_path))
    first = commit_statement(store, statement)
    second = commit_statement(store, statement)
    assert first["persisted"] and second["persisted"]
    assert first["node_ids"] == second["node_ids"]
    reloaded = ResearchGraphStore(directory=str(tmp_path))
    kinds = {n["type"] for n in reloaded.full()["nodes"]}
    assert "Deliverable" in kinds
    assert "Hypothesis" not in kinds


def test_step_identity_ignores_name_and_route():
    one = {
        "id": "EXP-1", "name": "first name", "route": "fedot_mas",
        "design": {"operation_ref": "OP-1", "step_key": "screen", "target_refs": ["DL-1"]},
        "launch_params": {"smiles": "CCO"},
        "expected_artifacts": [{"name": "out.csv", "role": "data", "media_type": "text/csv"}],
        "success_criteria": [{"criterion_id": "C1", "description": "rows present"}],
    }
    renamed = {**one, "name": "other name", "route": "coder"}
    assert step_contract(one["design"], one) == step_contract(renamed["design"], renamed)
    other = {
        **one,
        "design": {"operation_ref": "OP-1", "step_key": "dock", "target_refs": ["DL-1"]},
    }
    assert step_contract(one["design"], one) != step_contract(other["design"], other)


def test_service_stop_blocks_another_task_on_the_same_server():
    from types import SimpleNamespace

    state: dict = {}
    idle = SimpleNamespace(id="EXP-2", mcp_servers=[])
    assert service_block_reason(state, idle) == ""
    state["experiment_service_stops"] = {
        "chem": {"task_id": "EXP-1", "reason": "service_unavailable"},
    }
    other = SimpleNamespace(
        id="EXP-2",
        mcp_servers=[SimpleNamespace(server_id="chem", name="")],
    )
    assert service_block_reason(state, other)
    origin = SimpleNamespace(
        id="EXP-1",
        mcp_servers=[SimpleNamespace(server_id="chem", name="")],
    )
    assert service_block_reason(state, origin) == ""


def test_plan_stop_diagnosis_names_the_issue():
    import pytest

    try:
        from CoScientist.experiments.review import _diagnose_plan_stop
    except (ImportError, SyntaxError) as exc:
        pytest.skip(f"plan review imports the 3.12 runtime: {exc}")

    diagnosis = _diagnose_plan_stop("max_plan_revisions", [{
        "severity": "major",
        "category": "coverage",
        "message": "user obligation DL-1 has no covering task",
    }])
    assert diagnosis["blamed_complexity"] is False
    assert "DL-1" in diagnosis["primary_issue"]
