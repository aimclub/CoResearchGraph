"""Regressions found while rechecking the semantic requirement chain."""
import pytest

from CoScientist.requirements.routing import normalize_statement, merge_statements
from CoScientist.requirements.completion import evaluate_fulfillment


def _part(kind, quote, **extra):
    return {"kind": kind, "formulation": quote, "source": "user_request", "quote": quote, **extra}


def test_merge_preserves_two_hypotheses_sharing_a_quote():
    quote = "Предложи две гипотезы о причине эффекта"
    draft = {"parts": [
        _part("hypothesis", quote, formulation="Причина A", hypothesis_mode="formulate"),
        _part("hypothesis", quote, formulation="Причина B", hypothesis_mode="formulate"),
    ]}
    previous = normalize_statement(quote, draft)
    merged = merge_statements(previous, normalize_statement(quote, draft))
    assert [(p.id, p.formulation) for p in merged.parts] == [(p.id, p.formulation) for p in previous.parts]
    silence = merge_statements(previous, normalize_statement(quote, {"parts": []}))
    assert {p.id for p in silence.obligations()} == {p.id for p in previous.obligations()}


def test_global_volume_does_not_apply_to_a_separate_deliverable():
    source = "Нужны 100 кандидатов. Пришли пояснение."
    statement = normalize_statement(source, {"parts": [
        _part("deliverable", "Нужны 100 кандидатов", id="candidates"),
        _part("deliverable", "Пришли пояснение", id="explanation", criteria=[
            {"id": "document", "text": "пояснение", "origin": "user", "quote": "Пришли пояснение",
             "check": "contract", "evidence": "artifact"},
        ]),
    ], "volume": {"requested": 100, "requested_quote": "Нужны 100 кандидатов"}})
    done = evaluate_fulfillment(statement, [
        {"part_id": "candidates", "produced": 100, "artifacts": [{"inline": True, "content": {"rows": [1]}}]},
        {"part_id": "explanation", "artifacts": [{"inline": True, "content": {"rows": [1]}}], "criteria_checks": {"document": True}},
    ])
    assert done.fulfillment == "fulfilled"


def test_a_new_ungrounded_answer_cannot_borrow_old_grounds():
    source = "Каково среднее?"
    statement = normalize_statement(source, {"parts": [_part("question", source, id="q")]})
    done = evaluate_fulfillment(statement, [
        {"part_id": "q", "answer": "4", "grounded": True, "executed": True},
        {"part_id": "q", "answer": "100", "grounded": False, "executed": True},
    ])
    assert done.fulfillment == "unfulfilled"


def test_ungrounded_verdict_cannot_borrow_grounds_from_a_text_answer():
    source = "Проверь гипотезу"
    statement = normalize_statement(source, {"parts": [_part("hypothesis", source, id="h", hypothesis_mode="check")]})
    done = evaluate_fulfillment(statement, [
        {"part_id": "h", "answer": "данные получены", "grounded": True, "executed": True},
        {"part_id": "h", "verdict": "confirmed", "grounded": False, "executed": True},
    ])
    assert done.fulfillment == "unfulfilled"


def test_a_partial_catalog_update_keeps_the_requested_volume():
    source = "Нужны 100 кандидатов. Каков метод?"
    previous = normalize_statement(source, {"parts": [
        _part("deliverable", "Нужны 100 кандидатов", id="candidates"),
    ], "volume": {"requested": 100, "requested_quote": "Нужны 100 кандидатов"}})
    added = normalize_statement(source, {"parts": [_part("question", "Каков метод?", id="method")]})
    merged = merge_statements(previous, added)
    assert merged.volume.requested == 100
    assert merged.volume.requested_quote == "Нужны 100 кандидатов"
    assert {p.id for p in merged.obligations()} == {"candidates", "method"}


def test_plan_projection_does_not_display_borrowed_grounds():
    from CoScientist.requirements.coverage import project_requirements
    source = "Каково среднее?"
    statement = normalize_statement(source, {"parts": [_part("question", source, id="q")]})
    rows = project_requirements(
        [{"id": "task", "design": {"target_refs": ["q"],
         "target_links": [{"requirement_id": "q", "role": "delivers"}]}}],
        [p.model_dump() for p in statement.parts],
        [{"task_id": "task", "status": "success", "answer": "4", "grounded": True},
         {"task_id": "task", "status": "success", "answer": "100", "grounded": False}],
    )
    assert rows[0]["status"] == "open"
    assert rows[0]["grounds"] == ""


@pytest.mark.parametrize("grounded,answer_grounded,fulfilled", [
    (True, None, True), (False, None, False),
    (True, False, False), (False, True, True),
])
@pytest.mark.parametrize("canonical", [True, False])
def test_answer_grounding_survives_serialization_and_matches_projection(grounded, answer_grounded, fulfilled, canonical):
    from CoScientist.requirements.models import OutcomeRecord
    from CoScientist.requirements.coverage import project_requirements

    source = "Каково среднее?"
    statement = normalize_statement(source, {"parts": [_part("question", source, id="q")]})
    payload = {"part_id": "q", "executed": True, "answer": "42", "grounded": grounded}
    if answer_grounded is not None:
        payload["answer_grounded"] = answer_grounded
    outcome = OutcomeRecord.model_validate_json(OutcomeRecord(**payload).model_dump_json())
    done = evaluate_fulfillment(statement, [outcome])
    rows = project_requirements(
        [{"id": "task", "design": {"target_refs": ["q"]}}],
        [p.model_dump() for p in statement.parts] if canonical else [{"id": "q", "kind": "question"}],
        [{**outcome.model_dump(), "task_id": "task", "status": "success",
          "summary": "Среднее — 42", "artifacts": [{"inline": True, "content": {"rows": [1]}}]}],
    )
    assert (done.fulfillment == "fulfilled") is fulfilled
    assert (rows[0]["status"] == "met") is fulfilled
    assert bool(rows[0]["grounds"]) is fulfilled


def test_graph_answer_with_explicit_false_is_not_grounded_by_its_text():
    from CoScientist.requirements.completion import evaluate_nodes

    done = evaluate_nodes([{"id": "q", "type": "ResearchQuestion", "status": "closed",
                            "attrs": {"obligation": True, "answer": "42", "answer_grounded": False}}])
    assert done.fulfillment == "unfulfilled"


def test_shared_quote_reformulation_keeps_ids_and_retirement_history():
    from CoScientist.requirements.routing import retire_parts

    quote = "Предложи две гипотезы о причине эффекта"
    previous = normalize_statement(quote, {"parts": [
        _part("hypothesis", quote, id="a", formulation="Причина A", hypothesis_mode="formulate"),
        _part("hypothesis", quote, id="b", formulation="Причина B", hypothesis_mode="formulate"),
    ]})
    changed = normalize_statement(quote, {"parts": [
        _part("hypothesis", quote, id="b", formulation="Уточнение B", hypothesis_mode="formulate"),
    ]})
    merged = merge_statements(previous, changed)
    assert {p.id: p.formulation for p in merged.parts} == {"a": "Причина A", "b": "Уточнение B"}
    assert merged.parts[0].previous_formulation == "Причина B"
    retired = retire_parts(merged, ["a"])
    merged = merge_statements(retired, changed)
    assert {p.id for p in merged.obligations()} == {"b"}
    assert next(p for p in merged.parts if p.id == "a").retired


def test_explicit_volume_update_replaces_the_previous_request():
    previous = normalize_statement("Нужны 100 строк", {"parts": [_part("deliverable", "Нужны 100 строк")],
        "volume": {"requested": 100, "requested_quote": "Нужны 100 строк"}})
    new = normalize_statement("Нужны 20 строк", {"parts": [_part("deliverable", "Нужны 20 строк")],
        "volume": {"requested": 20, "requested_quote": "Нужны 20 строк"}})
    merged = merge_statements(previous, new, retire_ids=[previous.parts[0].id])
    assert merged.volume.requested == 20
    assert merged.volume.requested_quote == "Нужны 20 строк"
