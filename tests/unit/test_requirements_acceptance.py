"""Exact acceptance texts from the handoff, fed as drafts the way the model would.

The production prompt is not called here. These tests check faithful semantic
drafts and deterministic provenance validation. Parser accuracy needs model evals.
"""
from __future__ import annotations

from CoScientist.requirements.completion import evaluate_fulfillment, evaluate_nodes
from CoScientist.requirements.routing import merge_statements, normalize_statement, retire_parts

RUN_LIMIT = (
    "ОГРАНИЧЕНИЕ ПРОГОНА: если шаг сам порождает молекулы, конформеры, "
    "траектории, кандидатов или докинги и запрос требует больше 30, "
    "выполни ровно 15 и явно напиши, что это 15 из запрошенных N. "
    "Уже заданный входной набор не урезай. "
    "Важнее верная цепочка шагов, чем полный объём."
)
S = "Generate molecules which can bind to GSK-3beta protein."
C2 = "Synthesize a novel tyrosine hydroxylase activator with cellular specificity."


def _p(kind, formulation, quote, **extra):
    row = {
        "kind": kind,
        "formulation": formulation,
        "role": extra.pop("role", "user_obligation"),
        "source": extra.pop("source", "user_request"),
        "quote": quote,
    }
    row.update(extra)
    return row


def _kinds(statement):
    return [p.kind for p in statement.parts if p.obligation and not p.retired]


def test_a01_a02_direct_computation_and_format():
    asked = normalize_statement("Сколько будет 2+2?", {"parts": [
        _p("question", "Сколько будет 2+2?", "Сколько будет 2+2?"),
    ]})
    computed = normalize_statement("Вычисли 2+2", {"parts": [
        _p("question", "Вычисли 2+2", "Вычисли 2+2"),
    ]})
    formatted = normalize_statement("Вычисли 2+2. Ответь только числом.", {"parts": [
        _p("question", "Вычисли 2+2", "Вычисли 2+2", criteria=[
            {"text": "только числом", "origin": "user", "quote": "Ответь только числом."},
        ]),
    ]})
    assert _kinds(asked) == ["question"]
    assert _kinds(computed) == ["question"]
    assert _kinds(formatted) == ["question"]
    assert any("числом" in c.text for c in formatted.parts[0].criteria)


def test_a03_a11_explanation_does_not_adopt_the_quoted_claim():
    child = normalize_statement("Объясни ребёнку, почему 2+2=4.", {"parts": [
        _p("question", "Объясни ребёнку, почему 2+2=4.", "Объясни ребёнку, почему 2+2=4."),
    ]})
    quoted = normalize_statement("Объясни смысл фразы «Допустим, что 2+2=5».", {"parts": [
        _p("question", "Объясни смысл фразы «Допустим, что 2+2=5».",
           "Объясни смысл фразы «Допустим, что 2+2=5»."),
    ]})
    assert _kinds(child) == ["question"]
    assert _kinds(quoted) == ["question"]


def test_a04_a08_a09_hypothesis_modes():
    check = normalize_statement("Проверь утверждение, что 2+2=5.", {"parts": [
        _p("hypothesis", "2+2=5", "Проверь утверждение, что 2+2=5.", hypothesis_mode="check"),
    ]})
    done = evaluate_fulfillment(check, [{
        "part_id": check.parts[0].id, "executed": True, "verdict": "refuted", "grounded": True,
    }])
    assert check.parts[0].hypothesis_mode == "check"
    assert done.fulfillment == "fulfilled"

    science = "Проверь гипотезу, что A уменьшает ошибку относительно B на датасете D."
    one = normalize_statement(science, {"parts": [
        _p("hypothesis", science, science, hypothesis_mode="check"),
    ]})
    assert len(one.obligations()) == 1

    propose = "Предложи две проверяемые гипотезы о причинах роста ошибки. Пока не проверяй их."
    drafted = normalize_statement(propose, {"parts": [
        _p("hypothesis", "Рост ошибки вызван сдвигом распределения.", propose, hypothesis_mode="formulate"),
        _p("hypothesis", "Рост ошибки вызван утечкой признака.", propose, hypothesis_mode="formulate"),
    ]})
    assert len(drafted.obligations()) == 2
    assert all(p.hypothesis_mode == "formulate" for p in drafted.obligations())
    closed = evaluate_fulfillment(drafted, [
        {"part_id": p.id, "executed": True, "answer": p.formulation}
        for p in drafted.obligations()
    ])
    assert closed.fulfillment == "fulfilled"


def test_a05_a06_a07_a10_files_and_literature():
    mean = "Вычисли среднее по колонке temperature в приложенном CSV."
    measured = normalize_statement(mean, {"parts": [
        _p("question", mean, mean),
    ]})
    assert _kinds(measured) == ["question"]

    cleaned = "Очисти приложенную таблицу от дубликатов и верни CSV."
    table = normalize_statement(cleaned, {"parts": [
        _p("deliverable", "CSV без дубликатов", cleaned),
    ]})
    assert _kinds(table) == ["deliverable"]

    compared = "Сравни среднюю ошибку моделей A и B на этом датасете и верни CSV с результатами."
    both = normalize_statement(compared, {"parts": [
        _p("question", "Сравни среднюю ошибку моделей A и B", "Сравни среднюю ошибку моделей A и B"),
        _p("deliverable", "CSV с результатами", "верни CSV с результатами"),
    ]})
    assert _kinds(both) == ["question", "deliverable"]

    literature = "Найди исследования о влиянии температуры на выход реакции и кратко суммируй результаты."
    survey = normalize_statement(literature, {"parts": [
        _p("question", literature, literature),
    ]})
    assert _kinds(survey) == ["question"]


def test_a12_a13_s_does_not_invent_obligations():
    base = normalize_statement(S, {"parts": [
        _p("deliverable", "candidate molecules that can bind to GSK-3beta", S),
    ]})
    assert _kinds(base) == ["deliverable"]
    assert base.volume.requested is None
    assert not any(c.origin == "user" and c.threshold for p in base.parts for c in p.criteria)

    limited = normalize_statement(S + "\n" + RUN_LIMIT, {"parts": [
        _p("deliverable", "candidate molecules that can bind to GSK-3beta", S),
    ], "conditions": [{"text": RUN_LIMIT, "quote": RUN_LIMIT}],
        "volume": {"run_cap": 15, "run_cap_quote": "выполни ровно 15", "limit_scope": "generation"}})
    assert _kinds(limited) == ["deliverable"]
    assert limited.volume.requested is None
    assert limited.volume.run_cap == 15
    assert limited.conditions
    assert all("больше 30" not in p.formulation for p in limited.parts)


def test_a14_a15_volume_is_not_rewritten_by_the_run_cap():
    source = "Generate 100 candidate molecules. " + RUN_LIMIT
    statement = normalize_statement(source, {"parts": [
        _p("deliverable", "100 candidate molecules", "Generate 100 candidate molecules."),
    ], "volume": {"requested": 100, "requested_quote": "Generate 100 candidate molecules.",
                   "run_cap": 15, "run_cap_quote": "выполни ровно 15"}})
    assert statement.volume.requested == 100
    assert statement.volume.run_cap == 15
    partial = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "produced": 15,
        "artifacts": [{"kind": "table", "inline": True, "content": {"rows": [1]}}],
    }])
    assert partial.fulfillment == "partial"
    assert "15 of 100" in partial.remaining[0]["reason"]

    provided = (
        "Обработай предоставленный набор из 40 объектов. " + RUN_LIMIT
    )
    kept = normalize_statement(provided, {"parts": [
        _p("deliverable", "обработанный набор из 40 объектов",
           "Обработай предоставленный набор из 40 объектов."),
    ], "volume": {"input_count": 40, "input_count_quote": "набор из 40 объектов",
                   "run_cap": 15, "run_cap_quote": "выполни ровно 15"}})
    assert kept.volume.input_count == 40
    assert kept.volume.run_cap == 15
    assert kept.volume.requested != 15


def test_a16_c2_has_no_invented_hypotheses_or_thresholds():
    statement = normalize_statement(C2, {"parts": [
        _p("deliverable", C2, C2, physical_sample=True),
    ]})
    assert _kinds(statement) == ["deliverable"]
    assert statement.parts[0].physical_sample is True
    assert all(c.origin != "user" or c.threshold not in {"50", "95", "30", "0.4"}
               for p in statement.parts for c in p.criteria)
    smiles = evaluate_fulfillment(statement, [{
        "part_id": statement.parts[0].id,
        "artifacts": [{"kind": "table", "inline": True, "content": {"rows": [1]}, "text": "CCO"}],
    }])
    assert smiles.fulfillment != "fulfilled"


def test_a17_a18_a19_a20():
    csv = "Создай CSV минимум на 100 строк с колонками x и y."
    file_req = normalize_statement(csv, {"parts": [
        _p("deliverable", csv, csv, criteria=[
            {"text": "минимум на 100 строк", "origin": "user", "threshold": "100"},
            {"text": "колонки x и y", "origin": "user", "quote": "колонками x и y"},
        ]),
    ], "volume": {"requested": 100, "requested_quote": "минимум на 100 строк"}})
    assert file_req.volume.requested == 100
    assert any(c.origin == "user" and c.threshold == "100" for c in file_req.parts[0].criteria)
    short = evaluate_fulfillment(file_req, [{
        "part_id": file_req.parts[0].id,
        "produced": 15,
        "artifacts": [{"kind": "table", "inline": True, "content": {"rows": [1]}}],
    }])
    missing = evaluate_fulfillment(file_req, [{
        "part_id": file_req.parts[0].id,
        "artifacts": [{"kind": "table", "path": "/no/such.csv"}],
    }])
    assert short.fulfillment != "fulfilled"
    assert missing.fulfillment == "unfulfilled"

    soft = "Подбери хорошие кандидаты для связывания с X."
    picked = normalize_statement(soft, {"parts": [
        _p("deliverable", soft, soft),
        _p("deliverable", "score below -7", "связывания с X",
           criteria=[{"text": "score < -7", "origin": "user", "threshold": "-7"}]),
    ]})
    assert all(c.origin != "user" or c.threshold != "-7" for p in picked.parts for c in p.criteria)

    mixed = "Проверь, что A лучше B, и сохрани таблицу измерений."
    both = normalize_statement(mixed, {"parts": [
        _p("hypothesis", "A лучше B", "Проверь, что A лучше B", hypothesis_mode="check"),
        _p("deliverable", "таблица измерений", "сохрани таблицу измерений", criteria=[
            {"id": "table", "text": "таблица измерений", "origin": "user",
             "quote": "сохрани таблицу измерений", "check": "contract", "evidence": "artifact"},
        ]),
    ]})
    assert _kinds(both) == ["hypothesis", "deliverable"]
    finished = evaluate_fulfillment(both, [
        {"part_id": both.parts[0].id, "executed": True, "verdict": "refuted", "grounded": True},
        {"part_id": both.parts[1].id, "artifacts": [{"kind": "table", "inline": True, "content": {"rows": [1]}}], "criteria_checks": {"table": True}},
    ])
    assert finished.fulfillment == "fulfilled"

    protocol = "Напиши протокол проверки клеточной специфичности кандидата."
    doc = normalize_statement(protocol, {"parts": [_p("deliverable", protocol, protocol, protocol_only=True)]})
    assert doc.parts[0].protocol_only is True
    score = evaluate_fulfillment(doc, [{
        "part_id": doc.parts[0].id,
        "artifacts": [{"kind": "docking_score", "inline": True, "content": {"rows": [1]}}],
    }])
    written = evaluate_fulfillment(doc, [{
        "part_id": doc.parts[0].id,
        "artifacts": [{"kind": "document", "inline": True, "content": {"rows": [1]}, "text": "protocol"}],
    }])
    assert score.fulfillment != "fulfilled"
    assert written.fulfillment == "fulfilled"
    assert "not measured" in written.remaining[0]["reason"] if written.remaining else True


def test_b01_b03_b04_provenance_and_edits():
    source = "Generate molecules which can bind to X."
    rejected = normalize_statement(source, {"parts": [
        _p("hypothesis", "they bind", "bind", source="agent_proposal"),
        _p("deliverable", "molecules", "not in the source"),
    ]})
    assert rejected.uncertain
    assert rejected.obligations() == []

    first = normalize_statement(source, {"parts": [_p("deliverable", "molecules which can bind to X", source)]})
    again = normalize_statement(source, {"parts": [_p("deliverable", "molecules which can bind to X", source)]})
    assert again.parts[0].id == first.parts[0].id
    silent = merge_statements(first, normalize_statement(source, {"parts": []}))
    assert silent.obligations()[0].id == first.parts[0].id

    changed_source = "Нужно 20 вместо 100. Generate 20 candidate molecules."
    updated = normalize_statement(changed_source, {"parts": [
        _p("deliverable", "20 candidate molecules", "Generate 20 candidate molecules."),
    ]})
    merged = merge_statements(first, updated, retire_ids=[first.parts[0].id])
    live = [p for p in merged.parts if p.obligation]
    retired = [p for p in merged.parts if p.retired]
    assert len(live) == 1
    assert retired and retired[0].obligation is False
    assert live[0].previous_formulation


def test_b14_legacy_graph_is_not_rewritten_as_fulfilled():
    done = evaluate_nodes([
        {"id": "Q1", "type": "ResearchQuestion", "status": "open",
         "attrs": {"formulation": "old question"}},
        {"id": "H1", "type": "Hypothesis", "status": "formulated",
         "attrs": {"formulation": "old claim"}},
    ])
    assert done.asserted is False
    assert done.legacy is True
    assert done.fulfillment != "fulfilled"


def test_retire_parts_is_explicit():
    source = "Сколько будет 2+2?"
    statement = normalize_statement(source, {"parts": [_p("question", source, source)]})
    retired = retire_parts(statement, [statement.parts[0].id])
    assert retired.parts[0].retired is True
    assert retired.obligations() == []
