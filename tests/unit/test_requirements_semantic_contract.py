"""The normalizer validates a semantic contract without classifying wording."""
import pytest

from CoScientist.requirements.completion import evaluate_fulfillment, evaluate_nodes
from CoScientist.requirements.coverage import project_requirements, projection_from_state
from CoScientist.requirements.graph_io import commit_statement
from CoScientist.requirements.routing import normalize_statement
from CoScientist.graph.research.store import ResearchGraphStore


def part(kind, quote, **extra):
    return {"id": "H-test" if kind == "hypothesis" else "DL-test",
            "kind": kind, "formulation": quote, "source": "user_request",
            "quote": quote, **extra}


@pytest.mark.parametrize("quote,kind,extra", [
    ("Нужен архив измерений", "deliverable", {}),
    ("Je voudrais les données nettoyées", "deliverable", {}),
    ("¿Cuál es la media?", "question", {}),
    ("Установи, соответствует ли наблюдение предположению", "hypothesis", {"hypothesis_mode": "check"}),
    ("Нужны возможные объяснения наблюдаемого эффекта", "hypothesis", {"hypothesis_mode": "formulate"}),
])
def test_semantics_do_not_depend_on_a_verb_list(quote, kind, extra):
    statement = normalize_statement("Контекст. " + quote + ".", {"parts": [part(kind, quote, **extra)]})
    assert not statement.uncertain
    assert statement.parts[0].kind == kind
    assert statement.parts[0].hypothesis_mode == extra.get("hypothesis_mode")


def test_missing_mode_does_not_silently_choose_check_or_drop_catalog_uncertainty():
    source = "Проверь гипотезу. Нужен архив."
    statement = normalize_statement(source, {"parts": [
        part("hypothesis", "Проверь гипотезу"), part("deliverable", "Нужен архив"),
    ]})
    assert statement.uncertain
    assert len(statement.parts) == 1
    assert evaluate_fulfillment(statement, [{"part_id": "DL-test", "artifacts": [{"inline": True, "content": {"rows": [1]}}]}]).fulfillment == "unfulfilled"


def test_explicit_uncertainty_survives_a_valid_part():
    statement = normalize_statement("Нужен архив", {"parts": [part("deliverable", "Нужен архив")], "uncertainty": "another obligation is unclear"})
    assert statement.uncertain


@pytest.mark.parametrize("extra", [
    {"physical_sample": "false"}, {"physical_sample": True, "protocol_only": True},
    {"hypothesis_mode": "check"}, {"role": "invented"},
])
def test_invalid_semantic_fields_are_not_coerced_into_success(extra):
    statement = normalize_statement("Нужен архив", {"parts": [part("deliverable", "Нужен архив", **extra)]})
    assert statement.uncertain and not statement.parts


def test_result_properties_are_explicit_even_when_words_look_like_a_protocol():
    quote = "Доставь образец согласно протоколу"
    statement = normalize_statement(quote, {"parts": [part("deliverable", quote, physical_sample=True)]})
    assert statement.parts[0].physical_sample
    assert not statement.parts[0].protocol_only


def test_counts_conditions_and_generated_volume_are_not_inferred():
    source = "Generate 100 rows. If N > 30, run exactly 15."
    bare = normalize_statement(source, {"parts": [part("deliverable", source)]})
    assert bare.volume.requested is None and bare.volume.run_cap is None
    assert not bare.conditions
    parsed = normalize_statement(source, {"parts": [part("deliverable", "Generate 100 rows")],
        "volume": {"requested": 100, "requested_quote": "Generate 100 rows",
                   "run_cap": 15, "run_cap_quote": "If N > 30, run exactly 15", "produced": 15},
        "conditions": [{"text": "cap conditional on N > 30", "quote": "If N > 30, run exactly 15"}]})
    assert not parsed.uncertain
    assert parsed.volume.requested == 100 and parsed.volume.run_cap == 15
    assert parsed.volume.produced is None
    assert parsed.conditions[0].quote == "If N > 30, run exactly 15"


@pytest.mark.parametrize("count,quote", [(10, "100 rows"), (True, "100 rows"), (100, "not present"), (-1, "100 rows")])
def test_count_requires_a_matching_source_number(count, quote):
    statement = normalize_statement("100 rows", {"parts": [part("deliverable", "100 rows")],
        "volume": {"requested": count, "requested_quote": quote}})
    assert statement.uncertain and statement.volume.requested is None


def test_invalid_provenance_cannot_be_self_accepted():
    statement = normalize_statement("Нужен архив", {"parts": [
        part("deliverable", "not present", accepted=True),
    ]})
    assert statement.uncertain and not statement.obligations()


def test_frame_field_must_be_accepted_and_match_its_own_quote():
    draft = {"parts": [part("deliverable", "Архив", source="frame_field", field_id="result")]}
    assert not normalize_statement("q", draft, frame_fields={"result": {"value": "Архив", "accepted": True}}).uncertain
    assert normalize_statement("q", draft, frame_fields={"result": {"value": "Архив", "accepted": False}}).uncertain


def test_graph_roundtrip_retains_the_canonical_volume(tmp_path):
    source = "Нужен архив на 100 строк"
    statement = normalize_statement(source, {"parts": [part("deliverable", source)],
        "volume": {"requested": 100, "requested_quote": source}})
    store = ResearchGraphStore(directory=str(tmp_path))
    assert commit_statement(store, statement)["ok"]
    full = ResearchGraphStore(directory=str(tmp_path)).full()
    result = {"part_id": "DL-test", "produced": 15, "artifacts": [{"inline": True, "content": {"rows": [1]}}]}
    assert evaluate_nodes(full["nodes"], full["edges"], outcomes=[result]) == evaluate_fulfillment(statement, [result])
    assert evaluate_nodes(full["nodes"], full["edges"], outcomes=[result]).fulfillment == "partial"


@pytest.mark.parametrize("mode,outputs", [("formulate", {}), ("check", {"verdict": "refuted"})])
def test_plan_uses_the_same_assessor_as_completion(mode, outputs):
    source = "Нужны возможные объяснения эффекта"
    statement = normalize_statement(source, {"parts": [part("hypothesis", source, hypothesis_mode=mode)]})
    refs = [{**p.model_dump(), "volume": statement.volume.model_dump()} for p in statement.parts]
    rows = project_requirements([{"id": "EXP-1", "design": {"target_refs": ["H-test"], "target_links": [{"requirement_id": "H-test", "role": "delivers"}]}}], refs,
        [{"task_id": "EXP-1", "status": "success", "summary": "Объяснение", "outputs": outputs,
          "criteria_checks": [{"passed": True, "details": "observations recorded"}]}])
    done = evaluate_fulfillment(statement, [{"part_id": "H-test", "executed": True,
        "answer": "Объяснение", "grounded": True, "verdict": outputs.get("verdict", "")}])
    assert rows[0]["status"] == "met" and done.fulfillment == "fulfilled"


def test_state_projection_prefers_canonical_catalog_to_stale_context():
    statement = normalize_statement("Нужен архив", {"parts": [part("deliverable", "Нужен архив")]})
    rows = projection_from_state({"normalized_statement": statement.model_dump(),
        "experiment_context": {"requirement_refs": [{"id": "old", "kind": "question"}]}})
    assert [row["id"] for row in rows] == ["DL-test"]


def test_runtime_mirrors_plan_assessment_for_graph_and_report(tmp_path):
    from CoScientist.experiments.runtime.graph_bridge import _mirror_requirement_assessment
    source = "Нужен архив на 100 строк"
    statement = normalize_statement(source, {"parts": [part("deliverable", source)],
        "volume": {"requested": 100, "requested_quote": source}})
    store = ResearchGraphStore(directory=str(tmp_path))
    commit_statement(store, statement)
    state = {"normalized_statement": statement.model_dump(), "experiment_runtime": {
        "plan": {"tasks": [{"id": "EXP-1", "design": {"target_refs": ["DL-test"], "target_links": [{"requirement_id": "DL-test", "role": "delivers"}]}}]},
        "results": [{"task_id": "EXP-1", "status": "success", "summary": "15 rows",
                     "outputs": {"produced": 15}, "artifacts": [{"inline": True, "content": {"rows": [1]}}]}],
    }}
    _mirror_requirement_assessment(store, state)
    assert projection_from_state(state)[0]["status"] == "partial"
    full = ResearchGraphStore(directory=str(tmp_path)).full()
    done = evaluate_nodes(full["nodes"], full["edges"])
    assert done.fulfillment == "partial" and done.remaining[0]["reason"] == "15 of 100"
    # A repeated catalog write must not clear the assessed result.
    commit_statement(store, statement)
    assert evaluate_nodes(store.full()["nodes"]).fulfillment == "partial"
    state["experiment_runtime"]["results"][-1]["outputs"]["produced"] = 100
    _mirror_requirement_assessment(store, state)
    assert evaluate_nodes(store.full()["nodes"]).fulfillment == "fulfilled"


def test_uncertain_catalog_is_persisted_without_becoming_legacy_success(tmp_path):
    statement = normalize_statement("Нужен архив", None)
    store = ResearchGraphStore(directory=str(tmp_path))
    assert commit_statement(store, statement)["ok"]
    done = evaluate_nodes(ResearchGraphStore(directory=str(tmp_path)).full()["nodes"])
    assert done.fulfillment == "unfulfilled" and not done.legacy


def test_plan_schema_keeps_explicit_closing_links():
    from CoScientist.experiments.schemas.models import TaskDesign
    from CoScientist.requirements.coverage import closing_task_ids
    design = TaskDesign.model_validate({"target_links": [
        {"requirement_id": "DL-test", "role": "delivers"},
    ]})
    assert design.target_refs == ["DL-test"]
    task = {"id": "EXP-1", "design": design.model_dump()}
    assert closing_task_ids([task], "DL-test") == ["EXP-1"]
    task["design"]["target_links"][0]["role"] = "supports"
    assert closing_task_ids([task], "DL-test") == []


def test_explicit_closing_links_work_alongside_legacy_dependency_inference():
    from CoScientist.requirements.coverage import closing_task_ids
    prepare = {"id": "prepare", "design": {"target_refs": ["q"]}}
    deliver = {"id": "deliver", "depends_on": ["prepare"], "design": {
        "target_refs": ["q"], "target_links": [{"requirement_id": "q", "role": "delivers"}],
    }}
    support = {"id": "support", "depends_on": ["deliver"], "design": {
        "target_refs": ["q"], "target_links": [{"requirement_id": "q", "role": "supports"}],
    }}
    assert closing_task_ids([prepare, deliver, support], "q") == ["deliver"]


def test_ungrounded_answer_stays_open_after_outcome_aggregation():
    statement = normalize_statement("Среднее?", {"parts": [part("question", "Среднее?")]})
    done = evaluate_fulfillment(statement, [{"part_id": "DL-test", "executed": True,
        "answer": "42", "grounded": False}])
    assert done.fulfillment == "unfulfilled"


def test_agent_method_does_not_become_a_user_obligation_from_boolean_flag():
    source = 'Prepare a protocol'
    statement = normalize_statement(source, {'parts': [part('deliverable', source, criteria=[{
        'id': 'method', 'text': 'Use a proposed assay', 'origin': 'agent_method', 'obligation': True,
    }])]})
    criterion = statement.parts[0].criteria[0]
    assert criterion.origin == 'agent_method' and not criterion.obligation


def test_artifact_count_is_not_a_measurement_of_delivered_quantity():
    source = 'Generate 100 rows'
    statement = normalize_statement(source, {'parts': [part('deliverable', source)],
        'volume': {'requested': 100, 'requested_quote': source}})
    result = {'part_id': 'DL-test', 'artifacts': [{'inline': True, 'content': {'rows': [1]}} for _ in range(100)]}
    assert evaluate_fulfillment(statement, [result]).fulfillment == 'partial'


def test_duplicate_criterion_ids_make_references_ambiguous():
    source = 'Prepare a protocol'
    statement = normalize_statement(source, {'parts': [part('deliverable', source, criteria=[
        {'id': 'same', 'text': text, 'origin': 'user', 'quote': source} for text in ('Format', 'Contents')
    ])]})
    assert statement.uncertain
