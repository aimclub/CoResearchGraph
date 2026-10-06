"""Assess explicit requirement links, preserving versions and independent results."""
from copy import deepcopy

import pytest

from CoScientist.requirements.routing import normalize_statement
from CoScientist.requirements.coverage import project_requirements, projection_from_state
from CoScientist.requirements.completion import evaluate_fulfillment, evaluate_nodes
from CoScientist.requirements.graph_io import commit_statement
from CoScientist.experiments.runtime.graph_bridge import _mirror_requirement_assessment
from CoScientist.graph.research.store import ResearchGraphStore


def catalog():
    return normalize_statement("Give a table with measured values.", {"parts": [{
        "id": "result", "kind": "deliverable", "source": "user_request",
        "quote": "Give a table with measured values", "formulation": "Table",
        "criteria": [{"id": "values", "text": "Measured values", "origin": "user", "quote": "measured values"}],
    }]})


def task(tid, role="delivers", *, criterion=False):
    return {"id": tid, "design": {"target_refs": ["result"],
            "target_links": [{"requirement_id": "result", "role": role}]},
            "success_criteria": [{"criterion_id": tid + "-check", "requirement_id": "result",
                                  "requirement_criterion_id": "values"}] if criterion else []}


def result(tid, *, version=1, passed=True):
    return {"task_id": tid, "result_id": f"{tid}-v{version}", "result_version": version,
            "status": "success", "summary": "Measured table", "artifacts": [{"inline": True, "content": {"rows": [1]}}],
            "criteria_checks": [{"criterion_id": tid + "-check", "passed": passed, "details": "Measured against source"}]}


def projection(tasks, results):
    statement = catalog()
    return project_requirements(tasks, [p.model_dump() for p in statement.parts], results)[0]


def test_file_and_success_do_not_replace_a_required_criterion():
    row = projection([task("deliver")], [result("deliver")])
    assert row["status"] == "partial"
    assert row["criteria"][0]["passed"] is None
    assert "Measured values" in row["debt"]
    done = evaluate_fulfillment(catalog(), [{"part_id": "result", "artifacts": [{"inline": True, "content": {"rows": [1]}}]}])
    assert done.fulfillment == "partial"


def test_supporting_check_contributes_evidence_but_cannot_deliver():
    tasks = [task("measure", "supports", criterion=True), task("deliver")]
    assert projection(tasks, [result("measure")])["status"] == "open"
    done = projection(tasks, [result("measure"), result("deliver")])
    assert done["status"] == "met"
    assert done["closing_tasks"] == ["deliver"]
    assert done["criteria"][0]["passed"] is True


def test_every_required_closing_task_needs_its_own_current_result():
    tasks = [task("a", criterion=True), task("b")]
    partial = projection(tasks, [result("a")])
    assert partial["status"] == "partial"
    assert "b" in partial["debt"]
    assert projection(tasks, [result("a"), result("b")])["status"] == "met"
    failure = {**result("a", version=2), "status": "failure", "error_code": "service_unavailable"}
    assert projection(tasks, [failure, result("b"), result("a")])["status"] != "met"


def test_new_failed_check_cannot_borrow_old_evidence_even_if_results_reordered():
    tasks = [task("measure", "supports", criterion=True), task("deliver")]
    row = projection(tasks, [result("measure", version=2, passed=False), result("deliver"), result("measure")])
    assert row["status"] == "partial"
    assert row["criteria"][0]["passed"] is False


def test_redo_waits_for_the_new_version_without_mutating_saved_state():
    state = {"normalized_statement": catalog().model_dump(), "experiment_runtime": {
        "plan": {"tasks": [task("deliver", criterion=True)]}, "results": [result("deliver")],
        "tasks": {"deliver": {"operation_revision": 1, "status": "pending"}},
    }}
    before = deepcopy(state)
    assert projection_from_state(state)[0]["status"] == "open"
    assert state == before


def test_graph_completion_reads_the_same_criterion_assessment(tmp_path):
    statement = catalog()
    store = ResearchGraphStore(directory=str(tmp_path))
    commit_statement(store, statement)
    state = {"normalized_statement": statement.model_dump(), "experiment_runtime": {
        "plan": {"tasks": [task("deliver", criterion=True)]}, "results": [result("deliver", passed=False)],
    }}
    _mirror_requirement_assessment(store, state)
    assert evaluate_nodes(store.full()["nodes"]).fulfillment == "partial"
    assert projection_from_state(state)[0]["status"] == "partial"


def test_answer_is_not_inferred_from_a_success_summary_and_a_file():
    statement = normalize_statement("What is the mean?", {"parts": [{
        "id": "q", "kind": "question", "formulation": "Mean?", "source": "user_request", "quote": "What is the mean?",
    }]})
    tasks = [{"id": "answer", "design": {"target_refs": ["q"], "target_links": [{"requirement_id": "q", "role": "delivers"}]}}]
    records = [result("answer")]
    row = project_requirements(tasks, [p.model_dump() for p in statement.parts], records)[0]
    assert row["status"] == "open"
    records[0]["outputs"] = {"answer": "42", "answer_grounded": True}
    assert project_requirements(tasks, [p.model_dump() for p in statement.parts], records)[0]["status"] == "met"


def test_grounded_refutation_completes_but_conflicting_verdicts_do_not():
    statement = normalize_statement("Test the claim", {"parts": [{
        "id": "result", "kind": "hypothesis", "hypothesis_mode": "check",
        "source": "user_request", "quote": "Test the claim", "formulation": "The claim",
    }]})
    refs = [p.model_dump() for p in statement.parts]
    refuted = {**result("a"), "outputs": {"verdict": "refuted"}}
    assert project_requirements([task("a")], refs, [refuted])[0]["status"] == "met"
    confirmed = {**result("b"), "outputs": {"verdict": "confirmed"}}
    assert project_requirements([task("a"), task("b")], refs, [refuted, confirmed])[0]["status"] == "partial"


def test_scoped_answers_and_artifacts_cannot_be_borrowed_by_another_requirement():
    source = 'What are A and B?'
    statement = normalize_statement(source, {'parts': [
        {'id': ref, 'kind': 'question', 'formulation': ref, 'source': 'user_request', 'quote': source}
        for ref in ('a', 'b')
    ]})
    tasks = [{'id': 'both', 'design': {'target_refs': ['a', 'b'], 'target_links': [
        {'requirement_id': ref, 'role': 'delivers'} for ref in ('a', 'b')]}}]
    outcome = {**result('both'), 'answer': 'global answer', 'answer_grounded': True,
               'artifacts': [{'artifact_id': 'file-a', 'inline': True, 'content': {'rows': [1]}}],
               'outputs': {'answer': 'global answer', 'answer_grounded': True, 'requirements': {
                   'a': {'answer': 'A', 'answer_grounded': True, 'artifact_ids': ['file-a']},
                   'b': {'summary': 'B has not been determined'},
               }}}
    rows = project_requirements(tasks, [p.model_dump() for p in statement.parts], [outcome])
    assert [row['status'] for row in rows] == ['met', 'open']
    assert rows[1]['artifacts'] == [] and rows[1]['actual'] == ''


def test_scientific_check_is_only_a_verdict_for_its_addressed_hypothesis():
    source = 'Test both claims'
    statement = normalize_statement(source, {'parts': [
        {'id': ref, 'kind': 'hypothesis', 'hypothesis_mode': 'check',
         'formulation': ref, 'source': 'user_request', 'quote': source} for ref in ('a', 'b')
    ]})
    tasks = [{'id': 'both', 'design': {'target_refs': ['a', 'b']}}]
    outcome = {**result('both'), 'scientific_check': {'hypothesis_ref': 'a', 'status': 'supported'}}
    rows = project_requirements(tasks, [p.model_dump() for p in statement.parts], [outcome])
    assert [row['status'] for row in rows] == ['met', 'open']


def test_graph_hypothesis_alias_keeps_its_scientific_verdict_and_catalog_metadata():
    statement = normalize_statement('Test the claim', {'parts': [{
        'id': 'H-stable', 'kind': 'hypothesis', 'hypothesis_mode': 'check',
        'formulation': 'The original claim', 'source': 'user_request', 'quote': 'Test the claim'}]})
    outcome = {**result('check'), 'scientific_check': {'hypothesis_ref': 'H1', 'status': 'refuted'}}
    state = {'normalized_statement': statement.model_dump(),
             'experiment_context': {'hypothesis_refs': [{'id': 'H-stable', 'hypothesis_id': 'H1',
                                                        'kind': 'hypothesis', 'statement': 'Compact graph wording'}]},
             'experiment_runtime': {'plan': {'tasks': [{'id': 'check', 'design': {
                 'target_refs': ['H-stable'], 'target_links': [{'requirement_id': 'H-stable', 'role': 'delivers'}]}}]},
                                    'results': [outcome]}}
    rows = projection_from_state(state)
    assert len(rows) == 1
    assert rows[0]['status'] == 'met' and rows[0]['actual'] == 'refuted'
    assert rows[0]['formulation'] == 'The original claim' and rows[0]['obligation'] is True


def test_negative_scientific_threshold_does_not_invalidate_a_completed_check():
    source = 'Test the claim'
    statement = normalize_statement(source, {'parts': [{
        'id': 'result', 'kind': 'hypothesis', 'hypothesis_mode': 'check',
        'formulation': source, 'source': 'user_request', 'quote': source,
    }]})
    check_task = task('a')
    check_task['success_criteria'] = [{'criterion_id': 'threshold', 'purpose': 'assessment'}]
    outcome = {**result('a'), 'outputs': {'verdict': 'refuted'},
               'criteria_checks': [{'criterion_id': 'threshold', 'passed': False, 'details': 'Below the scientific threshold'}]}
    assert project_requirements([check_task], [p.model_dump() for p in statement.parts], [outcome])[0]['status'] == 'met'


def test_protocol_accepts_the_existing_runtime_report_artifact_contract():
    source = 'Prepare a protocol'
    statement = normalize_statement(source, {'parts': [{
        'id': 'result', 'kind': 'deliverable', 'protocol_only': True,
        'formulation': source, 'source': 'user_request', 'quote': source,
    }]})
    outcome = {**result('a'), 'artifacts': [{'role': 'report', 'external_url': 'https://storage.example/protocol.pdf', 'content_verified': True}]}
    assert project_requirements([task('a')], [p.model_dump() for p in statement.parts], [outcome])[0]['status'] == 'met'
    outcome['outputs'] = {'property_verified': False, 'limitation': 'Optimality is not established'}
    assessed = project_requirements([task('a')], [p.model_dump() for p in statement.parts], [outcome])[0]
    assert assessed['status'] == 'partial'


@pytest.mark.parametrize('artifact', [
    {'external_url': 'https://invalid.invalid/missing.csv'},
    {'uri': 's3://missing/table.csv'}, {'inline': True},
])
def test_unverified_location_or_inline_marker_does_not_deliver(artifact):
    statement = catalog()
    done = evaluate_fulfillment(statement, [{'part_id': 'result', 'artifacts': [artifact],
                                            'criteria_checks': {'values': True}}])
    assert done.fulfillment == 'unfulfilled'


def test_protocol_infrastructure_error_cannot_close_delivery():
    from CoScientist.requirements.completion import assess_requirement
    from CoScientist.requirements.models import OutcomeRecord
    part = catalog().parts[0].model_copy(update={'protocol_only': True, 'criteria': []})
    record = OutcomeRecord(part_id=part.id, infrastructure_error=True,
                           artifacts=[{'kind': 'document', 'inline': True, 'text': 'protocol'}])
    assert assess_requirement(part, record)[0] == 'unmet'


@pytest.mark.parametrize('second_ids,expected', [([str(i) for i in range(5, 10)], 'met'),
                                               ([str(i) for i in range(5)], 'partial'),
                                               (None, 'partial')])
def test_counted_delivery_unions_explicit_items_without_double_counting(second_ids, expected):
    source = 'Deliver 10 candidates'
    statement = normalize_statement(source, {'parts': [{
        'id': 'result', 'kind': 'deliverable', 'source': 'user_request', 'quote': source,
        'formulation': source,
    }], 'volume': {'requested': 10, 'requested_quote': '10'}})
    refs = [{**part.model_dump(), 'volume': statement.volume.model_dump()} for part in statement.parts]
    records = [{**result(tid), 'outputs': {'produced': 5, 'produced_ids': ids}}
               for tid, ids in [('a', [str(i) for i in range(5)]), ('b', second_ids)]]
    rows = project_requirements([task('a'), task('b')], refs, records)
    assert rows[0]['status'] == expected
    assert project_requirements([task('a'), task('b')], refs, records[:1])[0]['status'] != 'met'


def test_catalog_hypothesis_id_resolves_to_the_same_graph_node(tmp_path):
    from CoScientist.experiments.schemas.models import TaskDesign
    from CoScientist.experiments.runtime.graph_bridge import _task_hypothesis_ids
    source = 'Check the claim X'
    statement = normalize_statement(source, {'parts': [{
        'kind': 'hypothesis', 'hypothesis_mode': 'check', 'source': 'user_request',
        'quote': source, 'formulation': 'X',
    }]})
    part = statement.parts[0]
    design = TaskDesign(target_refs=[part.id])
    assert design.covered_hypothesis_ids([part.model_dump()]) == {part.id}
    store = ResearchGraphStore(directory=str(tmp_path))
    saved = commit_statement(store, statement)
    nid = saved['node_ids'][part.id]
    full = {node['id']: node for node in store.full()['nodes']}
    assert _task_hypothesis_ids(design.model_dump(), full) == [nid]
    assert design.covered_hypothesis_ids([{'id': part.id, 'kind': 'hypothesis', 'hypothesis_id': nid}]) == {nid}
