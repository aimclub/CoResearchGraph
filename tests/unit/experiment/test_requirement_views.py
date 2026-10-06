"""All requirement kinds stay visible before implementation details."""
from copy import deepcopy

import pytest

from CoScientist.experiments.plan_view import plan_to_view
from CoScientist.experiments.review import render_experiment_plan, render_experiment_results
from CoScientist.experiments.runtime import logical_operation_key
from CoScientist.experiments.critique import critique_plan, validate_and_critique_plan
from CoScientist.config.settings import ExperimentsSettings
from CoScientist.reporting.nir.build import build_outline, build_nir_values, NirProse, NirRequisites
from CoScientist.reporting.nir.evidence import NirEvidence
from CoScientist.requirements.coverage import project_requirements

from .helpers import _plan, _task, _design, _inventory
from .test_requirement_assessment import catalog, task, result


def question_plan():
    design = _design("")
    design.update(target_links=[{"requirement_id": "q", "role": "delivers"}],
                  question_ref="q", experiment_question="")
    return _plan(_task("EXP-1", design=design), hypotheses=[])


def test_question_is_linked_without_repeating_its_text_or_inventing_hypothesis():
    plan = question_plan()
    refs = [{"id": "q", "kind": "question", "formulation": "Which method is more accurate?"}]
    view = plan_to_view(plan, requirement_refs=refs)
    assert view["tasks"][0]["design"]["question_ref"] == "q"
    assert view["tasks"][0]["design"]["question"] is None
    text = render_experiment_plan(plan, requirement_refs=refs)
    assert text.count("Which method is more accurate?") == 1
    assert "(#requirement-q)" in text
    assert "Hypothesis" not in text
    assert text.index("## Requirements") < text.index("## EXP-1") < text.index("<details>")


def test_narrow_step_question_does_not_create_an_obligation():
    plan = question_plan()
    plan.tasks[0].design.experiment_question = "How sensitive is the comparison to noise?"
    refs = [{"id": "q", "kind": "question", "formulation": "Which method is more accurate?"}]
    view = plan_to_view(plan, requirement_refs=refs)
    assert [row["id"] for row in view["requirements"]] == ["q"]
    assert "Step question: How sensitive" in render_experiment_plan(plan, requirement_refs=refs)


def test_presentation_links_do_not_change_operation_or_attempt_identity():
    old = _task("EXP-1")
    revised = deepcopy(old)
    revised["design"]["question_ref"] = "q"
    revised["success_criteria"][0].update(requirement_id="q", requirement_criterion_id="accuracy")
    assert logical_operation_key(old) == logical_operation_key(revised)


def test_criterion_links_are_validated_by_id_not_wording():
    raw = _task("EXP-1")
    raw["design"] = {**_design(""), "target_refs": ["result"],
                     "target_links": [{"requirement_id": "result", "role": "delivers"}]}
    refs = [p.model_dump() for p in catalog().parts]
    plan = _plan(raw, hypotheses=[])
    critique = critique_plan(plan, settings=ExperimentsSettings(route_fedot=True), available_tools=_inventory(),
                             hypothesis_refs=[], requirement_refs=refs)
    assert any("no task checks" in issue.message for issue in critique.issues)
    plan.tasks[0].success_criteria[0].requirement_id = "result"
    plan.tasks[0].success_criteria[0].requirement_criterion_id = "values"
    critique = critique_plan(plan, settings=ExperimentsSettings(route_fedot=True), available_tools=_inventory(),
                             hypothesis_refs=[], requirement_refs=refs)
    assert not any("criterion reference" in issue.message or "no task checks" in issue.message for issue in critique.issues)


def test_criterion_purpose_and_catalog_reference_survive_plan_rendering():
    plan = question_plan()
    criterion = plan.tasks[0].success_criteria[0]
    criterion.purpose = 'assessment'
    criterion.requirement_id = 'q'
    criterion.requirement_criterion_id = 'accuracy'
    row = plan_to_view(plan)['tasks'][0]['success_criteria'][0]
    assert (row['purpose'], row['requirement_id'], row['requirement_criterion_id']) == ('assessment', 'q', 'accuracy')
    assert '[assessment → q/accuracy]' in render_experiment_plan(plan)


def test_new_catalog_requires_explicit_delivery_roles_but_old_plans_stay_readable():
    raw = _task('EXP-1', design={**_design(''), 'target_refs': ['result']})
    refs = [part.model_dump() for part in catalog().parts]
    critique = critique_plan(_plan(raw, hypotheses=[]), settings=ExperimentsSettings(route_fedot=True),
                             available_tools=_inventory(), requirement_refs=refs)
    assert any('explicit supports/delivers' in issue.message for issue in critique.issues)
    legacy = [{key: value for key, value in row.items() if key != 'provenance'} for row in refs]
    critique = critique_plan(_plan(raw, hypotheses=[]), settings=ExperimentsSettings(route_fedot=True),
                             available_tools=_inventory(), requirement_refs=legacy)
    assert not any('explicit supports/delivers' in issue.message for issue in critique.issues)


@pytest.mark.parametrize('target,operator,changed', [(0.8, '>=', False), (0.5, '>=', True), (0.8, '<=', True)])
def test_task_cannot_weaken_a_linked_numeric_requirement(target, operator, changed):
    refs = [part.model_dump() for part in catalog().parts]
    refs[0]['criteria'][0].update(text='Score >= 0.8', threshold='>= 0.8')
    raw = _task('EXP-1', design={**_design(''), 'target_links': [{'requirement_id': 'result', 'role': 'delivers'}]})
    raw['success_criteria'] = [{'criterion_id': 'check', 'description': 'A method description',
        'kind': 'threshold', 'purpose': 'assessment', 'metric': 'score', 'operator': operator, 'target': target,
        'verification': 'Read measured score', 'requirement_id': 'result', 'requirement_criterion_id': 'values'}]
    plan = _plan(raw, hypotheses=[])
    critique = critique_plan(plan, settings=ExperimentsSettings(route_fedot=True),
                             available_tools=_inventory(), requirement_refs=refs)
    assert any('changes catalog criterion' in issue.message for issue in critique.issues) is changed
    view = plan_to_view(plan, requirement_refs=refs)
    assert view['tasks'][0]['success_criteria'][0]['description'] == 'Score >= 0.8'
    assert 'A method description' not in render_experiment_plan(plan, requirement_refs=refs)


def test_executor_receives_the_catalog_criterion_with_its_verification_method():
    refs = [part.model_dump() for part in catalog().parts]
    payload = _plan(_task('EXP-1'), hypotheses=[]).model_dump(mode='json')
    payload['tasks'][0]['success_criteria'][0].update(
        requirement_id='result', requirement_criterion_id='values', description='Independent wording',
        verification='Inspect the measured values', purpose='assessment')
    payload['tasks'][0]['design']['target_links'] = [{'requirement_id': 'result', 'role': 'delivers'}]
    plan, _ = validate_and_critique_plan(payload, settings=ExperimentsSettings(route_fedot=True),
                                        available_tools=_inventory(), requirement_refs=iter(refs))
    criterion = plan.tasks[0].success_criteria[0]
    assert criterion.description == refs[0]['criteria'][0]['text']
    assert criterion.verification == 'Inspect the measured values'
    assert payload['tasks'][0]['success_criteria'][0]['description'] == 'Independent wording'


def test_a_task_cannot_assign_conflicting_roles_to_the_same_requirement():
    from pydantic import ValidationError
    from CoScientist.experiments.schemas.models import TaskDesign

    with pytest.raises(ValidationError, match='one role per requirement'):
        TaskDesign(target_links=[{'requirement_id': 'result', 'role': 'supports'},
                                 {'requirement_id': 'result', 'role': 'delivers'}])


def test_result_report_puts_requirement_assessment_before_attempt_log():
    state = {"normalized_statement": catalog().model_dump(), "experiment_runtime": {
        "plan": {"tasks": [task("deliver", criterion=True)]}, "results": [result("deliver", passed=False)],
    }, "report_language": "en"}
    text = render_experiment_results(state)
    assert text.index("## Requirements") < text.index("Execution details")
    assert "partial" in text and "Measured values" in text


def test_nir_preserves_all_requirements_and_assessments_even_when_author_disagrees():
    statement = catalog().model_dump()
    refs = statement["parts"]
    rows = project_requirements([task("deliver", criterion=True)], refs, [result("deliver", passed=False)])
    # More than the old six-line fallback: none of the requirements may disappear.
    rows = [{**rows[0], "id": f"R-{index}", "formulation": f"Requested result {index}"} for index in range(8)]
    ev = NirEvidence(original_request="Give the requested results", nodes=[{
        "id": "root", "type": "ResearchQuestion", "attrs": {"normalized_statement": statement,
                                                                "requirement_assessment": rows},
    }])
    outline = build_outline(ev)
    section = outline.sections[0]
    assert section.fixed_content
    assert all(f"Requested result {i}" in "\n".join(section.digest) for i in range(8))
    prose = NirProse(section_texts={section.id: ["Everything is fulfilled."]})
    values, _problems = build_nir_values(ev, NirRequisites(), prose, outline)
    content = str(values)
    assert "Everything is fulfilled" not in content
    assert "Requested result 7" in content


def test_shipped_plan_card_keeps_requirements_visible_when_a_document_is_attached(tmp_path):
    import json
    import runpy
    import shutil
    import subprocess
    from pathlib import Path
    import pytest

    if not shutil.which("node"):
        pytest.skip("node is required for the shipped JavaScript renderer")
    root = Path(__file__).resolve().parents[3]
    preview = runpy.run_path(str(root / "scripts/preview_plan_card.py"))
    shim = preview["_SHIM"].replace("function documentBlock() { return ''; }", "function documentBlock() { return 'DOCUMENT-LINK'; }")
    script = tmp_path / "render.cjs"
    script.write_text(shim)
    plan = question_plan()
    view = plan_to_view(plan, requirement_refs=[{"id": "q", "kind": "question", "formulation": "Which method is more accurate?"}])
    plan_file, output = tmp_path / "view.json", tmp_path / "render.json"
    plan_file.write_text(json.dumps(view))
    subprocess.run(["node", str(script), str(root), str(plan_file), "en", str(output)], check=True, capture_output=True)
    html = json.loads(output.read_text())["feed"]
    assert "DOCUMENT-LINK" in html
    assert html.count("Which method is more accurate?") == 1
    assert 'href="#requirement-preview-0-q"' in html
    assert html.index('id="requirement-preview-0-q"') < html.index('id="plan-tasks-preview-0"')
