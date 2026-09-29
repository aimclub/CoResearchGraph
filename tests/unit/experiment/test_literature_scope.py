"""Literature belongs to the parent roadmap, never to an EM execution route."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.context.builder import build_experiment_context
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.runtime import approve_plan, initialize_runtime, start_task
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.runtime.guards import guard_route_agent_tool
from CoScientist.experiments.scope import (
    is_literature_operation, partition_operations, skip_literature_only_experiment,
)

from .helpers import _inventory, _plan, _task


@pytest.mark.parametrize("statement", [
    "Собрать литературные данные о метаболитах H. sosnowskyi и их SMILES",
    "Собрать экспериментальные LD50",
    "Найти публикации о токсичности вещества",
    "Review published methods for the endpoint",
    "Literature review of existing evidence",
    "Search the clinical literature with search_pubmed",
])
def test_literature_operations_are_external(statement):
    assert is_literature_operation(statement)


@pytest.mark.parametrize("statement", [
    "Предсказать LD50 для мыши",
    "Выполнить структурную кластеризацию",
    "Train a model on supplied literature LD50 data",
    "Compute toxicity using the published method",
    "Оценить domain of applicability по статье",
    "Reproduce the experiments from the paper",
    "Collect a dataset and train a prediction model",
    "Extract PICO from the supplied abstract with get_pico",
    "Read the repository API documentation with tavily_extract",
    "Find the software repository with tavily_search",
])
def test_compute_and_supplied_data_analysis_are_not_removed(statement):
    assert not is_literature_operation(statement)


def test_mixed_context_keeps_original_ids_root_scope_and_supplied_data(monkeypatch):
    import CoScientist.experiments.context.builder as builder

    operations = [
        {"operation_id": "OP-3", "statement": "Собрать литературные данные о метаболитах"},
        {"operation_id": "OP-4", "statement": "Выполнить структурную кластеризацию"},
        {"operation_id": "OP-5", "statement": "Собрать экспериментальные LD50"},
        {"operation_id": "OP-6", "statement": "Предсказать отсутствующие LD50"},
    ]
    evidence = [{"id": "E1", "content": "Previously collected references"}]
    monkeypatch.setattr(builder, "research_graph_snapshot", lambda _ctx: {"prior_evidence": evidence})
    state = {
        "experiment_operations": copy.deepcopy(operations),
        "experiment_source_request": "Collect literature and compute LD50.",
        "experiment_data_refs": [{"ref": "cos-artifact:ld50.csv"}],
    }
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    context = state["experiment_context"]
    assert [op["operation_id"] for op in context["operations"]] == ["OP-4", "OP-6"]
    assert [op["operation_id"] for op in context["external_literature_operations"]] == ["OP-3", "OP-5"]
    assert state["experiment_operations"] == operations
    assert context["prior_evidence"] == evidence
    assert context["data_refs"] == [{"ref": "cos-artifact:ld50.csv"}]
    assert "available_research_capabilities" not in context
    prompt = json.loads(state["experiment_planner_context"])
    assert "available_research_capabilities" not in prompt
    assert prompt["external_literature_operations"] == context["external_literature_operations"]

    # Critique must not renumber OP-4 to OP-1 or demand the external OP-3/5.
    tasks = []
    for i, op in enumerate(context["operations"], 1):
        task = _task(f"EXP-{i}", route="react_tools")
        task["design"]["operation_ref"] = op["operation_id"]
        tasks.append(task)
    critique = critique_plan(_plan(*tasks), settings=ExperimentsSettings(),
                             available_tools=_inventory(), operations=operations)
    assert critique.verdict == "approve", critique
    subset = critique_plan(_plan(*tasks), settings=ExperimentsSettings(),
                           available_tools=_inventory(), operations=context["operations"])
    assert subset.verdict == "approve", subset


@pytest.mark.parametrize("route", ["research", "coder", "react_tools", "medical"])
def test_review_and_runtime_refuse_literature_disguised_as_other_routes(route):
    task = _task("EXP-1", route=route)
    task["description"] = "Search the literature for experimental LD50 values."
    plan = _plan(task)
    critique = critique_plan(plan, settings=ExperimentsSettings(), available_tools=_inventory())
    assert any(i.severity == "blocker" and "literature_outside_experiment_module" in i.message
               for i in critique.issues)
    # Imported, formerly approved snapshots still deserialize, but cannot start.
    state = {}
    initialize_runtime(state, plan, critique={"verdict": "approve", "issues": []})
    approve_plan(state)
    with pytest.raises(ExperimentRuntimeError) as error:
        start_task(state, "EXP-1")
    assert error.value.code == "literature_outside_experiment_module"
    runtime = state["experiment_runtime"]
    assert runtime["tasks"]["EXP-1"]["status"] == "blocked"
    assert runtime["tasks"]["EXP-1"]["attempt_order"] == []
    assert runtime.get("active_attempt_id") is None


def test_a_compute_label_cannot_hide_an_external_operation_ref():
    task = _task("EXP-1", route="coder")
    task["design"]["operation_ref"] = "OP-5"
    operations = [{"operation_id": "OP-5", "statement": "Собрать экспериментальные LD50"}]
    critique = critique_plan(_plan(task), settings=ExperimentsSettings(), operations=operations)
    assert any("literature_outside_experiment_module" in i.message for i in critique.issues)


@pytest.mark.parametrize("tool_name", ["search_papers", "search_pubmed"])
def test_bound_literature_tool_cannot_bypass_scope_check(tool_name):
    task = _task("EXP-1", route="react_tools", tool=tool_name)
    critique = critique_plan(_plan(task), settings=ExperimentsSettings(), available_tools=_inventory())
    assert any("literature_outside_experiment_module" in i.message for i in critique.issues)


def test_literature_inventory_is_external_even_if_explicitly_named_in_root_request():
    inventory = [*_inventory(), {
        "tool": "search_papers", "server_id": "srv-papers", "description": "Search literature",
    }]
    state = {
        "experiment_source_request": "Use search_papers, then use estimate_property.",
        "experiment_retrieved_capabilities": inventory,
    }
    build_experiment_context(SimpleNamespace(state=state, user_content=None))
    context = state["experiment_context"]
    for key in ("available_mcp_capabilities", "preferred_mcp_capabilities", "critique_mcp_capabilities"):
        assert {row["tool"] for row in context[key]} == {"estimate_property"}
    assert state["experiment_retrieved_capabilities"] == inventory  # discovery stays intact
    plan = _plan(_task("EXP-1", route="react_tools"))
    plan.source_request = state["experiment_source_request"]
    critique = critique_plan(plan, settings=ExperimentsSettings(), available_tools=inventory,
                             preferred_tools=inventory)
    assert critique.verdict == "approve", critique
    assert not any("search_papers" in issue.message for issue in critique.issues)


def test_old_executor_cannot_call_research_agent_directly():
    result = guard_route_agent_tool(SimpleNamespace(name="ResearchAgent"), {}, SimpleNamespace(state={}))
    assert result["error_code"] == "literature_outside_experiment_module"


def test_literature_only_delegation_returns_before_discovery_without_completing_scope():
    ops = [{"operation_id": "OP-2", "statement": "Collect experimental LD50 from literature"}]
    state = {"experiment_operations": ops, "experiment_source_request": ops[0]["statement"]}
    response = skip_literature_only_experiment(SimpleNamespace(state=state, user_content=None))
    assert response and "orchestrator" in response.parts[0].text
    assert state["experiment_module_outcome"]["status"] == "not_applicable"
    assert state["experiment_operations"] == ops
    assert "experiment_runtime" not in state


def test_mixed_delegation_still_enters_module():
    state = {"experiment_operations": [
        {"operation_id": "OP-1", "statement": "Review literature"},
        {"operation_id": "OP-2", "statement": "Predict LD50"},
    ]}
    assert skip_literature_only_experiment(SimpleNamespace(state=state, user_content=None)) is None
    assert partition_operations(state["experiment_operations"])[0][0]["operation_id"] == "OP-2"
