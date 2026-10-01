"""Dataset compatibility must not turn missing literature CSV into a global block."""
from types import SimpleNamespace

import pytest

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.capabilities.contracts import dataset_mismatch
from CoScientist.experiments.capabilities.inventory import get_grouped_mcp_inventory, index_inventory_tools
from CoScientist.experiments.context.builder import _cap_for_prompt, _normalize_capabilities
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.runtime import approve_plan, fallback_task, initialize_runtime, start_task
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.runtime.routing import inventory_covers_task
from CoScientist.experiments.schemas import ExperimentTask

from .helpers import _inventory, _plan, _task


def _cap(scope="requested_corpus", mode="fixed_dataset"):
    return {**_inventory()[0], "data_contract": {"input_mode": mode, "dataset_scope": scope},
            "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            "output_schema": {"type": "object", "required": ["answer"], "properties": {"answer": {"type": "object"}}}}


def _task_with_scope(task_id="EXP-1"):
    task = _task(task_id, route="react_tools")
    task["design"]["dataset"]["dataset_scope"] = "requested_corpus"
    task["launch_params"] = {}
    return task


def test_contract_survives_flat_grouped_and_prompt_projections():
    cap = _cap()
    normalized = _normalize_capabilities([cap])[0]
    assert normalized["data_contract"] == cap["data_contract"]
    assert index_inventory_tools([normalized])[cap["tool"]]["output_schema"] == cap["output_schema"]
    assert get_grouped_mcp_inventory([normalized])[0]["tools"][0]["data_contract"] == cap["data_contract"]
    assert _cap_for_prompt(normalized)["output_contract"]["required"] == ["answer"]


@pytest.mark.parametrize("cap", [_cap(), _cap(mode="caller_data", scope="training_corpus"), _inventory()[0]])
def test_matching_fixed_dataset_and_unknown_metadata_do_not_need_external_csv(cap):
    task = _task_with_scope()
    assert dataset_mismatch(task, cap) is None
    state = {"experiment_retrieved_capabilities": [cap]}
    initialize_runtime(state, _plan(task), critique={"verdict": "approve", "issues": []})
    approve_plan(state)
    assert start_task(state, "EXP-1")["route"] == "react_tools"


def test_wrong_declared_fixed_dataset_fails_review_and_has_bounded_coder_fallback():
    task = _task_with_scope()
    cap = _cap(scope="unrelated_corpus")
    plan = _plan(task)
    critique = critique_plan(plan, settings=ExperimentsSettings(), available_tools=[cap])
    assert any("tool_dataset_scope_mismatch" in issue.message for issue in critique.issues)
    state = {"experiment_retrieved_capabilities": [cap]}
    assert not inventory_covers_task(state, ExperimentTask.model_validate(task))
    initialize_runtime(state, plan, critique={"verdict": "approve", "issues": []})
    approve_plan(state)
    with pytest.raises(ExperimentRuntimeError) as caught:
        start_task(state, "EXP-1")
    assert caught.value.code == "tool_dataset_scope_mismatch"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["status"] == "fallback_pending"
    assert state["experiment_runtime"]["tasks"]["EXP-1"]["attempt_order"] == []
    assert fallback_task(state, "EXP-1", "Use code for the requested dataset")["route"] == "coder"
    assert start_task(state, "EXP-1")["route"] == "coder"


def test_wrong_dataset_blocks_only_affected_task_and_does_not_trust_forged_task_contract():
    wrong = _task_with_scope()
    wrong["mcp_servers"][0]["tools"][0]["data_contract"] = _cap()["data_contract"]
    independent = _task_with_scope("EXP-2")
    independent["design"]["dataset"]["dataset_scope"] = "unrelated_corpus"
    state = {"experiment_retrieved_capabilities": [_cap(scope="unrelated_corpus")]}
    initialize_runtime(state, _plan(wrong, independent), critique={"verdict": "approve", "issues": []})
    approve_plan(state)
    with pytest.raises(ExperimentRuntimeError):
        start_task(state, "EXP-1")
    assert start_task(state, "EXP-2")["route"] == "react_tools"
def test_registry_contracts_survive_real_retrieval_and_repeated_merge(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from CoScientist.storage import RetrievalToolResult
    from CoScientist.tools import retrieval_tools

    contract = {"input_mode": "fixed_dataset", "dataset_scope": "reference-compounds"}
    output = {"type": "object", "required": ["predictions"], "properties": {"predictions": {"type": "array"}}}

    class Registry:
        def __init__(self, *_args):
            pass

        async def initialize(self):
            pass

        async def close(self):
            pass

        async def get_server(self, _sid):
            return SimpleNamespace(protocol="http", url="http://registry.test/mcp")

        async def get_tools_by_server(self, _sid):
            return [SimpleNamespace(name="predict", description="Predict reference properties",
                                    input_schema={"type": "object", "x-data-contract": contract},
                                    output_schema=output)]

    monkeypatch.setattr(retrieval_tools, "PostgresClient", Registry)
    meta = asyncio.run(retrieval_tools._fetch_full_tool_meta({"srv"}))[("srv", "predict")]
    result = RetrievalToolResult(tool="predict", server_id="srv", score=1.0, **meta)
    accumulated = retrieval_tools._merge_accumulated_tools([], [result], query="reference properties")
    assert accumulated[0]["data_contract"] == contract
    assert accumulated[0]["output_schema"] == output
    # Legacy hits can be enriched by a later, authoritative registry read.
    del accumulated[0]["output_schema"]
    accumulated = retrieval_tools._merge_accumulated_tools(accumulated, [result], query="predict")
    assert accumulated[0]["output_schema"] == output
    assert len(accumulated) == 1
