"""Orchestrator gates: coalesce, module suppression, feasibility, retrieval budget."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import yaml

from .helpers import (
    _research_call_response,
)


def test_experiment_retrieval_budget_stops_repeated_llm_tool_calls():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.context import (
        enforce_experiment_retrieval_budget,
        reset_experiment_retrieval_budget,
    )

    context = SimpleNamespace(state={"retrieval_queries": ["prior request"]})
    reset_experiment_retrieval_budget(context)
    repeated_call = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="retrieve_tools",
                        args={"query": "KRAS G12C candidates"},
                    )
                )
            ],
        )
    )

    assert enforce_experiment_retrieval_budget(context, repeated_call) is None
    # Budget is 5 request-local retrieve_tools calls counted after the baseline.
    context.state["retrieval_queries"].extend(
        ["query one", "query two", "query three", "query four", "query five"]
    )
    stopped = enforce_experiment_retrieval_budget(context, repeated_call)

    assert stopped is not None
    assert "EXPERIMENT_RETRIEVAL_BUDGET_EXHAUSTED" in stopped.content.parts[0].text
    assert context.state["experiment_retrieval_budget_exhausted"] is True


def test_coalesce_merges_parallel_experiment_module_calls():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        coalesce_experiment_module_calls,
    )

    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Generate KRAS G12C inhibitors."},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Dock the generated molecules."},
                    )
                ),
            ],
        )
    )
    state = {}
    ctx = SimpleNamespace(state=state, user_content=None, agent_name="OrchestratorAgent")

    assert coalesce_experiment_module_calls(ctx, response) is None
    parts = response.content.parts
    assert len(parts) == 1
    merged = parts[0].function_call.args["request"]
    assert "Generate KRAS G12C inhibitors." in merged
    assert "Dock the generated molecules." in merged
    assert state.get("experiment_module_dispatched") is True


def test_targeted_redo_forces_one_module_hop_only_during_execution():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        coalesce_experiment_module_calls,
        suppress_experiment_module_after_completed,
    )

    state = {
        "experiment_runtime": {"phase": "execution"},
        "experiment_targeted_redo_pending": {
            "selected_task_ids": ["EXP-2"],
            "affected_task_ids": ["EXP-2", "EXP-3"],
        },
        # An explicit result-review redo is separate from the automatic
        # planning budget and must still be dispatchable when that is spent.
        "experiment_module_runs": 999,
    }
    ctx = SimpleNamespace(
        state=state,
        user_content=types.Content(role="user", parts=[types.Part(text="continue")]),
        agent_name="OrchestratorAgent",
    )
    prose = LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text="The experiment is finished.")],
    ))

    forced = coalesce_experiment_module_calls(ctx, prose)
    assert forced is not None
    call = forced.content.parts[0].function_call
    assert call.name == "ExperimentModuleAgent"
    assert "EXP-2" in call.args["request"]
    assert "cumulative attempt ledger" in call.args["request"]
    assert suppress_experiment_module_after_completed(ctx, forced) is None
    assert forced.content.parts[0].function_call.name == "ExperimentModuleAgent"
    assert state["experiment_module_runs"] == 999

    # A stale marker never causes a blind module replay outside execution.
    state["experiment_runtime"]["phase"] = "reporting"
    assert coalesce_experiment_module_calls(ctx, prose) is None


def test_suppress_experiment_module_after_completed_and_success():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        suppress_experiment_module_after_completed,
    )

    state = {
        "experiment_runtime": {
            "phase": "completed",
            "tasks_ok": True,
            "tasks": [{"id": "EXP-1", "status": "success", "result_ok": True}],
        }
    }
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Run more experiments."},
                    )
                )
            ],
        )
    )
    ctx = SimpleNamespace(state=state, user_content=None, agent_name="OrchestratorAgent")
    suppress_experiment_module_after_completed(ctx, response)
    # The call should be stripped/suppressed
    fcs = [
        p.function_call.name
        for p in response.content.parts
        if getattr(p, "function_call", None)
    ]
    assert "ExperimentModuleAgent" not in fcs


def test_suppress_experiment_module_allows_retry_when_tasks_failed():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        suppress_experiment_module_after_completed,
    )

    state = {
        "experiment_runtime": {
            "phase": "completed",
            "tasks_ok": False,
            "tasks": [{"id": "EXP-1", "phase": "failed", "result_ok": False}],
        }
    }
    response = LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="ExperimentModuleAgent",
                        args={"request": "Retry failed computation with alternative tool."},
                    )
                )
            ],
        )
    )
    ctx = SimpleNamespace(state=state, user_content=None, agent_name="OrchestratorAgent")
    suppress_experiment_module_after_completed(ctx, response)
    # The call should NOT be suppressed because tasks failed and orchestrator may retry
    fcs = [
        p.function_call.name
        for p in response.content.parts
        if getattr(p, "function_call", None)
    ]
    assert "ExperimentModuleAgent" in fcs


def test_suppress_completed_runtime_does_not_call_unaccepted_result_approved():
    from google.adk.models import LlmResponse
    from google.genai import types

    from CoScientist.experiments.runtime.coalesce import (
        suppress_experiment_module_after_completed,
    )

    state = {
        "experiment_runtime": {
            "phase": "completed", "tasks_ok": True,
            "tasks": [{"id": "EXP-1", "status": "success", "result_ok": True}],
        },
        "experiment_module_outcome": {
            "status": "blocked", "stage": "result_review",
            "reason": "result_review_timeout", "accepted": False,
        },
    }
    response = LlmResponse(content=types.Content(
        role="model", parts=[types.Part.from_function_call(
            name="ExperimentModuleAgent", args={"request": "retry everything"},
        )],
    ))

    suppress_experiment_module_after_completed(
        SimpleNamespace(state=state, agent_name="OrchestratorAgent"), response,
    )

    assert not any(getattr(part, "function_call", None) for part in response.content.parts)
    text = " ".join(part.text or "" for part in response.content.parts)
    assert "still unaccepted" in text
    assert "already terminal" not in text
    assert state["experiment_module_outcome"]["reason"] == "result_review_timeout"


def test_orchestrator_subordinates_clean_lanes():
    yaml_path = Path("CoScientist/agents/experiments.yaml")
    assert yaml_path.is_file()
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    agents = data.get("agents", {})
    orch = agents.get("OrchestratorAgent", {})
    subordinates = orch.get("subordinates", [])

    # The roster is main's, with the module standing where TaskExecutorAgent
    # stood — the profile is an overlay now, so the lanes it does not own are
    # main's to decide.
    assert "HypothesesAgent" in subordinates
    assert "ResearchAgent" in subordinates
    assert "ExperimentModuleAgent" in subordinates
    assert "PlannerAgent" in subordinates
    # Alembic is an internal execution route.  The orchestrator must not bypass
    # ExperimentModule review, URL pinning, or the Coder fallback policy.
    assert "McpBuilderAgent" not in subordinates

    # Code and tool-building are both reached through the module as reviewed
    # routes of its executor.
    assert "CoderAgent" not in subordinates
    assert "TaskExecutorAgent" not in subordinates


def test_early_feasibility_skips_check_for_explicit_module_call():
    """An orchestrator-chosen EM call is never second-guessed."""
    from CoScientist.experiments.context.builder import RETRIEVED_CAPABILITIES_KEY
    from CoScientist.experiments.runtime.guards import (
        NO_MATCHING_TOOL_STATE_KEY,
        assess_experiment_inventory_feasibility,
    )

    state = {
        "experiment_source_request": (
            "Сделай обзор литературы по роли гиппокампа в консолидации памяти."
        ),
        RETRIEVED_CAPABILITIES_KEY: [
            {
                "tool": "generate_case_mols",
                "server_id": "d36e",
                "description": "Generate case molecules with a GAN.",
            },
        ],
    }
    assess_experiment_inventory_feasibility(
        SimpleNamespace(state=state, agent_name="ToolPreparerAgent")
    )
    assert state.get(NO_MATCHING_TOOL_STATE_KEY) in (None, "")


def test_empty_inventory_does_not_block_the_coder_lane():
    from CoScientist.experiments.runtime.guards import (
        NO_MATCHING_TOOL_STATE_KEY,
        assess_experiment_inventory_feasibility,
    )
    from CoScientist.experiments.runtime.shared import GATE_ROUTED_STATE_KEY

    state = {GATE_ROUTED_STATE_KEY: True, "experiment_retrieved_capabilities": []}
    assess_experiment_inventory_feasibility(
        SimpleNamespace(state=state, agent_name="ToolPreparerAgent")
    )
    assert state.get(NO_MATCHING_TOOL_STATE_KEY) in (None, "")
