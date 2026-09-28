"""The pilot verifies observed ADK delegation without directing tool choice."""

import json
from types import SimpleNamespace

import pytest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from CoScientist.agents.callbacks import pilot_delegation
from CoScientist.agents.callbacks.pilot_delegation import require_pilot_delegations


REQUIRED = ("retrieve_tools", "ResearchAgent", "TaskExecutorAgent")


def _science_result(*names):
    return {"result": json.dumps({
        "status": "computed",
        "scientific_mcp_calls": [
            {"tool": name, "args": {}, "result": {"answer": {"n_reconstructed": 225}}}
            for name in names
        ],
    })}


def _event(name, *, invocation_id="run-1", result=None, include_call=True):
    call = types.FunctionCall(name=name, args={})
    response = types.FunctionResponse(
        name=name, response={"result": "ok"} if result is None else result
    )
    return SimpleNamespace(
        invocation_id=invocation_id,
        get_function_calls=lambda: [call] if include_call else [],
        get_function_responses=lambda: [response],
    )


def _context(*events, state=None):
    invocation = SimpleNamespace(
        invocation_id="run-1", session=SimpleNamespace(events=list(events))
    )
    return SimpleNamespace(_invocation_context=invocation, state=state or {})


def _model_response(*, tool=None, partial=False):
    part = (
        types.Part.from_function_call(name=tool, args={})
        if tool
        else types.Part(text="Pilot report")
    )
    return LlmResponse(
        content=types.Content(role="model", parts=[part]), partial=partial
    )


def test_pilot_accepts_extra_calls_and_required_delegations_in_any_order():
    context = _context(
        _event("retrieve_tools"),
        _event("HypothesesAgent"),
        _event("TaskExecutorAgent", result=_science_result(
            "dataset_overview_heracleum_tox", "chemical_space_clustering",
            "predict_ld50", "predict_molecule_profile"
        )),
        _event("ResearchAgent"),
    )
    report = _model_response()
    assert require_pilot_delegations(context, report) is None
    assert report.content.parts[0].text == "Pilot report"


def test_pilot_blocks_executor_handoff_that_drops_retrieved_science_tools():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")), remote_subagents=True)
    orchestrator = pilot.root
    executor = next(tool for tool in orchestrator.tools if getattr(tool, "name", None) == "TaskExecutorAgent")
    context = SimpleNamespace(state={"accumulated_tools": [
        {"tool": "predict_ld50", "server_id": "heracleum-server"},
        {"tool": "chemical_space_clustering", "server_id": "heracleum-server"},
        {"tool": "butina_clustering", "server_id": "unrelated-server"},
    ]})
    args = {"request": "Extract metabolites from the prepared heracleum-tox base"}

    responses = [
        callback(executor, args, context)
        for callback in orchestrator.canonical_before_tool_callbacks
    ]

    assert any(
        isinstance(response, dict)
        and "predict_ld50" in response.get("error", "")
        and "chemical_space_clustering" in response.get("error", "")
        and "butina_clustering" not in response.get("error", "")
        for response in responses
    )
    assert args["request"] == "Extract metabolites from the prepared heracleum-tox base"


def test_pilot_allows_executor_handoff_with_a_retrieved_science_tool():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")), remote_subagents=True)
    orchestrator = pilot.root
    executor = next(tool for tool in orchestrator.tools if getattr(tool, "name", None) == "TaskExecutorAgent")
    context = SimpleNamespace(state={"accumulated_tools": [
        {"tool": "predict_ld50", "server_id": "heracleum-server"},
    ]})
    args = {"request": "Run predict_ld50 on the prepared heracleum-tox MCP server "
                       "(server_id=heracleum-server)"}

    assert all(
        callback(executor, args, context) is None
        for callback in orchestrator.canonical_before_tool_callbacks
    )


def test_pilot_handoff_requires_every_discovered_name_and_server_id():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")), remote_subagents=True)
    executor = next(tool for tool in pilot.root.tools if getattr(tool, "name", None) == "TaskExecutorAgent")
    context = SimpleNamespace(state={"accumulated_tools": [
        {"tool": "predict_ld50", "server_id": "heracleum-server"},
        {"tool": "chemical_space_clustering", "server_id": "heracleum-server"},
    ]})
    assert any(
        "chemical_space_clustering" in response.get("error", "")
        for callback in pilot.root.canonical_before_tool_callbacks
        if isinstance(response := callback(executor, {
            "request": "Run predict_ld50 (server_id=heracleum-server)"
        }, context), dict)
    )
    assert all(
        callback(executor, {"request": (
            "Run predict_ld50 (server_id=heracleum-server); available tool "
            "chemical_space_clustering (server_id=heracleum-server)"
        )}, context) is None
        for callback in pilot.root.canonical_before_tool_callbacks
    )


def test_pilot_executor_blocks_coder_for_named_ready_mcp_tool():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")))
    executor = pilot.agent("TaskExecutorAgent")
    coder = next(tool for tool in executor.tools if getattr(tool, "name", None) == "CoderAgent")
    context = SimpleNamespace(
        state={},
        _invocation_context=SimpleNamespace(user_content=types.Content(
            role="user", parts=[types.Part(text=(
                "Run predict_ld50 from the prepared heracleum-tox MCP server"
            ))],
        )),
    )

    responses = [
        callback(coder, {"request": "Clone empiricalbase/heracleum-tox"}, context)
        for callback in executor.canonical_before_tool_callbacks
    ]
    assert any(
        isinstance(response, dict)
        and "ToolPipelineAgent" in response.get("error", "")
        for response in responses
    )


def test_pilot_executor_keeps_explicit_target_in_pipeline_request():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")))
    executor = pilot.agent("TaskExecutorAgent")
    pipeline = next(tool for tool in executor.tools if tool.name == "ToolPipelineAgent")
    context = SimpleNamespace(
        state={},
        _invocation_context=SimpleNamespace(user_content=types.Content(
            role="user", parts=[types.Part(text=(
                "Target tool: predict_molecule_profile (server_id=heracleum-server). "
                "Call it with name_or_smiles=xanthotoxin."
            ))],
        )),
    )
    args = {"request": "Compute a profile for xanthotoxin"}

    assert all(
        callback(pipeline, args, context) is None
        for callback in executor.canonical_before_tool_callbacks
    )
    assert args["request"].startswith("Target tool: predict_molecule_profile")
    assert "name_or_smiles=xanthotoxin" in args["request"]


def test_pilot_executor_preserves_explicit_repository_work():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")))
    executor = pilot.agent("TaskExecutorAgent")
    coder = next(tool for tool in executor.tools if getattr(tool, "name", None) == "CoderAgent")
    context = SimpleNamespace(
        state={},
        _invocation_context=SimpleNamespace(user_content=types.Content(
            role="user", parts=[types.Part(text=(
                "Clone https://github.com/example/analysis.git and inspect its source"
            ))],
        )),
    )

    assert all(
        callback(coder, {"request": "Clone the repository"}, context) is None
        for callback in executor.canonical_before_tool_callbacks
    )


def test_pilot_executor_keeps_named_mcp_work_out_of_coder_even_with_a_repo_url():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")))
    executor = pilot.agent("TaskExecutorAgent")
    coder = next(tool for tool in executor.tools if getattr(tool, "name", None) == "CoderAgent")
    context = SimpleNamespace(
        state={},
        _invocation_context=SimpleNamespace(user_content=types.Content(
            role="user", parts=[types.Part(text=(
                "Run predict_ld50 using the prepared MCP; the paper source is "
                "https://github.com/example/analysis.git"
            ))],
        )),
    )

    responses = [
        callback(coder, {"request": "Clone the paper source"}, context)
        for callback in executor.canonical_before_tool_callbacks
    ]
    assert any(
        isinstance(response, dict) and "ToolPipelineAgent" in response.get("error", "")
        for response in responses
    )


def test_pilot_executor_allows_separate_explicit_repository_subtask():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")))
    executor = pilot.agent("TaskExecutorAgent")
    coder = next(tool for tool in executor.tools if tool.name == "CoderAgent")
    url = "https://github.com/example/analysis.git"
    context = SimpleNamespace(state={}, _invocation_context=SimpleNamespace(
        user_content=types.Content(role="user", parts=[types.Part(text=(
            f"Run predict_ld50, then clone {url} and inspect its source"
        ))]),
    ))
    assert all(
        callback(coder, {"request": f"Clone {url} and inspect its source"}, context) is None
        for callback in executor.canonical_before_tool_callbacks
    )


def test_pilot_combines_scientific_receipts_across_executor_delegations():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent"),
        *(
            _event("TaskExecutorAgent", result=_science_result(name))
            for name in (
                "dataset_overview_heracleum_tox", "chemical_space_clustering",
                "predict_ld50", "predict_molecule_profile",
            )
        ),
    )
    assert require_pilot_delegations(context, _model_response()) is None


def _verified_report_context():
    return _context(
        _event("retrieve_tools"), _event("ResearchAgent"),
        _event("TaskExecutorAgent", result=_science_result(
            "dataset_overview_heracleum_tox", "chemical_space_clustering",
            "predict_ld50", "predict_molecule_profile",
        )),
    )


GROUNDED_REPORT = (
        "## Научный отчёт по Heracleum\n\n"
        "| Этап | Наблюдение |\n|---|---|\n"
        "| Обзор | MCP подтвердил 225 реконструированных соединений |\n"
        "| Кластеризация | Результат вычислен инструментом |\n"
        "| LD50 | Значения являются прогнозом модели |\n"
        "| Профиль | Получен отдельным вызовом MCP |\n\n"
        "Полные строки молекул и SMILES недоступны из агрегированного обзора. "
        "Экспериментальная проверка LD50 в доступных результатах не подтверждена. "
        "Тепловая карта и дендрограмма не приложены, потому что инструменты "
        "не вернули файлы изображений. Таблица отражает только реально "
        "полученные вычислительные результаты."
)


@pytest.mark.parametrize("claim", [
    "Тепловая карта создана (рисунок [[linkcb93]]).",
    "Кардиотоксичность составляет 87%.",
    "Тепловая карта построена и включена в отчёт.",
    "[Тепловая карта](results/heatmap.png) создана.",
    "Это согласуется с оригинальными экспериментальными данными.",
    "Источник: Supplementary Tables S1-S5.",
    "По литературе не найдено никаких публикаций.",
])
def test_pilot_rejects_each_unsupported_claim(claim):
    report = LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text=GROUNDED_REPORT + "\n" + claim)]
    ))
    with pytest.raises(RuntimeError, match="unsupported"):
        pilot_delegation.validate_pilot_report(_verified_report_context(), report)


def test_pilot_accepts_grounded_substantive_report():
    context = _verified_report_context()
    report_text = GROUNDED_REPORT
    report = LlmResponse(content=types.Content(
        role="model", parts=[types.Part(text=report_text)]
    ))
    assert pilot_delegation.validate_pilot_report(context, report) is None
    assert report.content.parts[0].text == report_text


def test_pilot_report_requires_molecule_level_limitation():
    report = LlmResponse(content=types.Content(role="model", parts=[types.Part(
        text=GROUNDED_REPORT.replace(
            "Полные строки молекул и SMILES недоступны из агрегированного обзора. ", ""
        )
    )]))
    with pytest.raises(RuntimeError, match="SMILES"):
        pilot_delegation.validate_pilot_report(_verified_report_context(), report)


def test_pilot_requests_missing_profile_before_accepting_final_report():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent"),
        _event("TaskExecutorAgent", result=_science_result(
            "dataset_overview_heracleum_tox", "chemical_space_clustering",
            "predict_ld50",
        )),
        state={"accumulated_tools": [
            {"tool": name, "server_id": "heracleum-server"}
            for name in (
                "dataset_overview_heracleum_tox", "chemical_space_clustering",
                "predict_ld50", "predict_molecule_profile",
            )
        ]},
    )

    correction = require_pilot_delegations(context, _model_response())
    call = correction.content.parts[0].function_call
    assert call.name == "TaskExecutorAgent"
    assert "Target tool: predict_molecule_profile" in call.args["request"]
    assert "name_or_smiles=xanthotoxin" in call.args["request"]
    assert "chemical_space_clustering" in call.args["request"]
    with pytest.raises(RuntimeError, match="predict_molecule_profile"):
        require_pilot_delegations(context, _model_response())


def test_pilot_recovers_missing_profile_after_reranker_clears_discovery():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = build_system(load_config(resolve_config_path("synapse_pilot")), remote_subagents=True)
    executor = next(
        tool for tool in pilot.root.tools
        if getattr(tool, "name", None) == "TaskExecutorAgent"
    )
    state = {"accumulated_tools": [
        {"tool": name, "server_id": "heracleum-server"}
        for name in (
            "dataset_overview_heracleum_tox", "chemical_space_clustering",
            "predict_ld50", "predict_molecule_profile",
        )
    ]}
    handoff = SimpleNamespace(
        state=state, _invocation_context=SimpleNamespace(invocation_id="run-1")
    )
    handoff_callback = next(
        callback for callback in pilot.root.canonical_before_tool_callbacks
        if callback.__name__ == "enforce_pilot_science_handoff"
    )
    handoff_callback(
        executor, {"request": "Run dataset_overview_heracleum_tox"}, handoff
    )
    state["accumulated_tools"] = []
    context = _context(
        _event("retrieve_tools"), _event("ResearchAgent"),
        _event("TaskExecutorAgent", result=_science_result(
            "dataset_overview_heracleum_tox", "chemical_space_clustering", "predict_ld50"
        )),
        state=state,
    )
    correction = require_pilot_delegations(context, _model_response())
    assert "Target tool: predict_molecule_profile" in (
        correction.content.parts[0].function_call.args["request"]
    )


def test_pilot_rejects_delegations_without_verified_scientific_computation():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent"),
        _event("TaskExecutorAgent", result={"result": "12,654 molecules, four CSV/PNG files"}),
    )
    with pytest.raises(RuntimeError, match="scientific computation"):
        require_pilot_delegations(context, _model_response())


def test_pilot_allows_intermediate_tool_calls():
    context = _context(_event("retrieve_tools"))
    assert (
        require_pilot_delegations(
            context, _model_response(tool="HypothesesAgent")
        )
        is None
    )
    assert require_pilot_delegations(context, _model_response(partial=True)) is None


@pytest.mark.parametrize("missing", REQUIRED)
def test_pilot_rejects_final_report_without_real_call_and_response(missing):
    events = [_event(name) for name in REQUIRED if name != missing]
    with pytest.raises(RuntimeError, match=missing):
        require_pilot_delegations(_context(*events), _model_response())


def test_pilot_ignores_other_invocations_and_failed_results():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent", result={"status": "error", "message": "failed"}),
        _event("ResearchAgent", invocation_id="old-run"),
        _event("TaskExecutorAgent"),
    )
    with pytest.raises(RuntimeError, match="ResearchAgent"):
        require_pilot_delegations(context, _model_response())


def test_pilot_requires_a_call_as_well_as_a_response():
    context = _context(
        _event("retrieve_tools"),
        _event("ResearchAgent", include_call=False),
        _event("TaskExecutorAgent"),
    )
    with pytest.raises(RuntimeError, match="ResearchAgent"):
        require_pilot_delegations(context, _model_response())


def test_pilot_profile_wires_observation_without_changing_regular_demo():
    from CoScientist.assembly import build_system
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = load_config(resolve_config_path("synapse_pilot"))
    regular = load_config(resolve_config_path("synapse_demo"))
    callbacks = pilot.agent("OrchestratorAgent").callbacks
    regular_callbacks = regular.agent("OrchestratorAgent").callbacks

    assert callbacks.after_model[0] == "require_pilot_delegations"
    assert "require_pilot_tool" not in callbacks.before_model
    assert "require_pilot_expected_tool" not in callbacks.before_tool
    assert "guard_unknown_tools" not in callbacks.after_model
    assert "require_pilot_delegations" not in regular_callbacks.after_model

    orchestrator = build_system(pilot, remote_subagents=True).root
    assert {tool.name for tool in orchestrator.tools if hasattr(tool, "name")} >= {
        "ResearchAgent",
        "TaskExecutorAgent",
    }
    assert require_pilot_delegations in orchestrator.canonical_after_model_callbacks


def test_pilot_reasoning_matches_gpt_oss_gateway_without_changing_regular_demo():
    from CoScientist.assembly.schema import load_config, resolve_config_path

    pilot = load_config(resolve_config_path("synapse_pilot"))
    regular = load_config(resolve_config_path("synapse_demo"))

    assert pilot.defaults.reasoning == "medium"
    assert pilot.agent("HypothesesAgent").reasoning == "medium"
    assert regular.defaults.reasoning is False
    assert regular.agent("HypothesesAgent").reasoning == "high"
