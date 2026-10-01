"""HITL pipeline-scope: pack/form/directive, callback gates, console SELECT+form."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from CoScientist.assembly.schema import load_config, resolve_config_path
from CoScientist.config import get_settings
from CoScientist.hitl.handler import ConsoleHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse
from CoScientist.hitl.pipeline_scope import (
    BASKET_LITERATURE,
    DIRECTIVE_KEY,
    FORM_BLOCK,
    LANES,
    OPTION_CUSTOM,
    OPTION_SKIP,
    SCOPE_BASKETS_ALL,
    SELECT_OPTIONS,
    SKIP_DIRECTIVE,
    STATE_KEY,
    as_bool,
    flatten_form_values,
    make_ask_pipeline_scope_callback,
    pack_scope,
    render_directive,
    scope_form,
)


class _ScriptedHandler:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def handle_request(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _ctx(state=None):
    state = {} if state is None else state
    session = SimpleNamespace(id="sess", user_id="user", state=state)
    return SimpleNamespace(
        state=state,
        agent_name="OrchestratorAgent",
        session=session,
        _invocation_context=SimpleNamespace(session=session),
    )


def _enable_scope_hitl(monkeypatch):
    web = get_settings().web
    monkeypatch.setattr(web, "scope_hitl", True)
    monkeypatch.setattr(web, "hitl_enabled", True)


def test_as_bool_uses_env_tokens_plus_da():
    assert as_bool("true") and as_bool("1") and as_bool("yes") and as_bool("да")
    assert as_bool("YES")
    assert not as_bool("") and not as_bool(None) and not as_bool("нет")
    assert not as_bool("false") and not as_bool("0")


def test_pack_scope_one_two_three_lanes():
    one = pack_scope({FORM_BLOCK: {"research": "да"}})
    assert one == {
        "research": True,
        "hypotheses": False,
        "experiments": False,
        "source": "hitl",
    }
    two = pack_scope({FORM_BLOCK: {"research": "yes", "experiments": "1"}})
    assert two["research"] and two["experiments"] and not two["hypotheses"]
    all_on = pack_scope({
        FORM_BLOCK: {"research": "true", "hypotheses": "да", "experiments": "yes"},
    })
    assert all(all_on[key] for key, _ in LANES)


def test_pack_scope_empty_or_all_no_is_none():
    assert pack_scope(None) is None
    assert pack_scope({}) is None
    assert pack_scope({FORM_BLOCK: {}}) is None
    assert pack_scope({FORM_BLOCK: {"research": "нет", "hypotheses": "", "experiments": "no"}}) is None


def test_flatten_accepts_flat_or_nested():
    assert flatten_form_values({"research": "да"})["research"] == "да"
    nested = flatten_form_values({FORM_BLOCK: {"experiments": "1"}})
    assert nested["experiments"] == "1"


def test_directive_lists_named_agents_not_skip_baskets():
    scope = pack_scope({FORM_BLOCK: {"research": "да"}})
    text = render_directive(scope)
    assert "1. ResearchAgent" in text
    assert "human-fixed scope" in text.lower()
    assert BASKET_LITERATURE not in text
    assert "no experiment requested" not in text
    assert "Do NOT call HypothesesAgent, ExperimentModuleAgent" in text
    assert "Full-cycle scientific research" not in text


def test_directive_experiments_and_mixed():
    scope = pack_scope({FORM_BLOCK: {"experiments": "да"}})
    text = render_directive(scope)
    assert "1. ExperimentModuleAgent" in text
    assert "Do NOT call ResearchAgent, HypothesesAgent" in text
    mixed = render_directive(pack_scope({FORM_BLOCK: {"research": "да", "experiments": "yes"}}))
    assert "ResearchAgent" in mixed and "ExperimentModuleAgent" in mixed
    assert "Do NOT call HypothesesAgent" in mixed
    assert "Full-cycle scientific research" not in mixed


def test_scope_form_fields_are_lane_keys():
    form = scope_form()
    names = [f["name"] for f in form["blocks"][0]["fields"]]
    assert names == [key for key, _ in LANES]
    assert form["title"] == "Пункты пайплайна"
    assert form["allow_skip"] is False
    for field in form["blocks"][0]["fields"]:
        assert field["kind"] == "toggle"
        assert field["value"] == "нет"


def test_callback_flags_off_does_not_call_handler(monkeypatch):
    web = get_settings().web
    monkeypatch.setattr(web, "scope_hitl", False)
    monkeypatch.setattr(web, "hitl_enabled", True)
    handler = _ScriptedHandler([])
    cb = make_ask_pipeline_scope_callback(handler)
    ctx = _ctx()

    async def run():
        assert await cb(ctx) is None

    asyncio.run(run())
    assert handler.requests == []
    assert STATE_KEY not in ctx.state

    monkeypatch.setattr(web, "scope_hitl", True)
    monkeypatch.setattr(web, "hitl_enabled", False)
    asyncio.run(run())
    assert handler.requests == []


def test_callback_skip_and_timeout_do_not_write_state(monkeypatch):
    _enable_scope_hitl(monkeypatch)

    async def run(response):
        handler = _ScriptedHandler([response])
        ctx = _ctx()
        cb = make_ask_pipeline_scope_callback(handler)
        assert await cb(ctx) is None
        assert STATE_KEY not in ctx.state
        assert ctx.state[DIRECTIVE_KEY] == SKIP_DIRECTIVE
        assert len(handler.requests) == 1
        assert handler.requests[0].options == list(SELECT_OPTIONS)

    asyncio.run(run(HITLResponse(
        action=HITLAction.SELECT, approved=True, selected_option=OPTION_SKIP,
    )))
    asyncio.run(run(HITLResponse(action=HITLAction.APPROVE, approved=True)))


def test_callback_custom_form_writes_scope_and_directive(monkeypatch):
    _enable_scope_hitl(monkeypatch)
    handler = _ScriptedHandler([
        HITLResponse(
            action=HITLAction.SELECT, approved=True, selected_option=OPTION_CUSTOM,
        ),
        HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            form_values={FORM_BLOCK: {"hypotheses": "да", "experiments": "yes"}},
        ),
    ])
    ctx = _ctx()
    cb = make_ask_pipeline_scope_callback(handler)

    async def run():
        assert await cb(ctx) is None

    asyncio.run(run())
    assert ctx.state[STATE_KEY]["hypotheses"] is True
    assert ctx.state[STATE_KEY]["experiments"] is True
    assert ctx.state[STATE_KEY]["research"] is False
    assert ctx.state[STATE_KEY]["source"] == "hitl"
    assert "preset" not in ctx.state[STATE_KEY]
    directive = ctx.state[DIRECTIVE_KEY]
    assert "HypothesesAgent" in directive
    assert "ExperimentModuleAgent" in directive
    assert "Do NOT call ResearchAgent" in directive
    assert "Full-cycle scientific research" not in directive
    assert handler.requests[1].form["blocks"][0]["title"] == FORM_BLOCK


def test_callback_empty_form_does_not_write_state(monkeypatch):
    _enable_scope_hitl(monkeypatch)
    handler = _ScriptedHandler([
        HITLResponse(
            action=HITLAction.SELECT, approved=True, selected_option=OPTION_CUSTOM,
        ),
        HITLResponse(action=HITLAction.APPROVE, approved=True, form_values={}),
    ])
    ctx = _ctx()
    cb = make_ask_pipeline_scope_callback(handler)
    asyncio.run(cb(ctx))
    assert STATE_KEY not in ctx.state
    assert ctx.state[DIRECTIVE_KEY] == SKIP_DIRECTIVE


def test_callback_repeat_entry_is_noop(monkeypatch):
    _enable_scope_hitl(monkeypatch)
    handler = _ScriptedHandler([])
    existing = {"research": True, "hypotheses": False, "experiments": False, "source": "hitl"}
    ctx = _ctx({STATE_KEY: existing, DIRECTIVE_KEY: "already"})
    cb = make_ask_pipeline_scope_callback(handler)
    asyncio.run(cb(ctx))
    assert handler.requests == []
    assert ctx.state[STATE_KEY] is existing


def _attach_console(monkeypatch) -> None:
    """Pretend a console is attached: the handler answers headless runs itself."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)


def test_console_select_by_option_number(monkeypatch):
    answers = iter(["2"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    _attach_console(monkeypatch)
    handler = ConsoleHITLHandler()

    async def run():
        return await handler.handle_request(HITLRequest(
            agent_name="X",
            action_type=HITLAction.SELECT,
            message="pick",
            options=["alpha", "beta"],
        ))

    resp = asyncio.run(run())
    assert resp.selected_option == "beta"
    assert resp.approved is True


def test_console_form_collects_nonempty_fields(monkeypatch):
    answers = iter(["да", "", "yes"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    _attach_console(monkeypatch)
    handler = ConsoleHITLHandler()
    form = {
        "title": "t",
        "intro": "i",
        "blocks": [{
            "title": FORM_BLOCK,
            "fields": [
                {"name": "research"},
                {"name": "hypotheses"},
                {"name": "experiments"},
            ],
        }],
    }

    async def run():
        return await handler.handle_request(HITLRequest(
            agent_name="X",
            action_type=HITLAction.APPROVE,
            message="form",
            form=form,
        ))

    resp = asyncio.run(run())
    assert resp.form_values == {FORM_BLOCK: {"research": "да", "experiments": "yes"}}
    assert pack_scope(resp.form_values)["research"] is True
    assert pack_scope(resp.form_values)["experiments"] is True
    assert pack_scope(resp.form_values)["hypotheses"] is False


def test_planner_context_copies_hitl_pipeline_scope():
    from CoScientist.experiments.context import build_experiment_context

    state = {
        "experiment_source_request": "Dock a molecule with ready MCP tools.",
        "pipeline_scope": {
            "research": False,
            "hypotheses": False,
            "experiments": True,
            "source": "hitl",
        },
    }
    ctx = SimpleNamespace(state=state, user_content=None)
    build_experiment_context(ctx)
    scope = ctx.state["experiment_context"]["pipeline_scope"]
    assert scope == {"research": False, "hypotheses": False, "experiments": True}
    assert '"pipeline_scope"' in ctx.state["experiment_planner_context"]
    assert '"research":false' in ctx.state["experiment_planner_context"]


def test_experiments_yaml_prepares_the_turn_before_asking_scope():
    config = load_config(resolve_config_path("experiments"))
    before = config.agent("OrchestratorAgent").callbacks.before_agent
    assert before[:2] == ["prepare_experiment_user_turn", "ask_pipeline_scope"]
    orch = config.agent("OrchestratorAgent").callbacks
    assert "inject_pipeline_scope_directive" not in orch.before_model
    # expand_link_refs owns the first slot — it is the egress point for the
    # agent's own text, and any callback that returns a response short-circuits
    # the rest of the chain. What matters for the scope is that it is enforced
    # before the module-call callbacks reshape the turn.
    assert orch.after_model[0] == "expand_link_refs"
    assert orch.after_model.index("enforce_pipeline_scope_hops") < (
        orch.after_model.index("coalesce_experiment_module_calls")
    )
    assert "mark_pipeline_scope_lane" in orch.after_tool


def test_prompt_scope_hitl_is_placeholder_instead_of_classifier(monkeypatch):
    from CoScientist.agents.prompts.templates import orchestrator
    from CoScientist.assembly.prompting import PromptContext

    config = load_config(resolve_config_path("experiments"))
    ctx = PromptContext(config=config.agent("OrchestratorAgent"), system=config, tool_entries=[])
    web = get_settings().web
    monkeypatch.setattr(web, "scope_hitl", False)
    monkeypatch.setattr(web, "hitl_enabled", False)
    off = orchestrator(ctx)
    assert SCOPE_BASKETS_ALL in off
    assert "{pipeline_scope_directive?}" not in off

    monkeypatch.setattr(web, "scope_hitl", True)
    monkeypatch.setattr(web, "hitl_enabled", True)
    on = orchestrator(ctx)
    assert "{pipeline_scope_directive?}" in on
    assert SCOPE_BASKETS_ALL not in on
    assert SKIP_DIRECTIVE not in on
    assert "Full-cycle scientific research" not in on
    assert "hops already ran" not in on
    assert "ALWAYS call `retrieve_tools` FIRST" not in on


def _llm(text=None, calls=None):
    from google.adk.models import LlmResponse
    from google.genai import types

    parts = []
    if text:
        parts.append(types.Part(text=text))
    for name, args in calls or []:
        parts.append(types.Part.from_function_call(name=name, args=args))
    return LlmResponse(content=types.Content(role="model", parts=parts))


def test_enforce_hops_forces_named_agent_on_prose():
    from CoScientist.hitl.pipeline_scope import enforce_pipeline_scope_hops

    ctx = _ctx({
        STATE_KEY: pack_scope({FORM_BLOCK: {"research": "да"}}),
        "research_frame": {"original_request": "find ligands"},
    })
    out = enforce_pipeline_scope_hops(ctx, _llm(text="the frame is ready"))
    fc = out.content.parts[0].function_call
    assert fc.name == "ResearchAgent"
    assert fc.args["request"] == "find ligands"


def test_enforce_hops_noop_on_skip_and_correct_call():
    from CoScientist.hitl.pipeline_scope import enforce_pipeline_scope_hops

    skip_ctx = _ctx({DIRECTIVE_KEY: SCOPE_BASKETS_ALL})
    assert enforce_pipeline_scope_hops(skip_ctx, _llm(text="x")) is None

    ctx = _ctx({STATE_KEY: pack_scope({FORM_BLOCK: {"research": "да"}})})
    already = _llm(calls=[("ResearchAgent", {"request": "find ligands"})])
    assert enforce_pipeline_scope_hops(ctx, already) is None


def test_enforce_hops_allows_retrieve_then_forces_em_after_lane_done():
    from CoScientist.hitl.pipeline_scope import (
        DONE_KEY,
        enforce_pipeline_scope_hops,
        mark_pipeline_scope_lane,
    )

    em = pack_scope({FORM_BLOCK: {"experiments": "да"}})
    ctx = _ctx({STATE_KEY: em})
    retrieve = _llm(calls=[("retrieve_tools", {"query": "docking"})])
    assert enforce_pipeline_scope_hops(ctx, retrieve) is None

    mixed = pack_scope({FORM_BLOCK: {"research": "да", "experiments": "да"}})
    tool_ctx = SimpleNamespace(state={STATE_KEY: mixed, DONE_KEY: []})
    mark_pipeline_scope_lane(SimpleNamespace(name="ResearchAgent"), {}, tool_ctx)
    assert tool_ctx.state[DONE_KEY] == ["ResearchAgent"]
    hop_ctx = _ctx(tool_ctx.state)
    hop_ctx.state["orchestrator_root_goal"] = "run docking"
    out = enforce_pipeline_scope_hops(hop_ctx, _llm(text="research done"))
    assert out.content.parts[0].function_call.name == "ExperimentModuleAgent"


def test_targeted_experiment_redo_reopens_lane_exactly_once():
    from CoScientist.hitl.pipeline_scope import DONE_KEY, mark_pipeline_scope_lane

    scope = pack_scope({FORM_BLOCK: {"experiments": "yes"}})
    state = {
        STATE_KEY: scope,
        DONE_KEY: ["ExperimentModuleAgent"],
        "experiment_module_outcome": {
            "resume_required": True,
            "reason": "targeted_redo_requested",
            "selected_task_ids": ["EXP-2"],
            "affected_task_ids": ["EXP-2", "EXP-3"],
        },
    }
    context = SimpleNamespace(state=state)
    tool = SimpleNamespace(name="ExperimentModuleAgent")

    mark_pipeline_scope_lane(tool, {}, context)

    assert "ExperimentModuleAgent" not in state[DONE_KEY]
    assert state["experiment_targeted_redo_pending"] == {
        "selected_task_ids": ["EXP-2"],
        "affected_task_ids": ["EXP-2", "EXP-3"],
    }
    assert state["experiment_module_outcome"]["resume_required"] is False
    assert state["experiment_module_outcome"]["resume_dispatched"] is True

    # The return from that explicit redo consumes the one-shot marker and
    # closes the lane only after result review accepted the updated stage;
    # stale review state or a failed AgentTool return cannot mark it DONE.
    state["experiment_runtime"] = {"phase": "completed"}
    state["experiment_module_outcome"] = {
        "status": "completed",
        "stage": "result_review",
        "reason": "result_approved",
        "accepted": True,
    }
    mark_pipeline_scope_lane(tool, {}, context)
    assert state["experiment_targeted_redo_pending"] is None
    assert state[DONE_KEY] == ["ExperimentModuleAgent"]
