"""Auto-approval of the experiment reviews, from the Approvals tab.

Observed 2026-09-21 on a live Heracleum run: the plan went up for review at
13:36:26 with a 300 s window, nobody answered, and at 13:41:26 the fail-closed
policy rejected it — "an experiment plan is never approved because nobody was
watching". The run then wrote its report without running a single experiment.
The only way through was an environment variable, which means restarting the
server, which the operator of a running session cannot do.

So both reviews get their own switch, each read at call time. Two properties
matter beyond "the flag works":

* the switches are per kind. Approving a plan unseen spends the run; accepting
  whatever came out of it spends the conclusions. One switch for both would
  make the cheaper choice buy the dearer one.
* they do NOT hang off the global "ask for my approval" switch. The experiment
  reviewer overrides ``_should_run_review`` and asks even when that switch is
  off, so if the UI greyed these out with it, the one way past a paused plan
  would be hidden exactly when it is needed. The last test pins that.
"""
from __future__ import annotations

import asyncio
import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from CoScientist.config import get_settings
from CoScientist.experiments import review as review_mod
from CoScientist.experiments.review import ExperimentReviewSessionAgent, _auto_approve
from CoScientist.hitl.models import HITLAction, HITLResponse
from CoScientist.hitl.session_agent import SessionAgent

from .helpers import _inventory, _plan, _task

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "CoScientist" / "web" / "static" / "js"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """The legacy one-switch override off, and the flags back as they were."""
    monkeypatch.delenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", raising=False)
    exp = get_settings().experiments
    before = (exp.plan_auto_approve, exp.result_auto_approve,
              exp.plan_review_timeout_s, exp.result_review_timeout_s)
    exp.plan_auto_approve = False
    exp.result_auto_approve = False
    yield exp
    (exp.plan_auto_approve, exp.result_auto_approve,
     exp.plan_review_timeout_s, exp.result_review_timeout_s) = before


def _review_plan(monkeypatch, state=None):
    plan = _plan(_task("EXP-1"))
    state = state if state is not None else {}
    state.setdefault("experiment_context", {
        "experiment_run_id": plan.experiment_run_id,
        "source_request": plan.source_request,
        "available_mcp_capabilities": _inventory(),
    })
    asked = []
    monkeypatch.setattr(review_mod, "approve_plan", lambda _s: None)
    monkeypatch.setattr(review_mod, "_publish_approved_plan_to_graph", lambda *_a, **_k: None)
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")

    # A console that records the question and answers yes. Whether it was
    # reached at all is what these tests are about.
    async def _console(request):
        asked.append(request)
        return HITLResponse(action=HITLAction.APPROVE, approved=True, instructions="")

    agent.hitl_handler = SimpleNamespace(handle_request=_console)
    ctx = SimpleNamespace(session=SimpleNamespace(state=state), invocation_id="inv-1")
    response = asyncio.run(agent._review_plan(ctx, plan.model_dump_json()))
    return response, asked, state


def test_the_legacy_plan_switch_no_longer_bypasses_the_mode(_clean_env, monkeypatch):
    _clean_env.plan_auto_approve = True
    response, asked, _ = _review_plan(monkeypatch)

    assert response.approved and asked, "the legacy switch bypassed the review"
    assert _auto_approve("plan") is False
    assert _auto_approve("result") is False


def test_a_plan_nobody_switched_on_is_still_put_up_for_review(_clean_env, monkeypatch):
    """The default: the operator is asked, and silence is not consent."""
    _, asked, _ = _review_plan(monkeypatch)
    assert len(asked) == 1, "the plan was approved without asking anyone"
    assert _auto_approve("plan") is False and _auto_approve("result") is False


def test_the_legacy_environment_switch_no_longer_covers_reviews(_clean_env, monkeypatch):
    monkeypatch.setenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", "1")
    assert not _auto_approve("plan") and not _auto_approve("result")

    monkeypatch.delenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE")
    _clean_env.result_auto_approve = True
    assert not _auto_approve("plan") and not _auto_approve("result")


def test_the_tab_keeps_legacy_review_fields_readable_but_inactive(_clean_env):
    from CoScientist.web.app import _apply_frontend_settings, _current_settings

    _apply_frontend_settings({"experimentModule": {
        "planAutoApprove": True, "resultAutoApprove": False,
        "planReviewTimeoutS": 1800, "resultReviewTimeoutS": 900,
    }})
    assert _auto_approve("plan") is False and _auto_approve("result") is False
    assert _clean_env.plan_review_timeout_s == 1800

    echoed = _current_settings()["experimentModule"]
    assert echoed == {
        "planAutoApprove": True, "resultAutoApprove": False,
        "planReviewTimeoutS": 1800, "resultReviewTimeoutS": 900,
        "routeFedot": get_settings().experiments.route_fedot,
        "routeAlembic": get_settings().experiments.route_alembic,
    }, echoed


def test_the_agents_tab_switches_fedot_for_the_next_session(monkeypatch):
    """The FEDOT.MAS route lands on the one switch every route decision reads,
    and reads back, so the modal does not show a stale toggle."""
    from CoScientist.experiments.runtime.state_machine import fedot_route_available
    from CoScientist.web.app import _apply_frontend_settings, _current_settings

    exp = get_settings().experiments
    monkeypatch.setattr(exp, "route_fedot", True)
    _apply_frontend_settings({"experimentModule": {"routeFedot": False}})
    assert exp.route_fedot is False
    assert fedot_route_available() is False
    assert _current_settings()["experimentModule"]["routeFedot"] is False

    _apply_frontend_settings({"experimentModule": {"routeFedot": True}})
    assert exp.route_fedot is True
    assert _current_settings()["experimentModule"]["routeFedot"] is True


def test_settings_switch_alembic_route_without_rebuilding_the_agent_tree(monkeypatch):
    from CoScientist.web.app import _apply_frontend_settings, _current_settings

    exp = get_settings().experiments
    monkeypatch.setattr(exp, "route_alembic", False)
    _apply_frontend_settings({"experimentModule": {"routeAlembic": True}})
    assert exp.route_alembic is True
    assert _current_settings()["experimentModule"]["routeAlembic"] is True


def test_the_agents_tab_switches_the_medical_agent(monkeypatch):
    """MEDICAL__ENABLED from the browser: it lands on the flag MedicalAgent's
    `enabled` reads, takes the experiment medical route with it, and reads back."""
    from CoScientist.experiments.runtime.state_machine import medical_route_available
    from CoScientist.web.app import _apply_frontend_settings, _current_settings

    web = get_settings().web
    monkeypatch.setattr(web, "medical_agent_enabled", True)
    _apply_frontend_settings({"medicalAgent": {"enabled": False}})
    assert web.medical_agent_enabled is False
    assert medical_route_available() is False
    assert _current_settings()["medicalAgent"] == {"enabled": False}

    _apply_frontend_settings({"medicalAgent": {"enabled": True}})
    assert web.medical_agent_enabled is True
    assert medical_route_available() is True


def test_the_medical_switch_is_in_the_agents_tab_with_its_own_scope_hint():
    modal = _read(WEB / "modals" / "settings.js")
    agents = modal[modal.index("id: 'agents'"):modal.index("id: 'models'")]
    row = agents[agents.index("id: 'medicalAgent'"):]
    row = row[:row.index("},")]
    assert "path: 'medicalAgent.enabled'" in row
    assert "env: 'MEDICAL__ENABLED'" in row
    assert "scopeHintKey: 'settings.f.medicalAgent.scopeHint'" in row
    assert "'settings.f.medicalAgent.scopeHint'" in _read(WEB / "i18n.js")


def test_the_fedot_switch_is_in_the_agents_tab_and_session_scoped():
    modal = _read(WEB / "modals" / "settings.js")
    agents = modal[modal.index("id: 'agents'"):modal.index("id: 'models'")]
    row = agents[agents.index("id: 'experimentRouteFedot'"):]
    row = row[:row.index("},")]
    assert "path: 'experimentModule.routeFedot'" in row
    assert "scope: 'session'" in row
    assert "env: 'EXPERIMENTS__ROUTE_FEDOT'" in row


@pytest.mark.parametrize("bad", [0, -1, "", "later", None])
def test_a_window_of_zero_does_not_become_wait_forever(_clean_env, bad):
    """`handler.py` reads a non-positive window as "no deadline", and these
    two reviews block the run while they wait. A field that fails validation
    must leave the previous window standing, not hand the run an open one."""
    from CoScientist.web.app import _apply_frontend_settings

    _clean_env.plan_review_timeout_s = 300.0
    _apply_frontend_settings({"experimentModule": {"planReviewTimeoutS": bad}})
    assert _clean_env.plan_review_timeout_s == 300.0


def test_the_experiment_review_asks_even_with_approvals_off(monkeypatch):
    """Why the two switches are not children of `general.hitlEnabled` in the
    modal: turning that off does not stop these reviews."""
    agent = ExperimentReviewSessionAgent(name="Reviewer", review_kind="plan")
    agent.hitl_handler = SimpleNamespace(handle_request=lambda req: None)
    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)

    assert agent._should_run_review() is True
    # The behaviour it overrides, for contrast: every other SessionAgent stops
    # asking when the global switch is off.
    assert SessionAgent._should_run_review(agent) is False


# ── the tab itself ──────────────────────────────────────────────────────────

def _read(path: Path) -> str:
    return io.open(path, encoding="utf-8").read()


def _js_block(src: str, key: str) -> str:
    """The text of the `key: { … }` object literal, by brace counting."""
    start = src.index(f"{key}: {{")
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i:j + 1]
    raise AssertionError(f"unbalanced braces after {key}")


def test_every_field_in_the_modal_has_a_label_and_a_home():
    """A field whose label key is missing renders its own key to the operator;
    a field whose path is not in `appSettings` is dropped by
    `mergeServerSettings`, which only copies keys it already knows — the
    switch would then read back as its default after every save."""
    modal = _read(WEB / "modals" / "settings.js")
    labels = set(re.findall(r"'settings\.f\.([A-Za-z0-9]+)\.label'",
                            _read(WEB / "i18n.js")))
    state = _read(WEB / "state.js")

    missing_label, missing_home = [], []
    for fid, path in re.findall(
            r"id: '([A-Za-z0-9]+)',\s*path: '([A-Za-z0-9.]+)'", modal):
        if fid not in labels:
            missing_label.append(fid)
        section, _, leaf = path.partition(".")
        if f"{leaf}:" not in _js_block(state, section):
            missing_home.append(path)

    assert not missing_label, f"no settings.f.<id>.label for: {missing_label}"
    assert not missing_home, f"not in appSettings: {missing_home}"


def test_the_two_experiment_switches_are_offered_in_the_approvals_tab():
    modal = _read(WEB / "modals" / "settings.js")
    approvals = modal[modal.index("id: 'approvals'"):modal.index("id: 'tools'")]
    for fid in ("experimentPlanAuto", "experimentResultAuto",
                "experimentPlanTimeout", "experimentResultTimeout"):
        assert f"id: '{fid}'" in approvals, f"{fid} is not in the Approvals tab"
    # Not parented: see test_the_experiment_review_asks_even_with_approvals_off.
    for fid in ("experimentPlanAuto", "experimentResultAuto"):
        row = approvals[approvals.index(f"id: '{fid}'"):]
        row = row[:row.index("},")]
        assert "parent:" not in row, f"{fid} was tied to the global switch"


# ── the window itself, and whose switch owns it ──────────────────────────────
# The docstring at the top of this file is the incident. Auto-approval was one
# answer to it; this is the other. The operator had already set
# HITL_AUTO_APPROVE_TIMEOUT=-1 — "I am at the console, do not decide without
# me" — and this was the one path that did not hear it: `handle_request` prefers
# a request's own window, and the review always set one.

_ANY = SimpleNamespace(hitl_handler=SimpleNamespace())


def test_basic_mode_gives_the_review_the_wait_it_advertises(monkeypatch):
    """The mode owns the wait outright. Taking the tighter of the two was the
    obvious thing and the wrong one: the default review window is 300 s, so the
    plan card — of all cards — got half the wait `basic` names, and an operator
    who set the field to an hour was capped to ten minutes with nothing saying
    so."""
    monkeypatch.setenv("HITL__MODE", "basic")
    assert ExperimentReviewSessionAgent._review_window(_ANY, 300.0) == 600.0
    assert ExperimentReviewSessionAgent._review_window(_ANY, 3600.0) == 600.0


def test_debug_mode_makes_the_review_wait_for_the_human(monkeypatch):
    """This was the one voice the operator's switch did not reach:
    `handle_request` prefers a request's own window, so someone who had turned
    every timeout off still had the plan review expire at 300 s and the run skip
    execution."""
    monkeypatch.setenv("HITL__MODE", "debug")
    assert ExperimentReviewSessionAgent._review_window(_ANY, 300.0) is None


def test_auto_mode_never_reaches_a_wait(monkeypatch):
    """`_auto_approve` answered first, so this is only about not returning
    something that would mean "refuse immediately" if a path ever got here."""
    monkeypatch.setenv("HITL__MODE", "auto")
    assert _auto_approve("plan") is True
    assert ExperimentReviewSessionAgent._review_window(_ANY, 300.0) == 0.0


def test_an_unreadable_mode_keeps_the_bounded_window(monkeypatch):
    """Fail safe, not fail open: if the mode cannot be read, the review keeps
    its deadline rather than silently becoming an indefinite wait."""
    import CoScientist.hitl.mode as mode_mod

    def explode():
        raise RuntimeError("настройки недоступны")

    monkeypatch.setattr(mode_mod, "wait_seconds", explode)
    assert ExperimentReviewSessionAgent._review_window(_ANY, 300.0) == 600.0


def test_the_window_reaches_the_request_the_handler_reads():
    """`_hitl` is the only place a window becomes a request, and the handler
    treats a non-positive one as no deadline at all."""
    stub = SimpleNamespace(name="ExperimentPlannerAgent")
    built = ExperimentReviewSessionAgent._hitl(
        stub, message="m", kind="plan", plan_id="PLAN-1", output="o",
        user_id="u", session_id="s", timeout_seconds=None)
    assert built.timeout_seconds is None
    assert built.context["experiment_review_kind"] == "plan"


def test_a_timed_out_review_with_no_deadline_still_audits():
    """`None` is a real window now, and `timed_out` can still come back under
    it: `FailClosedExperimentHITLHandler` answers that whenever no interactive
    reviewer is connected — a console run, an A2A call, a closed tab. The audit
    line formatted the window as a number and brought the whole review down
    with a TypeError.
    """
    from CoScientist.experiments.review import _window_word

    assert _window_word(None) == "none"
    assert _window_word(0) == "none"
    assert _window_word(300.0) == "300"


def test_the_research_frame_and_the_nir_report_are_in_the_research_tab_with_reasoning():
    """Both are stages of a study, not agents to pick: their switches and the
    reasoning of the agents behind them live in Research flow, and the Agents
    list leaves those agents out (the agentReasoning fields name them)."""
    modal = _read(WEB / "modals" / "settings.js")
    research = modal[modal.index("id: 'research'"):modal.index("id: 'approvals'")]
    agents = modal[modal.index("id: 'agents'"):modal.index("id: 'models'")]
    for field in ("contextInit", "nirReport"):
        assert f"id: '{field}'" in research and f"id: '{field}'" not in agents
    assert "id: 'contextInitReasoning', type: 'agentReasoning', agentsOf: 'general.contextInitEnabled'" in research
    assert "id: 'nirReportReasoning', type: 'agentReasoning', agentsOf: 'nirReport.enabled'" in research
    assert "!AGENTS_SHOWN_ELSEWHERE.has(a.enabledSetting)" in modal
