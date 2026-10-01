"""campaign_start и campaign_approve не проходят без карточки в веб-интерфейсе.

`hitl_before_campaign_start` показывает карточку перед campaign_start;
campaign_approve спрашивает сам, с планом кампании. Если HITL выключен или
веб-интерфейс не подключён, оба блокируются, а не выполняются молча.

Run from the repo root:  pytest tests/unit/test_campaign_start_hitl.py -q
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from CoScientist.hitl.callbacks import make_hitl_before_tool_callback
from CoScientist.microfluidics.a2a_optimization import adapter, campaign

_GATE = "CoScientist.hitl.human_gate.unavailable_reason"


def _settings(enabled: bool):
    return SimpleNamespace(web=SimpleNamespace(hitl_enabled=enabled))


def _handler(**answer):
    handler = MagicMock()
    decision = {"approved": False, "timed_out": False, "instructions": None,
                "free_input": None, **answer}
    handler.handle_request = AsyncMock(return_value=SimpleNamespace(**decision))
    return handler


# ── campaign_start ───────────────────────────────────────────────────────────

def _start(handler, name="campaign_start", unavailable=None):
    cb = make_hitl_before_tool_callback(
        handler, target_tools=("campaign_start",), require_hitl=True,
    )
    ctx = SimpleNamespace(agent_name="OptimizationAgent")
    with patch(_GATE, return_value=unavailable), \
         patch("CoScientist.hitl.callbacks.get_settings", return_value=_settings(True)), \
         patch("CoScientist.hitl.callbacks.session_key", return_value=("u", "s")):
        return asyncio.run(cb(tool=SimpleNamespace(name=name), args={}, tool_context=ctx))


def test_start_blocked_without_web_hitl():
    handler = _handler()
    result = _start(handler, unavailable="HITL is switched off")
    assert result["blocked_by"] == "hitl_unavailable"
    handler.handle_request.assert_not_awaited()


def test_start_waits_for_the_card():
    handler = _handler(instructions="нет")
    assert _start(handler)["blocked_by"] == "human"

    handler.handle_request.return_value.approved = True
    assert _start(handler) is None
    assert handler.handle_request.await_args.args[0].context["tool"] == "campaign_start"


def test_other_campaign_tools_are_not_gated():
    handler = _handler()
    assert _start(handler, name="campaign_get_status", unavailable="HITL is switched off") is None
    handler.handle_request.assert_not_awaited()


def test_default_callback_still_skips_when_hitl_is_off():
    handler = _handler()
    cb = make_hitl_before_tool_callback(handler, target_tools=("run_sandbox_task",))
    with patch("CoScientist.hitl.callbacks.get_settings", return_value=_settings(False)):
        assert asyncio.run(cb(tool=SimpleNamespace(name="run_sandbox_task"), args={},
                              tool_context=SimpleNamespace())) is None


# ── campaign_approve ─────────────────────────────────────────────────────────

def _waiting_for_approval():
    return SimpleNamespace(state={campaign.ACTIVE_KEY: {
        "task_id": "t1", "experiment_id": "campaign-1", "context_id": "ctx-1",
        "state": "input_required", "phase": "approval",
        "task": {"status": {"message": {"parts": [{"text": "plan"}]}}},
    }})


def _approve(handler, unavailable=None):
    context = _waiting_for_approval()
    sent = AsyncMock(return_value={"state": "working"})
    with patch("CoScientist.agents.common.hitl_handler", handler), \
         patch(_GATE, return_value=unavailable), \
         patch("CoScientist.graph.session_scope.session_key", return_value=("u", "s")), \
         patch.object(adapter, "_continue", sent):
        return asyncio.run(campaign.campaign_approve(context)), sent, context


def test_approve_blocked_without_web_hitl():
    handler = _handler()
    result, sent, context = _approve(handler, unavailable="the web interface is not attached")
    assert result["state"] == "error"
    handler.handle_request.assert_not_awaited()
    sent.assert_not_awaited()
    assert context.state[campaign.ACTIVE_KEY]["phase"] == "approval"


def test_approve_rejected_on_the_card_sends_nothing():
    handler = _handler(instructions="нет")
    result, sent, _ = _approve(handler)
    assert result["state"] == "rejected"
    sent.assert_not_awaited()


def test_approve_sent_only_after_the_card_is_approved():
    handler = _handler()
    handler.handle_request.return_value.approved = True
    result, sent, _ = _approve(handler)
    assert result == {"state": "working"}
    assert sent.await_args.args[2] == "Approve"
    assert handler.handle_request.await_args.args[0].context["plan"]
