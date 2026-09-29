"""Три режима подтверждений — и отказ, который наконец останавливает модуль.

До этого «спрашивать или нет» и «сколько ждать» решались в четырёх местах
независимо, и решения расходились: 34 из 45 истёкших карточек в сохранённых
сессиях закончились автоматическим «одобрено», причём уровень `side_effect` —
самый опасный — был единственным с обратным отсчётом, а `read` и `compute`
ждали человека вечно. Предсказать поведение по настройкам было нельзя.

Отдельно — отказ от плана эксперимента. «Отклонить» и «Доработать» уходили в
один и тот же `_feed_back`, потому что ни одна из двух ветвей не ставила
`stop_review_loop`, и планировщик просто писал план заново. Сказать «прекрати»
было нечем: только «попробуй ещё раз».

Run from the repo root:  pytest tests/unit/test_hitl_mode.py -q
"""
import asyncio

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.hitl import mode as mode_mod  # noqa: E402
from CoScientist.hitl.models import (  # noqa: E402
    HITLAction,
    HITLDecisionSource,
    HITLRequest,
    HITLResponse,
)


@pytest.fixture(autouse=True)
def _no_inherited_mode(monkeypatch):
    """Каждый тест называет режим сам: `.env` стенда не должен решать за него."""
    monkeypatch.delenv("HITL__MODE", raising=False)


# ── сами режимы ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("named,expected,wait", [
    ("auto", mode_mod.AUTO, 0.0),
    ("basic", mode_mod.BASIC, mode_mod.BASIC_WAIT_S),
    ("debug", mode_mod.DEBUG, None),
    ("AUTO", mode_mod.AUTO, 0.0),
    ("  basic  ", mode_mod.BASIC, mode_mod.BASIC_WAIT_S),
])
def test_the_named_mode_decides_the_wait(monkeypatch, named, expected, wait):
    monkeypatch.setenv("HITL__MODE", named)
    assert mode_mod.hitl_mode() == expected
    assert mode_mod.wait_seconds() == wait


def test_the_basic_wait_is_ten_minutes():
    assert mode_mod.BASIC_WAIT_S == 600.0


def test_silence_approves_in_no_mode():
    """Главное правило. Прогону, который должен идти без человека, назван
    отдельный режим — и он называется вслух, а не выводится из того, что никто
    не смотрел."""
    for named in mode_mod.MODES:
        assert mode_mod.silence_approves() is False, named


def test_a_misspelled_mode_falls_back_to_asking(monkeypatch):
    """Опечатка в настройке не должна молча включать автоподтверждение."""
    monkeypatch.setenv("HITL__MODE", "atuo")
    assert mode_mod.hitl_mode() == mode_mod.BASIC
    assert mode_mod.auto_approves() is False


# ── совместимость со стендами, настроенными до появления ручки ───────────────

def test_the_hitl_switch_is_not_a_mode(monkeypatch):
    """`HITL__ENABLED=false` must NOT read as `auto`, tempting as it looks. That
    switch removes the callbacks and the work-order tools, but the two
    experiment reviews ask even when it is off — deliberately, and the Approvals
    tab says so: greying them out with the global switch would hide the only way
    past a paused plan. Deriving `auto` from it silently approved a 7-task,
    230-minute experiment plan without ever drawing a card."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "hitl_enabled", False)
    monkeypatch.setattr(get_settings().web, "hitl_mode", "")
    monkeypatch.setattr(get_settings().web, "hitl_auto_approve_timeout", 300)
    assert mode_mod.hitl_mode() == mode_mod.BASIC
    assert mode_mod.auto_approves() is False

    from CoScientist.experiments.review import _auto_approve

    monkeypatch.setattr(get_settings().experiments, "plan_auto_approve", False)
    monkeypatch.setattr(get_settings().experiments, "result_auto_approve", False)
    monkeypatch.delenv("COSCIENTIST_EXPERIMENT_HITL_AUTO_APPROVE", raising=False)
    assert _auto_approve("plan") is False
    assert _auto_approve("result") is False


def test_a_non_positive_legacy_timeout_reads_as_debug(monkeypatch):
    """`HITL_AUTO_APPROVE_TIMEOUT=-1` дословно означает «ждать сколько
    угодно» — это и есть debug, и стенд с такой настройкой не должен менять
    поведение от того, что появился режим."""
    from CoScientist.config import get_settings

    monkeypatch.setattr(get_settings().web, "hitl_enabled", True)
    monkeypatch.setattr(get_settings().web, "hitl_mode", "")
    monkeypatch.setattr(get_settings().web, "hitl_auto_approve_timeout", -1)
    assert mode_mod.hitl_mode() == mode_mod.DEBUG

    monkeypatch.setattr(get_settings().web, "hitl_auto_approve_timeout", 300)
    assert mode_mod.hitl_mode() == mode_mod.BASIC


# ── что видит обработчик ─────────────────────────────────────────────────────

def _ask(agent="CoderAgent", action=HITLAction.APPROVE, **kw):
    from CoScientist.web.handler import WebHITLHandler

    handler = WebHITLHandler()
    request = HITLRequest(agent_name=agent, action_type=action,
                          message="Approve", **kw)
    return handler, asyncio.run(handler.handle_request(request))


def test_auto_draws_no_card_and_answers_yes(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "auto")
    handler, answer = _ask(timeout_seconds=600)
    assert answer.approved is True and answer.timed_out is False
    assert not handler._pending, "карточка не должна попадать в очередь ожидания"


def test_basic_refuses_when_nobody_answers(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "basic")
    monkeypatch.setattr(mode_mod, "wait_seconds", lambda: 0.001)
    _, answer = _ask(timeout_seconds=0.001)
    assert answer.approved is False
    assert answer.timed_out is True, "и об этом сказано прямо"


def test_the_refusal_puts_no_words_in_the_operators_mouth(monkeypatch):
    """`instructions` must stay EMPTY on a timeout, and that is load-bearing.
    Every consumer reads those fields as the operator's own words: the ТЗ
    interview recorded the explanation as the ANSWER to a question and put it
    into the document, and the sandbox handed it to an agent as "Follow user
    instructions: …". The fact that nobody answered travels in `timed_out`."""
    monkeypatch.setenv("HITL__MODE", "basic")
    monkeypatch.setattr(mode_mod, "wait_seconds", lambda: 0.001)
    _, answer = _ask(timeout_seconds=0.001)
    assert answer.timed_out is True
    assert not (answer.instructions or "")
    assert not (answer.free_input or "")


def test_a_headless_console_refuses_and_says_nobody_decided(monkeypatch):
    """The console handler is what every non-web run gets. It used to default to
    APPROVE with no console attached, which meant `debug` — the mode that exists
    so a run cannot leave without you — approved everything instantly on any box
    with no terminal. And its refusal must carry `timed_out`, or the experiment
    plan review records "rejected by the operator" against an operator who was
    never there."""
    import sys

    from CoScientist.hitl.handler import ConsoleHITLHandler

    monkeypatch.setenv("HITL__MODE", "debug")
    monkeypatch.delenv("HITL_HEADLESS_POLICY", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    answer = asyncio.run(ConsoleHITLHandler().handle_request(
        HITLRequest(agent_name="CoderAgent", action_type=HITLAction.APPROVE,
                    message="Approve")))
    assert answer.approved is False
    assert answer.timed_out is True, "nobody decided this"
    assert not (answer.instructions or "")

    from CoScientist.experiments.review import _is_refusal

    assert _is_refusal(answer) is False, "an absent human is not a rejection"


def test_auto_mode_does_not_block_a_console_run(monkeypatch):
    """The auto short-circuit used to live only in the web handler, so the one
    mode whose whole point is running unattended blocked on `input()` forever on
    an interactive CLI run."""
    import sys

    from CoScientist.hitl.handler import ConsoleHITLHandler

    monkeypatch.setenv("HITL__MODE", "auto")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    answer = asyncio.run(ConsoleHITLHandler().handle_request(
        HITLRequest(agent_name="CoderAgent", action_type=HITLAction.APPROVE,
                    message="Approve")))
    assert answer.approved is True and answer.timed_out is False


def test_auto_mode_makes_an_explicit_choice_on_a_select(monkeypatch):
    """«Approve everything» answered a SELECT with nothing selected, and every
    reader of one treats "no option" as the negative branch — so the NIR report
    card was silently DECLINED. The first option is taken: a card lists the
    cheaper, less outward-facing path first."""
    monkeypatch.setenv("HITL__MODE", "auto")
    _, answer = _ask(action=HITLAction.SELECT, options=["short", "full GOST"])
    assert answer.approved is True
    assert answer.selected_option == "short"


def test_auto_select_prefers_an_explicit_default(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "auto")
    _, answer = _ask(
        action=HITLAction.SELECT,
        options=["short", "full NIR"],
        default_option="full NIR",
    )
    assert answer.action == HITLAction.SELECT
    assert answer.selected_option == "full NIR"
    assert answer.decision_source == HITLDecisionSource.MODE_AUTO
    assert answer.system_reason == "hitl_mode_auto"


def test_auto_completes_forms_with_defaults_and_empty_values(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "auto")
    form = {"blocks": [{"title": "Details", "fields": [
        {"name": "with_value", "value": "kept"},
        {"name": "with_default", "default": "fallback"},
        {"name": "required_but_empty", "required": True},
    ]}]}
    _, answer = _ask(form=form)
    assert answer.approved is True
    assert answer.form_values == {"Details": {
        "with_value": "kept",
        "with_default": "fallback",
        "required_but_empty": "",
    }}


def test_auto_provide_input_continues_with_an_empty_value(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "auto")
    _, answer = _ask(action=HITLAction.PROVIDE_INPUT)
    assert answer.approved is True
    assert answer.action == HITLAction.PROVIDE_INPUT
    assert answer.instructions == ""
    assert answer.free_input == ""


def test_timeout_has_its_own_source_and_no_operator_words(monkeypatch):
    monkeypatch.setenv("HITL__MODE", "basic")
    monkeypatch.setattr(mode_mod, "wait_seconds", lambda: 0.001)
    _, answer = _ask(timeout_seconds=999)
    assert answer.decision_source == HITLDecisionSource.TIMEOUT
    assert answer.system_reason == "review_window_elapsed"
    assert not (answer.instructions or answer.free_input)


def test_auto_mode_records_both_halves_of_the_decision(monkeypatch):
    """A response with no request reads as a gap in the record, and a reload or
    an export rebuilds the chat from exactly these events."""
    monkeypatch.setenv("HITL__MODE", "auto")
    from CoScientist.web.handler import AUTO_MARK, WebHITLHandler

    handler = WebHITLHandler()
    recorded = []
    handler.set_recorder(lambda key, event: recorded.append(event))
    asyncio.run(handler.handle_request(HITLRequest(
        agent_name="CoderAgent", action_type=HITLAction.APPROVE, message="Approve",
        context={"_session": {"user_id": "u", "session_id": "s"}})))

    assert [e["type"] for e in recorded] == ["hitl_request", "hitl_response"]
    assert all(e.get("auto") == AUTO_MARK for e in recorded), (
        "a decision the mode took must be readable as such")


def test_the_handlers_window_speaks_the_modes_wait(monkeypatch):
    from CoScientist.web.handler import WebHITLHandler

    monkeypatch.setenv("HITL__MODE", "basic")
    assert WebHITLHandler().hitl_timeout_seconds == 600.0
    monkeypatch.setenv("HITL__MODE", "debug")
    # Вызывающие читают «<= 0» как «срока нет» — это их словарь.
    assert WebHITLHandler().hitl_timeout_seconds <= 0


def test_an_environment_mode_is_reported_as_locked_to_the_ui(monkeypatch):
    from CoScientist.web.app import _current_settings

    monkeypatch.setenv("HITL__MODE", "auto")
    assert _current_settings()["general"]["hitlModePinnedByEnv"] is True

    from pathlib import Path

    settings_js = (Path(__file__).resolve().parents[2] / "CoScientist" / "web"
                   / "static" / "js" / "modals" / "settings.js").read_text(
                       encoding="utf-8")
    assert "general.hitlModePinnedByEnv" in settings_js
    assert "settings.inactive.envPinned" in settings_js


# ── отказ от плана эксперимента ──────────────────────────────────────────────

def _response(action, **kw):
    return HITLResponse(action=action, approved=kw.pop("approved", False), **kw)


def test_a_rejection_is_told_apart_from_a_revision():
    from CoScientist.experiments.review import _is_refusal

    assert _is_refusal(_response(HITLAction.REJECT)) is True
    # «Доработать» приходит как EDIT — это просьба переписать, не отказ.
    assert _is_refusal(_response(HITLAction.EDIT, instructions="добавь контроль")) is False
    # Отказ С заметкой — всё равно отказ: заметка это причина, а не «ещё раз».
    assert _is_refusal(_response(HITLAction.REJECT, instructions="цель не та")) is True
    # Утверждение — не отказ ни в каком виде.
    assert _is_refusal(_response(HITLAction.APPROVE, approved=True)) is False


def test_a_timeout_is_not_the_operator_refusing():
    """Молчание теперь приходит как `reject`, но никто ничего не решал — у него
    своя ветка и своя причина паузы."""
    from CoScientist.experiments.review import _is_refusal

    assert _is_refusal(_response(HITLAction.REJECT, timed_out=True)) is False
    assert _is_refusal(_response(HITLAction.REJECT, stop_review_loop=True)) is False


def test_the_review_loop_stops_on_a_rejection_and_not_on_a_revision():
    """`SessionAgent` кормит обратно всё, что не утверждено, — значит «стоп»
    выражается единственным флагом, и обзор плана его не ставил."""
    import inspect

    from CoScientist.experiments.review import ExperimentReviewSessionAgent
    from CoScientist.hitl.session_agent import SessionAgent

    body = inspect.getsource(ExperimentReviewSessionAgent._review_plan)
    assert "_is_refusal(response)" in body
    assert 'stop_review_loop": True' in body
    assert "plan_rejected_by_operator" in body
    # И флаг действительно прекращает цикл.
    loop = inspect.getsource(SessionAgent._run_async_impl)
    assert "response.timed_out or response.stop_review_loop" in loop


def test_a_positive_answer_approves_even_with_notes():
    """У карточки плана есть отдельная кнопка «Доработать», поэтому сокращение
    «утвердил с заметкой = переписать» было чистым вредом: оператор жал
    «Утвердить», писал уточнение, и только что утверждённый план выбрасывался."""
    from pathlib import Path

    js = (Path(__file__).resolve().parents[2] / "CoScientist" / "web" / "static"
          / "js" / "hitl.js").read_text(encoding="utf-8")
    assert "function respondHITLApprove(requestId)" in js
    approve = js.split("function respondHITLApprove(requestId)")[1].split("\n}")[0]
    assert "action: 'approve'" in approve and "approved: true" in approve
    # Карточка плана эксперимента зовёт именно её, а не сокращение.
    assert "respondHITLApprove('${escJs(rid)}')" in js


def test_approving_with_a_note_does_not_replace_the_thing_approved():
    """The nastiest consequence of making a positive answer approve. `SessionAgent`
    treats `instructions` on an approval as the EDITED OUTPUT and writes it into
    `output_key` — and the experiment planner's output_key is `experiment_plan`.
    So one sentence of feedback replaced the whole approved plan in session
    state while the original plan went on running. Replacing the output is what
    «provide input» means; a plain approve carrying a remark is not that.
    """
    import inspect

    from CoScientist.hitl.session_agent import SessionAgent

    body = inspect.getsource(SessionAgent._run_async_impl)
    assert "response.action == HITLAction.PROVIDE_INPUT" in body, (
        "only «provide input» may replace the output it was shown")
    assert "response.action != HITLAction.EDIT" not in body.split(
        "if response.approved:")[1].split("if not response.free_input")[0], (
        "an APPROVE must not reach the overwrite")


def test_an_unanswered_work_order_is_not_a_rejected_one():
    """A persisted `rejected` order makes every later `declare_work_order`
    return that rejection without asking anyone, so one missed card used to kill
    the agent for the rest of the run — even after the operator came back."""
    import inspect

    from CoScientist.hitl.work_order_tools import WorkOrderToolset

    for name in ("_settle_declaration", "submit_work_report"):
        body = inspect.getsource(getattr(WorkOrderToolset, name))
        assert 'getattr(response, "timed_out", False)' in body, name
        assert '"status": "unanswered"' in body, name
        # …and it is asked BEFORE "not approved", or the rejection branch eats it:
        # a timed-out response is also an unapproved one.
        assert body.index('getattr(response, "timed_out", False)') < body.index(
            "not response.approved"), name
