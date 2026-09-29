"""``before_agent`` gate: ask the operator whether to produce a NIR report.

The question has to be mandatory, and a tool the model may or may not call is
not. So it lives in a callback that runs before the aggregator's first LLM
request, in the shape ``hitl.pipeline_scope.make_ask_pipeline_scope_callback``
established: gate on settings, stay idempotent through a state key, SELECT then
APPROVE-with-form, write state, and always return ``None`` so the agent runs
either way.

Silence means no. HITL off, the feature flag off, no server configured, a
timeout, a dismissal — each leaves ``enabled: False`` and an empty prompt block,
and the run produces exactly the Markdown report it produces today.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from CoScientist.config import get_settings
from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.graph.session_scope import session_key
from CoScientist.reporting.nir import hitl_form

logger = logging.getLogger(__name__)

#: Rendered into the aggregator prompt's ``{nir_block?}``. Empty unless the
#: operator asked for a report, so the prompt is byte-identical to today's
#: whenever the feature is not in play.
_PROMPT_BLOCK = """
### Отчёт о НИР по ГОСТ 7.32-2017

Оператор запросил, помимо краткого отчёта, нормативный документ.

Делегируй это `NirReportAgent` — передай ему одно указание: подготовить отчёт о
НИР по данным завершённого исследования. Он сам прочитает граф, напишет текст и
соберёт DOCX.

`NirReportAgent` вернёт ссылку на документ. Добавь её отдельным разделом в конец
своего отчёта, дословно, вместе с предупреждениями, которые он передал. Если он
вернул ошибку — напиши об этом одной строкой и не пытайся собрать документ сам.
"""


def _availability_problem() -> Optional[str]:
    """Why the NIR option must not be offered, or None when it may be.

    The runtime gate. ``system.yaml`` attaches ``NirReportAgent`` on a narrower
    one — ``nir_buildable``, the server alone — because attachment is decided
    once when the tree is assembled and cannot see a switch flipped later. So
    the agent may well be present here while this says no; every ``nir_report_*``
    tool refuses in that case, and the prompt below never mentions it.
    """
    settings = get_settings()
    if not settings.nir_ready:
        return (
            "NIR__ENABLED is off" if not settings.nir.enabled
            else "MCP__NORMCONTROL_URL is not set"
        )
    if not settings.web.hitl_enabled:
        return "HITL is off — nobody to ask"
    return None


def _disable(state: Any) -> None:
    state[hitl_form.STATE_REQUEST_KEY] = {"enabled": False}
    state[hitl_form.STATE_BLOCK_KEY] = ""


def make_ask_nir_report_callback(handler: AbstractHITLHandler):
    """Build the ``before_agent`` callback bound to one HITL handler."""

    async def before_agent_callback(callback_context, llm_request=None):
        state = callback_context.state

        # Asked once per session. The aggregator can run again (a retry, a
        # follow-up turn) and re-asking would show the operator the same form
        # after they already answered it.
        if state.get(hitl_form.STATE_DONE_KEY):
            return None

        problem = _availability_problem()
        if problem:
            logger.info("nir: not offering the report (%s)", problem)
            _disable(state)
            return None

        state[hitl_form.STATE_DONE_KEY] = True
        agent_name = getattr(callback_context, "agent_name", "ResultAggregatorAgent")
        user_id, session_id = session_key(callback_context)
        context = {"_session": {"user_id": user_id, "session_id": session_id}}

        try:
            choice = await handler.handle_request(HITLRequest(
                agent_name=agent_name,
                action_type=HITLAction.SELECT,
                message=hitl_form.SELECT_MESSAGE,
                options=hitl_form.choice_options(),
                default_option=hitl_form.OPTION_NIR,
                context=context,
                invoked_via="callback",
                trigger="ask_nir_report",
            ))
        except Exception as exc:  # noqa: BLE001 - a question must not sink a run
            logger.warning("nir: could not ask the operator (%s)", exc)
            _disable(state)
            return None

        if not hitl_form.wants_nir(getattr(choice, "selected_option", None)):
            _disable(state)
            return None

        try:
            answered = await handler.handle_request(HITLRequest(
                agent_name=agent_name,
                action_type=HITLAction.APPROVE,
                message=hitl_form.FORM_MESSAGE,
                form=hitl_form.requisites_form(),
                context=context,
                invoked_via="callback",
                trigger="nir_requisites",
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("nir: requisites form failed (%s)", exc)
            _disable(state)
            return None

        # The form allows skipping, and skipping is not a refusal: the operator
        # asked for the report and declined to fill the title page. That is a
        # draft with visible placeholders, which is a legitimate thing to want.
        requisites = hitl_form.parse_requisites(getattr(answered, "form_values", None))
        state[hitl_form.STATE_REQUEST_KEY] = {
            "enabled": True,
            "mode": get_settings().nir.mode,
            "requisites": hitl_form.requisites_to_state(requisites),
        }
        state[hitl_form.STATE_BLOCK_KEY] = _PROMPT_BLOCK
        logger.info("nir: operator requested a GOST report")
        return None

    return before_agent_callback


__all__ = ["make_ask_nir_report_callback"]
