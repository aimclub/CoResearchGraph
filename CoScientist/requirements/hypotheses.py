"""Hypothesis module: a claim to formulate or to test. Creating one is not success."""
from __future__ import annotations

from CoScientist.requirements.models import OutcomeRecord, RequirementPart

_VERDICTS = {"confirmed", "refuted"}


def hypothesis_met(part: RequirementPart, outcome: OutcomeRecord | None) -> tuple[bool, str]:
    """A requested check is done when the verdict is grounded, including refuted.

    An infrastructure failure is not a verdict. Formulate-only does not need
    ``confirmed``. ``inconclusive`` does not close a request for a definite answer.
    """
    if part.retired or not part.obligation:
        return True, ""
    if outcome is None or not outcome.executed:
        if part.hypothesis_mode == "formulate":
            return False, "the hypothesis was not formulated"
        return False, "the check was not run"
    if outcome.infrastructure_error:
        return False, "the service failed; that is not a scientific verdict"
    if part.hypothesis_mode == "formulate":
        if str(outcome.answer or "").strip():
            return True, ""
        return False, "the formulation was not recorded"
    verdict = str(outcome.verdict or "").strip().lower()
    if verdict in _VERDICTS and outcome.grounded:
        return True, ""
    if verdict == "inconclusive":
        return False, "the protocol ran but the claim is still undecided"
    if verdict in _VERDICTS and not outcome.grounded:
        return False, "a verdict without grounds does not close the check"
    return False, "the check has no verdict"


def service_error_is_not_refuted(verdict: str, *, infrastructure_error: bool) -> str:
    if infrastructure_error:
        return ""
    return verdict
