"""Question module: an unknown to answer, not a claim that must be invented first."""
from __future__ import annotations

from CoScientist.requirements.models import OutcomeRecord, RequirementPart


def question_met(part: RequirementPart, outcome: OutcomeRecord | None) -> tuple[bool, str]:
    if part.retired or not part.obligation:
        return True, ""
    if outcome is None or not outcome.executed:
        return False, "the question was not answered"
    if outcome.infrastructure_error:
        return False, "the answer is missing because execution failed"
    if str(outcome.verdict or "").strip().lower() == "inconclusive":
        return False, "the protocol ran but the question is still unanswered"
    grounded = outcome.grounded if outcome.answer_grounded is None else outcome.answer_grounded
    if not grounded or not str(outcome.answer or "").strip():
        return False, "no grounded answer was recorded"
    return True, ""


def root_is_the_question(root_mode: str) -> bool:
    """A single standalone question uses the graph root. It is not a second question."""
    return root_mode == "question"
