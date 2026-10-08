"""The research frame must not forge human provenance.

`уточнено оператором` is what makes a frame block count as human-sourced
(context_init/commit.py::_HUMAN_STATUSES -> source="human" on the graph node),
and the operator badge on renders keys off it. Only the operator form may write
it; an agent's own domain guess must be `предложено агентом`.
"""
from CoScientist.context_init.commit import _HUMAN_STATUSES, _block_source
from CoScientist.context_init.models import FrameBlock
from CoScientist.hitl.field_status import (
    AGENT_PROPOSED_STATUS,
    OPEN_STATUSES,
    OPERATOR_STATUS,
)


def _block(status: str) -> FrameBlock:
    return FrameBlock.model_validate(
        {"title": "Вопрос исследования",
         "fields": [{"name": "formulation", "value": "что-то", "status": status}]}
    )


def test_agent_proposed_is_not_human():
    assert AGENT_PROPOSED_STATUS not in _HUMAN_STATUSES
    assert _block_source(_block(AGENT_PROPOSED_STATUS)) != "human"


def test_operator_status_is_human():
    assert OPERATOR_STATUS in _HUMAN_STATUSES
    assert _block_source(_block(OPERATOR_STATUS)) == "human"


def test_agent_proposed_still_counts_as_a_set_value():
    """It is a usable working value — it just isn't human-provided."""
    assert AGENT_PROPOSED_STATUS not in OPEN_STATUSES
    assert _block(AGENT_PROPOSED_STATUS).set_fields()


def test_prompt_does_not_tell_the_agent_to_claim_operator_status():
    """Regression: the ContextInit prompt used to instruct the model to stamp its
    own domain guesses as «уточнено оператором», which forged human provenance."""
    from CoScientist.agents.prompts import templates

    src = open(templates.__file__, encoding="utf-8").read()
    i = src.find("Обоснованные рабочие значения из контекста домена")
    assert i != -1, "the frame-status instruction disappeared — update this test"
    instruction = src[i:i + 260]
    assert AGENT_PROPOSED_STATUS in instruction
    assert "ТОЛЬКО форма" in instruction
