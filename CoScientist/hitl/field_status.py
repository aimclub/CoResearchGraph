"""Per-field provenance status, shared by structured HITL intakes.

Lifted from the microfluidics ТЗ model (``microfluidics/models.py``) so every
structured intake — the ТЗ and the research frame (``context_init/models.py``) —
uses one status vocabulary. A field is "open" when it has no usable value yet.
That is what a HITL form uses to decide which fields to ask the human to fill.

Statuses are plain string Literals (not an Enum) so the dicts ADK stores in
session state and injects into downstream prompts render as readable text.
"""
from __future__ import annotations

from typing import Literal

# How a field value was obtained — downstream code treats these differently.
FieldStatus = Literal[
    "задано заказчиком",
    "уточнено оператором",
    "предложено агентом",
    "не задано",
    "свободный комментарий",
    "рассчитывается агентом",
]

# Statuses that mean "there is no usable value here yet".
OPEN_STATUSES = ("не задано", "рассчитывается агентом")

# The operator set this value by hand in a HITL form. ONLY apply_form_values may
# write this: it is what makes a node count as human-sourced downstream
# (context_init/commit.py::_HUMAN_STATUSES), so an agent stamping its own guess
# with it would forge the provenance the operator badge relies on.
OPERATOR_STATUS = "уточнено оператором"

# A working value the agent proposed from domain context — usable, but NOT human.
AGENT_PROPOSED_STATUS = "предложено агентом"


def is_open(status: str) -> bool:
    """True when the field still needs a value (drives the HITL form)."""
    return status in OPEN_STATUSES


__all__ = ["FieldStatus", "OPEN_STATUSES", "OPERATOR_STATUS",
           "AGENT_PROPOSED_STATUS", "is_open"]
