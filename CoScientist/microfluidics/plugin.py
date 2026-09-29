"""Entry point of the microfluidics profile.

Listed under ``plugins:`` in CoScientist/microfluidics/microfluidics.yaml and
imported by the core config loader before the system is built. Importing it
registers everything the profile adds to the shared core: prompts, tools,
callbacks, agent classes and output schemas, JSON-answer repairs, Work Order
rules, and the ТЗ panel's view of the stored ТЗ.
"""
import CoScientist.microfluidics.bindings  # noqa: F401
import CoScientist.microfluidics.json_answers  # noqa: F401
import CoScientist.microfluidics.prompts  # noqa: F401
import CoScientist.microfluidics.work_order  # noqa: F401
from CoScientist.hitl.tz_panel import set_tz_state_view


def _stored_tz_view(state):
    from CoScientist.microfluidics.tz_builder import TZ_STATE_KEY, load_tz
    from CoScientist.microfluidics.tz_review import tz_view

    tz = load_tz(state.get(TZ_STATE_KEY))
    return tz_view(tz) if tz is not None else None


set_tz_state_view(_stored_tz_view)
