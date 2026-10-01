"""Keep the CFD server's results in the state, as the server gave them.

``collect_cfd_result`` (after_tool, EquipmentAgent) files every answer of
``cfd_run_reactor_experiment`` and ``cfd_get_experiment_result`` under its
request id in ``cfd_runs``, so the journal, the report and a later optimization
module read the simulated numbers instead of the agent's retelling:

    cfd_runs = {request_id: {status, finished, reactor, design, residence_time_s,
                             results, trustworthy, blocking, error}}

A later answer for the same request id replaces the earlier one (``pending`` is
followed by the terminal status). The shape is the one recorded from the service
in ``tests/unit/microfluidics/fixtures/cfd_mcp/``; ``results`` is kept as the service returned it
(pressure drop, residence time, Damköhler number, conversion, outlet
concentrations, mixing index). Errors (``isError``, a failed transport) and the
listing tools change nothing.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from CoScientist.microfluidics.economics import structured_result

logger = logging.getLogger(__name__)

RUNS_KEY = "cfd_runs"

_RUN_TOOLS = frozenset({"cfd_run_reactor_experiment", "cfd_get_experiment_result"})

# The operating point a human compares runs by; the rest of `design` is defaults.
_DESIGN_FIELDS = (
    "inlet_speed_m_per_s",
    "concentration_a_mol_per_m3",
    "concentration_b_mol_per_m3",
    "rate_constant_m3_per_mol_s",
    "temperature_k",
    "turnovers",
    "reactant_a",
    "reactant_b",
    "product",
)


def run_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    design = result.get("design") or {}
    operating = (result.get("derived") or {}).get("operating_point") or {}
    return {
        "status": result.get("status"),
        "finished": result.get("finished"),
        "reactor": result.get("reactor") or design.get("reactor"),
        "design": {k: design[k] for k in _DESIGN_FIELDS if k in design},
        "residence_time_s": operating.get("residence_time_s_estimate"),
        "results": result.get("results") or {},
        "trustworthy": result.get("trustworthy"),
        "blocking": result.get("blocking") or [],
        "error": result.get("error"),
    }


def collect_cfd_result(
    tool: Any = None, args: Any = None, tool_context: Any = None, tool_response: Any = None,
    **kwargs: Any,
) -> None:
    """after_tool: file a CFD run under its request id."""
    name = str(getattr(tool, "name", "") or "")
    if name not in _RUN_TOOLS or tool_context is None:
        return None
    result = structured_result(tool_response)
    if not result:
        return None
    request_id = result.get("request_id") or (args or {}).get("request_id")
    if not request_id:
        return None
    try:
        runs = dict(tool_context.state.get(RUNS_KEY) or {})
        runs[str(request_id)] = run_summary(result)
        # Written whole: AgentTool forwards only whole-key deltas to the parent.
        tool_context.state[RUNS_KEY] = runs
    except Exception as exc:  # noqa: BLE001 — bookkeeping must never break the call
        logger.warning("cfd: could not record %s: %s", request_id, exc)
    return None


__all__ = ["RUNS_KEY", "collect_cfd_result", "run_summary"]
