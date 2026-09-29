#!/usr/bin/env python3
"""Deterministic check of the decision point that starts tool development.

When the tool-preparation loop finds nothing that covers the task, the
experiment executor must not fall back to a nearest-but-wrong tool: the
`redirect_when_no_tools` gate (CoScientist/agents/callbacks/tool_callbacks.py)
short-circuits the experiment agent and hands the step to tool *development*.

This script drives that gate with a synthetic session state and prints both the
log record and the structured decision it returns.
"""
import logging
import sys
import types

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

from CoScientist.agents.callbacks import tool_callbacks           # noqa: E402
from CoScientist.agents.callbacks.tool_callbacks import (      # noqa: E402
    TOOL_MATCH_STATE_KEY, redirect_when_no_tools,
)

# The package configures its own logging; make this module's records visible.
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("LOG: %(message)s"))
tool_callbacks.logger.addHandler(_h)
tool_callbacks.logger.setLevel(logging.INFO)
tool_callbacks.logger.propagate = False


def run(case: str, state: dict) -> None:
    ctx = types.SimpleNamespace(state=state)
    print(f"\n=== {case} ===")
    print(f"state: {state}")
    out = redirect_when_no_tools(ctx)
    if out is None:
        print("decision: continue with the ready-made tools (no escalation)")
        return
    print("decision: escalate — the experiment agent is stopped before the model call")
    print(out.parts[0].text)


run(
    "A. Ни один готовый инструмент не покрывает задачу",
    {TOOL_MATCH_STATE_KEY: {"matched": False, "best_score": 0.31},
     "filtered_tools": [], "filtered_mcps": []},
)
run(
    "B. Подходящий инструмент найден",
    {TOOL_MATCH_STATE_KEY: {"matched": True, "best_score": 0.88},
     "filtered_tools": [{"name": "calculate_docking"}], "filtered_mcps": []},
)
