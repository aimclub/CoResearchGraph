"""Shared fixtures builders for Experiment Module unit tests."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.critique import critique_plan
from CoScientist.experiments.runtime import (
    approve_plan,
    guard_route_agent_tool,
    initialize_runtime,
    on_route_agent_returned,
    start_task,
)
from CoScientist.experiments.schemas import ExperimentPlan

NOW = datetime(2026, 7, 31, 18, 0, tzinfo=timezone.utc).isoformat()


def _design(hypothesis_ref: str = "H1", *, question: str | None = None) -> dict:
    return {
        "hypothesis_ref": hypothesis_ref,
        "experiment_question": question
        or f"Does the ready MCP produce evidence for {hypothesis_ref}?",
        "dataset": {
            "name": "ready_mcp_inputs",
            "ref": None,
            "notes": "Synthetic fixture inputs for unit tests.",
        },
        "baselines": [
            {
                "name": "no-tool control",
                "kind": "method",
                "ref": None,
            }
        ],
        "metrics": [
            {
                "name": "artifact_present",
                "direction": "maximize",
                "threshold": 1,
                "test": "descriptive",
            }
        ],
        "analysis_artifacts": [
            {
                "name": "metrics_table.json",
                "role": "metrics_table",
                "prepare_via": "mcp",
                "path_or_tool": "metrics_table.json",
            }
        ],
    }


def _server(tool: str = "estimate_property", url: str = "http://127.0.0.1:8000/mcp") -> dict:
    return {
        "name": "chem-ready",
        "server_id": "srv-chem",
        "url": url,
        "tools": [
            {
                "name": tool,
                "description": "Compute the requested chemical property.",
                "input_schema": {
                    "type": "object",
                    "properties": {"smiles": {"type": "string"}},
                    "required": ["smiles"],
                },
            }
        ],
        "source": "registry",
    }


def _task(
    task_id: str,
    *,
    route: str = "fedot_mas",
    tool: str = "estimate_property",
    depends_on: list[str] | None = None,
    optional: bool = False,
    artifact_name: str | None = None,
    hypothesis_ref: str = "H1",
    design: dict | None = None,
) -> dict:
    artifact_name = artifact_name or f"{task_id.lower()}-result.csv"
    return {
        "id": task_id,
        "name": f"Chemical computation {task_id}",
        "description": "Compute a bounded chemical property using the ready MCP.",
        "rationale": "Produces direct computational evidence for the hypothesis.",
        "route": route,
        "design": design or _design(hypothesis_ref),
        "mcp_servers": [_server(tool)] if route in {"fedot_mas", "react_tools"} else [],
        "input_data": [],
        "launch_params": {"smiles": "CCO"},
        "success_criteria": [
            {
                "criterion_id": f"{task_id}-C1",
                "description": "The MCP execution completes.",
                "kind": "execution",
                "verification": "Check the structured route result status.",
            }
        ],
        "expected_artifacts": [
            {
                "name": artifact_name,
                "role": "data",
                "media_type": "text/csv",
                "description": "Managed result table.",
            }
        ],
        "est_duration_min": 1,
        "depends_on": depends_on or [],
        "optional": optional,
    }


def _plan(*tasks: dict, hypotheses: list[dict] | None = None) -> ExperimentPlan:
    refs = {
        str(task.get("design", {}).get("hypothesis_ref") or "H1").upper()
        for task in tasks
    }
    default_hypotheses = [
        {
            "hypothesis_id": hid,
            "statement": f"Fixture statement for {hid}.",
        }
        for hid in sorted(refs)
    ]
    return ExperimentPlan.model_validate(
        {
            "schema_version": "experiment-plan/1.0",
            "plan_id": "PLAN-acceptance",
            "experiment_run_id": "EXRUN-acceptance",
            "revision": 1,
            "source_request": "Run a bounded chemistry MCP experiment.",
            "goal": "Produce managed computational evidence.",
            "hypothesis": "The ready chemistry MCP can compute the requested property.",
            "hypotheses": hypotheses if hypotheses is not None else default_hypotheses,
            "methods": ["Exact ready-MCP execution"],
            "context_digest": "Ready chemistry MCP is registered.",
            "context_refs": ["TOOL-srv-chem"],
            "tasks": list(tasks),
            "risks": [],
            "assumptions": [],
            "total_est_duration_min": sum(task["est_duration_min"] for task in tasks),
            "created_at": NOW,
        }
    )


def _inventory() -> list[dict]:
    return [
        {
            "tool": "estimate_property",
            "server_id": "srv-chem",
            "description": "Compute the requested chemical property.",
            "input_schema": {},
        }
    ]


def _approved_state(plan: ExperimentPlan) -> dict:
    state: dict = {}
    critique = critique_plan(
        plan,
        # The fixtures default to fedot_mas tasks; FEDOT.MAS is off by default.
        settings=ExperimentsSettings(route_fedot=True),
        available_tools=_inventory(),
    )
    payload = critique.model_dump(mode="json")
    blocking = [
        i for i in critique.issues
        if i.severity in {"blocker", "major"}
    ]
    only_coder_inventory = bool(blocking) and all(
        "uses route=coder" in i.message and "inventory already" in i.message
        for i in blocking
    )
    if critique.verdict != "approve":
        assert only_coder_inventory, [i.message for i in blocking]
        payload = {
            "schema_version": "plan-critique/0.1",
            "critique_id": "CRIT-test",
            "plan_id": plan.plan_id,
            "verdict": "approve",
            "issues": [],
            "checked_at": NOW,
            "summary": "forced: coder+inventory is a repair concern",
        }
    initialize_runtime(state, plan, critique=payload)
    approve_plan(state)
    return state


def _tool_context(state: dict) -> SimpleNamespace:
    return SimpleNamespace(state=state)


def _route_return(state: dict, agent: str) -> None:
    tool = SimpleNamespace(name=agent)
    assert guard_route_agent_tool(tool, {}, _tool_context(state)) is None
    on_route_agent_returned(tool, {}, _tool_context(state), {"status": "success"})


def _success_result(task_id: str) -> dict:
    return {
        "status": "success",
        "summary": f"{task_id} completed with a managed result.",
        "criteria_checks": [
            {
                "criterion_id": f"{task_id}-C1",
                "passed": True,
                "details": "Structured route result reports success.",
            }
        ],
    }


def _research_call_response(request: str):
    from google.adk.models import LlmResponse
    from google.genai import types

    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="ResearchAgent",
                        args={"request": request},
                    )
                )
            ],
        )
    )


def _repo_fit_payload(route: str = "fedot_mas") -> dict:
    return {
        "source_request": "Generate molecules with high docking affinity.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Generate",
                "description": "Generate molecules",
                "route": route,
                "design": {
                    "analysis_artifacts": [
                        {"name": "out.json", "prepare_via": "mcp", "path_or_tool": "x"}
                    ]
                },
                "mcp_servers": [],
            }
        ],
    }


def _alembic_task(task_id: str = "EXP-1", *, hypothesis_ref: str = "H1") -> dict:
    task = _task(task_id, route="coder", hypothesis_ref=hypothesis_ref)
    task.update(
        {
            "route": "alembic_build",
            "repo_url": "https://github.com/whitead/synspace",
            "post_build_route": "react_tools",
            "mcp_servers": [],
            "code_assessment": {
                "requirement": "reuse",
                "evidence": "The repository exposes the required operation unchanged.",
                "entrypoints": ["synspace_score"],
            },
            "expected_artifacts": [
                {
                    "name": "mcp_endpoint",
                    "role": "mcp_server",
                    "description": "Served Alembic MCP URL",
                    "required": True,
                }
            ],
            "success_criteria": [
                {
                    "criterion_id": f"{task_id}-C1",
                    "description": "Alembic build exposes an MCP endpoint.",
                    "kind": "execution",
                    "verification": "outputs.mcp_url is an http(s) URL.",
                }
            ],
        }
    )
    return task


def _alembic_started_state() -> dict:
    plan = _plan(_alembic_task())
    state: dict = {}
    initialize_runtime(
        state, plan,
        critique={"verdict": "approve", "issues": [], "summary": "forced"},
    )
    approve_plan(state)
    start_task(state, "EXP-1", settings=ExperimentsSettings(route_alembic=True))
    return state


class _FakeResearchGraph:
    """Minimal stand-in for the research graph store used by hypotheses.py."""

    def __init__(self, hypothesis_statements: list[str], statuses: list[str] | None = None):
        self._nodes = [
            (f"H{i}", {
                "type": "Hypothesis",
                "status": (statuses[i - 1] if statuses and i <= len(statuses) else ""),
                "attrs": {"formulation": s},
            })
            for i, s in enumerate(hypothesis_statements, start=1)
        ]

    def nodes(self, data: bool = True):
        return list(self._nodes)


def _patch_research_graph(
    monkeypatch, statements: list[str], statuses: list[str] | None = None,
) -> None:
    import CoScientist.graph.research.store as research_store

    graph = _FakeResearchGraph(statements, statuses=statuses)
    monkeypatch.setattr(
        research_store,
        "get_research_graph",
        lambda ctx: SimpleNamespace(full_graph=lambda: graph),
    )


class _FakeInitGraph:
    """Stand-in research graph store for bootstrap_research_question_if_empty."""

    def __init__(self, empty: bool):
        self._empty = empty
        self.init_calls: list[dict] = []

    def is_empty(self) -> bool:
        return self._empty

    def init_research(self, source: str, question: str):
        self.init_calls.append({"source": source, "question": question})
        return {"ok": True, "root_id": "Q1"}

