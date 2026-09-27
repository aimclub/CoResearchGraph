"""The experiment module consumes literature; the orchestrator owns its collection.

Keep operation IDs intact when projecting a mixed research request. The root
scope is never shortened or marked complete merely because it is external to EM.
Legacy research routes remain deserializable, but are not executable here.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from CoScientist.experiments.capabilities.inventory import (
    FAMILY_RESEARCH, declared_family_capabilities,
)

LITERATURE_TOOLS = frozenset(
    row["tool"] for row in declared_family_capabilities(FAMILY_RESEARCH)
    # Generic web access may be needed for code/API documentation. It is not
    # literature collection by itself; the task's purpose is checked below.
    if row["tool"] not in {"tavily_search", "tavily_extract", "tavily_crawl"}
) | {"search_pubmed"}
LITERATURE_HANDOFF = (
    "Literature collection belongs to the orchestrator's ResearchAgent, outside "
    "ExperimentModuleAgent. Return this requirement to the orchestrator; consume "
    "its existing evidence/data refs as inputs. Do not replace it with Coder, "
    "MedicalAgent, Alembic or a literature MCP call inside the experiment."
)

# Conservative: keep mixed/ambiguous operations in scope, so a mention of
# literature-backed inputs never removes a requested calculation.
_COMPUTE = re.compile(
    r"\b(?:comput\w*|calculat\w*|predict\w*|cluster\w*|dock\w*|train\w*|"
    r"simulat\w*|implement\w*|reproduc\w*|benchmark\w*|"
    r"вычисл\w*|рассчита\w*|рассчит\w*|предска\w*|кластер\w*|докинг\w*|"
    r"обуч\w*|моделир\w*|реализ\w*|воспроизв\w*)\b", re.I,
)
_LITERATURE = re.compile(
    r"\b(?:literature|publications?|papers?|bibliograph\w*|pubmed|openalex|"
    r"литератур\w*|публикаци\w*|стать[яиюеёй]\w*|библиограф\w*)\b", re.I,
)
_COLLECT = re.compile(
    r"\b(?:collect\w*|search\w*|find|gather\w*|retriev\w*|review\w*|"
    r"собер\w*|собр\w*|сбор\w*|най[дт]\w*|поиск\w*|обзор\w*|изуч\w*)\b", re.I,
)
_OBSERVED_DATA = re.compile(
    r"\b(?:experimental|measured|published|экспериментальн\w*|измеренн\w*|"
    r"опубликованн\w*)\b", re.I,
)


def operation_rows(raw: Any) -> list[dict[str, str]]:
    """Normalize shape without renumbering already committed operation IDs."""
    if not isinstance(raw, (list, tuple)):
        from CoScientist.context_init.operations import normalize_operation_rows

        return normalize_operation_rows(raw)
    result = []
    for index, item in enumerate(raw, 1):
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        row = item if isinstance(item, Mapping) else {"statement": str(item or "")}
        statement = str(row.get("statement") or row.get("content") or row.get("text") or "").strip()
        if statement:
            result.append({
                "operation_id": str(row.get("operation_id") or f"OP-{index}").strip().upper(),
                "statement": statement,
            })
    return result


def is_literature_operation(statement: str) -> bool:
    text = str(statement or "")
    if _COMPUTE.search(text):
        return False
    if any(re.search(rf"\b{re.escape(name)}\b", text, re.I) for name in LITERATURE_TOOLS):
        return True
    return bool(
        re.match(r"\s*(?:literature|литератур\w*|обзор\s+литературы)\b", text, re.I)
        or (_COLLECT.search(text) and (_LITERATURE.search(text) or _OBSERVED_DATA.search(text)))
    )


def partition_operations(raw: Any) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    compute, literature = [], []
    for row in operation_rows(raw):
        (literature if is_literature_operation(row["statement"]) else compute).append(row)
    return compute, literature


def experiment_capabilities(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project discovery for EM without altering the shared discovery cache."""
    return [row for row in rows if str(row.get("tool") or row.get("name") or "").strip()
            not in LITERATURE_TOOLS]


def literature_task_reason(task: Any, operations: Any = ()) -> str | None:
    """Shared review/runtime boundary, including attempts to disguise the route."""
    row = task.model_dump(mode="json") if hasattr(task, "model_dump") else task
    design = row.get("design") or {}
    if row.get("route") == "research":
        return LITERATURE_HANDOFF
    for artifact in design.get("analysis_artifacts") or []:
        if artifact.get("prepare_via") == "research" or artifact.get("path_or_tool") in LITERATURE_TOOLS:
            return LITERATURE_HANDOFF
    for server in row.get("mcp_servers") or []:
        for tool in server.get("tools") or []:
            if (tool.get("name") if isinstance(tool, dict) else tool) in LITERATURE_TOOLS:
                return LITERATURE_HANDOFF
    ref = str(design.get("operation_ref") or "").strip().upper()
    indexed = {op["operation_id"]: op["statement"] for op in operation_rows(operations)}
    # Check both authoritative operation and the proposed task: a compute label
    # does not authorize a literature step, and prose cannot change a frame slot.
    texts = [indexed.get(ref, ""), str(row.get("name") or ""), str(row.get("description") or "")]
    if any(text and is_literature_operation(text) for text in texts):
        return LITERATURE_HANDOFF
    return None


def skip_literature_only_experiment(callback_context: Any) -> Any:
    """Return a misplaced literature-only delegation before discovery/LLM calls."""
    from google.genai import types
    from CoScientist.experiments.context.builder import _operations_from_state

    state = callback_context.state
    runtime = state.get("experiment_runtime") or {}
    if runtime.get("approved") and runtime.get("phase") == "execution":
        return None  # Resume existing work; start_task still enforces the boundary.
    request = str(state.get("experiment_source_request") or state.get("orchestrator_root_goal") or "")
    if not request:
        content = getattr(callback_context, "user_content", None)
        request = " ".join(str(getattr(part, "text", "") or "") for part in getattr(content, "parts", None) or [])
    compute, literature = partition_operations(_operations_from_state(state, request))
    if compute or not (literature or is_literature_operation(request)):
        return None
    state["experiment_module_outcome"] = {
        "status": "not_applicable", "stage": "scope",
        "reason": "literature_outside_experiment_module", "owner": "ResearchAgent",
        "external_literature_operations": literature,
    }
    return types.Content(role="model", parts=[types.Part(text=LITERATURE_HANDOFF)])
