"""Focused, bounded context builder for ExperimentPlannerAgent."""
from __future__ import annotations

import copy
import json
import logging
import os
import re
from typing import Any, Iterable
from uuid import uuid4

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types

from CoScientist.experiments.capabilities.inventory import get_grouped_mcp_inventory
from CoScientist.experiments.runtime.shared import audit

logger = logging.getLogger(__name__)
_MAX_RETRIEVAL_CALLS = 5
_DESC_LIMIT = 400
_PROMPT_DESC_LIMIT = 220
_CLEAR_ON_NEW_RUN = (
    "experiment_plan", "experiment_runtime", "experiment_task_results", "experiment_summary",
    "experiment_artifacts_manifest",
    "experiment_last_route_response", "experiment_active_envelope",
    "experiment_plan_validation_errors", "experiment_plan_review_paused",
    "experiment_plan_candidate", "experiment_plan_fallback_pending",
    "experiment_module_outcome",
    "experiment_review_pause_reason", "experiment_module_runs", "experiment_module_dispatched",
    "experiment_plan_revision_count", "experiment_inventory_blocker_hits",
    "experiment_no_matching_tool", "experiment_execution_summary",
    "experiment_repo_candidates",
    "accumulated_tools", "filtered_tools", "retrieval_queries",
)
DISCOVERED_CAPABILITIES_KEY = "experiment_discovered_capabilities"  # survives attempt clears
RETRIEVED_CAPABILITIES_KEY = "experiment_retrieved_capabilities"  # pre-rerank full set
PLANNER_CONTEXT_KEY = "experiment_planner_context"  # compact JSON for planner instruction
_H_LABEL_RE = re.compile(  # explicit H1/H2/… (domain-agnostic)
    r"(?:^|[\n\r•\-\*\u2022]\s*|(?<=\s))"
    r"(?P<id>H\d+)\s*[.:)\u2013\u2014\-]\s*(?P<statement>\S.*?)"
    r"(?=(?:\n\s*(?:H\d+\s*[.:)\u2013\u2014\-]|\u2022|•|\-|\*)|\Z))",
    re.IGNORECASE | re.DOTALL,
)
# System HypothesesAgent prose: "Hypothesis 1 (Parkinson's):" / "**Hypothesis 2:**"
_HYPOTHESIS_N_RE = re.compile(
    r"(?:^|[\n\r])\s*(?:\*\*)?Hypothesis\s*(?P<num>\d+)\s*"
    r"(?:\([^)]*\))?\s*(?:\*\*)?\s*[.:)\u2013\u2014\-]\s*"
    r"(?P<body>.+?)"
    r"(?=(?:\n\s*(?:\*\*)?Hypothesis\s*\d+|\n\s*H\d+\s*[.:)\u2013\u2014\-]|\Z))",
    re.IGNORECASE | re.DOTALL,
)
_STATEMENT_IN_BODY_RE = re.compile(
    r"(?:\*\*)?Statement(?:\*\*)?\s*:\s*(?P<statement>.+?)"
    r"(?=\n\s*(?:\*\*)?(?:VerificationMethod|ConfirmationCriteria|Hypothesis\s*\d+|H\d+\s*[.:)]|\*)|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_MAX_HYPOTHESIS_REFS = 8
_MAX_REPO_CANDIDATES = 8
_REPO_URL_RE = re.compile(
    r"(?P<url>https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org)/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+"
    r"|git@(?:github\.com|gitlab\.com|bitbucket\.org):[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(?:\.git)?)",
    re.IGNORECASE,
)
_PROMPT_OPTIONAL_KEYS = (
    "research_focus_id", "research_context", "hypotheses", "hypothesis_refs", "prior_results",
    "prior_evidence", "hypothesis_chain_context", "confirmation_criteria",
    "data_refs", "constraints", "operations", "explicit_mcp_servers", "repo_candidates",
    "revision_feedback", "unresolved_gaps", "pipeline_scope",
)
# Hypothesis statuses that still need experimental verification (typed read —
# refuted/postponed/confirmed nodes must not force new plan coverage).
_SNAPSHOT_ACTIVE_H_STATUSES = frozenset({"formulated", "under_verification"})

def _invocation_session(callback_context: CallbackContext) -> Any:
    inv = getattr(callback_context, "_invocation_context", None) or getattr(
        callback_context, "invocation_context", None
    )
    return getattr(inv, "session", None) if inv is not None else None

def reset_experiment_retrieval_budget(callback_context: CallbackContext) -> None:
    """Start a request-local retrieval budget; do not clear shared history."""
    queries = callback_context.state.get("retrieval_queries") or []
    callback_context.state["experiment_retrieval_query_baseline"] = len(queries)
    callback_context.state["experiment_retrieval_budget_exhausted"] = False
    try:
        from CoScientist.tools.retrieval_tools import clear_session_accumulated_tools
        sid = getattr(_invocation_session(callback_context), "id", None)
        clear_session_accumulated_tools(str(sid)) if sid else clear_session_accumulated_tools()
    except Exception:  # noqa: BLE001
        pass

def enforce_experiment_retrieval_budget(
    callback_context: CallbackContext, llm_response: LlmResponse,
) -> LlmResponse | None:
    """Cap discovery at four retrieve_tools calls."""
    content = getattr(llm_response, "content", None)
    parts = list(getattr(content, "parts", None) or [])
    is_retrieve = lambda p: getattr(getattr(p, "function_call", None), "name", None) == "retrieve_tools"
    call_parts = [p for p in parts if is_retrieve(p)]
    if not call_parts:
        return None
    state = callback_context.state
    baseline = int(state.get("experiment_retrieval_query_baseline") or 0)
    calls_used = max(0, len(state.get("retrieval_queries") or []) - baseline)
    remaining = max(0, _MAX_RETRIEVAL_CALLS - calls_used)
    if len(call_parts) <= remaining:
        return None
    if remaining:
        kept, kept_calls = [], 0
        for part in parts:
            if is_retrieve(part):
                if kept_calls >= remaining:
                    continue
                kept_calls += 1
            kept.append(part)
        logger.warning("EXPERIMENT_RETRIEVAL_BUDGET_TRIMMED used=%s allowed_now=%s", calls_used, remaining)
        return LlmResponse(
            content=types.Content(role=getattr(content, "role", None) or "model", parts=kept)
        )
    state["experiment_retrieval_budget_exhausted"] = True
    marker = (
        f"EXPERIMENT_RETRIEVAL_BUDGET_EXHAUSTED calls={_MAX_RETRIEVAL_CALLS}. "
        "Capability discovery is complete; use the accumulated exact tool metadata and end this retrieval stage."
    )
    audit(logger, marker, level=logging.WARNING)
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=marker)]))

def _user_text(callback_context: CallbackContext) -> str:
    content = getattr(callback_context, "user_content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    return "\n".join(c for p in (parts or []) if (c := getattr(p, "text", ""))).strip()


def _artifact_ref(node_attrs: dict[str, Any]) -> str:
    for key in ("source_ref", "path", "location", "url"):
        if val := str(node_attrs.get(key) or "").strip():
            return val
    return ""


def research_graph_snapshot(callback_context: CallbackContext) -> dict[str, Any]:
    """Typed, deterministic snapshot of the session research graph for the planner.

    Reads nodes BY TYPE (no prose parsing): active Hypothesis nodes committed by
    HypothesesAgent, the ContextInit frame star (Constraint / EmpiricalBase /
    ConfirmationCriteria) and prior Evidence/GeneratedData facts. Best-effort by
    contract: a disabled, empty or broken graph yields {} and planning proceeds
    on the session/text fallbacks — the snapshot must never raise.
    """
    try:
        from CoScientist.config import get_settings

        if not get_settings().research_graph.enabled:
            return {}
    except Exception:  # noqa: BLE001
        return {}
    # No ADK session (bare unit-test contexts) → no session-scoped graph.
    if _invocation_session(callback_context) is None:
        return {}
    try:
        from CoScientist.graph.research.store import get_research_graph

        store = get_research_graph(callback_context)
        if store.is_empty():
            return {}
        full = store.full()
        nodes = full.get("nodes", []) or []
        edges = full.get("edges", []) or []
        rendered = str(store.overview().get("rendered") or "")
    except Exception:  # noqa: BLE001
        return {}

    hypothesis_refs: list[dict[str, str]] = []
    constraints: list[dict[str, Any]] = []
    criteria: list[dict[str, Any]] = []
    data_refs: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    node_by_id = {
        str(node.get("id")): node for node in nodes
        if isinstance(node, dict) and node.get("id")
    }
    for node in nodes:
        if not isinstance(node, dict):
            continue
        ntype = node.get("type")
        attrs = node.get("attrs") or {}
        status = str(node.get("status") or "")
        if ntype == "Hypothesis":
            if status not in _SNAPSHOT_ACTIVE_H_STATUSES:
                continue
            eligible = getattr(store, "hypothesis_eligible", None)
            if callable(eligible) and not eligible(str(node.get("id") or "")):
                continue
            statement = str(attrs.get("formulation") or attrs.get("label") or "").strip()
            if not statement or len(hypothesis_refs) >= _MAX_HYPOTHESIS_REFS:
                continue
            hid = str(node.get("id") or "").strip().upper()
            if not re.fullmatch(r"H\d+", hid):
                hid = f"H{len(hypothesis_refs) + 1}"
            hypothesis_refs.append({
                "hypothesis_id": hid,
                "statement": re.sub(r"\s+", " ", statement)[:800],
            })
        elif ntype == "Constraint":
            if content := str(attrs.get("content") or "").strip():
                constraints.append({
                    "kind": "constraint",
                    "subtype": str(attrs.get("subtype") or ""),
                    "content": content[:_DESC_LIMIT],
                })
        elif ntype == "ConfirmationCriteria":
            row = {k: v for k, v in attrs.items() if v not in (None, "", [], {})}
            if row:
                criteria.append(_bounded(row, 8))
        elif ntype == "EmpiricalBase":
            if ref := _artifact_ref(attrs):
                data_refs.append({
                    "kind": "empirical_base",
                    "base_type": str(attrs.get("base_type") or ""),
                    "source_ref": ref,
                })
        elif ntype in ("Evidence", "GeneratedData"):
            content = str(attrs.get("content") or attrs.get("description") or "").strip()
            ref = _artifact_ref(attrs)
            if not content and not ref:
                continue
            evidence.append({
                "node_id": str(node.get("id") or ""),
                "kind": "generated_data" if ntype == "GeneratedData" else "evidence",
                "subtype": str(attrs.get("subtype") or ""),
                "content": content[:_DESC_LIMIT],
                "source_ref": ref,
                "status": status,
            })
    chain_context: list[dict[str, Any]] = []
    active_ids = {row["hypothesis_id"] for row in hypothesis_refs}
    for edge in edges:
        if not isinstance(edge, dict) or edge.get("type") != "conditional_successor":
            continue
        predecessor = str(edge.get("from") or "")
        successor = str(edge.get("to") or "")
        if successor not in active_ids:
            continue
        pred_node = node_by_id.get(predecessor) or {}
        pred_evidence_ids = {
            str(item.get("from") or "") for item in edges
            if isinstance(item, dict)
            and item.get("to") == predecessor
            and item.get("type") in {"supports", "refutes", "refines", "relates_to"}
        }
        pred_artifact_ids = {
            str(item.get("from") or "") for item in edges
            if isinstance(item, dict)
            and item.get("to") in pred_evidence_ids
            and item.get("type") == "derived_from"
        }
        chain_context.append({
            "successor_id": successor,
            "predecessor_id": predecessor,
            "predecessor_status": pred_node.get("status"),
            "predecessor_evidence": [
                node_by_id[node_id] for node_id in sorted(pred_evidence_ids)
                if node_id in node_by_id
            ][:8],
            "predecessor_artifacts": [
                node_by_id[node_id] for node_id in sorted(pred_artifact_ids)
                if node_id in node_by_id
            ][:8],
        })
    snapshot = {
        "hypothesis_refs": hypothesis_refs[:_MAX_HYPOTHESIS_REFS],
        "constraints": constraints[:20],
        "confirmation_criteria": criteria[:8],
        "data_refs": data_refs[:20],
        "prior_evidence": evidence[:8],
        "hypothesis_chain_context": chain_context[:4],
        "rendered": rendered[:4000],
    }
    return {key: value for key, value in snapshot.items() if value}


def _merge_constraint_rows(
    frame_rows: list[dict[str, Any]], graph_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Frame rows first, then graph-only rows; dedup by (subtype, content)."""
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in [*frame_rows, *graph_rows]:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("subtype") or "").strip().lower(),
            re.sub(r"\s+", " ", str(row.get("content") or "").strip().lower()),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged


def _constraints_from_state(state: Any) -> list[dict[str, Any]]:
    """Prefer explicit experiment_constraints; else the ContextInit research_frame."""
    existing = state.get("experiment_constraints") if hasattr(state, "get") else None
    if isinstance(existing, list) and existing:
        return [item for item in existing if isinstance(item, dict)]
    raw = state.get("research_frame") if hasattr(state, "get") else None
    if not raw:
        return []
    try:
        from CoScientist.context_init.agent import coerce_frame
        from CoScientist.context_init.commit import frame_constraint_rows
        return frame_constraint_rows(coerce_frame(raw))
    except Exception:  # noqa: BLE001 — missing/invalid frame must not block planning
        return []


def _operations_from_state(state: Any, source_request: str = "") -> list[dict[str, str]]:
    """Committed ops, then ContextInit frame, then numbered steps in the ask."""
    from CoScientist.context_init.operations import (
        normalize_operation_rows,
        parse_numbered_operations,
    )

    existing = state.get("experiment_operations") if hasattr(state, "get") else None
    rows = normalize_operation_rows(existing)
    if rows:
        return rows
    raw = state.get("research_frame") if hasattr(state, "get") else None
    if raw:
        try:
            from CoScientist.context_init.agent import coerce_frame
            from CoScientist.context_init.commit import frame_operations
            rows = frame_operations(coerce_frame(raw))
            if rows:
                return rows
        except Exception:  # noqa: BLE001 — missing/invalid frame must not block planning
            rows = []
    ask = source_request or (
        str(state.get("experiment_source_request") or "") if hasattr(state, "get") else ""
    )
    parsed = parse_numbered_operations(ask) if ask else []
    if len(parsed) >= 2:
        return normalize_operation_rows([op.model_dump() for op in parsed])
    return []

def extract_repo_candidates(
    source_request: str, *, limit: int = _MAX_REPO_CANDIDATES,
) -> list[dict[str, str]]:
    """Extract git repo URLs as tool-selection candidates (not alembic mandates)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    if not source_request:
        return out
    for match in _REPO_URL_RE.finditer(source_request):
        raw = match.group("url").rstrip(").,;]")
        url = raw[:-4] if raw.lower().endswith(".git") and raw.startswith("http") else raw
        if (key := url.lower()) in seen:
            continue
        seen.add(key)
        path = url.split("://", 1)[1] if "://" in url else (
            url.split(":", 1)[-1] if url.startswith("git@") else url
        )
        parts = [p for p in path.replace(".git", "").split("/") if p]
        host, owner = (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "")
        name = parts[2] if len(parts) > 2 else (parts[-1] if parts else "")
        out.append({
            "url": url if url.startswith("http") else f"https://{host}/{owner}/{name}".rstrip("/"),
            "host": host, "owner": owner, "repo_name": name.replace(".git", ""), "kind": "git_repo",
        })
        if len(out) >= limit:
            break
    return out


def _merge_repo_candidates(
    *groups: list[dict[str, Any]], limit: int = _MAX_REPO_CANDIDATES,
) -> list[dict[str, Any]]:
    """Dedupe by normalized URL; earlier groups win order priority."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip().rstrip("/")
            if not url:
                continue
            key = url.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= limit:
                return out
    return out


def _inventory_covers_ask(
    planner_caps: list[dict[str, Any]],
    source_request: str,
    operations: Iterable[Any] = (),
) -> bool:
    """True when this-run retrieve covers compute capabilities."""
    from CoScientist.experiments.capabilities.inventory import (
        index_inventory_tools,
        inventory_covers_capabilities,
        match_named_inventory_tool,
    )

    by_tool = index_inventory_tools(planner_caps)
    if not by_tool:
        return False
    if inventory_covers_capabilities(by_tool):
        return True
    return match_named_inventory_tool(source_request, by_tool) is not None


def resolve_repo_candidates(
    source_request: str,
    *,
    planner_caps: list[dict[str, Any]] | None = None,
    route_alembic: bool = False,
    cached: list[dict[str, Any]] | None = None,
    search: bool | None = None,
    search_limit: int = 5,
    operations: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Ask URLs + optional GitHub search when Alembic is on and inventory is empty.

    ``search=None`` → search when there are no this-run compute tools, or when
    there are no frame operations and the ask does not name an inventory tool.
    Pass ``search=False`` in unit tests to skip the network.
    """
    from_ask = extract_repo_candidates(source_request)
    # Explicit URLs are useful to Coder as well.  The feature switch controls
    # discovery/building, not whether a URL supplied by the operator survives
    # into the experiment context.
    if not route_alembic:
        return from_ask[:_MAX_REPO_CANDIDATES]
    caps = list(planner_caps or [])
    do_search = (
        (not _inventory_covers_ask(caps, source_request, operations))
        if search is None else bool(search)
    )
    if not do_search:
        return from_ask[:_MAX_REPO_CANDIDATES]

    if cached:
        merged = _merge_repo_candidates(from_ask, list(cached))
        if len(merged) >= min(search_limit, _MAX_REPO_CANDIDATES):
            return merged[:_MAX_REPO_CANDIDATES]

    from CoScientist.experiments.capabilities.repo_searcher import search_repos_sync

    found: list[dict[str, Any]] = []
    try:
        result = search_repos_sync(source_request, limit=search_limit)
        found = [c.to_context_item() for c in result.candidates]
        if result.errors:
            audit(
                logger,
                f"EXPERIMENT_REPO_SEARCH_ERRORS n={len(result.errors)} "
                f"sample={result.errors[0][:160]}",
                level=logging.WARNING,
            )
        audit(
            logger,
            f"EXPERIMENT_REPO_SEARCH hit={len(found)} raw={result.total_raw} "
            f"queries={result.search_queries}",
            stdout=(
                f"EXPERIMENT_REPO_SEARCH hit={len(found)} "
                f"urls={[c.get('url') for c in found[:3]]}"
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("repo search failed: %s", exc)
        audit(logger, f"EXPERIMENT_REPO_SEARCH_FAILED err={exc}", level=logging.WARNING)

    return _merge_repo_candidates(from_ask, found, list(cached or []))[:_MAX_REPO_CANDIDATES]

def _statement_from_hypothesis_body(body: str) -> str:
    """Prefer *Statement:* from system-HypothesesAgent prose; else first content line."""
    body = (body or "").strip()
    if not body:
        return ""
    if m := _STATEMENT_IN_BODY_RE.search(body):
        statement = re.sub(r"\s+", " ", m.group("statement")).strip()
        return re.sub(r"^\*+\s*", "", statement).strip()
    for line in body.splitlines():
        text = re.sub(r"^\*+|\*+$", "", line.strip()).strip()
        if not text:
            continue
        if re.match(r"(?i)^(verificationmethod|confirmationcriteria)\b", text):
            break
        if re.match(r"(?i)^statement\s*:", text):
            return re.sub(r"(?i)^statement\s*:\s*", "", text).strip()
        return text
    return re.sub(r"^\*+\s*", "", re.sub(r"\s+", " ", body).strip()).strip()


def extract_hypothesis_refs(
    source_request: str, *, legacy_hypotheses: Any = None, limit: int = _MAX_HYPOTHESIS_REFS,
) -> list[dict[str, str]]:
    """Merge explicit H* / Hypothesis-N labels with legacy session hypotheses."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(hid: str, statement: str) -> None:
        hid = str(hid or "").strip().upper()
        statement = re.sub(r"\s+", " ", str(statement or "").strip())
        if hid and statement and hid not in seen:
            seen.add(hid)
            out.append({"hypothesis_id": hid, "statement": statement[:800]})

    if source_request:
        for match in _H_LABEL_RE.finditer(source_request):
            _add(match.group("id"), match.group("statement"))
            if len(out) >= limit:
                return out
        # System HypothesesAgent style → normalize Hypothesis N → HN
        for match in _HYPOTHESIS_N_RE.finditer(source_request):
            statement = _statement_from_hypothesis_body(match.group("body"))
            if statement:
                _add(f"H{int(match.group('num'))}", statement)
            if len(out) >= limit:
                return out
    for index, item in enumerate(legacy_hypotheses or []):
        if len(out) >= limit:
            break
        if isinstance(item, str) and (text := item.strip()):
            m = re.match(r"^(H\d+)\s*[.:)\-]\s*(.+)$", text, re.I | re.DOTALL)
            if m:
                _add(*m.groups())
                continue
            m_n = re.match(
                r"^(?:\*\*)?Hypothesis\s*(\d+)\s*(?:\([^)]*\))?\s*(?:\*\*)?\s*[.:)\-]\s*(.+)$",
                text,
                re.I | re.DOTALL,
            )
            if m_n:
                _add(f"H{int(m_n.group(1))}", _statement_from_hypothesis_body(m_n.group(2)))
            else:
                _add(f"H{index + 1}", text)
        elif isinstance(item, dict):
            hid = item.get("hypothesis_id") or item.get("id") or item.get("key")
            statement = (
                item.get("statement") or item.get("text")
                or item.get("hypothesis") or item.get("content")
            )
            if statement:
                _add(str(hid) if hid else f"H{index + 1}", str(statement))
    return out[:limit]

def _bounded(value: Any, limit: int) -> Any:
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, list):
        return copy.deepcopy(value[:limit])
    if isinstance(value, dict):
        return copy.deepcopy(dict(list(value.items())[:limit]))
    return copy.deepcopy(value)

def _schema_brief(schema: Any) -> dict[str, Any]:
    """Keep required + param names; drop nested prose."""
    if not isinstance(schema, dict):
        return {}
    brief: dict[str, Any] = {}
    if required := schema.get("required"):
        brief["required"] = list(required)[:12]
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        brief["params"] = list(props.keys())[:16]
    return brief

def _normalize_capabilities(items: Any) -> list[dict[str, Any]]:
    """Project tool dicts into planner/critique inventory shape."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items or []:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or item.get("name") or "").strip()
        server_id = str(item.get("server_id") or "").strip()
        if not tool or not server_id or (key := (server_id, tool)) in seen:
            continue
        seen.add(key)
        schema = item.get("input_schema") or {}
        out.append({
            "tool": tool, "server_id": server_id,
            "description": str(item.get("description") or "")[:_DESC_LIMIT],
            "input_schema": _bounded(schema, 40) if isinstance(schema, dict) else {},
            "score": item.get("score"),
            "url": item.get("url"),
        })
        if len(out) >= 20:
            break
    return out

def _cap_for_prompt(cap: dict[str, Any]) -> dict[str, Any]:
    row = {
        "tool": cap["tool"], "server_id": cap["server_id"],
        "url": cap.get("url"),
        "description": str(cap.get("description") or "")[:_PROMPT_DESC_LIMIT],
    }
    if family := str(cap.get("family") or "").strip():
        row["family"] = family
    row.update(_schema_brief(cap.get("input_schema")))
    return row

def _prompt_context(context: dict[str, Any]) -> str:
    """Compact JSON for the planner instruction."""
    from CoScientist.experiments.capabilities.inventory import get_grouped_mcp_inventory

    available = context.get("available_mcp_capabilities") or []
    grouped_servers = context.get("available_mcp_servers") or get_grouped_mcp_inventory(available)
    prompt_servers = [
        {
            "name": s["name"],
            "server_id": s["server_id"],
            "url": s["url"],
            "tools": [
                t["name"] if isinstance(t, dict) else str(t)
                for t in s.get("tools") or []
            ],
        }
        for s in grouped_servers
    ]
    # Include both available_mcp_servers (grouped) and available_mcp_capabilities (flat) in prompt projection
    slim: dict[str, Any] = {
        "experiment_run_id": context.get("experiment_run_id"),
        "source_request": context.get("source_request"),
        "context_digest": context.get("context_digest") or "",
        "route_alembic": bool(context.get("route_alembic")),
        # route_fedot stays out: the planner prompt is built with or without the
        # FEDOT.MAS route, and a switched-off route should not reach it at all.
        "available_mcp_servers": prompt_servers,
        "available_mcp_capabilities": [
            _cap_for_prompt(c) for c in (context.get("available_mcp_capabilities") or [])
        ],
        "available_research_capabilities": [
            _cap_for_prompt(c) for c in (context.get("available_research_capabilities") or [])
        ],
    }
    # Only while the medical route is in the run: an empty list would still name
    # it to the planner.
    if medical := context.get("available_medical_capabilities"):
        slim["available_medical_capabilities"] = [_cap_for_prompt(c) for c in medical]
    for key in _PROMPT_OPTIONAL_KEYS:
        if key == "hypotheses" and context.get("hypothesis_refs"):
            continue
        if (val := context.get(key)) not in (None, "", [], {}):
            slim[key] = val
    return json.dumps(slim, ensure_ascii=False, separators=(",", ":"))

def _merge_capabilities(*sources: Any) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for source in sources:
        for cap in _normalize_capabilities(source):
            key = (str(cap["server_id"]), str(cap["tool"]))
            if key in seen:
                continue
            seen.add(key)
            merged.append(cap)
            if len(merged) >= 20:
                return merged
    return merged

def _resolve_capabilities(state: Any, previous_context: dict[str, Any]) -> list[dict[str, Any]]:
    """Union live discovery, durable snapshots, prior context. RETRIEVED first."""
    return _merge_capabilities(
        state.get(RETRIEVED_CAPABILITIES_KEY), state.get(DISCOVERED_CAPABILITIES_KEY),
        state.get("filtered_tools"), previous_context.get("available_mcp_capabilities"),
        state.get("accumulated_tools"),
    )

def _session_accumulated_raw(callback_context: CallbackContext | None = None) -> list[Any]:
    """Process-global discovery buffer (sid when given, else sole buffer)."""
    try:
        from CoScientist.tools.retrieval_tools import _SESSION_ACCUMULATED
        sid = ""
        if callback_context is not None:
            sid = str(getattr(_invocation_session(callback_context), "id", None) or "")
        if sid and sid in _SESSION_ACCUMULATED:
            return list(_SESSION_ACCUMULATED[sid])
        if len(_SESSION_ACCUMULATED) == 1:
            return list(next(iter(_SESSION_ACCUMULATED.values())))
    except Exception:  # noqa: BLE001
        pass
    return []


def _filter_caps_for_ask(caps: list[dict[str, Any]], ask: str) -> list[dict[str, Any]]:
    from CoScientist.experiments.capabilities.inventory import filter_inventory_to_needed

    return filter_inventory_to_needed(caps)


def stash_experiment_retrieved_capabilities(callback_context: CallbackContext) -> None:
    """Snapshot accumulated retrieval into durable inventory, then drop leftover families."""
    ask = str(callback_context.state.get("experiment_source_request") or "").strip()
    caps = _merge_capabilities(
        _session_accumulated_raw(callback_context), callback_context.state.get("accumulated_tools"),
    )
    if not caps and not ask:
        return
    prior = callback_context.state.get(RETRIEVED_CAPABILITIES_KEY)
    merged = _merge_capabilities(prior, caps)
    callback_context.state[RETRIEVED_CAPABILITIES_KEY] = _filter_caps_for_ask(merged, ask)

def snapshot_experiment_discovered_capabilities(callback_context: CallbackContext) -> None:
    """Persist registry tools so attempt clears do not erase critique inventory."""
    caps = _merge_capabilities(
        callback_context.state.get(RETRIEVED_CAPABILITIES_KEY),
        callback_context.state.get("filtered_tools"),
        callback_context.state.get(DISCOVERED_CAPABILITIES_KEY),
    )
    if caps:
        ask = str(callback_context.state.get("experiment_source_request") or "").strip()
        callback_context.state[DISCOVERED_CAPABILITIES_KEY] = _filter_caps_for_ask(caps, ask)

def skip_executor_without_runtime(callback_context: CallbackContext) -> types.Content | None:
    """Fail closed when plan review never approved a runtime."""
    state = callback_context.state
    runtime = state.get("experiment_runtime") or {}
    if isinstance(runtime, dict) and runtime.get("approved") and runtime.get("phase") == "execution":
        return None
    phase = runtime.get("phase") if isinstance(runtime, dict) else None
    # A phase is where the module stopped; the pause reason is why. On a review
    # nobody answered, "phase=awaiting_review" was all this said.
    from CoScientist.experiments.review import PAUSE_REASON_STATE_KEY
    pause_reason = str(state.get(PAUSE_REASON_STATE_KEY) or "")
    if state.get("experiment_plan_review_paused"):
        reason = f"plan review paused ({pause_reason})" if pause_reason else "plan review paused"
    elif pause_reason:
        reason = f"{pause_reason} (phase={phase or 'missing'})"
    else:
        reason = f"phase={phase or 'missing'}"
    message = f"Experiment execution skipped: no approved experiment runtime is active ({reason})."
    audit(logger, f"EXPERIMENT_EXECUTION_SKIPPED {reason}")
    state["experiment_execution_summary"] = message
    return types.Content(role="model", parts=[types.Part(text=message)])

def build_experiment_context(callback_context: CallbackContext) -> None:
    """Write one JSON-native context snapshot without copying raw chat history."""
    state = callback_context.state
    user_text = _user_text(callback_context)
    previous_context = state.get("experiment_context") or {}
    previous_runtime = state.get("experiment_runtime") or {}
    force_new_run = bool(state.get("experiment_force_new_run"))
    if not isinstance(previous_context, dict):
        previous_context = {}
    if not isinstance(previous_runtime, dict):
        previous_runtime = {}
    # One accepted experiment per session until HITL asks for replan.
    # A second EM hop after phase=completed must not wipe the runtime
    # (that is what started a fresh plan and the start_task loop).
    if previous_runtime.get("phase") == "completed" and not force_new_run:
        return
    persisted = str(state.get("experiment_source_request") or "").strip()
    prev_request = str(
        persisted or previous_context.get("source_request") or ""
    ).strip()
    critique = state.get("experiment_plan_critique") or {}
    if not isinstance(critique, dict):
        critique = {}
    # Critique/schema revise + HITL edits keep same run_id + inventory.
    planning_revision = not force_new_run and bool(
        state.get("experiment_plan_validation_errors")
        or critique.get("verdict") == "revise"
        or previous_runtime.get("phase") == "awaiting_review"
        or previous_runtime.get("phase") == "replan_requested"
    )
    source_request = persisted or (prev_request if planning_revision and prev_request else (user_text or prev_request))
    same_request = bool(source_request) and source_request == prev_request
    revising = planning_revision and same_request
    run_id = previous_context.get("experiment_run_id") if revising else f"EXRUN-{uuid4().hex}"
    if not revising:
        # Diagnostic for the re-planning loop: this is the only place that drops a
        # finished runtime, so record when it fires and what phase it destroyed.
        _prev_phase = previous_runtime.get("phase") if isinstance(previous_runtime, dict) else None
        audit(
            logger,
            "EXPERIMENT_NEW_RUN_CLEAR prev_phase=%r same_request=%s planning_revision=%s"
            % (_prev_phase, same_request, planning_revision),
        )
        for key in _CLEAR_ON_NEW_RUN:
            if key in state:
                state[key] = None
        # Wipe durable inventory only when ask changed from a prior ask (not first entry).
        if prev_request and not same_request:
            state[DISCOVERED_CAPABILITIES_KEY] = None
            state[RETRIEVED_CAPABILITIES_KEY] = None
            # Same rule for the replan budget: a genuinely new ask starts with a
            # full budget, a replan of the SAME ask keeps what it has spent.
            # This key is deliberately absent from _CLEAR_ON_NEW_RUN — clearing
            # it there would zero the counter on the very hop it must survive.
            from CoScientist.experiments.runtime.state_machine import REPLAN_ROUNDS_KEY
            state[REPLAN_ROUNDS_KEY] = 0
    if force_new_run:
        state["experiment_force_new_run"] = False
    # Mid-attempt filtered_tools is task-scoped; do not overwrite discovery.
    mid_attempt = bool(state.get("experiment_active_envelope"))
    live_discovery = _normalize_capabilities(state.get("filtered_tools")) if not mid_attempt else []
    if live_discovery:
        state[DISCOVERED_CAPABILITIES_KEY] = _merge_capabilities(
            state.get(RETRIEVED_CAPABILITIES_KEY), live_discovery,
            state.get(DISCOVERED_CAPABILITIES_KEY),
        )
    # If rerank cleared accumulated but forgot to stash, recover once (sole buffer).
    if not state.get(RETRIEVED_CAPABILITIES_KEY):
        stash_caps = _normalize_capabilities(state.get("accumulated_tools"))
        if not stash_caps:
            stash_caps = _normalize_capabilities(_session_accumulated_raw())
        if stash_caps:
            state[RETRIEVED_CAPABILITIES_KEY] = stash_caps
    capabilities = _resolve_capabilities(state, previous_context if (revising or same_request) else {})
    preferred = _normalize_capabilities(state.get("filtered_tools")) if not mid_attempt else []
    if not preferred:
        preferred = _normalize_capabilities(state.get(DISCOVERED_CAPABILITIES_KEY))
    planner_caps = capabilities if capabilities else preferred
    # Graph-first science input: typed nodes committed by previous agents
    # (ContextInit frame, HypothesesAgent H, prior Evidence). Empty when the
    # graph is disabled/empty — session/text fallbacks below still apply.
    snapshot = research_graph_snapshot(callback_context)
    # Typed snapshot fields go into the planner JSON. Do not also dump the
    # whole-graph prose overview — it duplicates nodes and blows the first turn.
    research_context = str(state.get("research_context") or "")[:4000]
    gaps = _bounded(state.get("experiment_unresolved_gaps") or [], 20)
    if not isinstance(gaps, list):
        gaps = []
    if not planner_caps:
        gap = "No ready MCP capabilities in planner inventory after discovery/rerank."
        if gap not in gaps:
            gaps = [*gaps, gap][:20]
    from CoScientist.config import get_settings
    from CoScientist.experiments.capabilities.inventory import (
        FAMILY_MEDICAL,
        FAMILY_RESEARCH,
        declared_family_capabilities,
    )
    from CoScientist.experiments.runtime.state_machine import (
        fedot_route_available,
        medical_route_available,
        session_route_agents,
    )
    experiments = get_settings().experiments
    # Routes as this session's executor holds them - the tree start_task hands
    # work to, not the YAML and switches as they are now.
    route_agents = session_route_agents(
        getattr(getattr(callback_context, "_invocation_context", None), "agent", None)
    )
    medical_on = medical_route_available(route_agents=route_agents)
    revision_feedback: list[dict[str, Any]] = []
    if state.get("experiment_plan_validation_errors"):
        revision_feedback.append({
            "kind": "schema", "errors": _bounded(state.get("experiment_plan_validation_errors"), 12),
        })
    if critique.get("verdict") == "revise":
        revision_feedback.append({
            "kind": "critique", "issues": _bounded(critique.get("issues") or [], 12),
        })
    cached_repos = state.get("experiment_repo_candidates")
    if not isinstance(cached_repos, list):
        cached_repos = previous_context.get("repo_candidates") if (revising or same_request) else []
    if not isinstance(cached_repos, list):
        cached_repos = []
    operations = _operations_from_state(state, source_request)
    repo_candidates = resolve_repo_candidates(
        source_request,
        planner_caps=planner_caps,
        route_alembic=bool(experiments.route_alembic),
        cached=cached_repos if (revising or same_request) else None,
        operations=operations,
    )
    if repo_candidates:
        state["experiment_repo_candidates"] = repo_candidates
    constraints = _merge_constraint_rows(
        _constraints_from_state(state), snapshot.get("constraints") or [],
    )
    if operations:
        state["experiment_operations"] = operations
    # Hypotheses source order is fixed: graph nodes → refs already published by
    # the HypothesesAgent bridge → text/legacy extraction as the last fallback.
    published_refs = [
        r for r in (state.get("hypothesis_refs") or [])
        if isinstance(r, dict) and str(r.get("statement") or "").strip()
    ]
    hypothesis_refs = (
        snapshot.get("hypothesis_refs")
        or published_refs
        or extract_hypothesis_refs(source_request, legacy_hypotheses=state.get("hypotheses"))
    )
    session_data_refs = [
        r for r in (state.get("experiment_data_refs") or []) if isinstance(r, dict)
    ]
    seen_data_refs = {
        str(r.get("source_ref") or r.get("ref") or "").strip() for r in session_data_refs
    }
    data_refs = [
        *session_data_refs,
        *[
            row for row in (snapshot.get("data_refs") or [])
            if str(row.get("source_ref") or "").strip() not in seen_data_refs
        ],
    ]
    context = {
        "experiment_run_id": run_id, "source_request": source_request,
        "research_focus_id": state.get("research_focus_id"), "research_context": research_context,
        "hypotheses": _bounded(state.get("hypotheses") or [], 20),
        "hypothesis_refs": hypothesis_refs,
        "operations": _bounded(operations, 20),
        "prior_results": _bounded(state.get("experiment_task_results") or [], 20),
        "prior_evidence": _bounded(snapshot.get("prior_evidence") or [], 20),
        "hypothesis_chain_context": _bounded(
            snapshot.get("hypothesis_chain_context") or [], 8),
        "confirmation_criteria": _bounded(snapshot.get("confirmation_criteria") or [], 20),
        "data_refs": _bounded(data_refs, 20),
        "constraints": _bounded(constraints, 20),
        "available_mcp_capabilities": planner_caps,
        "available_mcp_servers": get_grouped_mcp_inventory(planner_caps),
        "available_research_capabilities": declared_family_capabilities(FAMILY_RESEARCH),
        "available_medical_capabilities": (
            declared_family_capabilities(FAMILY_MEDICAL) if medical_on else []
        ),
        "preferred_mcp_capabilities": preferred if preferred else planner_caps,
        "critique_mcp_capabilities": capabilities if capabilities else planner_caps,
        "explicit_mcp_servers": _bounded(state.get("experiment_explicit_mcps") or [], 20),
        "repo_candidates": repo_candidates,
        "revision_feedback": revision_feedback, "unresolved_gaps": gaps[:20],
        "context_digest": research_context[:1500] or source_request[:1500],
        # The effective FEDOT.MAS answer (switch AND FedotAgent attached), not the
        # bare switch: the planner used to see route_fedot=true for an agent the
        # YAML had already removed.
        "route_alembic": bool(experiments.route_alembic),
        "route_fedot": fedot_route_available(experiments, route_agents=route_agents),
        "route_medical": medical_on,
    }
    scope = state.get("pipeline_scope")
    if isinstance(scope, dict) and scope.get("source") == "hitl":
        context["pipeline_scope"] = {
            "research": bool(scope.get("research")),
            "hypotheses": bool(scope.get("hypotheses")),
            "experiments": bool(scope.get("experiments")),
        }
    state["experiment_context"] = context
    state[PLANNER_CONTEXT_KEY] = _prompt_context(context)
    state["experiment_source_request"] = source_request
    if os.getenv("COSCIENTIST_EXPERIMENT_AUDIT_STDOUT") == "1":
        tools = [f"{c['server_id']}/{c['tool']}" for c in planner_caps]
        print(f"EXPERIMENT_INVENTORY_READY count={len(tools)} tools={tools}", flush=True)


__all__ = [
    "DISCOVERED_CAPABILITIES_KEY", "RETRIEVED_CAPABILITIES_KEY",
    "build_experiment_context", "enforce_experiment_retrieval_budget",
    "extract_hypothesis_refs", "extract_repo_candidates",
    "research_graph_snapshot",
    "resolve_repo_candidates",
    "reset_experiment_retrieval_budget",
    "skip_executor_without_runtime", "snapshot_experiment_discovered_capabilities",
    "stash_experiment_retrieved_capabilities",
]
