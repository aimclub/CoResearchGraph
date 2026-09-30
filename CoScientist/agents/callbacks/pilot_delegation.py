"""Verify the isolated pilot from observed ADK tool events.

The pilot keeps the normal tool roster and lets the orchestrator choose its
order. A final answer is valid only after each required tool has actually
been called and returned a successful response in the current invocation.
"""

import json
import re

from google.adk.models.llm_response import LlmResponse
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

from CoScientist.agents.callbacks.link_registry import (
    find_urls,
    link_id_for,
    register_user_links,
)

_REQUIRED = ("retrieve_tools", "ResearchAgent", "TaskExecutorAgent")
_PILOT_SCIENCE_TOOLS = (
    "dataset_overview_heracleum_tox",
    "chemical_space_clustering",
    "predict_ld50",
    "predict_molecule_profile",
)


def enforce_pilot_science_handoff(tool, args, tool_context):
    """Carry discovered pilot MCP names into the remote executor request."""
    if not isinstance(tool, AgentTool) or tool.name != "TaskExecutorAgent":
        return None
    request = args.get("request") if isinstance(args, dict) else None
    if not isinstance(request, str):
        return None
    retrieved = tool_context.state.get("accumulated_tools") or []
    science = {
        item["tool"]: item.get("server_id")
        for item in retrieved
        if isinstance(item, dict) and item.get("tool") in _PILOT_SCIENCE_TOOLS
    }
    invocation = getattr(tool_context, "_invocation_context", None)
    invocation_id = getattr(invocation, "invocation_id", None)
    prior = tool_context.state.get("_pilot_science_catalog") or {}
    known = dict(prior.get("tools") or {}) if prior.get("invocation_id") == invocation_id else {}
    known.update(science)
    tool_context.state["_pilot_science_catalog"] = {
        "invocation_id": invocation_id, "tools": known,
    }
    if not science or (
        all(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request) for name in science)
        and all(server_id in request for server_id in science.values() if server_id)
    ):
        return None
    available = ", ".join(
        f"{name} (server_id={server_id})" if server_id else name
        for name, server_id in science.items()
    )
    args["request"] = (
        f"{request}\n\nDiscovered pilot MCP tools: {available}. "
        "This list is routing context; execute only the computation requested "
        "above through ToolPipelineAgent. heracleum-tox is a prepared MCP "
        "service, not a GitHub repository."
    )
    return None


def enforce_pilot_executor_route(tool, args, tool_context):
    """Do not turn a named ready MCP computation into sandbox code."""
    if not isinstance(tool, AgentTool) or tool.name != "CoderAgent":
        return None
    subtask = args.get("request") if isinstance(args, dict) else None
    if not isinstance(subtask, str):
        return None
    invocation = getattr(tool_context, "_invocation_context", None)
    content = getattr(invocation, "user_content", None)
    request = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    names_ready_tool = any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", subtask)
        for name in _PILOT_SCIENCE_TOOLS
    )
    names_prepared_service = re.search(
        r"\b(?:heracleum[-_ ]tox|server_id)\b", subtask, re.IGNORECASE
    )
    reimplements_science = re.search(
        r"\b(?:predict|compute|calculate|estimate|run)\b.{0,100}?"
        r"\b(?:LD50|hepatotoxicity|DILI|cardiotoxicity|carcinogenicity|"
        r"chemical[ -]space|clustering|molecule[ -]profile)\b",
        subtask, re.IGNORECASE,
    ) and any(
        re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request)
        for name in _PILOT_SCIENCE_TOOLS
    )
    ambiguous_clone = (
        re.search(r"\b(?:clone|checkout|check out)\b", subtask, re.IGNORECASE)
        and not re.search(r"https?://github\.com/[^\s,;)]+", subtask, re.IGNORECASE)
        and any(re.search(rf"(?<!\w){re.escape(name)}(?!\w)", request)
                for name in _PILOT_SCIENCE_TOOLS)
    )
    if names_ready_tool or names_prepared_service or reimplements_science or ambiguous_clone:
        return {"error": (
            "The request names a ready scientific MCP tool. Delegate this "
            "computation to ToolPipelineAgent; CoderAgent cannot call MCP tools. "
            "A prepared MCP server label is not a GitHub repository."
        )}
    return None


def preserve_pilot_target(tool, args, tool_context):
    """Keep an explicit science target when the executor delegates downstream."""
    if not isinstance(tool, AgentTool) or tool.name != "ToolPipelineAgent":
        return None
    request = args.get("request") if isinstance(args, dict) else None
    if not isinstance(request, str):
        return None
    invocation = getattr(tool_context, "_invocation_context", None)
    content = getattr(invocation, "user_content", None)
    source = "\n".join(
        part.text or "" for part in (getattr(content, "parts", None) or [])
    )
    target = re.search(r"\bTarget tool:\s*([A-Za-z_][A-Za-z_0-9]*)\b", source)
    if not target or target.group(1) not in _PILOT_SCIENCE_TOOLS:
        return None
    target_pattern = re.compile(
        r"\bTarget tool:\s*" + re.escape(target.group(1))
        + r"\b(?:\s*\(server_id=([A-Za-z0-9_-]+)\))?"
    )
    source_target = target_pattern.search(source)
    request_target = target_pattern.search(request)
    if request_target and (
        not source_target.group(1)
        or request_target.group(1) == source_target.group(1)
    ):
        return None
    args["request"] = f"{source}\n\nExecutor subtask: {request}"
    return None


def _completed_delegations(callback_context):
    invocation = callback_context._invocation_context
    called = set()
    completed = {}
    for event in invocation.session.events:
        if event.invocation_id != invocation.invocation_id:
            continue
        called.update(call.name for call in event.get_function_calls())
        for response in event.get_function_responses():
            if response.name not in called:
                continue
            body = response.response
            if isinstance(body, dict) and (
                body.get("error") or body.get("status") in {"error", "failed"}
            ):
                continue
            completed.setdefault(response.name, []).append(body)
    return completed


def block_pilot_repeated_overview(tool, args, tool_context):
    """Stop another remote executor task from repeating a successful overview."""
    if not isinstance(tool, AgentTool) or tool.name != "TaskExecutorAgent":
        return None
    request = args.get("request") if isinstance(args, dict) else None
    if not isinstance(request, str):
        return None
    target = re.search(r"\bTarget tool:\s*([A-Za-z_][A-Za-z_0-9]*)\b", request)
    if target:
        if target.group(1) != "dataset_overview_heracleum_tox":
            return None
    else:
        task = request.split("Discovered pilot MCP tools:", 1)[0]
        if ("dataset_overview_heracleum_tox" not in task
                or any(name in task for name in _PILOT_SCIENCE_TOOLS[1:])):
            return None
    completed = _completed_delegations(tool_context)
    for body in completed.get("TaskExecutorAgent", []):
        result = body.get("result") if isinstance(body, dict) else None
        try:
            receipt = json.loads(result) if isinstance(result, str) else result
        except ValueError:
            continue
        if not isinstance(receipt, dict) or receipt.get("status") != "computed":
            continue
        for call in receipt.get("scientific_mcp_calls") or []:
            if (isinstance(call, dict)
                    and call.get("tool") == "dataset_overview_heracleum_tox"
                    and call.get("args") == {}
                    and call.get("result") not in (None, "", {})):
                return {
                    "error": "The aggregate overview already succeeded. A repeat "
                             "cannot provide molecule rows or SMILES; use a different "
                             "tool or report that the data are unavailable.",
                    "observed_result": call["result"],
                }
    return None


def _scientific_calls(bodies):
    calls = []
    for body in bodies:
        if not isinstance(body, dict):
            continue
        result = body.get("result")
        try:
            receipt = json.loads(result) if isinstance(result, str) else result
        except ValueError:
            continue
        if not isinstance(receipt, dict) or receipt.get("status") != "computed":
            continue
        if isinstance(receipt.get("scientific_mcp_calls"), list):
            calls.extend(receipt["scientific_mcp_calls"])
    return calls


def _verified_science_names(calls):
    return {
        call.get("tool") for call in calls
        if isinstance(call, dict)
        and call.get("result") not in (None, "", {})
        and not (
            isinstance(call.get("result"), dict)
            and call["result"].get("truncated")
        )
    }


def _request_missing_science(callback_context, name):
    state = getattr(callback_context, "state", None)
    invocation_id = callback_context._invocation_context.invocation_id
    discovered = (state.get("accumulated_tools") or []) if state is not None else []
    science = {
        item["tool"]: item.get("server_id")
        for item in discovered
        if isinstance(item, dict) and item.get("tool") in _PILOT_SCIENCE_TOOLS
    }
    prior = state.get("_pilot_science_catalog") or {}
    if prior.get("invocation_id") == invocation_id:
        science.update(prior.get("tools") or {})
    if not science.get(name):
        raise RuntimeError(
            "Pilot scientific computation contract: required MCP tool "
            f"{name} was not discovered with a server id"
        )

    progress = state.get("_pilot_science_recovery") or {}
    attempted = (
        list(progress.get("attempted") or [])
        if progress.get("invocation_id") == invocation_id else []
    )
    if name in attempted:
        raise RuntimeError(
            "Pilot scientific computation contract: no verified result from "
            f"{name} after a targeted executor attempt"
        )
    attempted.append(name)
    state["_pilot_science_recovery"] = {
        "invocation_id": invocation_id, "attempted": attempted,
    }

    available = ", ".join(
        f"{tool} (server_id={server_id})"
        for tool, server_id in science.items()
        if server_id
    )
    arguments = " with name_or_smiles=xanthotoxin" if name == "predict_molecule_profile" else ""
    cluster_member = (
        " xanthotoxin is a member of Heracleum cluster E."
        if name == "predict_molecule_profile" else ""
    )
    request = (
        f"Target tool: {name} (server_id={science[name]}). "
        f"Call {name}{arguments} through ToolPipelineAgent and return its actual "
        f"MCP result.{cluster_member} "
        f"Available pilot MCP tools: {available}. The other names are routing "
        "context, not additional targets for this request."
    )
    return LlmResponse(content=types.Content(role="model", parts=[
        types.Part.from_function_call(
            name="TaskExecutorAgent", args={"request": request}
        )
    ]))


def require_pilot_delegations(callback_context, llm_response):
    """Reject a final answer until all pilot tools have real call/response pairs."""
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(part.function_call for part in parts):
        return None
    completed = _completed_delegations(callback_context)
    missing = [name for name in _REQUIRED if name not in completed]
    if missing == ["TaskExecutorAgent"]:
        return _request_missing_science(callback_context, _PILOT_SCIENCE_TOOLS[0])
    if missing:
        raise RuntimeError(
            "Pilot delegation contract: final response before observed "
            "call and successful response for " + ", ".join(missing)
        )
    calls = _scientific_calls(completed["TaskExecutorAgent"])
    verified_names = _verified_science_names(calls)
    missing_science = [name for name in _PILOT_SCIENCE_TOOLS if name not in verified_names]
    if missing_science:
        return _request_missing_science(callback_context, missing_science[0])
    return None


def _validate_profile_cost_claims(report, calls):
    """Tie every stated synthesis cost to the molecule actually profiled."""
    costs = {}
    for call in calls:
        if not isinstance(call, dict) or call.get("tool") != "predict_molecule_profile":
            continue
        name = (call.get("args") or {}).get("name_or_smiles")
        result = call.get("result") or {}
        answer = result.get("answer") if isinstance(result, dict) else None
        synthesis = answer.get("synthesis_cost") if isinstance(answer, dict) else None
        cost = synthesis.get("usd_per_g") if isinstance(synthesis, dict) else None
        if isinstance(name, str) and isinstance(cost, (int, float)) and not isinstance(cost, bool):
            costs[re.sub(r"\s+", " ", name).strip().casefold()] = float(cost)

    if re.search(r"аналогичн\w*\s+вызов\w*|similar\s+(?:tool\s+)?calls", report, re.IGNORECASE):
        raise RuntimeError("Pilot report claims unenumerated profile calls")

    for amount in re.findall(
        r"(?<!\d)(\d+(?:[.,]\d+)?)\*{0,2}\s*(?:USD|US\$)\s*/\s*g\b",
        report, re.IGNORECASE,
    ):
        if float(amount.replace(",", ".")) not in costs.values():
            raise RuntimeError(f"Pilot report has an unsupported synthesis cost: {amount} USD/g")

    lines = report.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("|"):
            continue
        headings = [cell.strip().casefold() for cell in line.strip().strip("|").split("|")]
        cost_columns = [
            column for column, heading in enumerate(headings)
            if ("стоим" in heading or "cost" in heading)
            and re.search(r"usd\s*/\s*g", heading, re.IGNORECASE)
        ]
        if not cost_columns:
            continue
        cost_column = cost_columns[0]
        for row in lines[index + 1:]:
            if not row.lstrip().startswith("|"):
                break
            cells = [cell.strip().strip("*") for cell in row.strip().strip("|").split("|")]
            if len(cells) <= cost_column:
                continue
            amount = re.search(r"(?<!\d)\d+(?:[.,]\d+)?", cells[cost_column])
            if not amount:
                continue
            molecule = re.sub(r"\s+", " ", cells[0]).strip().casefold()
            observed = costs.get(molecule)
            if observed is None or float(amount.group().replace(",", ".")) != observed:
                raise RuntimeError(f"Pilot report has an unsupported synthesis cost for {cells[0]}")


def validate_pilot_report(callback_context, llm_response):
    """Keep the narrative report while refusing claims the observed work cannot support."""
    if llm_response.partial:
        return None
    parts = getattr(llm_response.content, "parts", None) or []
    if any(getattr(part, "function_call", None) for part in parts):
        return None
    report = "\n".join(part.text or "" for part in parts)
    if len(report.strip()) < 250 or sum(line.startswith("|") for line in report.splitlines()) < 3:
        raise RuntimeError("Pilot report is incomplete: a substantive report with a table is required")
    if report.lstrip().startswith(("Observed scientific MCP results:", "{")):
        raise RuntimeError("Pilot report is a raw receipt rather than a scientific report")

    completed = _completed_delegations(callback_context)
    calls = _scientific_calls(completed.get("TaskExecutorAgent", []))
    evidence = json.dumps(completed, ensure_ascii=False, default=str)
    registry = callback_context.state.get("user_links") or {}
    observed_urls = {
        url
        for call in calls
        if isinstance(call, dict)
        and call.get("result") not in (None, "", {})
        and not (
            isinstance(call.get("result"), dict)
            and call["result"].get("truncated")
        )
        for url, _, _ in find_urls(json.dumps(
            call["result"], ensure_ascii=False, default=str
        ))
    }
    for link_id in re.findall(r"\[\[link([0-9a-f]+)\]\]", report, re.IGNORECASE):
        ref = f"link{link_id.lower()}"
        entry = registry.get(ref)
        if not isinstance(entry, dict):
            matches = [url for url in observed_urls if link_id_for(url, registry) == ref]
            if len(matches) == 1:
                register_user_links(callback_context.state, matches[0], with_mentions=False)
                registry = callback_context.state.get("user_links") or {}
                entry = registry.get(ref)
        if not isinstance(entry, dict) or entry.get("url") not in evidence:
            raise RuntimeError(f"Pilot report has an unsupported artifact link: {link_id}")
    for target in re.findall(r"\]\(([^)]+\.(?:csv|tsv|png|svg|jpe?g|pdf)(?:\?[^)]*)?)\)", report, re.IGNORECASE):
        if target not in evidence:
            raise RuntimeError(f"Pilot report has an unsupported artifact: {target}")

    for percentage in re.findall(r"\d+(?:[.,]\d+)?[\s\u202f]*%", report):
        if percentage not in evidence:
            raise RuntimeError(f"Pilot report has an unsupported percentage: {percentage}")

    _validate_profile_cost_claims(report, calls)

    result_text = json.dumps(calls, ensure_ascii=False, default=str).lower()
    for line in report.splitlines():
        lowered = line.lower()
        figure_type = "heatmap" if "теплов" in lowered else (
            "dendrogram" if "дендрограмм" in lowered else None
        )
        if not figure_type or figure_type in result_text:
            continue
        if re.search(r"\bне\s+(?:создан|построен|приложен)|отсутств|недоступ", lowered):
            continue
        if re.search(r"✅|создан|построен|сгенерирован|см\.\s*рисунок", lowered):
            raise RuntimeError("Pilot report has an unsupported figure claim")

    if re.search(r"согласуетс[яь].{0,100}экспериментальн", report, re.IGNORECASE | re.DOTALL):
        research = json.dumps(completed.get("ResearchAgent", []), ensure_ascii=False, default=str)
        if not re.search(r"\b(?:10\.\d{4,9}/\S+|PMID[:\s]*\d+)\b", research, re.IGNORECASE):
            raise RuntimeError("Pilot report has an unsupported experimental comparison")

    for source in re.findall(r"\b10\.\d{4,9}/[^\s|)]+|Supplementary\s+Tables?\s+S\d+(?:[-–]\s*S?\d+)?", report, re.IGNORECASE):
        if source.rstrip(".,;*") not in evidence:
            raise RuntimeError(f"Pilot report cites an unsupported source: {source}")
    if re.search(r"не\s+найдено\s+никаких\s+публикац|отсутствуют\s+DOI/PMID", report, re.IGNORECASE):
        raise RuntimeError("Pilot report makes an unsupported categorical literature claim")

    if not (re.search(r"SMILES", report, re.IGNORECASE)
            and re.search(r"недоступ|не\s+доступ|отсутств|не\s+содерж|не\s+предостав", report, re.IGNORECASE)):
        raise RuntimeError("Pilot report must disclose the missing molecule-level SMILES data")
    return None
