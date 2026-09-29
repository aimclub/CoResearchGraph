"""Thin Alembic success adapter: inject MCP and reopen on post_build_route."""
from __future__ import annotations

import copy
import logging
import re
from typing import Any, Mapping, MutableMapping
from urllib.parse import urlparse

from CoScientist.config.settings import ExperimentsSettings
from CoScientist.experiments.schemas import ExecutionRoute, ExperimentTask
from CoScientist.experiments.runtime.shared import audit

logger = logging.getLogger(__name__)

_PLACEHOLDER_TOOL = "alembic_built_tool"
_ALEMBIC_DESC_MARKER = "Use the Alembic-built MCP"
_MCP_ENDPOINT_RE = re.compile(r"https?://[^\s)\]\"'<>]+/mcp\b", re.I)


def _server_id_from_url(repo_url: str | None, mcp_url: str) -> str:
    if repo_url:
        path = urlparse(repo_url).path.strip("/")
        name = path.split("/")[-1] if path else ""
        name = re.sub(r"[^A-Za-z0-9._-]", "-", name).strip("-")
        if name:
            return f"alembic-{name}"
    host = urlparse(mcp_url).hostname or "mcp"
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", host).strip("-") or "mcp"
    return f"alembic-{safe}"


def _tool_refs_from_outputs(
    outputs: dict[str, Any], *, placeholder: bool = True,
) -> list[dict[str, Any]]:
    raw = outputs.get("tools") or outputs.get("mcp_tools") or []
    tools: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and (name := item.strip()):
                tools.append({
                    "name": name,
                    "description": f"Alembic-built tool {name}",
                    "input_schema": None,
                    "required_for_task": True,
                })
            elif isinstance(item, dict) and (name := str(item.get("name") or "").strip()):
                tools.append({
                    "name": name,
                    "description": str(item.get("description") or f"Alembic-built tool {name}"),
                    "input_schema": item.get("input_schema"),
                    "required_for_task": bool(item.get("required_for_task", True)),
                })
    if not tools and placeholder:
        tools.append({
            "name": _PLACEHOLDER_TOOL,
            "description": "Primary tool exposed by the Alembic-built MCP server.",
            "input_schema": None,
            "required_for_task": True,
        })
    return tools


def _first_mcp_endpoint(text: str) -> str | None:
    match = _MCP_ENDPOINT_RE.search(text or "")
    if not match:
        return None
    return match.group(0).rstrip(".,;")


def extract_mcp_url(outputs: Any) -> str | None:
    """Pull an ``…/mcp`` URL from route outputs, nested dicts, or prose."""
    if isinstance(outputs, str):
        value = outputs.strip()
        if value.startswith("http") and "/mcp" in value.split("?", 1)[0]:
            return value.split()[0].rstrip(".,;")
        return _first_mcp_endpoint(outputs)
    if isinstance(outputs, Mapping):
        for key in ("mcp_url", "mcp_endpoint", "url", "endpoint"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip().startswith("http"):
                return value.strip()
        for value in outputs.values():
            if (found := extract_mcp_url(value)):
                return found
        return None
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            if (found := extract_mcp_url(item)):
                return found
    return None


def harvest_alembic_mcp_url(
    *blobs: Any,
    repo_url: str | None = None,
) -> str:
    """MCP URL from results/prose, else the local done-job registry."""
    for blob in blobs:
        found = extract_mcp_url(blob)
        if found:
            return found
    want = str(repo_url or "").rstrip("/").lower()
    if not want:
        return ""
    try:
        from CoScientist.tools.alembic_tools import web_list_builds
    except Exception:
        return ""
    repo_name = want.rsplit("/", 1)[-1]
    for build in web_list_builds() or []:
        if not isinstance(build, dict) or str(build.get("status") or "") != "done":
            continue
        url = str(build.get("mcp_url") or "").strip()
        if not url.startswith("http"):
            continue
        got = str(build.get("repo_url") or "").rstrip("/").lower()
        job_id = str(build.get("job_id") or "").lower()
        if got == want or (repo_name and repo_name in job_id):
            return url
    return ""


def mcp_url_from_task_runtime(task_runtime: Mapping[str, Any] | None) -> str:
    """Served Alembic MCP URL from post-build history or injected mcp_servers."""
    if not isinstance(task_runtime, Mapping):
        return ""
    for entry in reversed(list(task_runtime.get("route_history") or [])):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("reason") or "") != "alembic_post_build":
            continue
        url = str(entry.get("mcp_url") or "").strip()
        if url.startswith("http"):
            return url
    task = task_runtime.get("task") if isinstance(task_runtime.get("task"), dict) else {}
    for server in task.get("mcp_servers") or []:
        if not isinstance(server, dict):
            continue
        if str(server.get("source") or "") != "alembic":
            continue
        url = str(server.get("url") or "").strip()
        if url.startswith("http"):
            return url
    return ""


def tool_names_from_task(task: Mapping[str, Any] | None) -> list[str]:
    names: list[str] = []
    if not isinstance(task, Mapping):
        return names
    for server in task.get("mcp_servers") or []:
        if not isinstance(server, dict):
            continue
        for tool in server.get("tools") or []:
            name = str(tool.get("name") if isinstance(tool, dict) else tool or "").strip()
            if name and name != _PLACEHOLDER_TOOL and name not in names:
                names.append(name)
    return names


def scientific_ask(
    runtime: Mapping[str, Any] | None,
    task: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None = None,
) -> str:
    plan = runtime.get("plan") if isinstance(runtime, Mapping) else {}
    for candidate in (
        (state or {}).get("experiment_source_request") if isinstance(state, Mapping) else None,
        (plan or {}).get("source_request") if isinstance(plan, dict) else None,
        (task or {}).get("description") if isinstance(task, Mapping) else None,
        (task or {}).get("name") if isinstance(task, Mapping) else None,
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def alembic_post_build_context(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """None unless the active EM task already has a served Alembic MCP."""
    if not isinstance(state, Mapping):
        return None
    runtime = state.get("experiment_runtime")
    if not isinstance(runtime, dict):
        return None
    tid = runtime.get("active_task_id")
    task_runtime = (runtime.get("tasks") or {}).get(tid) if tid else None
    if not isinstance(task_runtime, dict):
        return None
    mcp_url = mcp_url_from_task_runtime(task_runtime)
    if not mcp_url:
        return None
    task = task_runtime.get("task") if isinstance(task_runtime.get("task"), dict) else {}
    return {
        "mcp_url": mcp_url,
        "ask": scientific_ask(runtime, task, state),
        "tools": tool_names_from_task(task),
        "task": task,
        "task_runtime": task_runtime,
        "runtime": runtime,
    }


def compose_alembic_fedot_task(ctx: Mapping[str, Any], original: str = "") -> str:
    """Authoritative post-build Fedot brief — never a hallucinated ``*.py`` task."""
    tools = ", ".join(ctx.get("tools") or []) or "list tools on the MCP server and call them"
    ask = str(ctx.get("ask") or "").strip() or (
        "Carry out the scientific experiment using the Alembic MCP."
    )
    body = (
        f"{ask}\n\n"
        "Alembic MCP is already served. Call its tools to satisfy the scientific "
        "ask. Do not write, execute, or invent a local Python script. Do not "
        "recommend CoderAgent.\n"
        f"mcp_url: {ctx.get('mcp_url')}\n"
        f"mcp_tools: {tools}\n"
    )
    orig = (original or "").strip()
    if orig and orig not in body:
        body += f"\nExecutor notes (non-authoritative): {orig}"
    return body


def pin_alembic_post_build_request(
    args: dict[str, Any],
    task_runtime: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
) -> bool:
    """Overwrite Fedot/ExperimentAgent request after Alembic (script-name leak)."""
    mcp_url = mcp_url_from_task_runtime(task_runtime)
    if not mcp_url:
        return False
    task = task_runtime.get("task") if isinstance(task_runtime.get("task"), dict) else {}
    tools = tool_names_from_task(task)
    ask = scientific_ask(runtime, task, state)
    payload = {
        "mcp_url": mcp_url,
        "task": compose_alembic_fedot_task(
            {"mcp_url": mcp_url, "ask": ask, "tools": tools}
        ),
        "instruction": (
            "Call the attached Alembic MCP tools to satisfy the scientific ask. "
            "Do not execute or invent a local Python script. Do not recommend "
            "CoderAgent. Missing input files → honest failure."
        ),
    }
    if tools:
        payload["mcp_tools"] = tools
    args["request"] = payload
    return True


def stamp_alembic_science_description(
    task: dict[str, Any], *, repo_url: str, source_request: str = "",
) -> None:
    """Keep the scientific ask; forbid 'write a Python script' as the alembic job."""
    desc = str(task.get("description") or "").strip()
    if _ALEMBIC_DESC_MARKER in desc:
        return
    ask = (source_request or desc or str(task.get("name") or "")).strip()
    suffix = (
        f" {_ALEMBIC_DESC_MARKER} from {repo_url} to carry out the scientific ask"
        f"{': ' + ask if ask and ask != desc else ''}; "
        "do not reimplement as a local Python script."
    )
    task["description"] = (f"{desc} {suffix}".strip() if desc else suffix.strip())


def apply_alembic_success(
    state: MutableMapping[str, Any],
    runtime: dict[str, Any],
    task_runtime: dict[str, Any],
    *,
    mcp_url: str,
    outputs: dict[str, Any] | None = None,
    settings: ExperimentsSettings | None = None,
) -> dict[str, Any]:
    """Inject Alembic MCP and reopen the task on post_build_route.

    A post_build_route of fedot_mas while FEDOT.MAS is off reopens on
    react_tools instead: the same bound server, called by ExperimentAgent.
    """
    outputs = outputs if isinstance(outputs, dict) else {}
    task = ExperimentTask.model_validate(task_runtime["task"])
    if task.route != ExecutionRoute.ALEMBIC_BUILD:
        raise ValueError("apply_alembic_success requires planned route alembic_build")
    if not task.post_build_route:
        raise ValueError("apply_alembic_success requires post_build_route on the task")
    if not (mcp_url or "").strip().startswith("http"):
        raise ValueError(f"invalid mcp_url for alembic success: {mcp_url!r}")

    from CoScientist.experiments.schemas.models import MCPServerRef

    tool_refs = _tool_refs_from_outputs(outputs, placeholder=False)
    if not tool_refs:
        from CoScientist.tools.alembic_tools import list_served_mcp_tools

        listed = list_served_mcp_tools(mcp_url.strip())
        tool_refs = _tool_refs_from_outputs({"tools": listed}, placeholder=True)

    server_id = _server_id_from_url(task.repo_url, mcp_url)
    server = MCPServerRef.model_validate({
        "name": server_id,
        "server_id": server_id,
        "url": mcp_url.strip(),
        "tools": tool_refs,
        "source": "alembic",
        "health": "healthy",
    })
    post_route = task.post_build_route
    if post_route == ExecutionRoute.FEDOT_MAS.value:
        # Lazy: state_machine imports this module inside its functions.
        from CoScientist.experiments.runtime.state_machine import fedot_route_available

        if not fedot_route_available(settings):
            post_route = ExecutionRoute.REACT_TOOLS.value
    updated = task.model_copy(update={
        "route": ExecutionRoute(post_route),
        "mcp_servers": [server],
        "post_build_route": None,
        "repo_url": task.repo_url,
    })
    updated = ExperimentTask.model_validate(updated.model_dump(mode="json"))

    task_runtime["task"] = updated.model_dump(mode="json")
    task_runtime["current_route"] = post_route
    task_runtime["route_history"].append(
        {"route": post_route, "reason": "alembic_post_build", "mcp_url": mcp_url.strip()}
    )
    task_runtime["status"] = "ready"
    task_runtime["last_message"] = (
        f"Alembic MCP ready at {mcp_url.strip()}; continuing via {post_route}."
    )
    audit(logger, f"EXPERIMENT_ALEMBIC_READY mcp_url={mcp_url.strip()} route={post_route}")

    plan = runtime.get("plan")
    if isinstance(plan, dict):
        tasks = plan.get("tasks") or []
        for idx, item in enumerate(tasks):
            if isinstance(item, dict) and item.get("id") == updated.id:
                tasks[idx] = copy.deepcopy(task_runtime["task"])
                break
        plan["tasks"] = tasks

    server_json = server.model_dump(mode="json")
    deployed = list(state.get("deployed_mcps") or [])
    deployed.append(copy.deepcopy(server_json))
    state["deployed_mcps"] = deployed

    return {
        "post_build_pending": True,
        "post_build_route": post_route,
        "mcp_url": mcp_url.strip(),
        "server_id": server.server_id or server.name,
    }


__all__ = [
    "alembic_post_build_context",
    "apply_alembic_success",
    "compose_alembic_fedot_task",
    "extract_mcp_url",
    "harvest_alembic_mcp_url",
    "mcp_url_from_task_runtime",
    "pin_alembic_post_build_request",
    "stamp_alembic_science_description",
]
