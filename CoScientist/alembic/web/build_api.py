"""FastAPI router for the MCP-build dashboard.

Serves per-job artifact reads and a live WebSocket per build. The socket tails
the build log, forwards structured ``ALEMBIC_EVENT`` lines, and answers the
page's requests: invoke a generated tool, stop the build. The alembic app
(``CoScientist.alembic.web.app``) includes this router; the main CoScientist web
UI mounts that whole app under ``/alembic``.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from CoScientist.alembic.web import artifacts
from CoScientist.tools import alembic_tools


router = APIRouter()


def _container_running(name: Optional[str]) -> bool:
    return bool(name) and alembic_tools._container_state(name)["running"]


def _controls_enabled() -> bool:
    """The start/stop/delete buttons. ALEMBIC_WEB_CONTROLS=0 turns them off on a
    shared deploy, where anyone who reaches the page could press them."""
    return os.getenv("ALEMBIC_WEB_CONTROLS", "1").strip().lower() not in ("0", "false", "no", "off")


def _annotate_container_status(builds: list) -> list:
    """Add the live docker state to each build snapshot.

    ``container_active`` is True/False, or None when docker is unreachable or
    the build has no container. ``runnable`` means the build's own image is on
    this host. ``controls`` tells the page whether to draw the buttons.
    """
    docker_ok = bool(shutil.which("docker"))
    inv = (alembic_tools.docker_inventory() if docker_ok
           else {"images": {}, "tags": {}, "containers": {}})
    controls = _controls_enabled()
    images: dict = {}
    for b in builds:
        name = b.get("container")
        state = inv["containers"].get(name or "")
        b["container_active"] = bool(state and state["running"]) if (name and docker_ok) else None
        b["container_exists"] = bool(state)
        image = alembic_tools.job_image(b, inv) if b.get("status") == "done" else None
        b["runnable"] = bool(image)
        b["image_size"] = inv["images"].get(image) if image else None
        b["controls"] = controls
        if image:
            images[b["job_id"]] = image
    # Older builds without a job tag can share one image, and deleting it takes
    # it from all of them; the page warns before that.
    for b in builds:
        image = images.get(b.get("job_id"))
        b["image_shared_with"] = [job for job, other in images.items()
                                  if image and other == image and job != b["job_id"]]
    return builds


# ── list + snapshot ────────────────────────────────────────────────────────
def _list_builds() -> list:
    return _annotate_container_status(alembic_tools.web_list_builds())


# docker ps can take seconds, so it runs off the event loop the chat shares.
@router.get("/api/builds")
async def api_builds():
    builds = await asyncio.to_thread(_list_builds)
    for b in builds:
        state = _action_state.get(b.get("job_id"))
        if state:
            b["action"] = dict(state)
    cleanup = _action_state.get(_CLEANUP)
    return JSONResponse({"builds": builds, "cleanup": dict(cleanup) if cleanup else None})


# Clearing non-runnable builds runs in the background as well: it can start a
# container to remove root-owned artifacts, which changes the network (see below).
_CLEANUP = "cleanup"  # key in _action_state; job ids always look like <repo>-<hex>


@router.post("/api/builds/cleanup")
async def api_clear_non_runnable():
    _require_controls()
    busy = _busy(_CLEANUP)
    if busy:
        return busy
    _start_background(_CLEANUP, "cleanup", alembic_tools.clear_non_runnable_builds)
    return JSONResponse({"ok": True, "accepted": True}, status_code=202)


@router.post("/api/builds")
async def api_start_build(payload: dict):
    """Kick off a fresh alembic pipeline for ``repo_url``. Delegates to the
    same daemon-subprocess launcher the agent flow uses, so the build is fully
    detached from this HTTP request: closing the browser, refreshing, or
    navigating elsewhere never touches the running pipeline."""
    repo_url = (payload.get("repo_url") or "").strip()
    force = bool(payload.get("force_rebuild"))
    if not repo_url:
        raise HTTPException(status_code=400, detail="repo_url is required")
    return JSONResponse(
        await alembic_tools.build_mcp_server(repo_url, force_rebuild=force)
    )


# Server controls of the builds list. An action runs in the background and the
# POST answers at once: while docker starts or stops a container it changes the
# host's network interfaces, and Firefox then drops every request that has not
# answered within 5 s (network.http.network-changed.timeout). A ~9 s restart
# was dropped that way every time. /api/builds reports progress and outcome.
_ACTIONS = {"start": "start_build_server", "restart": "restart_build_server",
            "stop": "stop_build_server", "delete-image": "delete_build_image"}
_action_state: dict = {}  # job id or _CLEANUP -> {name, running, started_at, finished_at, result}
_action_tasks: set = set()


def _require_controls() -> None:
    if not _controls_enabled():
        raise HTTPException(status_code=403,
                            detail="build controls are disabled (ALEMBIC_WEB_CONTROLS=0)")


def _busy(key: str) -> Optional[JSONResponse]:
    """A 409 while the action tracked under ``key`` is still running."""
    current = _action_state.get(key)
    if current and current["running"]:
        return JSONResponse({"ok": False, "error": f"{current['name']} is still running"},
                            status_code=409)
    return None


def _start_background(key: str, name: str, fn, *args, **kwargs) -> None:
    """Run ``fn`` in a worker thread; ``_action_state[key]`` tracks it for /api/builds."""
    loop = asyncio.get_running_loop()
    state = _action_state[key] = {"name": name, "running": True, "started_at": loop.time(),
                                  "finished_at": None, "result": None}

    async def run() -> None:
        try:
            result = await asyncio.to_thread(fn, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001  shown on the page, never raised
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        state.update(running=False, finished_at=loop.time(), result=result)

    task = asyncio.create_task(run())
    _action_tasks.add(task)  # a task nothing references can be garbage-collected mid-run
    task.add_done_callback(_action_tasks.discard)


@router.post("/api/builds/{job_id}/{action}")
async def api_build_action(job_id: str, action: str, include_shared: bool = False):
    _require_controls()
    if action not in _ACTIONS:
        raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
    if alembic_tools.web_build_snapshot(job_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown build {job_id!r}")
    busy = _busy(job_id)
    if busy:
        return busy
    options = {"include_shared": True} if action == "delete-image" and include_shared else {}
    _start_background(job_id, action, getattr(alembic_tools, _ACTIONS[action]), job_id, **options)
    return JSONResponse({"ok": True, "accepted": True, "action": action}, status_code=202)


def _resolve(job_id: str) -> tuple[dict, Path, str]:
    """(snapshot, workdir, repo_url) for a job, else raise 404. Also 404 when
    the job exists but has no persisted workdir (legacy build)."""
    snap = alembic_tools.web_build_snapshot(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown build {job_id!r}")
    workdir = alembic_tools.web_build_workdir(job_id)
    repo_url = snap.get("repo_url") or alembic_tools.web_build_repo_url(job_id)
    if workdir is None or not repo_url:
        raise HTTPException(status_code=404,
                            detail="build has no persisted artifacts (legacy build)")
    return snap, workdir, repo_url


@router.get("/api/builds/{job_id}")
async def api_build(job_id: str):
    snap = alembic_tools.web_build_snapshot(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown build {job_id!r}")
    await asyncio.to_thread(_annotate_container_status, [snap])
    return JSONResponse(snap)


# ── per-job artifact reads ─────────────────────────────────────────────────
@router.get("/api/builds/{job_id}/report")
async def api_report(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    r = artifacts.build_report(workdir, repo_url)
    return JSONResponse(r or {})


@router.get("/api/builds/{job_id}/tools")
async def api_tools(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    return JSONResponse(artifacts.build_tools(workdir, repo_url))


@router.get("/api/builds/{job_id}/examples")
async def api_examples(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    return JSONResponse(artifacts.build_examples(workdir, repo_url))


@router.get("/api/builds/{job_id}/files")
async def api_files(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    return JSONResponse({"files": artifacts.build_files(workdir, repo_url)})


@router.get("/api/builds/{job_id}/setup", response_class=PlainTextResponse)
async def api_setup(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    return PlainTextResponse(artifacts.build_setup(workdir, repo_url) or "")


@router.get("/api/builds/{job_id}/checks")
async def api_checks(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    return JSONResponse({"checks": artifacts.build_checks(workdir, repo_url)})


@router.get("/builds/{job_id}/artifacts.zip")
async def api_bundle(job_id: str):
    _, workdir, repo_url = _resolve(job_id)
    data = await asyncio.to_thread(artifacts.bundle_zip, workdir, repo_url)
    if data is None:
        raise HTTPException(status_code=404, detail="no built server for this build")
    name = artifacts._repo_name(repo_url)
    return StreamingResponse(
        io.BytesIO(data), media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="{name}-mcp-bundle.zip"'})


# ── live WebSocket: tail log + hydrate derived events ──────────────────────
async def _hydrate(ws: WebSocket, workdir: Path, repo_url: str) -> None:
    """Push the full set of derived panel events from on-disk state (safe to
    call at any time — the frontend treats each as an idempotent replace)."""
    report = artifacts.build_report(workdir, repo_url)
    if report:
        await ws.send_json({"type": "report", "report": "exploration", **report})
    await ws.send_json({"type": "server", **artifacts.build_tools(workdir, repo_url)})
    await ws.send_json({"type": "examples", **artifacts.build_examples(workdir, repo_url)})
    setup = artifacts.build_setup(workdir, repo_url)
    if setup is not None:
        await ws.send_json({"type": "setup", "content": setup})
    await ws.send_json({"type": "files", "files": artifacts.build_files(workdir, repo_url)})
    for chk in artifacts.build_checks(workdir, repo_url):
        await ws.send_json({"type": "check", **chk})


# ── live tool calls ─────────────────────────────────────────────────────────
# alembic.config.INVOKE_TIMEOUT (120 s) plus time to start a container.
_INVOKE_TIMEOUT = 180
_INVOKE_MARK = "ALEMBIC_INVOKE "
# Runs inside the build's image. docker commit blanks every .env key in the
# image (ENV KEY=), and alembic.config parses some of them as numbers, so the
# empty ones are dropped before alembic is imported.
_INVOKE_SCRIPT = (
    "import json, os, sys\n"
    "for k in [k for k, v in os.environ.items() if v == '']:\n"
    "    del os.environ[k]\n"
    "from alembic.tools.paths import set_current_repo\n"
    "from alembic.tools.invoke import _invoke_tool_function_sync\n"
    "set_current_repo(sys.argv[1])\n"
    "res = _invoke_tool_function_sync(sys.argv[2], json.loads(sys.argv[3]))\n"
    f"print({_INVOKE_MARK!r} + json.dumps(res, default=str), flush=True)\n"
)


def _invoke_in_container(snap: dict, tool: str, args: dict) -> dict:
    """Run a generated tool the way the validator does, inside the build's image.

    The tools venv was created in the container and points at container paths,
    so it cannot run on the host. The build's serve container is reused while it
    runs; otherwise a throwaway container starts from ``alembic-tool:<repo>``,
    which is the latest build of that repository.
    """
    repo_url = snap.get("repo_url")
    if not repo_url:
        return {"ok": False, "error": "this build has no repository URL"}
    opts = ["-e", "PYTHONPATH=/app", "-w", "/work"]
    argv = ["-c", _INVOKE_SCRIPT, repo_url, tool, json.dumps(args)]
    container = snap.get("container")
    if _container_running(container):
        cmd = ["docker", "exec", *opts, container, "python", *argv]
    else:
        image = f"alembic-tool:{alembic_tools._repo_name(repo_url)}"
        cmd = ["docker", "run", "--rm", *opts, "--entrypoint", "python", image, *argv]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=_INVOKE_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"no answer within {_INVOKE_TIMEOUT}s"}
    except OSError as exc:
        return {"ok": False, "error": f"docker is not available: {exc}"}
    for line in reversed(out.stdout.splitlines()):
        if line.startswith(_INVOKE_MARK):
            return json.loads(line[len(_INVOKE_MARK):])
    return {"ok": False, "error": "the tool runner returned no result",
            "stderr": (out.stderr or out.stdout)[-2000:]}


async def _with_mcp(url: str, fn):
    """Open an MCP session to ``url``, run ``fn(session)`` and return its result."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with asyncio.timeout(_INVOKE_TIMEOUT):
        async with streamablehttp_client(url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await fn(session)


def _mcp_unreachable(snap: dict) -> Optional[str]:
    """Why the build's MCP server cannot answer, or None when it may."""
    if not snap.get("mcp_url"):
        return "this build has no MCP address"
    container = snap.get("container")
    if container and not _container_running(container):
        return (f"the MCP server is not running (container {container} is stopped); "
                "Run in container still runs the tool code")
    return None


def _from_mcp_result(res) -> dict:
    """An MCP CallToolResult in the shape the page renders."""
    text = "\n".join(c.text for c in res.content if getattr(c, "type", None) == "text")
    if res.isError:
        return {"ok": False, "error": text or "the MCP tool reported an error"}
    if res.structuredContent is not None:
        return {"ok": True, "result": res.structuredContent}
    try:
        return {"ok": True, "result": json.loads(text)}
    except ValueError:
        return {"ok": True, "result": text}


async def _invoke_via_mcp(snap: dict, tool: str, args: dict) -> dict:
    """Call the tool on the build's served MCP endpoint, the way any agent would."""
    why = await asyncio.to_thread(_mcp_unreachable, snap)
    if why:
        return {"ok": False, "error": why}
    url = snap["mcp_url"]
    try:
        res = await _with_mcp(url, lambda s: s.call_tool(tool, args))
    except TimeoutError:
        return {"ok": False, "error": f"{url} gave no answer within {_INVOKE_TIMEOUT}s"}
    except Exception as exc:  # noqa: BLE001  refused connection, protocol error, ...
        return {"ok": False, "error": f"MCP call to {url} failed: {type(exc).__name__}: {exc}"}
    return _from_mcp_result(res)


@router.get("/api/builds/{job_id}/mcp_tools")
async def api_mcp_tools(job_id: str):
    """The tools as the served MCP server lists them, with their input schemas."""
    snap = alembic_tools.web_build_snapshot(job_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"unknown build {job_id!r}")
    why = await asyncio.to_thread(_mcp_unreachable, snap)
    if why:
        return JSONResponse({"tools": {}, "error": why})
    try:
        listed = await _with_mcp(snap["mcp_url"], lambda s: s.list_tools())
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"tools": {}, "error": f"{type(exc).__name__}: {exc}"})
    return JSONResponse({"tools": {t.name: t.inputSchema for t in listed.tools}})


# The generated MCP server adds these to every tool for the S3 key scope.
_SCOPE_ARGS = ("user_id", "session_id")


def _container_args(job_id: str, repo_url: str, tool: str, args: dict) -> dict:
    """``args`` for the tool function itself. The call form is filled from the
    MCP schema, where every tool also takes the S3 scope params, and the
    function accepts them only when it declares them. When its code cannot be
    read, a scope param is dropped if it is empty, as the form leaves it."""
    workdir = alembic_tools.web_build_workdir(job_id)
    declared = artifacts.declared_params(workdir, repo_url, tool) if workdir else None
    return {k: v for k, v in args.items()
            if k not in _SCOPE_ARGS or (k in declared if declared is not None else v != "")}


async def _answer_invoke(ws: WebSocket, job_id: str, msg: dict) -> None:
    tool, call_id = msg.get("tool"), msg.get("call_id")
    args = msg.get("args") or {}
    # MCP is the real interface; "container" is the debug path around the server.
    via = "container" if msg.get("mode") == "container" else "mcp"
    snap = alembic_tools.web_build_snapshot(job_id) or {}
    if snap.get("status") != "done":
        res = {"ok": False, "error": "tools can be called once the build is done"}
    elif not tool or not isinstance(args, dict):
        res = {"ok": False, "error": "a tool name and a JSON object of args are required"}
    elif via == "container":
        args = _container_args(job_id, snap.get("repo_url") or "", tool, args)
        res = await asyncio.to_thread(_invoke_in_container, snap, tool, args)
    else:
        res = await _invoke_via_mcp(snap, tool, args)
    ok = bool(res.get("ok"))
    await ws.send_json({
        "type": "invoke_result", "call_id": call_id, "tool": tool, "ok": ok, "via": via,
        "output": res.get("result") if ok else None,
        "reason": res.get("reason"),
        "error": None if ok else (res.get("error") or res.get("stderr") or "call failed"),
        "traceback": res.get("traceback"),
    })


async def _serve_client(ws: WebSocket, job_id: str) -> None:
    """Answer the page's requests (invoke a tool, stop the build) until it leaves."""
    calls: set = set()
    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")
            if kind == "invoke":
                task = asyncio.create_task(_answer_invoke(ws, job_id, msg))
                calls.add(task)
                task.add_done_callback(calls.discard)
            elif kind == "stop":
                res = await asyncio.to_thread(alembic_tools.cancel_build, job_id)
                if res.get("ok"):
                    await ws.send_json({"type": "pipeline", "status": "cancelled"})
                else:
                    await ws.send_json({"type": "error", "message": res.get("error")})
    finally:
        for task in calls:
            task.cancel()


async def _tail(ws: WebSocket, job_id: str, log_file: Path,
                workdir: Optional[Path], repo_url: Optional[str],
                client: asyncio.Task) -> None:
    """Forward the build log until the build ends, then send its final status."""
    pos = 0
    last_hydrate = 0.0
    while not client.done():
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if len(text) > pos:
            chunk = text[pos:]
            pos = len(text)
            if not chunk.endswith("\n"):
                last_nl = chunk.rfind("\n")
                if last_nl != -1:
                    pos -= len(chunk) - last_nl - 1
                    chunk = chunk[:last_nl + 1]
                else:
                    pos -= len(chunk)
                    chunk = ""
            stage_progressed = False
            for line in chunk.splitlines():
                ev = alembic_tools.parse_event_line(line)
                if ev is not None:
                    # Builds from before the pipeline fix report the plan's
                    # sample args, including keys the tool does not accept.
                    if ev.get("type") == "validation" and workdir is not None and repo_url:
                        ev["input"] = artifacts.call_args_for(
                            workdir, repo_url, ev.get("tool") or "", ev.get("input") or {})
                    await ws.send_json(ev)
                    if ev.get("type") == "stage" and ev.get("status") in ("done", "failed"):
                        stage_progressed = True
                elif line.strip():
                    await ws.send_json({"type": "log", "line": line})
            # Re-hydrate whenever a stage crosses a boundary: cheap disk reads.
            loop_now = asyncio.get_event_loop().time()
            if stage_progressed and workdir is not None and repo_url \
                    and loop_now - last_hydrate > 1.0:
                try:
                    await _hydrate(ws, workdir, repo_url)
                except Exception:  # noqa: BLE001
                    pass
                last_hydrate = loop_now

        snap = alembic_tools.web_build_snapshot(job_id)
        if snap and snap.get("status") in ("done", "failed"):
            if workdir is not None and repo_url:
                try:
                    await _hydrate(ws, workdir, repo_url)
                except Exception:  # noqa: BLE001
                    pass
            await ws.send_json({"type": "status", **snap})
            return

        await asyncio.sleep(0.6)


@router.websocket("/builds/ws/{job_id}")
async def build_ws(ws: WebSocket, job_id: str):
    await ws.accept()
    log_file = alembic_tools.web_build_log_file(job_id)
    if log_file is None:
        await ws.send_json({"type": "error", "message": f"unknown build {job_id}"})
        await ws.close()
        return

    snap = alembic_tools.web_build_snapshot(job_id) or {}
    workdir = alembic_tools.web_build_workdir(job_id)
    repo_url = snap.get("repo_url") or alembic_tools.web_build_repo_url(job_id)

    await ws.send_json({"type": "snapshot", **snap})
    if workdir is not None and repo_url:
        try:
            await _hydrate(ws, workdir, repo_url)
        except Exception as exc:  # noqa: BLE001 — never break the stream
            await ws.send_json({"type": "log",
                                "line": f"[hydrate] {type(exc).__name__}: {exc}"})

    client = asyncio.create_task(_serve_client(ws, job_id))
    try:
        # A finished build replays its log too, so its stage rail and feed fill in.
        await _tail(ws, job_id, log_file, workdir, repo_url, client)
        # The socket stays open after the build ends: the page calls tools over it.
        await client
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"[build_ws] error ({job_id}): {exc}")
    finally:
        client.cancel()


__all__ = ["router"]
