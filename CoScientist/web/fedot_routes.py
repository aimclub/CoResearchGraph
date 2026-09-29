"""Read-only FEDOT UI APIs. A session and (for detail/stream) run are mandatory."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from CoScientist.tools import fedot_runs


def local_trace(meta: dict, events: list[dict]) -> dict:
    """Render the captured ADK journal without querying somebody else's trace."""
    observations = []
    agents, models, tools = {}, {}, {}

    def iso(ts):
        return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else None

    def text(value):
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)

    root = {"id": meta["run_id"], "name": "FEDOT · " + meta.get("engine", ""),
            "parent_id": None, "type": "RUN", "start_time": iso(meta["started_at"]),
            "end_time": iso(meta.get("ended_at")), "input": meta.get("task"),
            "status_message": meta.get("error") or meta["status"]}
    observations.append(root)
    for ev in events:
        kind, agent = ev["type"], ev.get("agent", "")
        if kind in {"config", "run_start", "tokens"}:
            continue
        if kind == "run_end":
            root["output"] = text(ev.get("state") or ev.get("result") or "")
            root.update(end_time=iso(ev["ts"]), status_message=ev.get("error") or ev.get("status"))
            continue
        key = (ev.get("invocation_id"), ev.get("branch"), agent)
        parent = agents.get(ev.get("agent_span_id")) or agents.get(key, root)
        if kind == "agent_start":
            parent = agents.get(ev.get("parent_span_id"), root)
        node = {"id": f'{meta["run_id"]}-{ev["seq"]}', "parent_id": parent["id"],
                "type": kind.upper(), "name": agent or kind, "start_time": iso(ev["ts"])}
        if kind == "agent_start":
            node.update(type="AGENT", input=text({"instruction": ev.get("instruction"), "state": ev.get("incoming")}))
            agents[key] = node
            if ev.get("span_id"):
                agents[ev["span_id"]] = node
        elif kind == "agent_done":
            target = agents.get(ev.get("span_id")) or agents.get(key)
            if target:
                target.update(end_time=iso(ev["ts"]), output=text(ev.get("output")))
                continue
        elif kind == "model_start":
            node.update(type="GENERATION", input=text(ev.get("request")))
            models.setdefault(key, []).append(node)
        elif kind in {"model_end", "model_error"}:
            stack = models.get(key, [])
            if stack:
                target = stack.pop(0)
                target.update(end_time=iso(ev["ts"]), output=text(ev.get("response")),
                              status_message=ev.get("error"), level="ERROR" if ev.get("error") else "DEFAULT")
                continue
        elif kind == "tool":
            node.update(type="TOOL", name=ev.get("tool"), input=text(ev.get("args")))
            tools[(key, ev.get("call_id") or ev.get("tool"))] = node
        elif kind == "tool_result":
            target = tools.pop((key, ev.get("call_id") or ev.get("tool")), None)
            if target:
                target.update(end_time=iso(ev["ts"]), output=text(ev.get("text")),
                              level="ERROR" if ev.get("error") else "DEFAULT")
                continue
        else:
            node["output"] = text(ev.get("text") or ev)
        observations.append(node)
    return {"status": "ok", "source": "local", "trace_id": meta["run_id"],
            "name": root["name"], "timestamp": root["start_time"],
            "observations": observations, "run": meta}


def register_fedot_routes(app, runtime, web_dir, static_class):
    def scope(user_id, session_id):
        if not user_id or not session_id:
            raise HTTPException(400, "user_id and session_id are required; global FEDOT history is not exposed")
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return user_id, session_id

    def require_run(key, run_id):
        try:
            meta = fedot_runs.get_run(key, run_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if meta is None:
            raise HTTPException(404, "FEDOT run not found in this session")
        return meta

    @app.get("/fedot-demo")
    async def demo_redirect(request: Request):
        return RedirectResponse("/fedot-demo/" + ("?" + request.url.query if request.url.query else ""))

    app.mount("/fedot-demo", static_class(directory=str(web_dir / "static" / "fedot_demo"), html=True))

    @app.get("/api/users/{user_id}/sessions/{session_id}/fedot/runs")
    async def runs(user_id: str, session_id: str):
        return JSONResponse({"runs": await asyncio.to_thread(fedot_runs.list_runs, scope(user_id, session_id))},
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/users/{user_id}/sessions/{session_id}/fedot/runs/{run_id}")
    async def run(user_id: str, session_id: str, run_id: str):
        key = scope(user_id, session_id)
        meta = require_run(key, run_id)
        events, _ = await asyncio.to_thread(fedot_runs.read_events, key, run_id)
        return JSONResponse({"run": meta, "events": events}, headers={"Cache-Control": "no-store"})

    @app.get("/api/fedot-langfuse-trace")
    async def trace(user_id: str = "", session_id: str = "", run_id: str = ""):
        # Compatibility URL, now strictly run-scoped and available offline.
        key = scope(user_id, session_id)
        meta = require_run(key, run_id)
        events, _ = await asyncio.to_thread(fedot_runs.read_events, key, run_id)
        return JSONResponse(local_trace(meta, events), headers={"Cache-Control": "no-store"})

    @app.get("/api/fedot-live-stream")
    async def stream(request: Request, user_id: str = "", session_id: str = "", run_id: str = ""):
        key = scope(user_id, session_id)
        meta = require_run(key, run_id)
        try:
            after = max(0, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(400, "Invalid Last-Event-ID") from exc

        async def events():
            offset = 0
            while not await request.is_disconnected():
                batch, offset = await asyncio.to_thread(fedot_runs.read_events, key, run_id, offset)
                for event in batch:
                    if event["seq"] > after:
                        yield f'id: {event["seq"]}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n'
                if any(e["type"] == "run_end" for e in batch) or meta.get("imported") or meta["status"] != "running":
                    yield 'event: complete\ndata: {}\n\n'
                    return
                yield ": ping\n\n"
                await asyncio.sleep(1)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})
