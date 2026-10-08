"""The build page's socket. The page calls generated tools and stops builds over
it, so the socket has to answer both and stay open after the build ends.
A call goes to the served MCP server; the debug path runs the tool code in the
build's container. Docker and the MCP transport are faked.
"""

import asyncio

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.types import CallToolResult, TextContent

from CoScientist.alembic.web import build_api
from CoScientist.tools import alembic_tools

_JOB = "gget-938c68"
_REPO = "https://github.com/pachterlab/gget"
_SERVE = "alembic-serve-gget-b29990"
_URL = "http://localhost:20162/mcp"


def _client(monkeypatch, tmp_path, status="done", log="", workdir=None):
    log_file = tmp_path / "build.log"
    log_file.write_text(log, encoding="utf-8")
    snap = {"job_id": _JOB, "repo_url": _REPO, "status": status,
            "container": _SERVE, "mcp_url": _URL}
    monkeypatch.setattr(alembic_tools, "web_build_log_file", lambda jid: log_file)
    monkeypatch.setattr(alembic_tools, "web_build_snapshot", lambda jid: dict(snap))
    monkeypatch.setattr(alembic_tools, "web_build_workdir", lambda jid: workdir)
    app = FastAPI()
    app.include_router(build_api.router)
    return TestClient(app)


def _next(ws, kind):
    while True:
        msg = ws.receive_json()
        if msg["type"] == kind:
            return msg


def _call(monkeypatch, tmp_path, result, mode=None):
    used = {}

    def in_container(snap, tool, args):
        used.update(via="container", tool=tool, args=args)
        return result

    async def via_mcp(snap, tool, args):
        used.update(via="mcp", tool=tool, args=args)
        return result

    monkeypatch.setattr(build_api, "_invoke_in_container", in_container)
    monkeypatch.setattr(build_api, "_invoke_via_mcp", via_mcp)
    msg = {"type": "invoke", "tool": "info", "args": {"q": "brca1"}, "call_id": 7}
    if mode:
        msg["mode"] = mode
    with _client(monkeypatch, tmp_path).websocket_connect(f"/builds/ws/{_JOB}") as ws:
        _next(ws, "status")  # the build is over and the socket is still open
        ws.send_json(msg)
        return used, _next(ws, "invoke_result")


def test_a_call_goes_to_the_served_mcp_server(monkeypatch, tmp_path):
    used, res = _call(monkeypatch, tmp_path, {"ok": True, "result": {"n": 3}})

    assert used == {"via": "mcp", "tool": "info", "args": {"q": "brca1"}}
    assert res["via"] == "mcp"
    assert res["call_id"] == 7
    assert res["ok"] is True
    assert res["output"] == {"n": 3}


def test_the_debug_button_runs_the_tool_in_the_container(monkeypatch, tmp_path):
    used, res = _call(monkeypatch, tmp_path, {"ok": True, "result": 1}, mode="container")

    assert used["via"] == "container"
    assert res["via"] == "container"


def test_a_crashing_tool_comes_back_with_its_traceback(monkeypatch, tmp_path):
    _, res = _call(monkeypatch, tmp_path,
                   {"ok": False, "error": "ValueError: bad smiles", "traceback": "tb"})

    assert res["ok"] is False
    assert res["error"] == "ValueError: bad smiles"
    assert res["traceback"] == "tb"


def test_a_running_build_refuses_calls_and_can_be_stopped(monkeypatch, tmp_path):
    stopped = []
    monkeypatch.setattr(alembic_tools, "cancel_build",
                        lambda jid: stopped.append(jid) or {"ok": True})

    client = _client(monkeypatch, tmp_path, status="running")
    with client.websocket_connect(f"/builds/ws/{_JOB}") as ws:
        _next(ws, "snapshot")
        ws.send_json({"type": "invoke", "tool": "info", "args": {}, "call_id": 1})
        refused = _next(ws, "invoke_result")
        ws.send_json({"type": "stop"})
        cancelled = _next(ws, "pipeline")

    assert refused["ok"] is False
    assert "once the build is done" in refused["error"]
    assert stopped == [_JOB]
    assert cancelled["status"] == "cancelled"


def test_replayed_validation_input_matches_the_generated_signature(monkeypatch, tmp_path):
    """Plan sample args describe the repo function. The coder wrapped it without
    ``mols``, and prefilling the Call form with it made every call fail."""
    tools = tmp_path / "work" / "gget" / "output" / "tools"
    tools.mkdir(parents=True)
    (tools / "find_prop.py").write_text("def find_prop(smi, steps=(1, 1)):\n    return {}\n")
    event = ('ALEMBIC_EVENT {"type": "validation", "tool": "find_prop", "passed": true, '
             '"input": {"smi": "CCO", "mols": "mols_from_chemical_space"}}\n')

    client = _client(monkeypatch, tmp_path, log=event, workdir=tmp_path / "work")
    with client.websocket_connect(f"/builds/ws/{_JOB}") as ws:
        validation = _next(ws, "validation")

    assert validation["input"] == {"smi": "CCO"}



def _debug_call(monkeypatch, tmp_path, args, workdir=None):
    used = {}
    monkeypatch.setattr(build_api, "_invoke_in_container",
                        lambda snap, tool, a: used.update(args=a) or {"ok": True, "result": 1})
    client = _client(monkeypatch, tmp_path, workdir=workdir)
    with client.websocket_connect(f"/builds/ws/{_JOB}") as ws:
        _next(ws, "status")
        ws.send_json({"type": "invoke", "tool": "info", "mode": "container",
                      "call_id": 1, "args": args})
        _next(ws, "invoke_result")
    return used["args"]


def test_the_debug_run_leaves_out_scope_params_the_function_does_not_take(monkeypatch, tmp_path):
    """The call form is filled from the MCP schema, where every tool also has
    user_id/session_id for the S3 scope. The function itself would reject them."""
    tools = tmp_path / "work" / "gget" / "output" / "tools"
    tools.mkdir(parents=True)
    (tools / "info.py").write_text("def info(q, user_id=None):\n    return {}\n")

    args = _debug_call(monkeypatch, tmp_path, {"q": "brca1", "user_id": "u", "session_id": ""},
                       workdir=tmp_path / "work")

    assert args == {"q": "brca1", "user_id": "u"}


def test_without_readable_code_the_debug_run_drops_only_empty_scope_params(monkeypatch, tmp_path):
    args = _debug_call(monkeypatch, tmp_path, {"q": "brca1", "user_id": "", "session_id": "s"})

    assert args == {"q": "brca1", "session_id": "s"}


# ── MCP path ─────────────────────────────────────────────────────────────────


def test_a_structured_mcp_result_is_the_output():
    res = CallToolResult(content=[TextContent(type="text", text='{"svg": "<svg/>"}')],
                         structuredContent={"svg": "<svg/>"})

    assert build_api._from_mcp_result(res) == {"ok": True, "result": {"svg": "<svg/>"}}


def test_an_mcp_tool_error_comes_back_as_an_error():
    res = CallToolResult(
        content=[TextContent(type="text", text="Error calling tool 'find_prop': TypeError")],
        isError=True)

    out = build_api._from_mcp_result(res)

    assert out["ok"] is False
    assert "TypeError" in out["error"]


def _invoke_mcp(monkeypatch, running, with_mcp):
    monkeypatch.setattr(build_api, "_container_running", lambda name: name in running)
    monkeypatch.setattr(build_api, "_with_mcp", with_mcp)
    return asyncio.run(build_api._invoke_via_mcp(
        {"mcp_url": _URL, "container": _SERVE}, "info", {}))


def test_a_stopped_server_is_reported_without_connecting(monkeypatch):
    async def must_not_connect(url, fn):
        raise AssertionError("connected to a stopped server")

    res = _invoke_mcp(monkeypatch, set(), must_not_connect)

    assert res["ok"] is False
    assert "not running" in res["error"]


def test_an_unreachable_server_is_an_error_and_not_a_crash(monkeypatch):
    async def refused(url, fn):
        raise ConnectionError("connection refused")

    res = _invoke_mcp(monkeypatch, {_SERVE}, refused)

    assert res["ok"] is False
    assert "connection refused" in res["error"]


# ── debug path: the command it would run ────────────────────────────────────


class _Done:
    def __init__(self, stdout, stderr=""):
        self.stdout, self.stderr = stdout, stderr


def _invoke(monkeypatch, running, stdout, stderr=""):
    ran = []
    monkeypatch.setattr(build_api, "_container_running", lambda name: name in running)
    monkeypatch.setattr(build_api.subprocess, "run",
                        lambda cmd, **kw: ran.append(cmd) or _Done(stdout, stderr))
    res = build_api._invoke_in_container(
        {"repo_url": _REPO, "container": _SERVE}, "info", {"q": "x"})
    return ran[0], res


def test_a_debug_run_uses_the_live_serve_container(monkeypatch):
    cmd, res = _invoke(monkeypatch, {_SERVE},
                       'loader noise\nALEMBIC_INVOKE {"ok": true, "result": 1}\n')

    assert cmd[:2] == ["docker", "exec"] and _SERVE in cmd
    assert res == {"ok": True, "result": 1}


def test_without_a_live_container_a_debug_run_starts_one_from_the_image(monkeypatch):
    cmd, _ = _invoke(monkeypatch, set(), 'ALEMBIC_INVOKE {"ok": true, "result": 1}\n')

    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "alembic-tool:gget" in cmd


def test_a_runner_that_prints_no_result_is_reported(monkeypatch):
    _, res = _invoke(monkeypatch, set(), "", stderr="ModuleNotFoundError: alembic")

    assert res["ok"] is False
    assert "no result" in res["error"]
    assert "ModuleNotFoundError" in res["stderr"]
