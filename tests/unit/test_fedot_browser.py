"""Optional real-browser regression; set COSCIENTIST_TEST_BROWSER to Chromium/Edge.

No model/provider requests. A small local app serves the real viewer and run API.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest


@pytest.mark.skipif(not os.getenv("COSCIENTIST_TEST_BROWSER"), reason="optional headless browser smoke")
def test_real_viewer_switches_runs_without_stale_graphs_or_answers(tmp_path):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from websockets.sync.client import connect
    from CoScientist.tools.fedot_live import FedotLiveBroadcaster
    from CoScientist.tools import fedot_runs
    from CoScientist.web.fedot_routes import register_fedot_routes

    web = Path(__file__).resolve().parents[2] / "CoScientist" / "web"
    app = FastAPI()
    def require_session(user, session):
        if (user, session) not in {("u", "s"), ("u", "other"), ("u", "imported")}:
            raise KeyError(session)
    register_fedot_routes(app, SimpleNamespace(registry=SimpleNamespace(require_session=require_session)), web, StaticFiles)
    app.mount("/static", StaticFiles(directory=web / "static"))
    @app.get("/fedot-trace")
    def trace_page():
        return HTMLResponse((web / "templates" / "fedot_trace.html").read_text(encoding="utf-8"))

    bus = FedotLiveBroadcaster()
    first = bus.begin_run(("u", "s"), task="first task", engine="mas")
    first.publish_config({"coordinator": {"name": "coordinator", "instruction": "route", "tools": []},
                          "workers": [{"name": "worker", "instruction": "work", "tools": [], "output_key": "answer"}]})
    first.event({"type": "agent_start", "agent": "worker", "instruction": "work", "span_id": "span"})
    first.event({"type": "agent_done", "agent": "worker", "output": "FIRST ANSWER", "output_key": "answer", "span_id": "span"})
    first.event({"type": "run_end", "status": "success", "state": {"answer": "FIRST ANSWER"}})
    failed = bus.begin_run(("u", "s"), task="failed before config", engine="mas")
    failed.event({"type": "run_end", "status": "error", "error": "CONFIG FAILED"})
    foreign = bus.begin_run(("u", "other"), task="FOREIGN SECRET", engine="mas")
    foreign.event({"type": "run_end", "status": "success", "state": {"answer": "FOREIGN SECRET"}})
    fedot_runs.restore(("u", "imported"), fedot_runs.snapshot(("u", "s")))

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    profile = tmp_path / "edge"
    browser = subprocess.Popen(
        [os.environ["COSCIENTIST_TEST_BROWSER"], "--headless=new", "--disable-gpu", "--no-first-run",
         "--disable-extensions", "--remote-debugging-port=0", f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ws = None
    try:
        deadline = time.monotonic() + 20
        while not (profile / "DevToolsActivePort").exists():
            assert browser.poll() is None, "Browser exited before opening DevTools"
            assert time.monotonic() < deadline, "Browser startup timed out"
            time.sleep(0.1)
        dev_port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        request = Request(f"http://127.0.0.1:{dev_port}/json/new?about:blank", method="PUT")
        with urlopen(request, timeout=5) as response:
            target = json.load(response)
        ws = connect(target["webSocketDebuggerUrl"], open_timeout=5)
        serial, errors = 0, []
        def call(method, params=None):
            nonlocal serial
            serial += 1
            ws.send(json.dumps({"id": serial, "method": method, "params": params or {}}))
            while True:
                reply = json.loads(ws.recv(timeout=10))
                if reply.get("method") == "Runtime.exceptionThrown":
                    details = reply["params"]["exceptionDetails"]
                    errors.append(details.get("exception", {}).get("description") or details)
                if reply.get("id") == serial:
                    assert "error" not in reply, reply
                    return reply.get("result", {})
        def evaluate(expression):
            result = call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            assert "exceptionDetails" not in result, result
            return result["result"].get("value")
        def wait_for(expression):
            end = time.monotonic() + 10
            while not evaluate(expression):
                assert not errors, errors
                assert time.monotonic() < end, expression
                time.sleep(0.1)
        def navigate(path):
            call("Page.navigate", {"url": f"http://127.0.0.1:{port}" + path})
        call("Runtime.enable")
        call("Page.enable")
        navigate(f"/fedot-demo/?user_id=u&session_id=s&run_id={first.run_id}")
        wait_for("document.querySelector('#answer')?.textContent.includes('FIRST ANSWER')")
        assert evaluate("document.querySelectorAll('#graph .node').length") == 2
        assert evaluate("document.querySelector('#fedot-runs select').options.length") == 2
        assert "FOREIGN SECRET" not in evaluate("document.body.textContent")
        evaluate(f"document.querySelector('#fedot-runs select').value='{failed.run_id}'; document.querySelector('#fedot-runs select').dispatchEvent(new Event('change'))")
        wait_for("document.querySelector('#feed')?.textContent.includes('CONFIG FAILED')")
        assert evaluate("document.querySelectorAll('#graph .node').length") == 0
        assert "FIRST ANSWER" not in evaluate("document.querySelector('#answer').textContent")
        assert failed.run_id in evaluate("document.querySelector('#fedot-runs a').href")
        navigate(f"/fedot-trace?user_id=u&session_id=s&run_id={first.run_id}")
        wait_for("document.querySelector('#content')?.textContent.includes('FIRST ANSWER')")
        assert evaluate("document.querySelector('#fedot-runs select').options.length") == 2
        assert "FOREIGN SECRET" not in evaluate("document.body.textContent")
        navigate(f"/fedot-demo/?user_id=u&session_id=s&run_id={foreign.run_id}")
        wait_for("document.querySelector('#fedot-runs')?.textContent.includes('Указанного запуска нет')")
        assert evaluate("document.querySelectorAll('#graph .node').length") == 0
        assert "FOREIGN SECRET" not in evaluate("document.body.textContent")
        navigate(f"/fedot-demo/?user_id=u&session_id=imported&run_id={first.run_id}")
        wait_for("document.querySelector('#answer')?.textContent.includes('FIRST ANSWER')")
        assert evaluate("document.querySelector('#fedot-runs select').options.length") == 2
        assert "session_id=imported" in evaluate("document.querySelector('#fedot-runs a').href")
        navigate("/fedot-demo/?user_id=u&session_id=s")
        wait_for("document.querySelector('#feed')?.textContent.includes('CONFIG FAILED')")
        third = bus.begin_run(("u", "s"), task="MAW live", engine="maw")
        third.publish_config({"agents": [{"name": "writer", "instruction": "write", "tools": []}],
                              "pipeline": {"type": "agent", "agent_name": "writer"}})
        third.event({"type": "run_end", "status": "success", "state": {"answer": "THIRD ANSWER"}})
        wait_for("document.querySelector('#answer')?.textContent.includes('THIRD ANSWER')")
        assert evaluate("document.querySelector('#fedot-runs select').options.length") == 3
        assert evaluate("document.querySelectorAll('#graph .node').length") == 1
        assert not errors, errors
    finally:
        if ws:
            ws.close()
        browser.terminate()
        browser.wait(timeout=10)
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()
