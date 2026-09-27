"""Optional Chromium regressions for the session MAS graph.

The page, JavaScript and CSS are real.  The first test serves a deterministic
graph fixture; the second uses the real session APIs.  Neither test runs an
agent, a language model or an MCP service.
"""

from __future__ import annotations

import base64
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
from urllib.request import Request, urlopen

import pytest


DESIGNS = ("atlas", "workshop")
VIEWPORTS = ((1440, 900), (1280, 720), (800, 900))


def _assert_flow_connections(evaluate):
    """Every arrow touches its named cards and avoids all card interiors."""
    defects = evaluate(r"""
      (() => {
        const cards = new Map([...document.querySelectorAll('g.agent-node')].map(g => {
          const shape = g.querySelector('.node-card');
          return [g.dataset.viewId, {x:+g.dataset.x, y:+g.dataset.y,
            w:+g.dataset.width || +shape?.getAttribute('width'),
            h:+g.dataset.height || +shape?.getAttribute('height'), shape:g.dataset.shape}];
        }));
        const border = (p, r) => r && (r.shape === 'circle'
          ? Math.abs(Math.hypot(p.x-r.x-r.w/2, p.y-r.y-r.h/2)-r.w/2) < .75
          : r.shape === 'hexagon'
            ? Math.abs(Math.max(Math.abs(p.y-r.y-r.h/2)/(r.h/2),
                Math.abs(p.x-r.x-r.w/2)/(r.w/2)+Math.abs(p.y-r.y-r.h/2)/r.h)-1) < .02
            : p.x >= r.x-.75 && p.x <= r.x+r.w+.75 && p.y >= r.y-.75 && p.y <= r.y+r.h+.75 &&
              Math.min(Math.abs(p.x-r.x), Math.abs(p.x-r.x-r.w),
                Math.abs(p.y-r.y), Math.abs(p.y-r.y-r.h)) < .75);
        const inside = (p, r) => r.shape === 'circle'
          ? Math.hypot(p.x-r.x-r.w/2, p.y-r.y-r.h/2) < r.w/2-1
          : r.shape === 'hexagon'
            ? Math.abs(p.y-r.y-r.h/2) < r.h/2-1 &&
              Math.abs(p.x-r.x-r.w/2)+Math.abs(p.y-r.y-r.h/2)*r.w/(2*r.h) < r.w/2-1
            : p.x > r.x+1 && p.x < r.x+r.w-1 && p.y > r.y+1 && p.y < r.y+r.h-1;
        return [...document.querySelectorAll('path.edge[marker-end]')].flatMap(edge => {
          const source=cards.get(edge.dataset.from), target=cards.get(edge.dataset.to);
          const length=edge.getTotalLength(), problems=[];
          if (!border(edge.getPointAtLength(0), source)) problems.push('detached start');
          if (!border(edge.getPointAtLength(length), target)) problems.push('detached end');
          for (let distance=3; distance < length-2; distance+=3) {
            if ([...cards.values()].some(card => inside(edge.getPointAtLength(distance), card))) {
              problems.push('crosses a card'); break;
            }
          }
          return problems.length ? [{from:edge.dataset.from,to:edge.dataset.to,problems}] : [];
        });
      })()
    """)
    assert not defects, defects


def _assert_no_node_overlaps(evaluate):
    assert evaluate(r"""
      (() => {
        const nodes=[...document.querySelectorAll('g.agent-node')].map(n =>
          ({x:+n.dataset.x,y:+n.dataset.y,w:+n.dataset.width,h:+n.dataset.height}));
        return nodes.every((a,i) => nodes.slice(i+1).every(b =>
          a.x+a.w <= b.x || b.x+b.w <= a.x || a.y+a.h <= b.y || b.y+b.h <= a.y));
      })()
    """)


def _screenshot(call, directory: Path | None, name: str):
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        data = call("Page.captureScreenshot", {"format": "png"})["data"]
        (directory / name).write_bytes(base64.b64decode(data))


def _real_click(call, evaluate, selector: str):
    """Click a visible control through CDP, panning the SVG to a remote card."""
    encoded = json.dumps(selector)
    for attempt in range(3):
        stable_deadline = time.monotonic() + 2
        while evaluate("document.querySelector('#network')?.getAttribute('aria-busy') === 'true'"):
            assert time.monotonic() < stable_deadline, "Graph layout did not settle before click"
            time.sleep(.05)
        call("Runtime.evaluate", {"awaitPromise":True,"expression":
             "new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))"})
        position = evaluate(f"""
          (() => {{
            const elements=[...document.querySelectorAll({encoded})];
            let element=elements.find(candidate => {{
              const r=candidate.getBoundingClientRect(), hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);
              return r.width>0&&r.height>0&&hit&&(hit===candidate||candidate.contains(hit));
            }}) || elements[0];
            if (!element) return null;
            if (!element.closest('.agent-node')) element.scrollIntoView({{block:'nearest',inline:'nearest'}});
            const r=element.getBoundingClientRect(), scene=document.querySelector('#network').getBoundingClientRect();
            const hit=document.elementFromPoint(r.left+r.width/2,r.top+r.height/2);
            return {{x:r.left+r.width/2,y:r.top+r.height/2,
              dx:r.left+r.width/2-scene.left-scene.width/2,dy:r.top+r.height/2-scene.top-scene.height/2,
              card:!!element.closest('.agent-node'),
              visible:r.width>0&&r.height>0&&r.left+r.width/2>scene.left&&r.left+r.width/2<scene.right&&
                r.top+r.height/2>scene.top&&r.top+r.height/2<scene.bottom&&hit&&(hit===element||element.contains(hit))}};
          }})()
        """)
        assert position, selector
        if position["card"] and not position["visible"]:
            scene = evaluate("(() => {const r=document.querySelector('#network').getBoundingClientRect();return {x:r.left+r.width/2,y:r.top+r.height/2}})()")
            call("Input.dispatchMouseEvent", {"type":"mouseWheel","x":scene["x"],"y":scene["y"],
                                               "deltaX":position["dx"],"deltaY":position["dy"]})
            time.sleep(.2)
            continue
        call("Input.dispatchMouseEvent", {"type":"mouseMoved","x":position["x"],"y":position["y"]})
        call("Input.dispatchMouseEvent", {"type":"mousePressed","x":position["x"],"y":position["y"],
                                           "button":"left","buttons":1,"clickCount":1})
        call("Input.dispatchMouseEvent", {"type":"mouseReleased","x":position["x"],"y":position["y"],
                                           "button":"left","buttons":0,"clickCount":1})
        if not position["card"]:
            return
        time.sleep(.15)
        if evaluate(f"[...document.querySelectorAll({encoded})].some(n=>n.classList.contains('selected'))"):
            return
    raise AssertionError(f"CDP click did not select {selector}")


@contextmanager
def _run_browser(tmp_path, app, query="user_id=u&session_id=s&design=workshop"):
    import uvicorn
    from websockets.sync.client import connect

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    ready_deadline = time.monotonic() + 12
    while True:
        try:
            with urlopen(f"http://127.0.0.1:{port}/agent-tree?{query}", timeout=1):
                break
        except OSError:
            assert time.monotonic() < ready_deadline, "Local browser fixture did not start"
            time.sleep(.1)
    profile = tmp_path / "chromium-profile"
    browser = subprocess.Popen(
        [os.environ["COSCIENTIST_TEST_BROWSER"], "--headless=new", "--disable-gpu",
         "--no-first-run", "--disable-extensions", "--remote-debugging-port=0",
         f"--user-data-dir={profile}", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ws = None
    try:
        deadline = time.monotonic() + 20
        while not (profile / "DevToolsActivePort").exists():
            assert browser.poll() is None, "Chromium exited before opening DevTools"
            assert time.monotonic() < deadline, "Chromium startup timed out"
            time.sleep(.1)
        dev_port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        with urlopen(Request(f"http://127.0.0.1:{dev_port}/json/new?about:blank", method="PUT"), timeout=5) as response:
            target = json.load(response)
        ws = connect(target["webSocketDebuggerUrl"], open_timeout=5, max_size=16 * 1024 * 1024)
        serial, errors = 0, []

        def call(method, params=None):
            nonlocal serial
            serial += 1
            current = serial
            ws.send(json.dumps({"id": current, "method": method, "params": params or {}}))
            while True:
                reply = json.loads(ws.recv(timeout=15))
                if reply.get("method") == "Runtime.exceptionThrown":
                    details = reply["params"]["exceptionDetails"]
                    errors.append(details.get("exception", {}).get("description") or details)
                if reply.get("id") == current:
                    assert "error" not in reply, reply
                    return reply.get("result", {})

        def evaluate(expression):
            result = call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            assert "exceptionDetails" not in result, result
            return result.get("result", {}).get("value")

        def wait_for(expression, timeout=20):
            end = time.monotonic() + timeout
            while not evaluate(expression):
                assert time.monotonic() < end, expression
                time.sleep(.1)

        call("Runtime.enable")
        call("Page.enable")
        call("Emulation.setDeviceMetricsOverride", {"width":1440,"height":900,"deviceScaleFactor":1,"mobile":False})
        call("Page.navigate", {"url":f"http://127.0.0.1:{port}/agent-tree?{query}"})
        wait_for("document.querySelector('svg[data-scale]') && document.querySelectorAll('g.agent-node').length")
        wait_for("document.querySelector('#network').getAttribute('aria-busy') === 'false'")
        yield call, evaluate, wait_for, errors, port
    finally:
        if ws:
            ws.close()
        browser.terminate()
        browser.wait(timeout=10)
        server.should_exit = True
        thread.join(timeout=10)
        sock.close()


@pytest.mark.skipif(not os.getenv("COSCIENTIST_TEST_BROWSER"), reason="optional headless browser smoke")
def test_agent_tree_two_designs_geometry_camera_and_tools(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from CoScientist.config import get_settings, settings_scope
    from CoScientist.web.agent_settings import agents_catalog
    from CoScientist.web.agent_tree import project_agent_tree

    monkeypatch.setenv("COSCIENTIST_CONFIG", "hypotheses")
    web = Path(__file__).resolve().parents[2] / "CoScientist" / "web"
    app = FastAPI()
    snapshot = get_settings().model_copy(deep=True)
    snapshot.web.start_mode = "planner"
    snapshot.context_init.enabled = True
    snapshot.orchestrator.use_planner = True
    snapshot.nir.enabled = False
    snapshot.mcp.normcontrol_url = None
    phases = [project_agent_tree(snapshot, desired_revision=1, active_revision=1, running=False),
              project_agent_tree(snapshot, desired_revision=1, active_revision=1, running=True)]
    nir_snapshot = snapshot.model_copy(deep=True)
    nir_snapshot.nir.enabled = True
    nir_snapshot.mcp.normcontrol_url = "http://normcontrol.invalid/mcp"
    phases.append(project_agent_tree(nir_snapshot, desired_revision=2, active_revision=1, running=True))
    state = {"phase": 0, "tool_calls": 0}

    @app.get("/agent-tree")
    def page():
        return HTMLResponse((web / "templates" / "agent_tree.html").read_text(encoding="utf-8"))

    @app.get("/api/users/{user_id}/sessions/{session_id}/agent-tree")
    def graph(user_id: str, session_id: str):
        assert (user_id, session_id) == ("u", "s")
        return JSONResponse(copy.deepcopy(phases[state["phase"]]), headers={"Cache-Control":"no-store"})

    @app.get("/api/users/{user_id}/sessions/{session_id}/agents/catalog")
    def catalog(user_id: str, session_id: str):
        current = nir_snapshot if state["phase"] == 2 else snapshot
        with settings_scope(current):
            return agents_catalog()

    @app.get("/api/users/{user_id}/sessions/{session_id}/agents/{agent_name}/tools")
    def tools(user_id: str, session_id: str, agent_name: str):
        assert (user_id, session_id, agent_name) == ("u", "s", "ExperimentAgent")
        state["tool_calls"] += 1
        if state["tool_calls"] == 1:
            time.sleep(1.1)
            technical_name = "late-tool"
        else:
            technical_name = "fresh-tool"
        return {"agent":agent_name,"dynamic":False,"selectionReady":True,"tools":[{
            "id":f"local:{technical_name}","name":technical_name,
            "display_name":{"ru":"FEDOT AutoML — обучить модель","en":"FEDOT AutoML — train model"},
            "summary":{"ru":"Автоматически подбирает и обучает модель машинного обучения.","en":"Trains a model."},
            "status":"available","pinned":True}]}

    @app.post("/__test__/phase")
    def set_phase(payload: dict):
        state["phase"] = int(payload.get("phase", 0))
        return state

    app.mount("/static", StaticFiles(directory=web / "static"), name="static")

    with _run_browser(tmp_path, app, "user_id=u&session_id=s&design=atlas") as (call, evaluate, wait_for, errors, port):
        screenshots = Path(os.environ["COSCIENTIST_TEST_SCREENSHOTS"]) if os.getenv("COSCIENTIST_TEST_SCREENSHOTS") else None
        assert evaluate("[...document.querySelectorAll('.design-options button')].map(b=>b.dataset.design)") == list(DESIGNS)
        assert evaluate("document.querySelector('#network').dataset.design") == "atlas"
        assert evaluate("Number(document.querySelector('#network').dataset.scale)") == pytest.approx(1)
        expected = sorted(item["id"] for item in phases[0]["workflow"]["instances"])
        expected.append(phases[0]["workflow"]["configuration"]["ownerInstanceId"] + "/configuration")
        assert evaluate("[...document.querySelectorAll('.agent-node')].map(n=>n.dataset.viewId).sort()") == sorted(expected)
        assert evaluate("document.querySelector('.agent-node[data-agent=__mas_session__] .node-title').textContent.includes('Конфигуратор')")
        assert evaluate("parseFloat(getComputedStyle(document.querySelector('.node-title')).fontSize) >= 14")
        _assert_no_node_overlaps(evaluate)
        _assert_flow_connections(evaluate)

        _real_click(call, evaluate, "#zoom-in")
        scale = evaluate("Number(document.querySelector('#network').dataset.scale)")
        camera = evaluate("document.querySelector('#network').getAttribute('viewBox')")
        evaluate("fetch('/__test__/phase',{method:'POST',headers:{'content-type':'application/json'},body:'{\"phase\":1}'})")
        wait_for("document.querySelector('#state')?.textContent.includes('Запрос выполняется')")
        assert evaluate("Number(document.querySelector('#network').dataset.scale)") == pytest.approx(scale)
        evaluate("fetch('/__test__/phase',{method:'POST',headers:{'content-type':'application/json'},body:'{\"phase\":2}'})")
        wait_for("!!document.querySelector('.agent-node[data-agent=NirReportAgent]')")
        assert evaluate("Number(document.querySelector('#network').dataset.scale)") == pytest.approx(scale)
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == camera

        _real_click(call, evaluate, ".agent-node[data-agent=ExperimentAgent]")
        wait_for("!!document.querySelector('#details.open #open-tools')")
        _real_click(call, evaluate, "#open-tools")
        wait_for("!document.querySelector('#tools-modal').classList.contains('hidden')")
        _real_click(call, evaluate, "#tools-close")
        _real_click(call, evaluate, "#details-close")
        time.sleep(1.3)
        assert "late-tool" not in (evaluate("document.querySelector('#tools-list').textContent") or "")
        _real_click(call, evaluate, ".agent-node[data-agent=ExperimentAgent]")
        _real_click(call, evaluate, "#open-tools")
        wait_for("document.querySelector('#tools-list').textContent.includes('FEDOT AutoML')")
        assert "обучить модель" in evaluate("document.querySelector('#tools-list').textContent")
        assert "fresh-tool" in evaluate("document.querySelector('#tools-list').textContent")
        _real_click(call, evaluate, "#tools-close")

        ids = evaluate("[...document.querySelectorAll('.agent-node')].map(n=>n.dataset.viewId).sort()")
        selected = evaluate("document.querySelector('.agent-node.selected').dataset.agent")
        cameras = {}
        for design in DESIGNS:
            _real_click(call, evaluate, f"[data-design={design}]")
            wait_for(f"document.querySelector('#network').dataset.design === '{design}' && document.querySelector('#network').getAttribute('aria-busy') === 'false'")
            assert evaluate("[...document.querySelectorAll('.agent-node')].map(n=>n.dataset.viewId).sort()") == ids
            assert evaluate("document.querySelector('.agent-node.selected').dataset.agent") == selected
            assert evaluate("localStorage.getItem('mas-graph-design')") == design
            _assert_no_node_overlaps(evaluate)
            _assert_flow_connections(evaluate)
            assert evaluate("[...document.querySelectorAll('path.edge')].every(p=>!/[QC]/.test(p.getAttribute('d')))")
            _real_click(call, evaluate, "#fit")
            _screenshot(call, screenshots, f"agent-tree-{design}-overview-1440x900.png")
            _real_click(call, evaluate, "#zoom-reset")
            _real_click(call, evaluate, "#zoom-in")
            cameras[design] = evaluate("document.querySelector('#network').getAttribute('viewBox')")
            _screenshot(call, screenshots, f"agent-tree-{design}-details-1440x900.png")
        for design in DESIGNS:
            _real_click(call, evaluate, f"[data-design={design}]")
            wait_for(f"document.querySelector('#network').dataset.design === '{design}'")
            assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == cameras[design]

        for width, height in VIEWPORTS:
            call("Emulation.setDeviceMetricsOverride", {"width":width,"height":height,"deviceScaleFactor":1,"mobile":False})
            time.sleep(.2)
            wait_for("document.querySelector('#network').getAttribute('aria-busy') === 'false'")
            assert evaluate("document.body.scrollWidth <= innerWidth && document.body.scrollHeight <= innerHeight")
            _assert_no_node_overlaps(evaluate)
            _assert_flow_connections(evaluate)
            _screenshot(call, screenshots, f"agent-tree-workshop-{width}x{height}.png")

        call("Page.navigate", {"url":f"http://127.0.0.1:{port}/agent-tree?user_id=u&session_id=s&design=classic"})
        wait_for("document.querySelector('#network')?.dataset.design === 'workshop'")
        evaluate("localStorage.setItem('mas-graph-design','constellation')")
        call("Page.navigate", {"url":f"http://127.0.0.1:{port}/agent-tree?user_id=u&session_id=s"})
        wait_for("document.querySelector('#network')?.dataset.design === 'workshop'")
        assert evaluate("localStorage.getItem('mas-graph-design')") == "workshop"
        assert not errors, errors


@pytest.mark.skipif(not os.getenv("COSCIENTIST_TEST_BROWSER"), reason="optional headless browser smoke")
@pytest.mark.parametrize("design", DESIGNS)
def test_agent_tree_real_session_controls_and_persistence(tmp_path, monkeypatch, design):
    from CoScientist.agents import common
    from CoScientist.web.app import create_app

    async def no_proxy_preflight():
        return None

    monkeypatch.setattr(common, "verify_proxy_reachable", no_proxy_preflight)
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COSCIENTIST_CONFIG", "hypotheses")
    app = create_app()
    runtime = app.state.runtime
    user = runtime.registry.create_user("MAS browser")
    session = runtime.registry.create_session(user["id"], "Agent architecture")
    key = (user["id"], session["id"])
    runtime.save_agent_configuration(key, {"general":{"startMode":"planner","contextInitEnabled":True},
        "medicalAgent":{"enabled":False},"agents":{"overrides":{"ResearchAgent":{"enabled":True}}}})
    query = f"user_id={user['id']}&session_id={session['id']}&design={design}"

    with _run_browser(tmp_path, app, query) as (call, evaluate, wait_for, errors, port):
        screenshots = Path(os.environ["COSCIENTIST_TEST_SCREENSHOTS"]) if os.getenv("COSCIENTIST_TEST_SCREENSHOTS") else None
        if evaluate("document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded')!=='true'"):
            _real_click(call, evaluate, "#agent-menu-toggle")
        wait_for("document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded') === 'true'")
        wait_for("!!document.querySelector('[data-agent-toggle=ResearchAgent]')")
        assert not evaluate("!!document.querySelector('[data-agent-toggle=TaskExecutorAgent]')")
        for required in ("OrchestratorAgent", "ExperimentModuleAgent", "ExperimentPlannerAgent",
                         "ExperimentExecutorAgent", "ExperimentResultReviewAgent"):
            assert evaluate(f"document.querySelector('[data-agent-toggle={required}]')?.disabled === true")
        assert evaluate("document.querySelector('[data-agent-toggle=ExperimentModuleAgent]').closest('.agent-menu-row').textContent.includes('Обязательный')")
        assert not evaluate("!!document.querySelector('[data-agent-toggle=ToolPreparerAgent]')")

        _real_click(call, evaluate, ".agent-node[data-agent=__mas_session__]")
        wait_for("!!document.querySelector('#details.open #open-agent-menu')")
        assert "Управляет составом агентов выбранной сессии" in evaluate("document.querySelector('#details').textContent")
        _real_click(call, evaluate, "#agent-menu-toggle")
        _real_click(call, evaluate, "#open-agent-menu")
        assert evaluate("document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded')") == "true"
        _real_click(call, evaluate, "#agent-menu-toggle")
        wait_for("document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded') === 'false'")

        _real_click(call, evaluate, ".agent-node[data-agent=ResearchAgent]")
        wait_for("!!document.querySelector('#details.open #agent-toggle:not(:disabled)')")
        title = evaluate("document.querySelector('#details h2').textContent")
        description = evaluate("document.querySelector('#details .description').textContent")
        _real_click(call, evaluate, "#zoom-in")
        camera = evaluate("document.querySelector('#network').getAttribute('viewBox')")
        assert "Отключить" in evaluate("document.querySelector('#agent-toggle').textContent")
        _real_click(call, evaluate, "#agent-toggle")
        wait_for("!document.querySelector('.agent-node[data-agent=ResearchAgent]') && document.querySelector('#agent-toggle')?.textContent.includes('Включить')")
        assert evaluate("document.querySelector('#details h2').textContent") == title
        assert evaluate("document.querySelector('#details .description').textContent") == description
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == camera
        assert not evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').checked")
        _real_click(call, evaluate, "#agent-toggle")
        wait_for("document.querySelectorAll('.agent-node[data-agent=ResearchAgent]').length >= 2 && document.querySelector('#agent-toggle')?.textContent.includes('Отключить')")
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == camera

        _real_click(call, evaluate, "#agent-menu-toggle")
        wait_for("document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded') === 'true'")

        evaluate("window.realFetch=window.fetch;window.fetch=(url,options)=>options?.method==='POST'?Promise.resolve(new Response('{}',{status:503})):window.realFetch(url,options)")
        _real_click(call, evaluate, "[data-agent-toggle=ResearchAgent]")
        wait_for("document.querySelector('#agent-menu-status').classList.contains('failed') && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        assert evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').checked")
        assert evaluate("document.querySelectorAll('.agent-node[data-agent=ResearchAgent]').length >= 2")
        evaluate("window.fetch=window.realFetch")
        _real_click(call, evaluate, "#agent-menu-retry")
        wait_for("!document.querySelector('#agent-menu-status').classList.contains('failed')")

        # A late successful save may update the graph, but must not replace the
        # inspector content of an agent selected while that request was pending.
        evaluate("""
          window.realFetch=window.fetch;
          window.fetch=(url,options) => options?.method==='POST'
            ? new Promise(resolve => { window.releaseAgentSave=() => window.realFetch(url,options).then(resolve); })
            : window.realFetch(url,options)
        """)
        _real_click(call, evaluate, "[data-agent-toggle=ResearchAgent]")
        _real_click(call, evaluate, ".agent-node[data-agent=ExperimentAgent]")
        wait_for("document.querySelector('#details h2')?.textContent.includes('ReAct')")
        evaluate("window.releaseAgentSave();window.fetch=window.realFetch")
        wait_for("!document.querySelector('.agent-node[data-agent=ResearchAgent]') && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        assert "ReAct" in evaluate("document.querySelector('#details h2').textContent")
        _real_click(call, evaluate, "[data-agent-toggle=ResearchAgent]")
        wait_for("document.querySelectorAll('.agent-node[data-agent=ResearchAgent]').length >= 2")

        _real_click(call, evaluate, "[data-agent-toggle=MedicalAgent]")
        wait_for("!!document.querySelector('.agent-node[data-agent=MedicalAgent]') && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        call("Page.navigate", {"url":f"http://127.0.0.1:{port}/agent-tree?{query}"})
        wait_for("!!document.querySelector('.agent-node[data-agent=MedicalAgent]') && document.querySelector('[data-agent-toggle=MedicalAgent]')?.checked")
        _assert_flow_connections(evaluate)

        _real_click(call, evaluate, ".agent-node[data-agent=ExperimentPlannerAgent]")
        wait_for("!!document.querySelector('#details.open')")
        assert not evaluate("!!document.querySelector('#details #agent-toggle')")
        assert "Обязательный участник" in evaluate("document.querySelector('#details').textContent")

        for width, height in VIEWPORTS:
            call("Emulation.setDeviceMetricsOverride", {"width":width,"height":height,"deviceScaleFactor":1,"mobile":False})
            time.sleep(.2)
            wait_for("document.querySelector('#network').getAttribute('aria-busy') === 'false'")
            assert evaluate("document.body.scrollWidth <= innerWidth && document.body.scrollHeight <= innerHeight")
            assert evaluate("(() => {const r=document.querySelector('#agent-menu').getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&r.right<=innerWidth})()")
            _assert_no_node_overlaps(evaluate)
            _assert_flow_connections(evaluate)
            _screenshot(call, screenshots, f"agent-tree-{design}-panel-{width}x{height}.png")

        evaluate("document.querySelector('#agent-menu-search').value='литературы';document.querySelector('#agent-menu-search').dispatchEvent(new Event('input'))")
        assert evaluate("document.querySelectorAll('.agent-menu-row').length") == 1
        evaluate("document.querySelector('#agent-menu-search').dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))")
        assert evaluate("document.querySelector('#agent-menu-content').classList.contains('hidden')")
        assert not errors, errors
