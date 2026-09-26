"""Optional Chromium smoke tests for the session MAS tree.

The browser is pointed at a small local FastAPI app.  The page, JavaScript,
layout module and CSS are the real application files; only the session graph
and tool responses are fixtures, so this test never needs a provider or MCP.
"""

from __future__ import annotations

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


def _assert_flow_connections(evaluate):
    """Visible flow arrows must connect their named cards, not empty scopes."""
    defects = evaluate("""
      (() => {
        const cards = new Map([...document.querySelectorAll('g.agent-node')].map(g => {
          const rect = g.querySelector('.node-card');
          return [g.dataset.viewId, {x: +g.dataset.x, y: +g.dataset.y,
            w: +(g.dataset.width || rect.getAttribute('width')), h: +(g.dataset.height || rect.getAttribute('height')),
            shape: g.dataset.shape}];
        }));
        const onBorder = (p, r) => r && (r.shape === 'circle'
          ? Math.abs(Math.hypot(p.x - r.x - r.w / 2, p.y - r.y - r.h / 2) - r.w / 2) < .5
          : r.shape === 'hexagon' ? Math.abs(Math.max(Math.abs(p.y - r.y - r.h / 2) / (r.h / 2),
            Math.abs(p.x - r.x - r.w / 2) / (r.w / 2) + Math.abs(p.y - r.y - r.h / 2) / r.h) - 1) < .01
          :
          p.x >= r.x - .5 && p.x <= r.x + r.w + .5 &&
          p.y >= r.y - .5 && p.y <= r.y + r.h + .5 &&
          Math.min(Math.abs(p.x - r.x), Math.abs(p.x - r.x - r.w),
            Math.abs(p.y - r.y), Math.abs(p.y - r.y - r.h)) < .5);
        const selector = document.querySelector('#network').dataset.design === 'classic' ? 'path.edge.flow[marker-end]' : 'path.edge[marker-end]';
        return [...document.querySelectorAll(selector)].flatMap(edge => {
          const start = edge.getPointAtLength(0), length = edge.getTotalLength();
          const end = edge.getPointAtLength(length), source = cards.get(edge.dataset.from), target = cards.get(edge.dataset.to);
          const problems = [];
          if (!onBorder(start, source)) problems.push('start is detached');
          if (!onBorder(end, target)) problems.push('end is detached');
          for (let distance = 3; distance < length - 2; distance += 3) {
            const p = edge.getPointAtLength(distance);
            if ([...cards.values()].some(r => r.shape === 'circle'
              ? Math.hypot(p.x - r.x - r.w / 2, p.y - r.y - r.h / 2) < r.w / 2 - 1
              : r.shape === 'hexagon' ? Math.abs(p.y - r.y - r.h / 2) < r.h / 2 - 1 &&
                Math.abs(p.x - r.x - r.w / 2) + Math.abs(p.y - r.y - r.h / 2) * r.w / (2 * r.h) < r.w / 2 - 1
              : p.x > r.x + 1 && p.x < r.x + r.w - 1 && p.y > r.y + 1 && p.y < r.y + r.h - 1)) {
              problems.push('crosses a card'); break;
            }
          }
          return problems.length ? [{from: edge.dataset.from, to: edge.dataset.to, problems}] : [];
        });
      })()
    """)
    assert not defects, defects


@contextmanager
def _run_browser(tmp_path, app, query="user_id=u&session_id=s&design=classic"):
    """Yield a CDP connection and its small command/evaluation API."""
    import uvicorn
    from websockets.sync.client import connect

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()

    profile = tmp_path / "chromium-profile"
    browser = subprocess.Popen(
        [
            os.environ["COSCIENTIST_TEST_BROWSER"],
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--disable-extensions",
            "--remote-debugging-port=0",
            f"--user-data-dir={profile}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ws = None
    try:
        deadline = time.monotonic() + 20
        while not (profile / "DevToolsActivePort").exists():
            assert browser.poll() is None, "Chromium exited before opening DevTools"
            assert time.monotonic() < deadline, "Chromium startup timed out"
            time.sleep(0.1)
        dev_port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        request = Request(f"http://127.0.0.1:{dev_port}/json/new?about:blank", method="PUT")
        with urlopen(request, timeout=5) as response:
            target = json.load(response)
        # Photographic desk textures make a lossless screenshot exceed the
        # websocket library's default 1 MiB, even at ordinary desktop sizes.
        ws = connect(target["webSocketDebuggerUrl"], open_timeout=5, max_size=16 * 1024 * 1024)
        serial = 0
        errors = []

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

        def wait_for(expression, timeout=12):
            end = time.monotonic() + timeout
            while not evaluate(expression):
                assert time.monotonic() < end, expression
                time.sleep(0.1)

        call("Runtime.enable")
        call("Page.enable")
        call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
        call("Page.navigate", {"url": f"http://127.0.0.1:{port}/agent-tree?{query}"})
        wait_for("document.querySelector('svg[data-scale]') && document.querySelectorAll('g.agent-node').length")
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
def test_agent_tree_browser_geometry_updates_and_tools(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from CoScientist.config import get_settings
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
    phases = [
        project_agent_tree(snapshot, desired_revision=1, active_revision=1, running=False),
        project_agent_tree(snapshot, desired_revision=1, active_revision=1, running=True),
    ]
    nir_snapshot = snapshot.model_copy(deep=True)
    nir_snapshot.nir.enabled = True
    # This is only a build-time gate for the YAML projection; no request is made.
    nir_snapshot.mcp.normcontrol_url = "http://normcontrol.invalid/mcp"
    phases.append(project_agent_tree(nir_snapshot, desired_revision=2, active_revision=1, running=True))
    state = {"phase": 0, "tool_calls": 0}

    @app.get("/agent-tree")
    def agent_tree_page():
        return HTMLResponse((web / "templates" / "agent_tree.html").read_text(encoding="utf-8"))

    @app.get("/api/users/{user_id}/sessions/{session_id}/agent-tree")
    def agent_tree_api(user_id: str, session_id: str):
        assert (user_id, session_id) == ("u", "s")
        return JSONResponse(copy.deepcopy(phases[state["phase"]]), headers={"Cache-Control": "no-store"})

    @app.get("/api/users/{user_id}/sessions/{session_id}/agents/{agent_name}/tools")
    def tools_api(user_id: str, session_id: str, agent_name: str):
        assert (user_id, session_id, agent_name) == ("u", "s", "ExperimentAgent")
        state["tool_calls"] += 1
        if state["tool_calls"] == 1:
            time.sleep(1.3)
            name = "late-tool"
        else:
            name = "fresh-tool"
        return JSONResponse({
            "agent": agent_name,
            "dynamic": False,
            "selectionReady": True,
            "tools": [{
                "id": f"local:{name}",
                "name": name,
                "display_name": {"ru": "FEDOT AutoML — обучить модель", "en": "FEDOT AutoML — train a model"},
                "summary": {"ru": "Автоматически подбирает и обучает модель машинного обучения.", "en": "Automatically selects and trains a machine learning model."},
                "status": "available",
                "pinned": True,
            }],
        })

    @app.get("/api/users/{user_id}/sessions/{session_id}/agents/catalog")
    def catalog_api(user_id: str, session_id: str):
        from CoScientist.config import settings_scope
        from CoScientist.web.agent_settings import agents_catalog
        with settings_scope(nir_snapshot if state["phase"] == 2 else snapshot):
            return agents_catalog()

    @app.post("/__test__/phase")
    def set_phase(payload: dict):
        state["phase"] = int(payload.get("phase", 0))
        return {"phase": state["phase"]}

    app.mount("/static", StaticFiles(directory=web / "static"), name="static")

    with _run_browser(tmp_path, app) as (call, evaluate, wait_for, errors, port):
        screenshot_dir = Path(os.environ["COSCIENTIST_TEST_SCREENSHOTS"]) if os.getenv("COSCIENTIST_TEST_SCREENSHOTS") else None
        assert evaluate("document.querySelector('svg').dataset.layout") == "elk"
        initial_scale = evaluate("Number(document.querySelector('svg').dataset.scale)")
        assert 0.45 <= initial_scale < 1
        rows = evaluate("""
          [...document.querySelectorAll('g.agent-node')].map(g => {
            const card = g.querySelector('rect.node-card');
            return {agent: g.dataset.agent, viewId: g.dataset.viewId,
              x: Number(g.dataset.x), y: Number(g.dataset.y),
              width: Number(card?.getAttribute('width')), height: Number(card?.getAttribute('height'))};
          })
        """)
        assert rows
        workflow_agents = {item["agentId"] for item in phases[0]["workflow"]["instances"]}
        rendered_agents = {row["agent"] for row in rows}
        assert rendered_agents <= workflow_agents | {"__mas_session__"}
        assert rendered_agents >= workflow_agents | {"__mas_session__"}
        assert all(row["viewId"] for row in rows)
        assert all(row["width"] == 224 and row["height"] >= 72 for row in rows)

        assert sum(row["agent"] == "ResearchAgent" for row in rows) >= 2, rows
        by_agent = {name: next(row for row in rows if row["agent"] == name) for name in (
            "ExperimentPlannerAgent", "ExperimentExecutorAgent", "ExperimentResultReviewAgent", "ResultAggregatorAgent"
        )}
        assert by_agent["ExperimentPlannerAgent"]["x"] < by_agent["ExperimentExecutorAgent"]["x"] < by_agent["ExperimentResultReviewAgent"]["x"]
        assert by_agent["ExperimentResultReviewAgent"]["x"] < by_agent["ResultAggregatorAgent"]["x"]
        config = next(row for row in rows if row["agent"] == "__mas_session__")
        orchestrator = next(row for row in rows if row["agent"] == "OrchestratorAgent")
        assert config["y"] == orchestrator["y"]
        _assert_flow_connections(evaluate)
        assert "доступен в нескольких ветках" not in evaluate("document.body.textContent").lower()
        assert "Общий агент" not in evaluate("document.body.textContent")
        delegate_paths = evaluate("[...document.querySelectorAll('path.edge.delegate')].map(edge => edge.getAttribute('d'))")
        assert len(delegate_paths) == len(set(delegate_paths)), delegate_paths
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                assert left["x"] + left["width"] <= right["x"] or right["x"] + right["width"] <= left["x"] or left["y"] + left["height"] <= right["y"] or right["y"] + right["height"] <= left["y"], (left, right)
        if screenshot_dir:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            image = call("Page.captureScreenshot", {"format": "png"})["data"]
            (screenshot_dir / "agent-tree-wide-overview-1440x900.png").write_bytes(__import__("base64").b64decode(image))

        context_point = evaluate("""
          (() => { const r = document.querySelector('g.agent-node[data-agent="ContextInitAgent"]').getBoundingClientRect();
            return {x: r.left + r.width / 2, y: r.top + r.height / 2}; })()
        """)
        call("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": context_point["x"], "y": context_point["y"], "button": "none", "pointerType": "mouse"})
        call("Input.dispatchMouseEvent", {"type": "mousePressed", "x": context_point["x"], "y": context_point["y"], "button": "left", "buttons": 1, "clickCount": 1, "pointerType": "mouse"})
        call("Input.dispatchMouseEvent", {"type": "mouseReleased", "x": context_point["x"], "y": context_point["y"], "button": "left", "buttons": 0, "clickCount": 1, "pointerType": "mouse"})
        wait_for("!!document.querySelector('#details.open')")
        wait_for("!!document.querySelector('#details-close')")
        evaluate("document.querySelector('#details-close').dispatchEvent(new MouseEvent('click', {bubbles:true}))")

        assert evaluate("['#zoom-in', '#zoom-out', '#zoom-reset', '#fit'].every(id => document.querySelector(id))")
        evaluate("document.querySelector('#zoom-in').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        scale_before_status = evaluate("Number(document.querySelector('svg').dataset.scale)")
        assert scale_before_status > initial_scale

        evaluate("fetch('/__test__/phase', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({phase:1})})")
        wait_for("document.querySelector('#state')?.textContent.includes('Запрос выполняется')")
        scale_after_status = evaluate("Number(document.querySelector('svg').dataset.scale)")
        assert scale_after_status == pytest.approx(scale_before_status)

        evaluate("fetch('/__test__/phase', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({phase:2})})")
        wait_for("!!document.querySelector('g.agent-node[data-agent=\"NirReportAgent\"]')")
        _assert_flow_connections(evaluate)
        assert evaluate("Number(document.querySelector('svg').dataset.scale)") == pytest.approx(scale_before_status)

        evaluate("document.querySelector('g.agent-node[data-agent=\"ExperimentAgent\"]').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        wait_for("!!document.querySelector('#details.open') && !!document.querySelector('#details-close')")
        evaluate("[...document.querySelectorAll('#details button')].find(button => button.id !== 'details-close').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        wait_for("!document.querySelector('#tools-modal').classList.contains('hidden')")
        evaluate("document.querySelector('#details-close').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        assert evaluate("!document.querySelector('#details').classList.contains('open')")
        evaluate("document.querySelector('#tools-close')?.dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        time.sleep(1.6)
        assert evaluate("document.querySelector('#tools-modal').classList.contains('hidden')")
        assert "late-tool" not in (evaluate("document.querySelector('#tools-list').textContent") or "")

        evaluate("document.querySelector('g.agent-node[data-agent=\"ExperimentAgent\"]').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        wait_for("!!document.querySelector('#details.open')")
        evaluate("[...document.querySelectorAll('#details button')].find(button => button.id !== 'details-close').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        wait_for("document.querySelector('#tools-list').textContent.includes('FEDOT AutoML')")
        tools_text = evaluate("document.querySelector('#tools-list').textContent")
        assert "обучить модель" in tools_text
        assert "fresh-tool" in tools_text
        assert evaluate("!!document.querySelector('#tools-list .pinned, .tool .badge.pinned')")

        if screenshot_dir:
            image = call("Page.captureScreenshot", {"format": "png"})["data"]
            (screenshot_dir / "agent-tree-tools.png").write_bytes(__import__("base64").b64decode(image))
        evaluate("document.querySelector('#tools-close').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        evaluate("document.querySelector('#fit').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        wait_for("document.querySelector('svg').dataset.scale")
        if screenshot_dir:
            image = call("Page.captureScreenshot", {"format": "png"})["data"]
            (screenshot_dir / "agent-tree-fit-1440x900.png").write_bytes(__import__("base64").b64decode(image))

        for width, height in ((1440, 900), (1280, 720), (800, 900)):
            call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
            time.sleep(.15)  # Deliver ResizeObserver before checking async layout.
            wait_for("document.querySelector('#network').getAttribute('aria-busy') === 'false'")
            _assert_flow_connections(evaluate)
            evaluate("document.querySelector('#zoom-reset').click()")
            wait_for("document.querySelector('#network') && document.querySelector('#details') && document.querySelectorAll('g.agent-node').length")
            dimensions = evaluate("""
              (() => { const n = document.querySelector('#network').getBoundingClientRect();
                const d = document.querySelector('#details').getBoundingClientRect();
                return {bodyW: document.body.scrollWidth, bodyH: document.body.scrollHeight,
                  networkW: n.width, networkH: n.height, detailsW: d.width, detailsH: d.height,
                  viewportW: innerWidth, viewportH: innerHeight}; })()
            """)
            assert dimensions["bodyW"] <= width and dimensions["bodyH"] <= height
            assert dimensions["networkW"] > 0 and dimensions["networkH"] > 0
            assert dimensions["detailsW"] > 0 and dimensions["detailsH"] > 0
            if width >= 1000:
                assert dimensions["networkW"] + dimensions["detailsW"] == pytest.approx(width, abs=2)
            if screenshot_dir:
                if width == 800:
                    evaluate("document.querySelector('#details-close').click()")
                image = call("Page.captureScreenshot", {"format": "png"})["data"]
                (screenshot_dir / f"agent-tree-{width}x{height}.png").write_bytes(__import__("base64").b64decode(image))

        # A useful review image of the experiment, at readable scale rather than
        # the explicitly requested whole-graph fit overview.
        call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
        evaluate("document.querySelector('g.agent-node[data-agent=\"ExperimentPlannerAgent\"]').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        time.sleep(.15)
        wait_for("document.querySelector('#network').getAttribute('aria-busy') === 'false'")
        evaluate("document.querySelector('#zoom-reset').click()")
        evaluate("""
          (() => { const scene = document.querySelector('#network');
            const card = document.querySelector('g.agent-node[data-agent="ExperimentModuleAgent"]');
            const deltaY = card.getBoundingClientRect().top - scene.getBoundingClientRect().top - 70;
            scene.dispatchEvent(new WheelEvent('wheel', {deltaY, bubbles:true, cancelable:true})); })()
        """)
        if screenshot_dir:
            image = call("Page.captureScreenshot", {"format": "png"})["data"]
            (screenshot_dir / "agent-tree-experiment-1440x900.png").write_bytes(__import__("base64").b64decode(image))

        # Every design renders the same occurrence IDs and keeps the current
        # selection. New links attach to cards (or circles), never to empty space.
        expected_ids = evaluate("[...document.querySelectorAll('.agent-node')].map(n => n.dataset.viewId).sort()")
        selected_id = evaluate("document.querySelector('.agent-node.selected').dataset.viewId")
        classic_camera = evaluate("document.querySelector('#network').getAttribute('viewBox')")
        assert evaluate("document.querySelectorAll('.design-options button').length") == 7
        for design in ("studio", "atlas", "blueprint", "constellation", "architecture", "workshop"):
            evaluate(f"document.querySelector('.design-options button[data-design=\"{design}\"]').click()")
            wait_for(f"document.querySelector('#network').dataset.design === '{design}' && document.querySelector('#network').getAttribute('aria-busy') === 'false'")
            assert evaluate("[...document.querySelectorAll('.agent-node')].map(n => n.dataset.viewId).sort()") == expected_ids
            assert evaluate("document.querySelector('.agent-node.selected').dataset.viewId") == selected_id
            assert evaluate("document.querySelectorAll('.design-options button[aria-pressed=\"true\"]').length") == 1
            assert evaluate("localStorage.getItem('mas-graph-design')") == design
            if design == 'workshop':
                assert not evaluate("[...document.querySelectorAll('.node-title')].some(n => n.textContent.includes('…'))")
                evaluate("document.querySelector('.agent-node[data-agent=ExperimentAgent]').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
                evaluate("document.querySelector('#open-tools').click()")
                wait_for("document.querySelector('#tools-list')?.textContent.includes('FEDOT AutoML')")
                evaluate("document.querySelector('#tools-close').click()")
                evaluate(f"document.querySelector('.agent-node[data-view-id=\"{selected_id}\"]').dispatchEvent(new MouseEvent('click', {{bubbles:true}}))")
            _assert_flow_connections(evaluate)
            paths = evaluate("[...document.querySelectorAll('.edge')].map(p => p.getAttribute('d'))")
            assert all('Q' not in path and 'C' not in path for path in paths)
            assert sum(path.count('L') == 1 for path in paths) >= len(paths) * .8
            # No overlapping nodes, including the full circle bounds.
            assert evaluate("""
              (() => { const rows = [...document.querySelectorAll('.agent-node')].map(n =>
                ({x:+n.dataset.x, y:+n.dataset.y, w:+n.dataset.width, h:+n.dataset.height}));
                return rows.every((a, i) => rows.slice(i + 1).every(b =>
                  a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y)); })()
            """)
            if screenshot_dir:
                image = call("Page.captureScreenshot", {"format": "png"})["data"]
                (screenshot_dir / f"agent-tree-design-{design}-1440x900.png").write_bytes(__import__("base64").b64decode(image))
                evaluate("document.querySelector('#zoom-reset').click()")
                evaluate("""
                  (() => { const scene = document.querySelector('#network');
                    const node = document.querySelector('.agent-node[data-agent="ExperimentModuleAgent"]');
                    const deltaY = node.getBoundingClientRect().top - scene.getBoundingClientRect().top - 60;
                    scene.dispatchEvent(new WheelEvent('wheel', {deltaY, bubbles:true, cancelable:true})); })()
                """)
                image = call("Page.captureScreenshot", {"format": "png"})["data"]
                (screenshot_dir / f"agent-tree-design-{design}-detail.png").write_bytes(__import__("base64").b64decode(image))
                evaluate("document.querySelector('#fit').click()")
            for width, height in ((1280, 720), (800, 900), (1440, 900)):
                call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
                time.sleep(.1)
                wait_for("document.querySelector('#network').getAttribute('aria-busy') === 'false'")
                assert evaluate("document.body.scrollWidth <= innerWidth && document.body.scrollHeight <= innerHeight")
            if design == "studio":
                evaluate("document.querySelector('#zoom-in').click()")
                studio_camera = evaluate("document.querySelector('#network').getAttribute('viewBox')")

        evaluate("document.querySelector('.design-options button[data-design=\"studio\"]').click()")
        wait_for("document.querySelector('#network').dataset.design === 'studio'")
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == studio_camera
        evaluate("document.querySelector('.design-options button[data-design=\"classic\"]').click()")
        wait_for("document.querySelector('#network').dataset.design === 'classic'")
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == classic_camera

        evaluate("document.querySelector('.design-options button[data-design=\"constellation\"]').click()")
        wait_for("document.querySelector('#network').dataset.design === 'constellation'")
        call("Page.navigate", {"url": f"http://127.0.0.1:{port}/agent-tree?user_id=u&session_id=s"})
        wait_for("document.querySelector('#network')?.dataset.design === 'constellation'")
        _assert_flow_connections(evaluate)
        # Switching back to an asynchronously laid out view must not overwrite
        # the final choice when buttons are pressed quickly.
        evaluate("['studio','classic','atlas'].forEach(name => document.querySelector('.design-options button[data-design=\"' + name + '\"]').click())")
        wait_for("document.querySelector('#network').dataset.design === 'atlas' && document.querySelector('#network').getAttribute('aria-busy') === 'false'")
        evaluate("document.querySelector('.design-options button[data-design=\"classic\"]').click()")
        wait_for("document.querySelector('#network').dataset.design === 'classic' && document.querySelector('#network').getAttribute('aria-busy') === 'false'")

        call("Page.addScriptToEvaluateOnNewDocument", {"source": "Object.defineProperty(window, 'ELK', {get: () => undefined, set: () => {}, configurable: true});"})
        call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 900, "deviceScaleFactor": 1, "mobile": False})
        call("Page.navigate", {"url": f"http://127.0.0.1:{port}/agent-tree?user_id=u&session_id=s"})
        wait_for("document.querySelector('svg[data-scale]') && document.querySelectorAll('g.agent-node').length")
        wait_for("document.querySelector('svg').dataset.layout === 'fallback'")
        _assert_flow_connections(evaluate)

        # With every optional caller disabled, the coordinator still needs a
        # real output port and enough space to route around session settings.
        result = call("Runtime.evaluate", {"awaitPromise": True, "returnByValue": True, "expression": """
          (async () => {
            const a = (id) => ({id, agentId: id, type: 'agent'});
            const root = {id: 'workflow', type: 'sequence', children: [a('owner'), a('report')]};
            const geometry = await AgentTreeLayout.layout({root, instances: root.children,
              configuration: {ownerInstanceId: 'owner'}});
            const flow = geometry.edges.find(edge => edge.relation === 'flow' && edge.arrow);
            const owner = geometry.nodes.find(node => node.id === 'owner');
            const report = geometry.nodes.find(node => node.id === 'report');
            const start = flow.points[0], end = flow.points[flow.points.length - 1];
            return {connected: start.y === owner.y + owner.h && start.x > owner.x && start.x < owner.x + owner.w &&
              end.x === report.x && end.y === report.y + report.h / 2,
              bounded: geometry.edges.every(edge => edge.points.every(p => p.x >= 0 && p.x <= geometry.w && p.y >= 0 && p.y <= geometry.h))};
          })()
        """})
        assert result["result"]["value"] == {"connected": True, "bounded": True}

        assert not errors, errors


@pytest.mark.skipif(not os.getenv("COSCIENTIST_TEST_BROWSER"), reason="optional headless browser smoke")
@pytest.mark.parametrize("design", ["architecture", "workshop"])
def test_architecture_menu_changes_real_session_and_keeps_camera(tmp_path, monkeypatch, design):
    """Use real session APIs and persistence; no agent or external model runs."""
    from CoScientist.web.app import create_app

    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COSCIENTIST_CONFIG", "hypotheses")
    app = create_app()
    runtime = app.state.runtime
    user = runtime.registry.create_user("Blueprint browser")
    session = runtime.registry.create_session(user["id"], "Architecture")
    key = (user["id"], session["id"])
    runtime.save_agent_configuration(key, {"general": {"startMode": "planner", "contextInitEnabled": True},
        "medicalAgent": {"enabled": False}, "agents": {"overrides": {"ResearchAgent": {"enabled": True}}}})
    query = f"user_id={user['id']}&session_id={session['id']}&design={design}"
    with _run_browser(tmp_path, app, query) as (call, evaluate, wait_for, errors, port):
        wait_for(f"document.querySelector('#network').dataset.design === '{design}'")
        wait_for("!!document.querySelector('[data-agent-toggle=ResearchAgent]')")
        evaluate("if (document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded') !== 'true') document.querySelector('#agent-menu-toggle').click()")
        assert evaluate("document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded')") == 'true'
        assert evaluate("document.querySelector('[data-agent-toggle=OrchestratorAgent]').disabled")
        assert not evaluate("document.querySelector('[data-agent-toggle=ToolPreparerAgent]') !== null")
        assert evaluate("document.querySelectorAll('.agent-node[data-agent=ResearchAgent]').length") >= 2
        evaluate("document.querySelector('.agent-node[data-agent=ExperimentAgent]').dispatchEvent(new MouseEvent('click', {bubbles:true}))")
        time.sleep(.15)
        assert evaluate("document.querySelector('#details h2').textContent") == 'ReAct: MCP-инструменты'
        assert evaluate("document.querySelector('#details .description').textContent").startswith('Выполняет вычисления')
        assert evaluate("!!document.querySelector('#open-tools')")
        selected = evaluate("document.querySelector('.agent-node.selected').dataset.viewId")
        camera = evaluate("document.querySelector('#network').getAttribute('viewBox')")
        evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').click()")
        wait_for("!document.querySelector('.agent-node[data-agent=ResearchAgent]') && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == camera
        assert evaluate("document.querySelector('.agent-node.selected').dataset.viewId") == selected
        assert not evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').checked")
        evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').click()")
        wait_for("document.querySelectorAll('.agent-node[data-agent=ResearchAgent]').length >= 2 && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == camera
        # Repeated polling must preserve the user's camera and switch state.
        time.sleep(3.2)
        assert evaluate("document.querySelector('#network').getAttribute('viewBox')") == camera
        assert evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').checked")
        # A failed save must not remove nodes or leave the switch flipped.
        evaluate("window.originalFetch = window.fetch; window.fetch = (url, options) => options?.method === 'POST' ? Promise.resolve(new Response('{}', {status: 503})) : window.originalFetch(url, options)")
        evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').click()")
        wait_for("document.querySelector('#agent-menu-status').classList.contains('failed') && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        assert evaluate("document.querySelector('[data-agent-toggle=ResearchAgent]').checked")
        assert evaluate("document.querySelectorAll('.agent-node[data-agent=ResearchAgent]').length") >= 2
        evaluate("window.fetch = window.originalFetch; document.querySelector('#agent-menu-retry').click()")
        wait_for("!document.querySelector('#agent-menu-status').classList.contains('failed')")
        # A setting-backed route is added immediately and survives reloading.
        evaluate("document.querySelector('[data-agent-toggle=MedicalAgent]').click()")
        wait_for("document.querySelector('.agent-node[data-agent=MedicalAgent]') && document.querySelector('#agent-menu').getAttribute('aria-busy') === 'false'")
        call("Page.navigate", {"url": f"http://127.0.0.1:{port}/agent-tree?{query}"})
        wait_for("document.querySelector('.agent-node[data-agent=MedicalAgent]') && document.querySelector('[data-agent-toggle=MedicalAgent]')?.checked")
        evaluate("if (document.querySelector('#agent-menu-toggle').getAttribute('aria-expanded') !== 'true') document.querySelector('#agent-menu-toggle').click()")
        _assert_flow_connections(evaluate)
        screenshots = os.getenv("COSCIENTIST_TEST_SCREENSHOTS")
        for width, height in ((1440, 900), (1280, 720), (800, 900)):
            call("Emulation.setDeviceMetricsOverride", {"width": width, "height": height, "deviceScaleFactor": 1, "mobile": False})
            time.sleep(.15)
            evaluate("document.querySelector('#fit').click()")
            assert evaluate("document.body.scrollWidth <= innerWidth && document.body.scrollHeight <= innerHeight")
            assert evaluate("(() => { const r = document.querySelector('#agent-menu').getBoundingClientRect(); return r.top >= 0 && r.bottom <= innerHeight && r.right <= innerWidth; })()")
            if screenshots:
                target = Path(screenshots); target.mkdir(parents=True, exist_ok=True)
                image = call("Page.captureScreenshot", {"format": "png"})["data"]
                (target / f"{design}-menu-{width}x{height}.png").write_bytes(__import__('base64').b64decode(image))
        # Search and keyboard dismissal do not modify configuration.
        evaluate("document.querySelector('#agent-menu-search').value = 'литературы'; document.querySelector('#agent-menu-search').dispatchEvent(new Event('input'))")
        assert evaluate("document.querySelectorAll('.agent-menu-row').length") == 1
        evaluate("document.querySelector('#agent-menu-search').dispatchEvent(new KeyboardEvent('keydown', {key:'Escape', bubbles:true}))")
        assert evaluate("document.querySelector('#agent-menu-content').classList.contains('hidden')")
        assert not errors, errors
