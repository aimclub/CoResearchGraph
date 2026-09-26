from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi.testclient import TestClient

from CoScientist.tools import mcp_catalog


def _server(server_id: str, name: str, *, status: str = "active"):
    return {
        "server_id": server_id,
        "name": name,
        "protocol": "http",
        "description": f"{name} description",
        "status": status,
        "last_synced": "2026-09-26T12:00:00+00:00",
    }


def _tool(server_id: str, name: str, description: str = "Readable description"):
    return {
        "tool_id": f"{server_id}:{name}",
        "server_id": server_id,
        "name": name,
        "description": description,
        "input_schema": {"type": "object"},
        "output_schema": None,
        "status": "active",
    }


def _endpoint(server: dict, *, secret: str = "secret"):
    return {
        "id": server["server_id"],
        "registry_id": server["server_id"],
        "name": server["name"],
        "url": f"https://internal.invalid/mcp?key={secret}",
        "headers": {"Authorization": f"Bearer {secret}"},
        "tool_filter": None,
        "sources": ["registry"],
        "aliases": [server["name"]],
        "registry": server,
        "url_key": "private",
    }


def test_live_list_is_current_and_registry_difference_is_explicit():
    server = _server("science", "Science")
    registry = {
        "status": "ready",
        "servers": [server],
        "tools": [_tool("science", "still_here"), _tool("science", "removed")],
    }
    probe = {
        "status": "reachable",
        "checked_at": "2026-09-26T13:00:00+00:00",
        "server_info": {"name": "Science", "version": "2"},
        "tools": [_tool("science", "still_here"), _tool("science", "new_live")],
    }

    result = mcp_catalog._assemble_catalog(
        registry, [_endpoint(server)], [probe], checked_at="2026-09-26T13:00:00+00:00",
    )

    assert {tool["name"] for tool in result["tools"]} == {"still_here", "new_live"}
    assert result["summary"]["current_tools"] == 2
    difference = result["servers"][0]["discrepancy"]
    assert difference == {
        "has_difference": True,
        "live_only": ["new_live"],
        "indexed_only": ["removed"],
    }
    new_tool = next(tool for tool in result["tools"] if tool["name"] == "new_live")
    assert new_tool["display_name"]["ru"] == "Инструмент"
    assert new_tool["summary"]["ru"] == "Описание на русском пока не добавлено."
    assert new_tool["display_name"]["en"] == "New live"


def test_unavailable_server_keeps_saved_metadata_without_claiming_availability():
    server = _server("offline", "Offline")
    registry = {"status": "ready", "servers": [server], "tools": [_tool("offline", "predict")]}
    result = mcp_catalog._assemble_catalog(
        registry,
        [_endpoint(server)],
        [{"status": "unavailable", "error_type": "TimeoutError", "tools": []}],
        checked_at="2026-09-26T13:00:00+00:00",
    )

    assert result["summary"]["current_tools"] == 0
    assert result["tools"][0]["status"] == "saved"
    assert result["servers"][0]["discovery_status"] == "unavailable"


def test_same_tool_name_on_different_servers_keeps_distinct_identity():
    servers = [_server("one", "One"), _server("two", "Two")]
    registry = {
        "status": "ready",
        "servers": servers,
        "tools": [_tool("one", "reproduce_all"), _tool("two", "reproduce_all")],
    }
    probes = [
        {"status": "reachable", "tools": [_tool(server["server_id"], "reproduce_all")]}
        for server in servers
    ]
    result = mcp_catalog._assemble_catalog(
        registry, [_endpoint(server) for server in servers], probes,
    )

    assert [tool["id"] for tool in result["tools"]] == [
        "one:reproduce_all", "two:reproduce_all",
    ]


def test_public_snapshot_never_contains_connection_secrets():
    secret = "do-not-publish"
    server = _server("private", "Private")
    result = mcp_catalog._assemble_catalog(
        {"status": "ready", "servers": [server], "tools": [_tool("private", "calculate")]},
        [_endpoint(server, secret=secret)],
        [{"status": "reachable", "tools": [_tool("private", "calculate")]}],
    )

    serialized = json.dumps(result)
    assert secret not in serialized
    assert "Authorization" not in serialized
    assert "internal.invalid" not in serialized


def test_fedot_server_is_featured_first_with_required_wording():
    fedot = _server("d6dac6fb09066080", "AutoMLTools")
    another = _server("other", "Another")
    registry = {
        "status": "ready",
        "servers": [another, fedot],
        "tools": [
            _tool("other", "calculate"),
            _tool("d6dac6fb09066080", "train_ml"),
            _tool("d6dac6fb09066080", "predict_ml"),
        ],
    }
    result = mcp_catalog._assemble_catalog(
        registry,
        [_endpoint(another), _endpoint(fedot)],
        [
            {"status": "reachable", "tools": [_tool("other", "calculate")]},
            {"status": "unavailable", "error_type": "ConnectError", "tools": []},
        ],
    )

    first = result["servers"][0]
    assert first["registry_id"] == "d6dac6fb09066080"
    assert first["featured"] is True
    assert "автоматического машинного обучения для научных задач" in first["display_name"]["ru"]
    assert [tool["name"] for tool in result["tools"][:2]] == ["train_ml", "predict_ml"]


def test_hybrid_epidemiology_server_uses_live_endpoint_and_russian_copy():
    server = _server("6d7e3471063c3f95", "hybrid-surrogate-epidemics")
    tool = _tool("6d7e3471063c3f95", "run_hybrid_model")
    result = mcp_catalog._assemble_catalog(
        {"status": "ready", "servers": [server], "tools": [tool]},
        [_endpoint(server)],
        [{"status": "reachable", "tools": [tool]}],
    )

    assert result["servers"][0]["display_name"]["ru"] == (
        "Гибридное и суррогатное моделирование эпидемий"
    )
    assert result["tools"][0]["display_name"]["ru"] == (
        "Запустить гибридную эпидемиологическую модель"
    )

    root = Path(__file__).resolve().parents[2]
    configured = json.loads(
        (root / "scripts/rag_tools/servers.json").read_text(encoding="utf-8")
    )
    epid = next(item for item in configured if item["name"] == "hybrid-surrogate-epidemics")
    assert epid["url"] == "http://10.32.11.22:7332/mcp"


def test_page_assets_and_navigation_are_wired():
    root = Path(__file__).resolve().parents[2]
    page = (root / "CoScientist/web/templates/tools.html").read_text(encoding="utf-8")
    rail = (root / "CoScientist/web/static/js/activity_rail.js").read_text(encoding="utf-8")
    script = (root / "CoScientist/web/static/js/tool_catalog.js").read_text(encoding="utf-8")

    assert "/static/css/tool_catalog.css" in page
    assert "/static/js/tool_catalog.js" in page
    assert 'name: "ToolCatalogue"' in rail and "window.open('/tools'" in rail
    assert "'/api/mcp-tools/refresh'" in script
    assert "details[data-details-id]" in script  # open cards survive polling/language changes
    assert "Рекомендуемое" not in script
    assert "badge(text('featured')" not in script
    assert "badge(tool.status" not in script
    assert "featured-status" not in script
    assert "lang === 'en' && tool.original_description" in script
    assert 'id="availability-filter"' in page
    assert 'id="status-filter"' not in page
    assert "Сохранённое описание" not in page
    assert "Проверено сейчас" not in page
    assert "function needsAttention(tool)" in script
    assert "server.discrepancy?.has_difference" in script


def test_refresh_remains_available_in_memory_when_snapshot_disk_is_read_only(tmp_path, monkeypatch):
    async def registry():
        return {"status": "ready", "servers": [], "tools": []}

    def cannot_write(_payload):
        raise PermissionError("read-only")

    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(mcp_catalog, "_read_registry", registry)
    monkeypatch.setattr(mcp_catalog, "_merge_endpoints", lambda _registry: [])
    monkeypatch.setattr(mcp_catalog, "_write_snapshot", cannot_write)

    result = asyncio.run(mcp_catalog.refresh_catalog())

    assert result["snapshot_id"]
    assert mcp_catalog._snapshot == result
    assert not (tmp_path / "state" / "tool_catalog.json").exists()


def test_web_routes_serve_versioned_page_and_start_single_refresh(tmp_path, monkeypatch):
    payload = mcp_catalog._empty_snapshot(refreshing=False)
    calls = []

    async def fake_get_catalog(*, refresh_if_stale=True):
        calls.append(("get", refresh_if_stale))
        return dict(payload)

    async def fake_request_refresh(*, force=False):
        calls.append(("refresh", force))
        return True

    monkeypatch.setattr(mcp_catalog, "get_catalog", fake_get_catalog)
    monkeypatch.setattr(mcp_catalog, "request_refresh", fake_request_refresh)
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web-state"))

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        page = client.get("/tools")
        assert page.status_code == 200
        assert "no-store" in page.headers["cache-control"]
        assert "/static/js/tool_catalog.js?v=" in page.text

        result = client.get("/api/mcp-tools")
        assert result.status_code == 200
        assert "no-store" in result.headers["cache-control"]
        assert calls[-1] == ("get", True)

        refresh = client.post("/api/mcp-tools/refresh")
        assert refresh.status_code == 202
        assert refresh.json()["refresh_started"] is True
        assert calls[-2:] == [("refresh", True), ("get", False)]
