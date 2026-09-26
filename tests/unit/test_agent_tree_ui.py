import asyncio

import pytest

from fastapi.testclient import TestClient

from CoScientist.config import get_settings, settings_scope
from CoScientist.web import app as web_app
from CoScientist.web.app import create_app


def _create_user(client, nickname):
    response = client.post("/api/users", json={"nickname": nickname})
    assert response.status_code == 201
    return response.json()["user"]


def _create_session(client, user_id, title):
    response = client.post(f"/api/users/{user_id}/sessions", json={"title": title})
    assert response.status_code == 201
    return response.json()["session"]


def test_agent_tree_is_session_scoped_and_hides_internal_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web-state"))
    app = create_app()
    with TestClient(app) as client:
        user = _create_user(client, "Tree tester")
        first = _create_session(client, user["id"], "First")
        second = _create_session(client, user["id"], "Second")
        first_api = f"/api/users/{user['id']}/sessions/{first['id']}"
        second_api = f"/api/users/{user['id']}/sessions/{second['id']}"

        settings = client.get(first_api + "/settings").json()
        settings["agents"]["overrides"]["ResearchAgent"] = {"enabled": False}
        saved = client.post(first_api + "/settings", json=settings)
        assert saved.status_code == 200
        assert saved.json()["agentConfiguration"]["desiredRevision"] == 1

        first_tree = client.get(first_api + "/agent-tree").json()
        second_tree = client.get(second_api + "/agent-tree").json()
        first_names = {node["id"] for node in first_tree["nodes"]}
        second_names = {node["id"] for node in second_tree["nodes"]}

        assert "ResearchAgent" not in first_names
        assert "ResearchAgent" in second_names
        assert "PlanningPipelineAgent" not in second_names
        assert "ToolPreparerAgent" not in second_names
        assert "ExperimentAgent" in second_names
        assert second_tree["desiredRevision"] == 0
        assert all(edge["from"] in second_names and edge["to"] in second_names
                   for edge in second_tree["edges"])


def test_react_tools_pin_fedot_automl_first(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web-state"))
    app = create_app()
    with TestClient(app) as client:
        user = _create_user(client, "FEDOT tool tester")
        session = _create_session(client, user["id"], "Tools")
        url = (
            f"/api/users/{user['id']}/sessions/{session['id']}"
            "/agents/ExperimentAgent/tools"
        )
        response = client.get(url)
        assert response.status_code == 200
        tools = response.json()["tools"]
        assert tools[0]["id"] == "d6dac6fb09066080:train_ml"
        assert tools[0]["name"] == "train_ml"
        assert tools[0]["pinned"] is True


def test_architecture_switch_preserves_session_settings_and_accepted_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COSCIENTIST_CONFIG", "hypotheses")
    app = create_app()
    with TestClient(app) as client:
        user = _create_user(client, "Architecture tester")
        first = _create_session(client, user["id"], "First")
        second = _create_session(client, user["id"], "Second")
        api = f"/api/users/{user['id']}/sessions/{first['id']}"
        other = f"/api/users/{user['id']}/sessions/{second['id']}"
        overrides = {"ResearchAgent": {"reasoning": "high"}, "CoderAgent": {"model": "custom-model"}}
        assert client.post(api + "/settings", json={"agents": {"overrides": overrides}}).status_code == 200
        runtime = app.state.runtime
        key = (user["id"], first["id"])
        revision, accepted = runtime.settings_snapshot(key)
        runtime.manager_agent_revisions[key] = revision
        global_before = get_settings().model_dump()

        off = client.post(api + "/agents/ResearchAgent/enabled", json={"enabled": False})
        assert off.status_code == 200
        assert off.json()["agentConfiguration"]["pending"] is True
        assert off.json()["agentConfiguration"]["activeRevision"] == revision
        catalog = {a["name"]: a for a in off.json()["agents"]}
        assert catalog["ResearchAgent"]["effectiveEnabled"] is False
        assert catalog["ResearchAgent"]["title"]["ru"] == "Исследователь литературы"
        assert catalog["ResearchAgent"]["descriptionLocalized"]["ru"].startswith("Ищет")
        assert "ResearchAgent" not in {n["agentId"] for n in client.get(api + "/agent-tree").json()["workflow"]["instances"]}
        assert "ResearchAgent" in {n["agentId"] for n in client.get(other + "/agent-tree").json()["workflow"]["instances"]}
        assert accepted.agents.overrides["ResearchAgent"].enabled is None
        settings = client.get(api + "/settings").json()
        assert settings["agents"]["overrides"]["ResearchAgent"] == {"enabled": False, "reasoning": "high"}
        assert settings["agents"]["overrides"]["CoderAgent"] == {"model": "custom-model"}
        assert client.post(api + "/agents/ResearchAgent/enabled", json={"enabled": True}).status_code == 200
        assert "ResearchAgent" in {n["agentId"] for n in client.get(api + "/agent-tree").json()["workflow"]["instances"]}

        # Setting-backed routes must change the route, not a dead override.
        assert client.post(api + "/agents/MedicalAgent/enabled", json={"enabled": True}).status_code == 200
        settings = client.get(api + "/settings").json()
        assert settings["medicalAgent"]["enabled"] is True
        assert "enabled" not in settings["agents"]["overrides"].get("MedicalAgent", {})
        assert client.post(api + "/agents/MedicalAgent/enabled", json={"enabled": False}).status_code == 200
        assert get_settings().model_dump() == global_before
        assert client.get(other + "/agent-tree").json()["desiredRevision"] == 0

        before = client.get(api + "/agent-tree").json()["desiredRevision"]
        for name in ("OrchestratorAgent", "PlannerAgent", "ToolPreparerAgent"):
            assert client.post(api + f"/agents/{name}/enabled", json={"enabled": False}).status_code == 409
        assert client.post(api + "/agents/UnknownAgent/enabled", json={"enabled": False}).status_code == 404
        assert client.post(api + "/agents/ResearchAgent/enabled", json={"enabled": "false"}).status_code == 422
        wrong = f"/api/users/not-the-owner/sessions/{first['id']}/agents/ResearchAgent/enabled"
        assert client.post(wrong, json={"enabled": False}).status_code == 404
        assert client.get(api + "/agent-tree").json()["desiredRevision"] == before


def test_agent_tree_page_and_navigation_are_wired(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web-state"))
    app = create_app()
    with TestClient(app) as client:
        page = client.get("/agent-tree")
        assert page.status_code == 200
        assert "/static/js/agent_tree.js?v=" in page.text
        assert "/static/css/agent_tree.css?v=" in page.text

    from CoScientist.web.app import WEB_DIR
    nav = (WEB_DIR / "static" / "js" / "activity_rail.js").read_text(encoding="utf-8")
    sessions = (WEB_DIR / "static" / "js" / "sessions.js").read_text(encoding="utf-8")
    assert 'name: "AgentTopology"' in nav
    assert "id: \"agent-tree-link\"" in nav
    assert "`/agent-tree?user_id=${encodeURIComponent(user.id)}" in sessions


def test_settings_scope_is_isolated_between_async_tasks():
    base = get_settings()
    first = base.model_copy(deep=True)
    second = base.model_copy(deep=True)
    first.web.start_mode = "orchestrator"
    second.web.start_mode = "orchestrator_planner"

    async def read(value):
        with settings_scope(value):
            await asyncio.sleep(0)
            return get_settings().web.start_mode

    async def scenario():
        return await asyncio.gather(read(first), read(second))

    assert asyncio.run(scenario()) == ["orchestrator", "orchestrator_planner"]
    assert get_settings() is base


def test_accepted_revision_is_kept_until_the_next_request(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web-state"))

    async def scenario():
        runtime = web_app.WebRuntime()
        user = runtime.registry.create_user("Revision tester")
        session = runtime.registry.create_session(user["id"], "Revision")
        key = (user["id"], session["id"])
        old_revision, old_snapshot = runtime.settings_snapshot(key)
        rebuilt = []

        class StubManager:
            settings_override = old_snapshot

            async def rebuild_agent_tree(self):
                rebuilt.append(self.settings_override.web.start_mode)

        manager = StubManager()
        runtime.managers[key] = manager
        runtime.manager_agent_revisions[key] = old_revision
        runtime.save_agent_configuration(
            key,
            {"general": {"startMode": "orchestrator"}},
        )

        # A request accepted before the save keeps the exact old snapshot.
        assert await runtime.get_manager(
            *key,
            settings_snapshot=old_snapshot,
            agent_revision=old_revision,
        ) is manager
        assert rebuilt == []

        # The next request receives the desired revision and rebuilds in place.
        new_revision, new_snapshot = runtime.settings_snapshot(key)
        assert new_revision == old_revision + 1
        assert new_snapshot.web.start_mode == "orchestrator"
        assert await runtime.get_manager(
            *key,
            settings_snapshot=new_snapshot,
            agent_revision=new_revision,
        ) is manager
        assert rebuilt == ["orchestrator"]

    asyncio.run(scenario())


def test_domain_profile_uses_its_yaml_root(monkeypatch):
    from CoScientist.web.agent_tree import project_agent_tree

    monkeypatch.setenv("COSCIENTIST_CONFIG", "microfluidics")
    snapshot = get_settings().model_copy(deep=True)
    snapshot.web.start_mode = "orchestrator"
    tree = project_agent_tree(
        snapshot,
        desired_revision=0,
        active_revision=None,
        running=False,
    )

    assert tree["profile"] == "microfluidics"
    assert "RootOrchestrator" in {node["id"] for node in tree["nodes"]}


@pytest.mark.parametrize("suffix", ["/settings", "/agents/catalog", "/agent-tree", "/agents/ExperimentAgent/tools"])
def test_configuration_reads_reject_foreign_and_missing_sessions(tmp_path, monkeypatch, suffix):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "state"))
    with TestClient(create_app()) as client:
        owner = _create_user(client, "Owner")
        other = _create_user(client, "Other")
        session = _create_session(client, owner["id"], "Private session")
        wrong_owner = f"/api/users/{other['id']}/sessions/{session['id']}"
        missing = f"/api/users/{owner['id']}/sessions/missing"
        assert client.get(wrong_owner + suffix).status_code == 404
        assert client.get(missing + suffix).status_code == 404


def test_mcp_tools_use_catalog_metadata_and_server_identity():
    from CoScientist.assembly.schema import SystemConfig
    from CoScientist.web.agent_tree import agent_tools_payload

    config = SystemConfig.model_validate({"agents": {
        "Reader": {"root": True, "tools": ["paper_analysis"]},
        "React": {"tools": ["dynamic_tools"]},
    }})
    catalog = {
        "servers": [{"id": "papers", "name": "paper-analysis"}, {"id": "other", "name": "Other"}],
        "tools": [
            {"id": "papers:search", "server_id": "papers", "name": "search", "status": "saved",
             "display_name": {"ru": "Найти научные статьи"}, "summary": {"ru": "Поиск в базе статей."}},
            {"id": "other:search", "server_id": "other", "name": "search", "status": "available",
             "display_name": {"ru": "Другой поиск"}, "summary": {"ru": "Другой сервер."}},
        ],
    }
    snapshot = get_settings().model_copy(deep=True)
    snapshot.mcp.paper_analysis_url = "http://papers.invalid/mcp"
    with settings_scope(snapshot):
        tools = agent_tools_payload(config, "Reader", {}, catalog)["tools"]
        assert [tool["id"] for tool in tools] == ["papers:search"]
        assert tools[0]["status"] == "saved"
        assert tools[0]["display_name"]["ru"] == "Найти научные статьи"
        selected = agent_tools_payload(config, "React", {"filtered_tools": [{"server_id": "papers", "name": "search"}]}, catalog)["tools"]
        assert "papers:search" in {tool["id"] for tool in selected}
        assert "other:search" not in {tool["id"] for tool in selected}
        assert selected[0]["pinned"] is True and selected[0]["selected"] is False
        snapshot.mcp.paper_analysis_url = None
        assert agent_tools_payload(config, "Reader", {}, catalog)["tools"] == []
