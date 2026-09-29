"""Stages of a linear pipeline — what the web status indicator counts
("Этап k из N"). See SystemConfig.linear_stages and status_indicator.js."""
from fastapi.testclient import TestClient

from CoScientist.assembly.schema import (
    DEFAULT_CONFIG_PATH,
    SystemConfig,
    load_config,
    resolve_config_path,
)


def test_microfluidics_delegates_experiments_as_one_external_stage():
    config = load_config(resolve_config_path("microfluidics"))

    stages = config.linear_stages()

    assert [s["agent"] for s in stages] == [
        "TZSpecAgent", "TZQueryGenAgent", "LiteratureOrchestrator",
        "LiteratureSynthesisAgent", "EvidenceVerifierAgent", "RouteSelectionAgent",
        "MolDesignAgent", "EconomicsAgent", "OptimizationAgent", "ReportAgent",
    ]
    assert stages[0]["title"] == "Техническое задание"
    # Everything working inside a stage counts as that stage.
    by_agent = {s["agent"]: s["members"] for s in stages}
    assert "ResearchAgent" in by_agent["LiteratureOrchestrator"]
    assert "PlannerAgent" not in by_agent
    # ModuleC_Reactor is disabled: the pipeline stops after the optimization campaign.
    assert "ReactorAgent" not in by_agent
    assert stages[-2]["title"] == "Оптимизация условий синтеза"


def test_a_freely_routing_orchestrator_has_no_stages():
    assert load_config(DEFAULT_CONFIG_PATH).linear_stages() == []


def _config(**pipeline):
    return SystemConfig.model_validate({
        "pipeline": pipeline,
        "agents": {
            "Root": {"class": "llm", "root": True, "subordinates": ["Module", "Last"]},
            "Module": {"class": "sequential", "children": ["First", "Off", "Second"]},
            "First": {"class": "llm", "title": "Первый", "subordinates": ["Helper"]},
            "Off": {"class": "llm", "enabled": False},
            "Second": {"class": "llm"},
            "Helper": {"class": "llm"},
            "Last": {"class": "llm", "subordinates": ["Helper"]},
            "Pre": {"class": "llm"},
        },
    })


def test_sequential_modules_unroll_and_disabled_steps_drop_out():
    stages = _config(linear=True, pre=["Pre"]).linear_stages()

    assert [(s["agent"], s["title"]) for s in stages] == [
        ("Pre", "Pre"), ("First", "Первый"), ("Second", "Second"), ("Last", "Last"),
    ]
    # A shared subordinate counts toward the first stage that uses it.
    assert stages[1]["members"] == ["First", "Helper"]
    assert stages[3]["members"] == ["Last"]


def test_not_linear_unless_declared():
    assert _config().linear_stages() == []


def test_the_session_snapshot_carries_the_stages(monkeypatch):
    from CoScientist.web import app as web_app

    stages = [{"agent": "A", "title": "Этап А", "members": ["A"]}]
    monkeypatch.setattr(web_app, "_pipeline_stages", lambda: stages)
    app = web_app.create_app()
    with TestClient(app) as client:
        user = client.post("/api/users", json={"nickname": "Gleb"}).json()["user"]
        session = client.post(
            f"/api/users/{user['id']}/sessions", json={"title": "Work"},
        ).json()["session"]
        with client.websocket_connect(
            f"/ws?user_id={user['id']}&session_id={session['id']}"
        ) as websocket:
            websocket.receive_json()
            snapshot = websocket.receive_json()

    assert snapshot["type"] == "session_snapshot"
    assert snapshot["pipeline_stages"] == stages
