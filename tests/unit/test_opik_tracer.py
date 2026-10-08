import os
import pytest
from opik.integrations.adk import OpikTracer, track_adk_agent_recursive
from CoScientist.config import get_settings
from CoScientist.logging.opik_tracer import get_multi_agent_tracer
from CoScientist.assembly import build_system
from CoScientist.agents import build_for_mode


def test_opik_tracer_disabled(monkeypatch):
    monkeypatch.setenv("OPIK__ENABLED", "false")
    settings = get_settings()
    monkeypatch.setattr(settings.web, "opik_enabled", False)
    monkeypatch.setattr(settings.opik, "enabled", False)

    tracer = get_multi_agent_tracer()
    assert tracer is None
    assert os.getenv("OPIK_TRACK_DISABLE") == "true"


def test_opik_tracer_enabled_env_vars(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.web, "opik_enabled", True)
    monkeypatch.setattr(settings.opik, "enabled", True)
    monkeypatch.setattr(settings.opik, "api_key", "test_key_123")
    monkeypatch.setattr(settings.opik, "url_override", "https://opik.custom.internal/api")
    monkeypatch.setattr(settings.opik, "opik_project_name", "adk-coscientist")

    tracer = get_multi_agent_tracer()
    assert tracer is not None
    assert isinstance(tracer, OpikTracer)
    assert os.getenv("OPIK_API_KEY") == "test_key_123"
    assert os.getenv("OPIK_URL_OVERRIDE") == "https://opik.custom.internal/api"
    assert os.getenv("OPIK_PROJECT_NAME") == "adk-coscientist"
    assert os.getenv("OPIK_TRACK_DISABLE") is None


def test_agent_system_run_root_cached():
    system = build_system()
    run_root1 = system.run_root
    run_root2 = system.run_root
    assert run_root1 is run_root2


def test_build_for_mode_tracks_run_root(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings.web, "opik_enabled", True)
    monkeypatch.setattr(settings.opik, "enabled", True)
    monkeypatch.setattr(settings.opik, "opik_project_name", "adk-coscientist")

    system = build_for_mode()
    run_root = system.run_root

    # Verify that run_root has Opik callbacks attached
    assert run_root.before_agent_callback is not None
    assert run_root.after_agent_callback is not None

    # If run_root has sub_agents (e.g. SequentialAgent ResearchPipeline), verify they also have callbacks
    if hasattr(run_root, "sub_agents"):
        for sub_agent in run_root.sub_agents:
            assert sub_agent.before_agent_callback is not None
            assert sub_agent.after_agent_callback is not None
