"""Keep Experiment Module unit tests independent of ambient EXPERIMENTS__* flags."""
from __future__ import annotations

import pytest

from CoScientist.config.settings import ExperimentsSettings


@pytest.fixture(autouse=True)
def _experiment_module_defaults(monkeypatch):
    """Keep unit tests independent of ambient EXPERIMENTS__* kill-switches."""
    monkeypatch.setenv("EXPERIMENTS__ROUTE_FEDOT", "true")
    monkeypatch.delenv("EXPERIMENTS__ROUTE_CODER_MCP", raising=False)
    monkeypatch.delenv("EXPERIMENTS__ROUTE_ALEMBIC", raising=False)
    from CoScientist.config import get_settings

    # get_settings() returns an import-time singleton that may already have
    # absorbed EXPERIMENTS__ROUTE_FEDOT=false from the ambient shell.
    experiments = ExperimentsSettings(
        route_fedot=True,
        route_coder_mcp=False,
        route_alembic=False,
    )
    monkeypatch.setattr(get_settings(), "experiments", experiments, raising=False)
    # The suite runs on system.yaml, which has no ExperimentExecutorAgent, so
    # fedot_route_available asks FedotAgent's own `enabled` there - and main
    # gates that on EXECUTOR__FEDOT_FALLBACK, which a shell may have turned off.
    monkeypatch.setattr(get_settings().web, "fedot_fallback_enabled", True)
    # The medical route follows MedicalAgent, which MEDICAL__ENABLED gates.
    monkeypatch.setattr(get_settings().web, "medical_agent_enabled", True)
    yield
