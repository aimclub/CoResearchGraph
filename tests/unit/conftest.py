"""Keep unit tests off state the developer actually uses.

Three stores persist to disk: the session registry (WEB_STATE_DIR, default
graph_runs/web_state), the sandbox bindings (SANDBOX_BINDINGS_FILE, default
graph_runs/sandbox_bindings.json) and the default research blackboard
(RESEARCH_GRAPH__DIR, default graph_runs/research_active.json). Without
isolation a test both pollutes real state and inherits from earlier runs —
nickname-uniqueness tests failed with 409 on a second run, binding tests wrote
fake container ids into the file the running system reads to decide which
sandbox to continue in, and the hypothesis-commit tests read whatever
hypotheses the developer's own session happened to hold that minute.
"""
import os
import tempfile

import pytest

# The unit suite asserts about the DEFAULT profile (CoScientist/agents/system.yaml):
# PlannerAgent's roster, the plan critic's agent list, the aggregator's prompt.
# `COSCIENTIST_CONFIG` selects the profile and is read from `.env` like any other
# setting, so a developer who pins a profile there — a perfectly reasonable thing
# to do to run the web UI on `experiments` — silently repointed the whole suite
# and got eighteen failures about agents that profile does not have.
#
# Set before collection, because `get_config()` is lru_cached and a test module
# can call it at import time. Tests that want another profile still load it
# explicitly (`load_config(resolve_config_path("experiments"))`), which does not
# consult the environment.
# A conftest is imported before any test module, so nothing has imported
# CoScientist — and called get_config() — yet. Clearing the cache here instead
# would mean importing the app at configure time, which builds the whole agent
# system before the isolation fixtures below have run.
os.environ["COSCIENTIST_CONFIG"] = ""

# Same reason, same timing: `research_graph` is a module-level singleton built
# at import, so its directory has to be redirected before anything imports
# CoScientist. A fixture would be too late — the store would already have read
# the developer's live blackboard, and a test that commits a hypothesis would
# read back whatever their running session holds.
# A fresh directory per run, not a fixed name: a shared one would hand the next
# run whatever this one committed, which is the same complaint one line up.
os.environ["RESEARCH_GRAPH__DIR"] = tempfile.mkdtemp(prefix="coscientist_graph_")

# Imported after the two environment writes above, on purpose: this is the
# first import of CoScientist in the run, and it reads both of them.
from CoScientist.config import get_settings


@pytest.fixture(autouse=True)
def _isolated_web_state(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web_state"))
    monkeypatch.setenv("SANDBOX_BINDINGS_FILE",
                       str(tmp_path / "sandbox_bindings.json"))


@pytest.fixture(autouse=True)
def _let_caplog_see_our_logs():
    """Let pytest's caplog capture the application's own logger.

    ``CoScientist.logging.logger`` sets ``propagate = False`` deliberately: the
    application owns its file handler and must not hijack the root logger for
    every third-party library. But caplog listens on root, so with propagation
    off a test asserting on a warning we do emit sees an empty log — and which
    tests hit it depends on whether anything imported the logging module first,
    which is not something a test should depend on.
    """
    import logging

    app_logger = logging.getLogger("CoScientist")
    previous = app_logger.propagate
    app_logger.propagate = True
    try:
        yield
    finally:
        app_logger.propagate = previous


@pytest.fixture(autouse=True)
def _auth_disabled_by_default(monkeypatch):
    """Let every other test go on testing what it was written to test.

    ``create_app`` now installs a deny-by-default gate, so a TestClient without
    a session gets 401 on every route. Tests about graph scoping or report
    links have nothing to say about authentication, and threading a cookie
    through each of them would only obscure what they assert.

    The gate itself is covered by tests/unit/test_web_auth.py, whose own
    autouse fixture turns it back on — a module fixture runs after this one, so
    it wins.
    """
    monkeypatch.setattr(get_settings().auth, "enabled", False)
