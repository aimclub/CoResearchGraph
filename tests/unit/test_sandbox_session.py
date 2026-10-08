"""One OpenHands sandbox per session, reused across calls.

The sandbox is a stateful machine: an experiment is built up over several
calls, so repeated calls in one session must land in the SAME sandbox, a
different session must get its own, and a clean one must be reachable on
demand. The sandbox rides on the coder workspace id, so the pins that govern
the code workspace (A2A ``CODER_WORKSPACE_ID``, ADK session state) govern it
too.
"""
import asyncio
import inspect
import os
from types import SimpleNamespace

import pytest
from dotenv import load_dotenv

load_dotenv()

from CoScientist.tools.coder_tools import CoderToolset, _WORKSPACE_STATE_KEY  # noqa: E402
from CoScientist.tools.coder_tools import openhands_sandbox as client  # noqa: E402
from CoScientist.tools.coder_tools import sandbox_tools  # noqa: E402


def _ctx(state, session_id="adk-session-1"):
    """A CoderAgent sub-session sharing the forwarded top-level state."""
    return SimpleNamespace(
        state=state,
        _invocation_context=SimpleNamespace(
            session=SimpleNamespace(id="sub-" + os.urandom(3).hex())
        ),
    )


def setup_function():
    os.environ.pop("CODER_WORKSPACE_ID", None)


# ── session keying ───────────────────────────────────────────────────────────

def test_sandbox_shares_the_coder_workspace_identity():
    state = {}
    expected = CoderToolset._workspace_id(_ctx(state))
    assert sandbox_tools._session(_ctx(state)) == expected
    assert state[_WORKSPACE_STATE_KEY] == expected


def test_repeated_calls_in_one_session_share_a_key():
    state = {}
    keys = {sandbox_tools._session(_ctx(state)) for _ in range(3)}
    assert len(keys) == 1


def test_sessions_get_different_keys():
    assert sandbox_tools._session(_ctx({})) != sandbox_tools._session(_ctx({}))


def test_a2a_pin_wins_so_multi_step_work_survives_per_call_sessions():
    """Over A2A each delegation is a fresh session; the pin keeps one sandbox."""
    os.environ["CODER_WORKSPACE_ID"] = "a2a_shared"
    try:
        assert sandbox_tools._session(_ctx({})) == sandbox_tools._session(_ctx({}))
    finally:
        os.environ.pop("CODER_WORKSPACE_ID", None)


# ── binding lifecycle (client level, no server) ──────────────────────────────

def test_binding_is_per_session_and_rebindable():
    client.clear_binding("sess-1")
    client.clear_binding("sess-2")

    assert client.read_binding("sess-1") is None       # new session -> no sandbox
    client.write_binding("sess-1", "sandbox-a")
    client.write_binding("sess-2", "sandbox-b")

    assert client.read_binding("sess-1") == "sandbox-a"  # reused by default
    assert client.read_binding("sess-2") == "sandbox-b"  # isolated

    client.clear_binding("sess-1")                       # new_sandbox / reset
    assert client.read_binding("sess-1") is None
    assert client.read_binding("sess-2") == "sandbox-b"


def test_state_pin_survives_a_sub_session_without_registry_knowledge():
    ctx = _ctx({})
    session = sandbox_tools._session(ctx)
    client.write_binding(session, "sandbox-x", ctx)

    sub = _ctx(dict(ctx.state))          # AgentTool copies parent state down
    client._REGISTRY.drop(session)       # ...but the process registry is cold
    assert client.read_binding(session, sub) == "sandbox-x"


# ── ADK adapter contract ─────────────────────────────────────────────────────

def test_tools_are_async_so_they_never_block_the_event_loop():
    """ADK calls a sync function tool directly on the loop; a blocking sandbox
    wait would freeze the web UI, log streaming and every other agent."""
    for tool in (sandbox_tools.run_sandbox_task,
                 sandbox_tools.check_sandbox_task,
                 sandbox_tools.list_sandbox_files):
        assert inspect.iscoroutinefunction(tool), tool.__name__


def test_run_passes_the_session_and_bounded_wait_through(monkeypatch):
    captured = {}

    async def fake_run(task, **kwargs):
        captured.update(kwargs, task=task)
        return {"status": "cooldown", "summary": "ok", "sandbox_id": "s1",
                "reused": True, "watch_url": "w", "vscode_url": "v"}

    monkeypatch.setattr(sandbox_tools.sandbox, "arun_sandbox_task", fake_run)

    ctx = _ctx({})
    result = asyncio.run(sandbox_tools.run_sandbox_task("train it", tool_context=ctx))

    assert captured["session_id"] == sandbox_tools._session(ctx)
    assert captured["timeout"] == sandbox_tools.RUN_WAIT  # bounded, not forever
    assert captured["new_sandbox"] is False               # reuse is the default
    assert result["status"] == "success"
    assert result["summary"] == "ok"


# ── live links (they are only useful while the sandbox runs) ─────────────────

def _stub_server(monkeypatch, order):
    """A sandbox server that accepts the task, then a run that never ends here."""

    class _Accepted:
        status_code = 200

        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"task_id": "s1", "watch_url": "w", "vscode_url": "v"}

    async def fake_post(self, url, json=None):
        order.append("submit")
        return _Accepted()

    async def fake_wait(**kwargs):
        order.append("wait")
        return {"status": "cooldown", "summary": "done"}

    monkeypatch.setenv("SANDBOX_URL", "http://sandbox.test")
    monkeypatch.setattr(client.httpx.AsyncClient, "post", fake_post)
    monkeypatch.setattr(client, "_await_completion", fake_wait)


def test_client_announces_the_sandbox_before_it_starts_waiting(monkeypatch):
    """The call returns when the job is over; a watcher must be told earlier."""
    order = []
    seen = []
    _stub_server(monkeypatch, order)

    result = asyncio.run(client.arun_sandbox_task(
        "train it",
        session_id="sess-links",
        on_start=lambda info: (order.append("announce"), seen.append(info)),
    ))

    assert order == ["submit", "announce", "wait"]
    assert seen[0]["watch_url"] == "w" and seen[0]["vscode_url"] == "v"
    assert result["status"] == "cooldown"


def test_a_failing_watcher_never_takes_the_run_down(monkeypatch):
    """Nobody watching (dead socket, no tab) is not a reason to lose the job."""
    order = []
    _stub_server(monkeypatch, order)

    def boom(info):
        raise RuntimeError("no socket")

    result = asyncio.run(client.arun_sandbox_task(
        "train it", session_id="sess-links", on_start=boom,
    ))

    assert order == ["submit", "wait"]
    assert result["status"] == "cooldown"


def test_links_reach_the_host_session_while_the_task_still_runs(monkeypatch):
    """The tool hands the URLs to the host (Web UI) mid-run, not on return."""
    delivered = []

    async def fake_run(task, **kwargs):
        await kwargs["on_start"]({
            "sandbox_id": "s1", "watch_url": "w", "vscode_url": "v", "reused": False,
        })
        return {"status": "cooldown", "summary": "ok"}

    async def sink(host_key, info):
        delivered.append((host_key, info))

    monkeypatch.setattr(sandbox_tools.sandbox, "arun_sandbox_task", fake_run)
    sandbox_tools.set_sandbox_start_sink(sink)
    try:
        ctx = _ctx({"graph_scope_user_id": "u1", "graph_scope_session_id": "web-1"})
        asyncio.run(sandbox_tools.run_sandbox_task("t", tool_context=ctx))
    finally:
        sandbox_tools.set_sandbox_start_sink(None)

    assert delivered == [(("u1", "web-1"), {
        "sandbox_id": "s1", "watch_url": "w", "vscode_url": "v", "reused": False,
    })]


def test_no_sink_means_the_tool_behaves_exactly_as_before(monkeypatch):
    """CLI / A2A hosts register nothing; the notifier must stay a no-op."""
    async def fake_run(task, **kwargs):
        await kwargs["on_start"]({"sandbox_id": "s1", "watch_url": "w"})
        return {"status": "cooldown", "summary": "ok"}

    monkeypatch.setattr(sandbox_tools.sandbox, "arun_sandbox_task", fake_run)
    sandbox_tools.set_sandbox_start_sink(None)
    result = asyncio.run(sandbox_tools.run_sandbox_task("t", tool_context=_ctx({})))
    assert result["status"] == "success"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"status": "cooldown"}, "success"),
        ({"status": "completed"}, "success"),
        ({"status": "timeout"}, "running"),
        ({"status": "submitted"}, "running"),
        ({"status": "busy"}, "busy"),
        ({"status": "error"}, "error"),
        ({"status": "cancelled"}, "error"),
    ],
)
def test_status_is_mapped_to_the_coder_vocabulary(raw, expected):
    assert sandbox_tools._shape(raw, waited=60)["status"] == expected


def test_running_result_tells_the_model_how_to_pick_it_up():
    shaped = sandbox_tools._shape({"status": "timeout"}, waited=600)
    assert "check_sandbox_task" in shaped["next_step"]


def test_expiry_is_surfaced_so_the_model_knows_the_files_are_gone():
    shaped = sandbox_tools._shape(
        {"status": "cooldown", "sandbox_expired": True}, waited=60,
    )
    assert "NEW empty one" in shaped["note"]


def test_tools_drop_out_when_no_sandbox_is_configured(monkeypatch):
    monkeypatch.setattr(sandbox_tools, "sandbox_configured", lambda: False)
    assert sandbox_tools.get_sandbox_tools() == []


# ── The dataset archive attached in the web UI ────────────────────────────────
# The user attaches one .zip per session. It reaches the agent as a prompt
# section, not as an auto-filled argument: passing it to the sandbox is the
# agent's own call, since only it knows whether a given step needs that data.

def _dataset_ctx(url):
    from CoScientist.agents.callbacks import tool_callbacks
    state = {tool_callbacks.DATASET_URL_STATE_KEY: url} if url else {}
    return SimpleNamespace(state=state)


def test_dataset_prompt_block_names_the_link_and_is_empty_without_one():
    from CoScientist.agents.callbacks import inject_dataset_context
    from CoScientist.agents.callbacks.tool_callbacks import DATASET_CONTEXT_STATE_KEY

    url = "https://example.org/data/dataset.zip"
    attached = _dataset_ctx(url)
    inject_dataset_context(attached)
    block = attached.state[DATASET_CONTEXT_STATE_KEY]
    assert url in block and "dataset_url" in block

    empty = _dataset_ctx("")
    inject_dataset_context(empty)
    assert empty.state[DATASET_CONTEXT_STATE_KEY] == ""


@pytest.mark.parametrize("coder_mode", ["local", "openhands"])
def test_the_dataset_block_reaches_the_coder_in_both_tool_setups(coder_mode):
    """Since nothing auto-fills the argument, the prompt is the ONLY way in.

    The coder has two prompts — one for the local toolset, one for the
    sandbox-relay setup the web UI switch leaves it in — and the relay one used
    to omit the block, so an attached archive was invisible to the very agent
    that had to forward it.
    """
    from CoScientist.agents import build_for_mode
    from CoScientist.config import get_settings

    settings = get_settings()
    previous = settings.web.coder_mode
    settings.web.coder_mode = coder_mode
    try:
        system = build_for_mode()
        for name in ("CoderAgent", "DatasetCollectorAgent"):
            assert "{dataset_context?}" in system.agents[name].instruction, (
                f"{name} cannot see the attached archive with "
                f"coder_mode={coder_mode}"
            )
    finally:
        settings.web.coder_mode = previous


def test_nothing_fills_dataset_url_in_behind_the_agent():
    """No callback may pre-fill the argument — the agent sends it or it stays out."""
    from CoScientist.agents import build_for_mode

    system = build_for_mode()
    for name in ("CoderAgent", "DatasetCollectorAgent"):
        callbacks = system.agents[name].canonical_before_tool_callbacks
        assert not any(
            "dataset" in getattr(cb, "__name__", "") for cb in callbacks
        ), f"{name} must not auto-attach the dataset link"
