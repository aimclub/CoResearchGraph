"""Provider-facing wrappers reserve each actual attempt against one run."""
import asyncio
from types import SimpleNamespace

import litellm

from CoScientist.agents import common
from CoScientist.execution_control import RunController, bind_run
from CoScientist.graph import agent_summary


def test_retrying_adk_wrapper_counts_failed_attempts_and_disables_sdk_retries(
    tmp_path, monkeypatch
):
    calls = []

    async def flaky(self, llm_request, stream=False):
        calls.append(stream)
        if len(calls) == 1:
            raise TimeoutError("request timed out")
        yield "answer"

    monkeypatch.setattr(common.LiteLlm, "generate_content_async", flaky)
    monkeypatch.setattr(common, "_proxy_verified", True)
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run("r")
    llm = common.make_llm()

    async def scenario():
        with bind_run(handle):
            return [item async for item in llm.generate_content_async(object())]

    assert asyncio.run(scenario()) == ["answer"]
    assert controller.status("r").attempts_used == 2
    assert llm._additional_args["num_retries"] == 0


def test_direct_summary_fallback_reserves_both_provider_attempts(
    tmp_path, monkeypatch
):
    import CoScientist.config as config_module

    settings = SimpleNamespace(llm=SimpleNamespace(
        agent_summary_model="openrouter/retired",
        summary_url=None,
        main_model="openrouter/main",
        main_url="http://main",
        openai_api_key="key",
        request_timeout=5,
    ))
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    calls = []

    async def fake_completion(**kwargs):
        calls.append(kwargs)
        if "retired" in kwargs["model"]:
            raise litellm.NotFoundError(
                "retired", model=kwargs["model"], llm_provider="openrouter"
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(litellm, "acompletion", fake_completion)
    controller = RunController(tmp_path / "control.sqlite3")
    handle = controller.create_run("r")

    async def scenario():
        with bind_run(handle):
            return await agent_summary._complete("system", "trace")

    assert asyncio.run(scenario()) == ("ok", "openrouter/main")
    assert controller.status("r").attempts_used == 2
    assert [call["num_retries"] for call in calls] == [0, 0]


def test_pause_gate_precedes_proxy_network_probe(tmp_path, monkeypatch):
    probes = []

    async def probe(force=False):
        probes.append(force)
        monkeypatch.setattr(common, "_proxy_verified", True)

    async def answer(self, llm_request, stream=False):
        yield "answer"

    monkeypatch.setattr(common, "_proxy_verified", False)
    monkeypatch.setattr(common, "verify_proxy_reachable", probe)
    monkeypatch.setattr(common.LiteLlm, "generate_content_async", answer)
    controller = RunController(tmp_path / "control.sqlite3", poll_interval=0.01)
    handle = controller.create_run("r")
    controller.request_pause("r", "manual")
    llm = common.make_llm()

    async def scenario():
        with bind_run(handle):
            task = asyncio.create_task(
                _collect(llm.generate_content_async(object()))
            )
            await asyncio.sleep(0.03)
            assert probes == []
            assert controller.status("r").attempts_used == 0
            controller.resume("r", "manual")
            assert await asyncio.wait_for(task, 1) == ["answer"]

    asyncio.run(scenario())
    assert probes == [False]
    assert controller.status("r").attempts_used == 1


async def _collect(source):
    return [item async for item in source]
