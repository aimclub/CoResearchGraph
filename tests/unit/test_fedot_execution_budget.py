"""FEDOT's own model factory must participate in the same root quota."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from CoScientist.execution_control import RunController, bind_run
from CoScientist.tools.fedot_mas_patch import ensure_fedot_openai_proxy_compat


def test_fedot_nonproxy_factory_counts_once_and_disables_hidden_retries(monkeypatch, tmp_path):
    async def scenario():
        ensure_fedot_openai_proxy_compat()
        from fedotmas.common.llm import make_llm
        from fedotmas._settings import ModelConfig

        model = make_llm(ModelConfig(model="openai/test-model", api_key="test-key"))
        calls = []
        async def fake(*args, **kwargs):
            calls.append(kwargs)
            return {"ok": True}
        monkeypatch.setattr(model.llm_client.client, "acompletion", fake)
        controller = RunController(tmp_path / "budget.db")
        handle = controller.create_run(initial_budget=1)
        with bind_run(handle):
            result = await model.llm_client.acompletion(model="openai/test-model", messages=[], tools=[])
        assert result == {"ok": True}
        assert calls[0]["num_retries"] == 0
        assert handle.status().attempts_used == 1
    asyncio.run(scenario())


def test_fedot_proxy_is_not_double_counted(monkeypatch, tmp_path):
    async def scenario():
        ensure_fedot_openai_proxy_compat()
        from fedotmas.common.llm import make_llm
        from fedotmas._settings import ModelConfig

        model = make_llm(ModelConfig(model="openai/test-model", api_key="test-key", api_base="http://example.invalid/v1"))
        calls = []
        async def fake(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[], model_dump=lambda: {
                "model": "test-model", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "test"}}]})
        monkeypatch.setattr(model.llm_client._client.chat.completions, "create", fake)
        controller = RunController(tmp_path / "budget.db")
        handle = controller.create_run(initial_budget=1)
        with bind_run(handle):
            await model.llm_client.acompletion(model="openai/test-model", messages=[], tools=[])
        assert handle.status().attempts_used == 1
        assert calls[0]["model"] == "test-model"
        assert model.llm_client._client.max_retries == 0
        await model.llm_client._client.close()
    asyncio.run(scenario())
