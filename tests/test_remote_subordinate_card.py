import asyncio

from CoScientist.a2a.config import AGENT_CARD_URLS, AGENT_URLS
from CoScientist.assembly.assembler import _subordinate_instance
from CoScientist.assembly.schema import get_config


def test_remote_subordinate_resolves_docker_host_without_fetching_card(monkeypatch):
    sub = get_config().agent("ResearchAgent")
    endpoint = "http://coscientist-real:8003/"
    monkeypatch.setitem(AGENT_URLS, sub.a2a.key, endpoint)
    monkeypatch.setitem(
        AGENT_CARD_URLS, sub.a2a.key, endpoint + ".well-known/agent.json"
    )
    remote = _subordinate_instance(sub, {}, remote_subagents=True)

    async def resolve():
        try:
            return await remote._ensure_resolved()
        finally:
            await remote.cleanup()

    client = asyncio.run(resolve())

    assert client is not None
    assert remote._agent_card.url == endpoint
