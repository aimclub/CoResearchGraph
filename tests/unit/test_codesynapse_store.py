import asyncio

import pytest

from CoScientist.integrations.codesynapse.models import A2ATaskRecord, IntegrationRun, TraceEvent
from CoScientist.integrations.codesynapse.store import DuplicateIdentityError, InMemoryIntegrationStore


def test_store_enforces_unique_external_runs_and_event_ids():
    async def scenario():
        store = InMemoryIntegrationStore()
        run = IntegrationRun(external_run_id="external-1", tenant_id="tenant-1", project_id="project-1")
        await store.create_run(run)

        with pytest.raises(DuplicateIdentityError):
            await store.create_run(run)

        event = TraceEvent(
            event_id="event-1", run_id="run-1", sequence=1,
            tenant_id="tenant-1", project_id="project-1", type="run.started",
        )
        assert await store.append_event(event)
        assert not await store.append_event(event)

    asyncio.run(scenario())


def test_store_replays_events_in_sequence_order():
    async def scenario():
        store = InMemoryIntegrationStore()
        for sequence in (2, 1, 3):
            await store.append_event(
                TraceEvent(
                    event_id=f"event-{sequence}", run_id="run-1", sequence=sequence,
                    tenant_id="tenant-1", project_id="project-1", type="tool.completed",
                )
            )

        replay = await store.replay_events("run-1", after_sequence=1)
        assert [event.sequence for event in replay] == [2, 3]

    asyncio.run(scenario())


def test_store_allows_only_one_live_mongo_style_run_lease():
    async def scenario():
        store = InMemoryIntegrationStore()
        assert await store.acquire_run_lease("external-1", "worker-a", 60)
        assert not await store.acquire_run_lease("external-1", "worker-b", 60)
        await store.release_run_lease("external-1", "worker-a")
        assert await store.acquire_run_lease("external-1", "worker-b", 60)

    asyncio.run(scenario())
