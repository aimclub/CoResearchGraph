"""A slow callback must not hold checkpoint capture or the shared event loop."""

import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from CoScientist.checkpoints import capture, synapse
from CoScientist.checkpoints.notifications import SnapshotNotifier
from CoScientist.checkpoints.store import LocalZipStore


def test_slow_delivery_does_not_delay_capture_or_event_loop(
    monkeypatch, tmp_path, notifier
):
    monkeypatch.setattr(
        synapse,
        "_synapse_cfg",
        lambda: SimpleNamespace(enabled=True, callback_url="http://platform.test"),
    )
    monkeypatch.setattr(synapse, "_bundle_base_url", lambda: None)
    monkeypatch.setattr(capture, "_collect_store_parts", dict)
    monkeypatch.setattr(capture, "collect_pins", dict)

    def slow_post(url, **kwargs):
        time.sleep(0.3)
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", slow_post)
    session = SimpleNamespace(
        app_name="app", id="ctx-slow", user_id="u", state={}, events=[]
    )
    store = LocalZipStore(str(tmp_path))

    async def scenario():
        started = time.monotonic()

        async def heartbeat():
            await asyncio.sleep(0.01)
            return time.monotonic() - started

        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        manifest = await capture.capture_checkpoint(
            session=session, label="T0", store=store, validator_pending=False
        )
        capture_seconds = time.monotonic() - started
        lag = await beat
        assert manifest is not None
        assert store.bundle_path(manifest.checkpoint_id).exists()
        assert (
            capture_seconds < 0.2
        ), f"capture waited for callback: {capture_seconds:.3f}s"
        assert lag < 0.2, f"shared event loop blocked: {lag:.3f}s"

    asyncio.run(scenario())


@pytest.fixture
def notifier(monkeypatch):
    from CoScientist.checkpoints import notifications

    worker = SnapshotNotifier()
    monkeypatch.setattr(notifications, "_notifier", worker)
    yield worker
    worker.close()


def test_queue_is_bounded_and_shutdown_is_bounded(monkeypatch, caplog):
    entered, release = threading.Event(), threading.Event()
    delivered = []

    def blocked_post(url, json, timeout):
        entered.set()
        assert release.wait(2)
        delivered.append(json["point_id"])
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", blocked_post)
    worker = SnapshotNotifier(capacity=1)
    try:
        worker.submit("http://platform.test", {"point_id": "one", "run_id": "r"})
        assert entered.wait(1)
        worker.submit("http://platform.test", {"point_id": "two", "run_id": "r"})
        worker.submit("http://platform.test", {"point_id": "overflow", "run_id": "r"})
        assert "queue full" in caplog.text and "overflow" in caplog.text
        started = time.monotonic()
        worker.close(timeout=0.02)
        assert time.monotonic() - started < 0.2
        assert "delivery incomplete" in caplog.text
        worker.submit("http://platform.test", {"point_id": "closed", "run_id": "r"})
        assert "worker closed" in caplog.text
    finally:
        release.set()
        worker.close()
    assert delivered == ["one", "two"]


@pytest.mark.parametrize("failure", ["timeout", "http503"])
def test_delivery_failure_is_visible_and_worker_continues(monkeypatch, caplog, failure):
    delivered = []

    def post(url, json, timeout):
        assert timeout == 5.0
        if json["point_id"] == "failed":
            if failure == "timeout":
                raise httpx.ReadTimeout("secret-callback-url")
            return httpx.Response(503, request=httpx.Request("POST", url))
        delivered.append(json["point_id"])
        return httpx.Response(200, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    worker = SnapshotNotifier()
    worker.submit("http://secret-callback-url", {"point_id": "failed", "run_id": "r"})
    worker.submit("http://secret-callback-url", {"point_id": "next", "run_id": "r"})
    worker.close()
    assert delivered == ["next"]
    assert "delivery failed" in caplog.text
    assert "secret-callback-url" not in caplog.text
