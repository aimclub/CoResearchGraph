"""Bounded, best-effort snapshot notifications outside the agent event loop."""

from __future__ import annotations

import atexit
import logging
import queue
import threading

import httpx

logger = logging.getLogger(__name__)


class SnapshotNotifier:
    """One process worker, at most capacity queued requests plus one in flight.

    Local bundles are authoritative. Delivery is one attempt per item; overflow,
    HTTP failures and incomplete shutdown are logged, never hidden or retried.
    """

    def __init__(self, capacity: int = 64) -> None:
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="snapshot-notifications",
            daemon=True,
        )
        self._thread.start()

    def submit(self, url: str, body: dict) -> None:
        with self._lock:
            if self._closed.is_set():
                logger.warning(
                    "[SNAPSHOT_NOTIFY] point_id=%s — worker closed; bundle remains local",
                    body["point_id"],
                )
                return
            try:
                self._queue.put_nowait((url, body))
            except queue.Full:
                logger.warning(
                    "[SNAPSHOT_NOTIFY] point_id=%s — queue full; bundle remains local",
                    body["point_id"],
                )

    def _run(self) -> None:
        while not self._closed.is_set() or not self._queue.empty():
            try:
                url, body = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                response = httpx.post(url, json=body, timeout=5.0)
                response.raise_for_status()
                logger.info(
                    "[SNAPSHOT_NOTIFY] point_id=%s run_id=%s — delivered",
                    body["point_id"],
                    body["run_id"],
                )
            except Exception as exc:  # noqa: BLE001 — never kill the delivery worker
                # Callback URLs can contain secrets. Never log URL/body/exception text.
                status = (
                    exc.response.status_code
                    if isinstance(exc, httpx.HTTPStatusError)
                    else None
                )
                logger.warning(
                    "[SNAPSHOT_NOTIFY] point_id=%s error=%s status=%s — delivery failed; bundle remains local",
                    body["point_id"],
                    type(exc).__name__,
                    status,
                )
            finally:
                self._queue.task_done()

    def close(self, timeout: float = 5.0) -> None:
        """Drain on normal process exit for at most timeout seconds."""
        with self._lock:
            self._closed.set()
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning(
                "[SNAPSHOT_NOTIFY] queued=%s — shutdown deadline exceeded; delivery incomplete",
                self._queue.qsize(),
            )


_notifier: SnapshotNotifier | None = None
_lock = threading.Lock()


def enqueue_snapshot(url: str, body: dict) -> None:
    global _notifier
    with _lock:
        if _notifier is None:
            _notifier = SnapshotNotifier()
            atexit.register(_notifier.close)
        notifier = _notifier
    notifier.submit(url, body)
