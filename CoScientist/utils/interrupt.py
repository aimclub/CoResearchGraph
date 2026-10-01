"""Make Ctrl+C always terminate the process.

A graceful shutdown alone is not enough. ADK runs synchronous tools in a
``ThreadPoolExecutor`` whose worker threads are non-daemon and are joined at
interpreter exit, so a tool blocked in I/O (an HTTP call, a subprocess, a
parser) keeps the process alive after the event loop has already stopped.
Python cannot kill a thread, only the whole process.

Policy for every entry point:

* the first Ctrl+C (or SIGTERM) starts the normal graceful shutdown and arms a
  watchdog that hard-exits the process if it is still alive after
  ``COSCIENTIST_SHUTDOWN_GRACE`` seconds (default 5);
* a second Ctrl+C exits immediately.
"""
from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
from types import FrameType
from typing import Generator, Optional

import uvicorn

SHUTDOWN_GRACE = float(os.getenv("COSCIENTIST_SHUTDOWN_GRACE", "5"))
EXIT_CODE = 130  # conventional exit status for "terminated by SIGINT"

# No lock: every caller runs on the main thread (signal handlers), and a lock
# held when the second Ctrl+C lands would deadlock the handler.
_armed = False


def _say(message: str) -> None:
    # os.write, not print: this runs inside signal handlers, and print() would
    # raise "reentrant call" if the main thread was interrupted mid-print.
    try:
        os.write(sys.stderr.fileno(), f"\n[CoScientist] {message}\n".encode())
    except (OSError, ValueError, AttributeError):
        pass


def force_exit(code: int = EXIT_CODE) -> None:
    """Terminate now, skipping thread joins and pending cleanup."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001 — exiting regardless
            pass
    os._exit(code)


def _watchdog(grace: float) -> None:
    _say(f"Graceful shutdown did not finish in {grace:g}s; forcing exit.")
    force_exit()


def request_exit(
    grace: float = SHUTDOWN_GRACE, sig: int = signal.SIGINT
) -> None:
    """Register one interrupt: arm the watchdog on the first, exit on the next.

    Only a repeated SIGINT forces the exit. A SIGTERM after Ctrl+C is not a
    user asking twice: the --reload supervisor sends one to its worker, which
    also got the terminal's SIGINT.
    """
    global _armed
    if _armed:
        if sig == signal.SIGINT:
            _say("Forced exit.")
            force_exit()
        return
    _armed = True
    _say(f"Shutting down (Ctrl+C again to force; forced in {grace:g}s)...")
    timer = threading.Timer(grace, _watchdog, args=(grace,))
    # A daemon thread keeps running while the interpreter joins the stuck
    # non-daemon tool threads, which is exactly when it has to fire.
    timer.daemon = True
    timer.start()


def install_sigint_exit() -> None:
    """SIGINT/SIGTERM handler for non-uvicorn entry points (e.g. the REPL).

    Arms the watchdog and raises ``KeyboardInterrupt``. Install it before
    ``asyncio.run``: the runner only adds its own SIGINT handler when the
    default one is still in place.
    """

    def _handler(sig: int, frame: Optional[FrameType]) -> None:
        request_exit(sig=sig)
        raise KeyboardInterrupt

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)


class InterruptibleServer(uvicorn.Server):
    """``uvicorn.Server`` whose shutdown is guaranteed to end the process."""

    def handle_exit(self, sig: int, frame: Optional[FrameType]) -> None:
        request_exit(sig=sig)
        super().handle_exit(sig, frame)


class SharedSignalServer(uvicorn.Server):
    """``uvicorn.Server`` that leaves signal handling to its owner.

    uvicorn ignores ``config.install_signal_handlers`` and makes every
    ``serve()`` replace the process-wide SIGINT handler with its own. When
    several servers share one process, only the last one to start would then
    see Ctrl+C. The owner installs one handler and stops all servers instead.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield
