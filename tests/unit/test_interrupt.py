import signal

import pytest

from CoScientist.utils import interrupt


class _Timer:
    started: list = []

    def __init__(self, delay, fn, args=()):
        self.delay, self.fn, self.args = delay, fn, args
        self.daemon = False

    def start(self):
        _Timer.started.append(self)


class _Exited(Exception):
    pass


@pytest.fixture
def policy(monkeypatch):
    _Timer.started = []
    monkeypatch.setattr(interrupt, "_armed", False)
    monkeypatch.setattr(interrupt.threading, "Timer", _Timer)

    def fake_exit(code=interrupt.EXIT_CODE):
        raise _Exited(code)

    monkeypatch.setattr(interrupt, "force_exit", fake_exit)
    return _Timer.started


def test_first_interrupt_arms_daemon_watchdog(policy):
    interrupt.request_exit(grace=3)
    assert len(policy) == 1
    timer = policy[0]
    assert timer.delay == 3 and timer.daemon
    with pytest.raises(_Exited):
        timer.fn(*timer.args)


def test_second_sigint_exits_immediately(policy):
    interrupt.request_exit()
    with pytest.raises(_Exited):
        interrupt.request_exit(sig=signal.SIGINT)


def test_sigterm_after_sigint_keeps_graceful_shutdown(policy):
    interrupt.request_exit(sig=signal.SIGINT)
    interrupt.request_exit(sig=signal.SIGTERM)
    assert len(policy) == 1


def test_shared_signal_server_leaves_handlers_alone():
    import uvicorn

    before = signal.getsignal(signal.SIGINT)
    server = interrupt.SharedSignalServer(uvicorn.Config(app=None))
    with server.capture_signals():
        assert signal.getsignal(signal.SIGINT) is before
