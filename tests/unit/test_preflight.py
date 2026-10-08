"""The preflight checklist: know before the run, not an hour in.

Probes are injected, so the check aggregation, env gating, exit code, and render
are verified without a VPN, Docker, or the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "scripts" / "preflight.py"
_spec = importlib.util.spec_from_file_location("preflight", _MOD_PATH)
pf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pf
_spec.loader.exec_module(pf)


def _all_ok_probes():
    return (
        lambda h, p: (True, f"{h}:{p} ok"),  # tcp
        lambda c, a: (True, f"ctx {c} ok"),  # docker
        lambda u: (True, f"{u} ok"),  # http
    )


def test_env_check_flags_missing():
    res = pf.check_env(["PRESENT", "MISSING"], {"PRESENT": "x"})
    by = {r.name: r for r in res}
    assert by["env PRESENT"].ok is True
    assert by["env MISSING"].ok is False and "MISSING" in by["env MISSING"].detail


def test_env_check_treats_blank_as_missing():
    res = pf.check_env(["BLANK"], {"BLANK": "   "})
    assert res[0].ok is False


def test_run_preflight_all_green():
    tcp, docker, http = _all_ok_probes()
    spec = pf.PreflightSpec(
        required_env=["OPENROUTER_API_KEY"],
        hosts=[("build-host2.nsslab", 22), ("gpu-host", 2376)],
        contexts=[("bdgx", "1.43")],
        mcp_urls=["http://gpu-host:20001/mcp"],
    )
    report = pf.run_preflight(
        spec, env={"OPENROUTER_API_KEY": "k"}, tcp=tcp, docker=docker, http=http
    )
    assert report.ok is True
    assert report.exit_code == 0
    assert len(report.results) == 5  # 1 env + 2 hosts + 1 ctx + 1 mcp
    assert "READY" in report.render() and "5/5" in report.render()


def test_run_preflight_one_failure_is_red_and_nonzero():
    _tcp, docker, http = _all_ok_probes()

    # dgx reachable, build-host down
    def tcp_mixed(h, p):
        return (h != "build-host2.nsslab", f"{h} probed")

    spec = pf.PreflightSpec(
        required_env=["OPENROUTER_API_KEY"],
        hosts=[("build-host2.nsslab", 22), ("gpu-host", 2376)],
        contexts=[("bdgx", "1.43")],
    )
    report = pf.run_preflight(
        spec, env={"OPENROUTER_API_KEY": "k"}, tcp=tcp_mixed, docker=docker, http=http
    )
    assert report.ok is False
    assert report.exit_code == 1
    assert "NOT READY" in report.render()
    failed = [r for r in report.results if not r.ok]
    assert len(failed) == 1 and "build-host2.nsslab" in failed[0].name


def test_run_preflight_missing_env_fails_even_if_infra_up():
    tcp, docker, http = _all_ok_probes()
    spec = pf.PreflightSpec(
        required_env=["OPENROUTER_API_KEY"], contexts=[("bdgx", "1.43")]
    )
    report = pf.run_preflight(spec, env={}, tcp=tcp, docker=docker, http=http)
    assert report.ok is False and report.exit_code == 1


def test_docker_probe_pins_api_version(monkeypatch):
    seen = {}

    def fake_runner(cmd, **kw):
        seen["cmd"] = cmd
        seen["api"] = kw.get("env", {}).get("DOCKER_API_VERSION")

        class R:
            returncode = 0
            stdout = "24.0.2"
            stderr = ""

        return R()

    ok, detail = pf.docker_probe("bdgx", "1.43", runner=fake_runner)
    assert ok is True
    assert seen["cmd"][:3] == ["docker", "--context", "bdgx"]
    assert seen["api"] == "1.43"
    assert "24.0.2" in detail and "API 1.43" in detail


def test_docker_probe_reports_daemon_error():
    def failing(cmd, **kw):
        class R:
            returncode = 1
            stdout = ""
            stderr = "client version 1.52 is too new"

        return R()

    ok, detail = pf.docker_probe("bdgx", None, runner=failing)
    assert ok is False and "too new" in detail
