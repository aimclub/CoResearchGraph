"""The MCP hub on Docker Hub: settings from .env, the description that carries a
server's metadata, the checks and pushes of an upload, a pull that becomes a
build record, and the hub step of build_mcp_server. Docker, Docker Hub and the
network are fakes.
"""
import asyncio
import json
import subprocess
import types
import urllib.error
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from CoScientist.alembic.web import build_api
from CoScientist.tools import alembic_hub as hub
from CoScientist.tools import alembic_tools as at

_MORDRED = "https://github.com/mordred-descriptor/mordred"
_META = {"repo_url": _MORDRED, "job_id": "mordred-babc36",
         "tools": [{"name": "calculate_descriptors", "description": "Mordred descriptors | all 1613"}],
         "tool_counts": {"tools_total": 3, "tools_passed": 3}, "uploaded_at": 1}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Write the project .env the hub reads."""
    path = tmp_path / ".env"

    def write(**values):
        path.write_text("".join(f"{k}={v}\n" for k, v in values.items()), encoding="utf-8")
        monkeypatch.setattr(hub, "ENV_FILE", path)

    monkeypatch.setattr(hub, "_cache", {})
    return write


@pytest.fixture
def builds(tmp_path, monkeypatch):
    monkeypatch.setattr(at, "LOG_DIR", tmp_path / "builds")
    monkeypatch.setattr(at, "_JOBS", {})
    (tmp_path / "builds").mkdir()


# ── settings and names ───────────────────────────────────────────────────────

def test_credentials_come_from_the_env_file_only(env, monkeypatch):
    monkeypatch.setenv("DOCKERHUB_USERNAME", "from-shell")
    monkeypatch.setenv("DOCKERHUB_TOKEN", "shell-token")
    env(DOCKERHUB_USERNAME="HubUser", DOCKERHUB_TOKEN="dckr_pat_x")

    cfg = hub.HubConfig.load()
    assert (cfg.namespace, cfg.username, cfg.can_upload) == ("hubuser", "HubUser", True)

    env(DOCKERHUB_NAMESPACE="someone-else")
    cfg = hub.HubConfig.load()
    assert (cfg.namespace, cfg.can_upload) == ("someone-else", False)  # browsing needs no credentials

    env()
    assert hub.HubConfig.load() is None


def test_repository_names_are_valid_on_docker_hub():
    assert hub.repository_name("https://github.com/owner/PathFinderCRC.git") == "alembic-tool-pathfindercrc"
    assert hub.repository_name("https://github.com/owner/my repo+x/") == "alembic-tool-my-repo-x"
    assert hub.build_tag("mordred-babc36") == "mordred-babc36"


def test_the_short_description_says_what_the_server_does(tmp_path):
    """A listing shows this line alone: "Alembic MCP server for x/y, 5 tools"
    told a reader nothing about the server."""
    (tmp_path / "exploration.md").write_text(
        "# mordred\n\n## Description\nmordred is a molecular descriptor calculator for "
        "chemoinformatics / QSAR. Given an RDKit molecule it computes 1826 descriptors.\n\n"
        "## Entry points\n- mordred.Calculator\n", encoding="utf-8")

    summary = hub.summarise_repository(tmp_path, "mordred")
    short, _ = hub.render_description("ns", "alembic-tool-mordred", {**_META, "summary": summary})

    assert summary == "Molecular descriptor calculator for chemoinformatics / QSAR"
    assert short == summary  # one line everywhere: no "Alembic", no tool count
    assert len(short) <= 100


def test_a_long_summary_stays_within_the_hub_limit(tmp_path):
    """Docker Hub takes 100 characters for the short description, so the line is
    cut between words and the listings show it whole."""
    (tmp_path / "exploration.md").write_text(
        "## Description\ncytopus is a Python package that ships a curated single-cell-genomics "
        "KnowledgeBase of cell-type hierarchies, cellular processes and identities.\n",
        encoding="utf-8")

    summary = hub.summarise_repository(tmp_path, "cytopus")
    short, _ = hub.render_description("ns", "alembic-tool-cytopus", {**_META, "summary": summary})

    assert len(summary.split()) <= 15 and len(summary) <= 100
    assert short == summary and not short.endswith("…")


def test_a_summary_is_cut_at_a_word_that_can_end_it(tmp_path):
    (tmp_path / "exploration.md").write_text(
        "## Description\n`synspace` generates a local chemical space around a given molecule "
        "(SMILES) using forward and retro synthesis rules.\n", encoding="utf-8")

    assert hub.summarise_repository(tmp_path, "synspace") == \
        "Generates a local chemical space around a given molecule using forward and retro synthesis rules"


def test_an_aside_in_brackets_does_not_eat_the_summary(tmp_path):
    """"BioPsyKit is a Python package (PyPI: biopsykit, v0.13.2) for the analysis
    of biopsychological data" was cut to "...for the analysis"."""
    (tmp_path / "exploration.md").write_text(
        "## Description\nBioPsyKit is a Python package (PyPI: `biopsykit`, v0.13.2) for the "
        "analysis of biopsychological data. It provides pipelines for ECG and EEG.\n",
        encoding="utf-8")

    assert hub.summarise_repository(tmp_path, "BioPsyKit") == \
        "Python package for the analysis of biopsychological data"


def test_the_repository_name_goes_even_when_the_report_spells_it_its_own_way(tmp_path):
    """"NeuroKit" cut off a prefix and left "2 is a user-friendly Python toolbox";
    the report of nnUNet calls it nnU-Net, which no name rule would catch."""
    def summary(text, name):
        (tmp_path / "exploration.md").write_text(f"## Description\n{text}\n", encoding="utf-8")
        return hub.summarise_repository(tmp_path, name)

    assert summary("NeuroKit2 is a user-friendly Python toolbox for biosignal processing. More.",
                   "NeuroKit") == "User-friendly Python toolbox for biosignal processing"
    assert summary("nnU-Net is a self-configuring segmentation framework for medical images. More.",
                   "nnUNet") == "Self-configuring segmentation framework for medical images"


def test_without_a_report_the_description_names_the_repository(tmp_path):
    assert hub.summarise_repository(tmp_path, "mordred") is None
    short, _ = hub.render_description("ns", "alembic-tool-mordred", _META)
    assert short == "MCP server for mordred-descriptor/mordred"


def test_the_description_carries_the_metadata_back():
    short, full = hub.render_description("ns", "alembic-tool-mordred", {**_META, "tools": [
        {"name": "calculate_descriptors", "description": "closes a comment --> here"}]})

    assert len(short) <= 100
    assert "`calculate_descriptors`" in full and "3 of 3" in full
    assert f"serve {_MORDRED}" in full
    assert hub.parse_meta(full)["tools"][0]["description"] == "closes a comment --> here"
    assert hub.parse_meta("# a README written by hand") is None


def test_a_pipe_in_a_tool_description_is_escaped_once():
    meta = hub._build_meta({"repo_url": _MORDRED, "job_id": "mordred-babc36",
                            "tools": [{"name": "pick", "description": "a | b\nmore"}]})

    _, full = hub.render_description("ns", "alembic-tool-mordred", meta)

    assert "| `pick` |  | a \\| b |" in full
    assert hub.parse_meta(full)["tools"][0]["description"] == "a | b"


def test_the_description_says_which_tools_actually_work():
    """A reader picking a server off the hub has to see that, without pulling
    several GB: mordred's three tools are not equally trustworthy."""
    meta = {**_META, "tools": [
        {"name": "calculate_descriptors", "description": "all 1613", "status": "passed", "invoc": "2/3"},
        {"name": "run_mordred_cli", "description": "the CLI", "status": "failed"},
        {"name": "calculate_descriptors_pandas", "description": "a frame", "status": "perfect", "invoc": "2/2"}]}

    _, full = hub.render_description("ns", "alembic-tool-mordred", meta)

    assert "| Tool | Validation | Description |" in full
    assert "| `calculate_descriptors` | passed (2/3 calls) | all 1613 |" in full
    assert "| `run_mordred_cli` | failed | the CLI |" in full
    assert "may or may not work" in full
    assert [(t["name"], t.get("status")) for t in hub.parse_meta(full)["tools"]] == [
        ("calculate_descriptors", "passed"), ("run_mordred_cli", "failed"),
        ("calculate_descriptors_pandas", "perfect")]


def test_the_verdicts_come_from_the_validation_report(tmp_path):
    (tmp_path / "validation.json").write_text(json.dumps({"tools": [
        {"name": "calculate_descriptors", "status": "passed", "invoc_passed": 2, "invoc_total": 3},
        {"name": "run_mordred_cli", "status": "failed", "invoc_passed": 0, "invoc_total": 0},
        {"name": "nameless_row"}]}), encoding="utf-8")

    verdicts = hub.tool_verdicts(tmp_path)

    assert verdicts["calculate_descriptors"] == {"status": "passed", "invoc": "2/3"}
    assert verdicts["run_mordred_cli"] == {"status": "failed"}  # no calls attempted, no ratio
    assert hub.tool_verdicts(tmp_path / "missing") == {}


def test_a_long_tool_list_still_fits_the_description_limit():
    tools = [{"name": f"tool_{i}", "description": "x" * 190, "status": "passed"} for i in range(300)]
    _, full = hub.render_description("ns", "alembic-tool-big", {**_META, "tools": tools})

    assert len(full) <= 25000
    parsed = hub.parse_meta(full)["tools"]
    # Descriptions are what gets dropped; the verdicts stay.
    assert [t["name"] for t in parsed] == [f"tool_{i}" for i in range(300)]
    assert {t.get("status") for t in parsed} == {"passed"}


# ── listing ──────────────────────────────────────────────────────────────────

def _hub_api(monkeypatch, routes):
    seen = []

    def request(method, url, token=None, body=None):
        seen.append((method, url, token, body))
        if isinstance(routes, Exception):
            raise routes
        return routes[url]

    monkeypatch.setattr(hub, "_request", request)
    return seen


def test_the_hub_lists_alembic_servers_with_their_metadata(env, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    _, full = hub.render_description("ns", "alembic-tool-mordred", _META)
    api = hub.HUB_API
    _hub_api(monkeypatch, {
        f"{api}/repositories/ns/?page_size=100": {"next": None, "results": [
            {"name": "alembic-tool-medsam", "last_updated": "2026-08-04T00:00:00Z"},
            {"name": "alembic-tool-mordred", "last_updated": "2026-09-16T00:00:00Z", "pull_count": 3},
            {"name": "some-other-image", "last_updated": "2026-09-17T00:00:00Z"}]},
        f"{api}/repositories/ns/alembic-tool-mordred/": {"full_description": full},
        f"{api}/repositories/ns/alembic-tool-medsam/": {"full_description": ""},
    })

    listing = hub.list_servers()

    assert [s["name"] for s in listing["servers"]] == ["alembic-tool-mordred", "alembic-tool-medsam"]
    assert listing["servers"][0]["meta"]["job_id"] == "mordred-babc36"
    assert listing["servers"][1]["meta"] is None
    assert hub.find_for_repo("https://github.com/Mordred-Descriptor/mordred.git")["name"] == "alembic-tool-mordred"
    # An image uploaded by hand has no metadata and matches by name; the pull checks it.
    assert hub.find_for_repo("https://github.com/bowang-lab/MedSAM")["name"] == "alembic-tool-medsam"
    assert hub.find_for_repo("https://github.com/owner/unknown") is None


def test_an_unreachable_hub_finds_nothing(env, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    _hub_api(monkeypatch, urllib.error.URLError("Name or service not known"))

    listing = hub.list_servers()

    assert listing["ok"] is False and "not reachable" in listing["error"]
    assert hub.find_for_repo(_MORDRED) is None


# ── upload ───────────────────────────────────────────────────────────────────

class _Docker:
    """docker for an image with ``env`` and ``files`` a published image must not have."""

    def __init__(self, env=None, files=""):
        self.env = env if env is not None else [
            "PATH=/usr/local/bin", "GPG_KEY=7169605F62C751356D054A26A821E680E5FA6305", "OPENROUTER_API_KEY="]
        self.files = files
        self.calls = []

    def __call__(self, *args, timeout=120):
        self.calls.append(args)
        if args[:2] == ("image", "inspect") and "Config.Env" in args[3]:
            return subprocess.CompletedProcess(args, 0, json.dumps(self.env), "")
        if args[0] == "run":
            return subprocess.CompletedProcess(args, 0, self.files, "")
        return subprocess.CompletedProcess(args, 0, "", "")


def _finished_build(monkeypatch, docker):
    at._JOBS["mordred-babc36"] = {
        "job_id": "mordred-babc36", "repo_url": _MORDRED, "status": "done", "started_at": 1.0,
        "finished_at": 2.0, "log_file": str(at.LOG_DIR / "mordred-babc36.log"),
        "image_id": "sha256:img", "tool_counts": _META["tool_counts"],
        "tools": [{"name": "calculate_descriptors", "description": "Mordred descriptors"}]}
    monkeypatch.setattr(at, "_docker", docker)
    monkeypatch.setattr(at, "docker_inventory",
                        lambda: {"images": {"sha256:img": "3.31GB"}, "tags": {}, "containers": {}})


def _uploading(monkeypatch, env):
    """A finished build (3/3 valid, see _META) and a hub that takes the upload:
    the docker commands run, and the description requests."""
    env(DOCKERHUB_USERNAME="ns", DOCKERHUB_TOKEN="dckr_pat_secret")
    _finished_build(monkeypatch, _Docker())
    ran = []
    monkeypatch.setattr(hub, "_run", lambda cmd, **kw: ran.append((cmd, kw)) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(hub, "_hub_token", lambda cfg: "jwt")
    return ran, _hub_api(monkeypatch, {f"{hub.HUB_API}/repositories/ns/alembic-tool-mordred/": {}})


def test_an_upload_pushes_both_tags_and_writes_the_description(env, builds, monkeypatch):
    ran, written = _uploading(monkeypatch, env)

    result = hub.upload_build("mordred-babc36")

    assert result["ok"] is True
    login, *pushes = ran
    config_dir = login[0][login[0].index("--config") + 1]
    assert login[0][-1] == "--password-stdin" and login[1]["input"] == "dckr_pat_secret"
    assert not any("dckr_pat_secret" in part for cmd, _ in ran for part in cmd)
    assert [cmd for cmd, _ in pushes] == [
        ["docker", "--config", config_dir, "push", "ns/alembic-tool-mordred:mordred-babc36"],
        ["docker", "--config", config_dir, "push", "ns/alembic-tool-mordred:latest"]]
    assert not Path(config_dir).exists()  # the login never outlives the upload
    [(method, url, token, body)] = written
    assert (method, token) == ("PATCH", "jwt")
    assert hub.parse_meta(body["full_description"])["tools"][0]["name"] == "calculate_descriptors"
    assert at._JOBS["mordred-babc36"]["hub"]["tags"] == ["mordred-babc36", "latest"]
    assert json.loads((at.LOG_DIR / "mordred-babc36.json").read_text())["hub"]["status"] == "uploaded"


def _hub_serves(monkeypatch, passed, perfect):
    counts = {"tools_total": 4, "tools_passed": passed, "tools_perfect": perfect}
    monkeypatch.setattr(hub, "list_servers", lambda refresh=False: {"ok": True, "servers": [
        {"name": "alembic-tool-mordred", "meta": {"tool_counts": counts}}]})


def _pushes(monkeypatch, env):
    ran, _ = _uploading(monkeypatch, env)
    return lambda: [cmd for cmd, _ in ran if "push" in cmd]


def test_auto_upload_keeps_a_better_build_on_the_hub(env, builds, monkeypatch):
    pushes = _pushes(monkeypatch, env)
    _hub_serves(monkeypatch, passed=4, perfect=4)

    result = hub.upload_build("mordred-babc36", keep_better=True)

    assert result["ok"] is False and result["skipped"] is True
    assert "better build" in result["error"]
    assert pushes() == []
    assert at._JOBS["mordred-babc36"]["hub"]["status"] == "skipped"


def test_auto_upload_replaces_a_build_that_is_not_better(env, builds, monkeypatch):
    pushes = _pushes(monkeypatch, env)
    ours = _META["tool_counts"]
    _hub_serves(monkeypatch, passed=ours["tools_passed"], perfect=ours.get("tools_perfect"))

    assert hub.upload_build("mordred-babc36", keep_better=True)["ok"] is True
    assert len(pushes()) == 2


def test_the_upload_button_replaces_even_a_better_build(env, builds, monkeypatch):
    pushes = _pushes(monkeypatch, env)
    _hub_serves(monkeypatch, passed=4, perfect=4)

    assert hub.upload_build("mordred-babc36")["ok"] is True
    assert len(pushes()) == 2


def test_an_image_with_secrets_or_logs_is_not_uploaded(env, builds, monkeypatch):
    env(DOCKERHUB_USERNAME="ns", DOCKERHUB_TOKEN="dckr_pat_secret")
    _finished_build(monkeypatch, _Docker(env=["OPENROUTER_API_KEY=sk-or-v1-abc"],
                                         files="/work/.alembic/mordred/pipeline.log\n"))
    monkeypatch.setattr(hub, "_run", lambda cmd, **kw: pytest.fail("docker login or push ran"))

    result = hub.upload_build("mordred-babc36")

    assert result["ok"] is False
    assert "OPENROUTER_API_KEY" in result["error"] and "pipeline.log" in result["error"]


def test_uploading_needs_the_token_in_env(env, builds, monkeypatch):
    env(DOCKERHUB_NAMESPACE="ns")
    assert "DOCKERHUB_TOKEN" in hub.upload_build("mordred-babc36")["error"]


# ── pull ─────────────────────────────────────────────────────────────────────

class _Inline:
    def __init__(self, target, args, **kw):
        self.target, self.args = target, args

    def start(self):
        self.target(*self.args)


class _PulledImage(_Docker):
    def __init__(self, repo_url):
        super().__init__()
        self.repo_url = repo_url

    def __call__(self, *args, timeout=120):
        self.calls.append(args)
        out = ""
        if args[:2] == ("image", "inspect"):
            out = "sha256:pulled " + json.dumps({"alembic.repo_url": self.repo_url})
        elif args[0] == "run":
            out = "mordred\n"
        return subprocess.CompletedProcess(args, 0, out, "")


def _pull(monkeypatch, labelled):
    docker = _PulledImage(labelled)
    monkeypatch.setattr(at, "_docker", docker)
    monkeypatch.setattr(hub, "threading", types.SimpleNamespace(Thread=_Inline))
    monkeypatch.setattr(hub, "_run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(at, "_import_image_artifacts",
                        lambda job_id, container, repo: ("/tmp/wd", {"tools_total": 3, "tools_passed": 3}))
    started = []
    monkeypatch.setattr(at, "start_build_server", lambda job_id: started.append(job_id) or
                        {"ok": True, "mcp_url": "http://localhost:27000/mcp"})
    monkeypatch.setattr(at, "_await_answer", lambda rec, url: True)
    return docker, started


def test_a_pulled_server_becomes_a_build_and_is_served(env, builds, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    docker, started = _pull(monkeypatch, _MORDRED)

    snap = hub.start_pull("alembic-tool-mordred", "latest", scope=["u", "s"], repo_url=_MORDRED)

    job_id = snap["job_id"]
    assert snap["status"] == "running" and f"check_mcp_build('{job_id}')" in snap["note"]
    rec = at._JOBS[job_id]
    assert (rec["status"], rec["origin"], rec["image"]) == ("done", "hub", f"alembic-tool:{job_id}")
    assert rec["scopes"] == [["u", "s"]] and rec["hub"]["status"] == "pulled"
    assert ("tag", "ns/alembic-tool-mordred:latest", f"alembic-tool:{job_id}") in docker.calls
    assert started == [job_id]


def test_an_image_without_a_label_is_read_from_its_own_artifacts(tmp_path):
    """Images built before the provenance labels carry the repository only inside:
    in the plan, or in a report that links it."""
    work = tmp_path / "workdir" / "cytopus"
    (work / "reports").mkdir(parents=True)
    (work / "output").mkdir()
    (work / "output" / "server.py").write_text(
        'mcp = FastMCP("cytopus")\n\n'
        '@mcp.tool()\ndef get_spectra_genesets(celltypes: list) -> dict:\n'
        '    """Gene sets for cell types.\n\n    More prose.\n    """\n    return {}\n\n'
        '@mcp.tool()\nasync def list_kb(x: int = 1) -> list:\n    return []\n', encoding="utf-8")
    (work / "reports" / "validation.json").write_text(json.dumps({"tools": [
        {"name": "get_spectra_genesets", "status": "perfect", "invoc_passed": 2, "invoc_total": 2}]}),
        encoding="utf-8")
    (work / "reports" / "exploration.md").write_text(
        "Built from https://github.com/wallet-maker/cytopus.git, which cites "
        "https://github.com/dpeerlab/spectra as related work.", encoding="utf-8")

    details = hub.image_artifacts(str(tmp_path / "workdir"), "cytopus")

    # The related project is named in the same report; only the matching name counts.
    assert details["repo_url"] == "https://github.com/wallet-maker/cytopus.git"
    assert details["tools"] == [
        {"name": "get_spectra_genesets", "description": "Gene sets for cell types.",
         "status": "perfect", "invoc": "2/2"},
        {"name": "list_kb", "description": ""}]  # the report says nothing about it


def test_a_pull_that_raises_does_not_stay_running(env, builds, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    _pull(monkeypatch, _MORDRED)
    monkeypatch.setattr(at, "_import_image_artifacts",
                        lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(hub.logger, "exception", lambda *a, **k: None)  # keep the traceback out of logs/app.log

    rec = at._JOBS[hub.start_pull("alembic-tool-mordred", repo_url=_MORDRED)["job_id"]]

    assert rec["status"] == "failed" and "disk full" in rec["error"]


def test_a_pull_whose_local_tag_fails_is_not_served(env, builds, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    docker, started = _pull(monkeypatch, _MORDRED)
    tag_fails = lambda *args, timeout=120: (subprocess.CompletedProcess(args, 1, "", "no space")
                                            if args[0] == "tag" else docker(*args, timeout=timeout))
    monkeypatch.setattr(at, "_docker", tag_fails)

    rec = at._JOBS[hub.start_pull("alembic-tool-mordred", repo_url=_MORDRED)["job_id"]]

    assert rec["status"] == "failed" and "no space" in rec["error"] and started == []


def test_an_image_whose_report_is_markdown_alone_still_counts_its_tools(tmp_path):
    """The builds page shows "x/y tools" from the validator's counts, which the
    oldest images do not carry as JSON; the per-tool verdicts give the same."""
    work = tmp_path / "workdir" / "cytopus"
    (work / "reports").mkdir(parents=True)
    (work / "output").mkdir()
    (work / "output" / "server.py").write_text(
        'mcp = FastMCP("cytopus")\n\n@mcp.tool()\ndef list_kb():\n    return []\n\n'
        '@mcp.tool()\ndef export_kb():\n    return []\n\n'
        '@mcp.tool()\ndef get_identities():\n    return []\n', encoding="utf-8")
    (work / "reports" / "validation.md").write_text(
        "## Tool Invocations\n- **list_kb** — PASSED\n- **export_kb** — FAILED\n"
        "- **get_identities** — PASSED\n", encoding="utf-8")

    details = hub.image_artifacts(str(tmp_path / "workdir"), "cytopus")

    assert details["tool_counts"] == {"tools_total": 3, "tools_passed": 2, "tools_perfect": 0}


def test_a_pull_of_an_image_that_names_no_repository_says_so(env, builds, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    _, started = _pull(monkeypatch, "")
    monkeypatch.setattr(hub, "image_artifacts",
                        lambda workdir, repo: {"repo_url": None, "tools": [], "tool_counts": None})

    snap = hub.start_pull("alembic-tool-cytopus")

    rec = at._JOBS[snap["job_id"]]
    assert rec["status"] == "failed" and "pull it again with that repository URL" in rec["error"]
    assert started == []


def test_a_pulled_image_of_another_repository_is_not_served(env, builds, monkeypatch):
    env(DOCKERHUB_USERNAME="ns")
    _, started = _pull(monkeypatch, "https://github.com/someone/mordred")

    snap = hub.start_pull("alembic-tool-mordred", repo_url=_MORDRED)

    rec = at._JOBS[snap["job_id"]]
    assert rec["status"] == "failed" and "not from" in rec["error"]
    assert started == []


# ── build_mcp_server ─────────────────────────────────────────────────────────

@pytest.fixture
def no_local(builds, monkeypatch):
    monkeypatch.setattr(at, "_repo_exists", lambda url, timeout=20: (True, "ok"))
    monkeypatch.setattr(at, "_reuse_from_host", lambda repo_url, scope: None)
    monkeypatch.setattr(at, "_runner", lambda rec: None)
    pulls = []
    monkeypatch.setattr(hub, "find_for_repo", lambda repo_url: {"name": "alembic-tool-mordred"})
    monkeypatch.setattr(hub, "start_pull", lambda name, tag, scope=None, repo_url=None: pulls.append(name) or
                        {"status": "running", "job_id": "mordred-hub-abc123", "origin": "hub"})
    return pulls


def test_build_mcp_server_pulls_from_the_hub_before_building(no_local, monkeypatch):
    monkeypatch.setattr(hub, "search_enabled", lambda: True)

    result = asyncio.run(at.build_mcp_server(_MORDRED))

    assert result["job_id"] == "mordred-hub-abc123"
    assert no_local == ["alembic-tool-mordred"] and at._JOBS == {}  # no build started


@pytest.mark.parametrize("search, force", [(False, False), (True, True)])
def test_the_hub_is_skipped_when_search_is_off_or_a_rebuild_is_forced(no_local, monkeypatch, search, force):
    monkeypatch.setattr(hub, "search_enabled", lambda: search)

    result = asyncio.run(at.build_mcp_server(_MORDRED, force_rebuild=force))

    assert result["status"] == "running" and result["job_id"] != "mordred-hub-abc123"
    assert [rec["origin"] for rec in at._JOBS.values()] == ["builder"]
    assert no_local == []


# ── settings and web routes ──────────────────────────────────────────────────

def test_an_agent_cannot_start_a_conversion_when_the_setting_is_off(no_local, monkeypatch, builds):
    """A conversion takes tens of minutes, and a demo must not start one by
    itself. The builds page still converts: it passes no tool_context."""
    monkeypatch.setattr(hub, "search_enabled", lambda: True)
    monkeypatch.setattr(hub, "find_for_repo", lambda repo_url: None)
    monkeypatch.setattr(at, "_agent_may_build", lambda: False)
    context = types.SimpleNamespace(state={"graph_scope_user_id": "u", "graph_scope_session_id": "s"})

    refused = asyncio.run(at.build_mcp_server(_MORDRED, tool_context=context))
    assert refused["status"] == "error" and "turned off for agents" in refused["error"]
    assert at._JOBS == {}

    # The operator's own Run on the builds page goes through.
    assert asyncio.run(at.build_mcp_server(_MORDRED))["status"] == "running"


def test_the_hub_settings_round_trip_through_the_settings_api(monkeypatch):
    from CoScientist.config import get_settings
    from CoScientist.web.app import _apply_frontend_settings, _settings_payload

    web = get_settings().web
    monkeypatch.setattr(web, "alembic_hub_search_enabled", True)
    monkeypatch.setattr(web, "alembic_hub_auto_upload", False)

    monkeypatch.setattr(web, "alembic_agent_build_enabled", False)

    _apply_frontend_settings({"alembicHub": {"searchEnabled": False, "autoUpload": True,
                                             "agentBuildEnabled": True}})

    assert _settings_payload()["alembicHub"] == {"searchEnabled": False, "autoUpload": True,
                                                 "agentBuildEnabled": True}
    assert (hub.search_enabled(), hub.auto_upload_enabled(), at._agent_may_build()) == (False, True, True)


def _client():
    app = FastAPI()
    app.include_router(build_api.router)
    return TestClient(app)


def test_the_hub_page_shows_which_servers_are_on_this_host(monkeypatch):
    monkeypatch.setattr(hub, "list_servers", lambda refresh=False: {
        "ok": True, "namespace": "ns", "can_upload": True,
        "servers": [{"name": "alembic-tool-mordred", "image": "ns/alembic-tool-mordred", "meta": _META}]})
    monkeypatch.setattr(at, "web_list_builds", lambda: [
        {"job_id": "mordred-hub-abc123", "origin": "hub", "status": "done",
         "hub": {"image": "ns/alembic-tool-mordred"}},
        {"job_id": "gget-938c68", "status": "done"}])

    body = _client().get("/api/hub").json()

    assert body["servers"][0]["local_builds"] == [
        {"job_id": "mordred-hub-abc123", "origin": "hub", "status": "done"}]


def test_pulling_from_the_page_obeys_the_controls_switch(monkeypatch):
    monkeypatch.setattr(hub, "start_pull", lambda *a, **kw: pytest.fail("pulled"))
    monkeypatch.setenv("ALEMBIC_WEB_CONTROLS", "0")

    assert _client().post("/api/hub/pull", json={"name": "alembic-tool-mordred"}).status_code == 403
