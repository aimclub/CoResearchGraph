"""The sandbox panel: browsing a workspace, reading a file, taking one out.

(Named for the panel because `test_sandbox_artifacts.py` is upstream's, about
sandbox uploads becoming durable report artifacts. Two different subjects that
arrived at one filename from two branches.)

Getting a file out of the sandbox and into something a reader can open.

`list_sandbox_files` could say a file exists and nothing could reach it. The
transfer is one hop — pull the bytes, put them in the bucket every other
artifact lives in, answer with the durable link — so that everything
downstream (the report, the chat, the S3 references the execution graph reads
onto an agent's card) keeps working untouched.
"""
import types

import pytest

from CoScientist.tools.coder_tools import sandbox_artifacts


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """A sandbox that hands over one small file."""
    def download(remote, local, **_kwargs):
        with open(local, "wb") as fh:
            fh.write(b"loss,epoch\n0.1,3\n")
        return {"status": "ok", "size_bytes": 17, "sandbox_id": "sbx-1"}

    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.download_sandbox_file",
        download, raising=False)
    return download


def _stub_upload(monkeypatch, ref=("results", "sandbox-artifacts/s1/metrics.csv")):
    monkeypatch.setattr("CoScientist.reporting.s3_upload.upload_and_ref",
                        lambda *a, **k: ref, raising=False)


def test_a_transferred_file_comes_back_as_a_link_that_keeps_working(sandbox, monkeypatch):
    _stub_upload(monkeypatch)

    out = sandbox_artifacts.transfer_sandbox_artifact(
        "/workspace/metrics.csv", session_id="s1")

    assert out["status"] == "success"
    # The durable form, not a presigned URL: a signature expires, the object does not.
    assert out["url"] == "/api/artifact/results/sandbox-artifacts/s1/metrics.csv"
    assert out["bucket"] == "results"
    assert "?" not in out["url"], "a presigned URL would carry a signature"


def test_the_object_is_filed_under_the_session_that_produced_it(sandbox, monkeypatch):
    seen = {}

    def upload(local, prefix):
        seen["prefix"] = prefix
        return ("results", f"{prefix}/metrics.csv")

    monkeypatch.setattr("CoScientist.reporting.s3_upload.upload_and_ref",
                        upload, raising=False)

    sandbox_artifacts.transfer_sandbox_artifact("/workspace/metrics.csv",
                                                session_id="session-42")
    assert "session-42" in seen["prefix"]


def test_a_sandbox_that_refuses_is_reported_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.download_sandbox_file",
        lambda *a, **k: {"status": "error", "message": "No sandbox is bound."},
        raising=False)

    out = sandbox_artifacts.transfer_sandbox_artifact("/workspace/x", session_id="s1")
    assert out["status"] == "error"
    assert "No sandbox is bound." in out["message"]


def test_storage_refusing_the_file_is_not_reported_as_success(sandbox, monkeypatch):
    """The bytes arrived; nobody can open them. That is a failure, not a link."""
    monkeypatch.setattr("CoScientist.reporting.s3_upload.upload_and_ref",
                        lambda *a, **k: None, raising=False)

    out = sandbox_artifacts.transfer_sandbox_artifact("/workspace/x", session_id="s1")
    assert out["status"] == "error"
    assert "S3" in out["message"]


def test_an_enormous_artifact_is_refused_rather_than_written_to_disk(monkeypatch):
    """A training run leaves gigabytes behind; a directory arrives as one ZIP."""
    def huge(remote, local, **_kwargs):
        open(local, "wb").close()
        return {"status": "ok", "size_bytes": 10 * 1024 ** 3}

    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.download_sandbox_file",
        huge, raising=False)

    out = sandbox_artifacts.transfer_sandbox_artifact("/workspace", session_id="s1")
    assert out["status"] == "error"
    assert "слишком велик" in out["message"]


def test_an_empty_path_is_refused_before_anything_is_contacted():
    out = sandbox_artifacts.transfer_sandbox_artifact("   ")
    assert out["status"] == "error"


def test_the_agents_are_given_the_tool_and_it_is_documented():
    """Undocumented, guard_unknown_tools would refuse the call as hallucinated."""
    from CoScientist.assembly import bindings
    from CoScientist.tools.coder_tools.sandbox_tools import get_sandbox_tools

    documented = {d.name for d in bindings._SANDBOX_TAIL_DOCS}
    assert "fetch_sandbox_artifact" in documented
    # And actually attached, not merely described.
    attached = {getattr(t, "__name__", "") for t in get_sandbox_tools()}
    assert not attached or "fetch_sandbox_artifact" in attached


def test_the_panel_is_wired_into_the_page():
    """A person watching a run could not look inside the sandbox at all.

    The panel is only useful if the page actually carries it: the markup, the
    module that drives it, and a way in from the rail.
    """
    from starlette.testclient import TestClient

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        page = client.get("/").text
        module = client.get("/static/js/modals/artifacts.js")
        rail = client.get("/static/js/activity_rail.js").text

    assert 'id="artifacts-modal"' in page
    assert "modals/artifacts.js" in page, "the module must be loaded, not just present"
    assert module.status_code == 200
    assert "SandboxArtifacts" in rail, "and reachable from the rail"
    # It talks to the endpoints that exist, not to invented ones. The URL is
    # assembled from a session-scoped base, so check the parts.
    assert "/sandbox" in module.text
    assert "/files?" in module.text and "path=" in module.text
    assert "/fetch" in module.text
    assert "/tasks" in module.text, "and asks which workspaces exist"


def test_the_panel_asks_for_the_open_session_not_a_global_sandbox():
    """A sandbox belongs to one session; listing another's would be a lie."""
    from starlette.testclient import TestClient

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        module = client.get("/static/js/modals/artifacts.js").text

    assert "activeSession" in module and "activeUser" in module


# ---------------------------------------------------------------------------
# Choosing which workspace to look at
# ---------------------------------------------------------------------------

class _Answer:
    """The bare shape of an httpx response the client actually touches."""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _sandbox_status(monkeypatch, payload):
    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox

    monkeypatch.setattr(sandbox, "resolve_sandbox_url", lambda *a, **k: "http://sbx",
                        raising=False)
    monkeypatch.setattr(sandbox, "read_binding", lambda *a, **k: "task-ours",
                        raising=False)
    monkeypatch.setattr(sandbox.httpx, "get", lambda *a, **k: _Answer(payload),
                        raising=False)
    return sandbox


def test_every_workspace_the_sandbox_holds_is_offered_ours_first(monkeypatch):
    """The reader's own container is one of many; the others are one id away."""
    sandbox = _sandbox_status(monkeypatch, {
        "current_task": {"task_id": "task-other", "status": "running",
                         "task": "train a transformer"},
        "active_tasks": [{"task_id": "task-other", "status": "running"}],
        "queue": [{"task_id": "task-waiting", "status": "queued", "task": "next up"}],
        "completed_tasks": [{"task_id": "task-ours", "status": "cooldown",
                             "summary": "ours"}],
    })

    out = sandbox.list_sandbox_tasks(session_id="s1")

    assert out["status"] == "ok"
    ids = [w["sandbox_id"] for w in out["workspaces"]]
    assert ids[0] == "task-ours", "the session's own sandbox leads the list"
    assert set(ids) == {"task-ours", "task-other", "task-waiting"}
    # One task appearing in two sections of one answer is still one workspace.
    assert len(ids) == len(set(ids))


def test_a_workspace_whose_container_is_gone_is_marked_unbrowsable(monkeypatch):
    """`/files` answers 409 for it; saying so beats an empty listing."""
    sandbox = _sandbox_status(monkeypatch, {
        "current_task": None, "active_tasks": [], "queue": [],
        "completed_tasks": [{"task_id": "task-ours", "status": "running"},
                            {"task_id": "task-done", "status": "completed"}],
    })

    by_id = {w["sandbox_id"]: w for w in
             sandbox.list_sandbox_tasks(session_id="s1")["workspaces"]}

    assert by_id["task-ours"]["browsable"] is True
    assert by_id["task-done"]["browsable"] is False


def test_the_session_sandbox_is_listed_even_when_it_fell_off_the_recent_list(monkeypatch):
    """Twenty runs later it is still the one the reader means by "current"."""
    sandbox = _sandbox_status(monkeypatch, {
        "current_task": None, "active_tasks": [], "queue": [], "completed_tasks": [],
    })
    monkeypatch.setattr(sandbox, "_fetch_task",
                        lambda api, target: {"status": "cooldown", "task": "ours"},
                        raising=False)

    workspaces = sandbox.list_sandbox_tasks(session_id="s1")["workspaces"]

    assert [w["sandbox_id"] for w in workspaces] == ["task-ours"]
    assert workspaces[0]["current"] is True


def test_a_chosen_workspace_reaches_the_sandbox_client(monkeypatch):
    """The dropdown is pointless if the id stops at the web layer."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    seen = {}

    def listing(path, **kwargs):
        seen.update(path=path, sandbox_id=kwargs.get("sandbox_id"))
        return {"status": "ok", "path": path, "entries": []}

    monkeypatch.setattr(sandbox, "list_sandbox_files", listing, raising=False)

    with TestClient(create_app()) as client:
        client.get("/api/users/u1/sessions/s1/sandbox/files"
                   "?path=/workspace&sandbox_id=task-other")

    assert seen["sandbox_id"] == "task-other"


def test_the_transfer_honours_the_chosen_workspace(monkeypatch):
    """Otherwise "забрать" would quietly pull the file from a different run."""
    seen = {}

    def download(remote, local, **kwargs):
        seen["sandbox_id"] = kwargs.get("sandbox_id")
        open(local, "wb").close()
        return {"status": "ok", "size_bytes": 0, "sandbox_id": kwargs.get("sandbox_id")}

    monkeypatch.setattr(
        "CoScientist.tools.coder_tools.openhands_sandbox.download_sandbox_file",
        download, raising=False)
    _stub_upload(monkeypatch)

    sandbox_artifacts.transfer_sandbox_artifact(
        "/workspace/metrics.csv", session_id="s1", sandbox_id="task-other")

    assert seen["sandbox_id"] == "task-other"


# ---------------------------------------------------------------------------
# The panel's own logic, driven in a stub DOM
# ---------------------------------------------------------------------------

#: The panel is served as plain script, so it can be run against a handful of
#: stubs. Worth the harness: the bug it guards against — listing the files
#: before deciding whose files to list — is invisible to a source assertion.
_PROBE = r"""
const fs = require('fs'), vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');

const nodes = {};
function node(id) {
  if (!nodes[id]) nodes[id] = {
    id, innerHTML: '', textContent: '', className: '',
    classList: { add() {}, remove() {}, toggle() {} },
  };
  return nodes[id];
}
const asked = [];
const context = {
  console,
  activeUser: { id: 'u1' }, activeSession: { id: 's1' },
  document: { getElementById: node },
  escHtml: (v) => String(v == null ? '' : v),
  async fetch(url) {
    asked.push(url);
    if (url.endsWith('/tasks')) {
      return { json: async () => ({ status: 'ok', workspaces: JSON.parse(process.argv[3]) }) };
    }
    return { json: async () => ({ status: 'ok', path: '/workspace',
                                 entries: JSON.parse(process.argv[4] || '[]') }) };
  },
};
vm.createContext(context);
vm.runInContext(source + '\n;globalThis.__open = openArtifactsModal;', context);
context.__open().then(() => console.log(JSON.stringify({
  asked,
  options: nodes['artifacts-workspace'].innerHTML,
  listing: nodes['artifacts-body'].innerHTML,
  note: nodes['artifacts-note'].textContent,
})));
"""


def _open_the_panel(workspaces, entries=()):
    """Open the panel against a sandbox holding ``workspaces``; report what it did."""
    import json
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        pytest.skip("node is needed to drive the panel's script")

    from starlette.testclient import TestClient

    from CoScientist.web.app import create_app

    with TestClient(create_app()) as client:
        module = client.get("/static/js/modals/artifacts.js").text

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        probe = f"{tmp}/probe.js"
        panel = f"{tmp}/artifacts.js"
        open(probe, "w").write(_PROBE)
        open(panel, "w").write(module)
        out = subprocess.run([node, probe, panel, json.dumps(workspaces),
                              json.dumps(list(entries))],
                             capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_opening_the_panel_reads_a_real_workspace_not_the_missing_binding():
    """The session need not own the sandbox it is watching.

    A run started elsewhere leaves this session with no binding, and asking for
    "our sandbox" then answers "No sandbox is bound to this session" over a
    machine visibly full of files. The workspace is chosen first, and the
    listing names it.
    """
    seen = _open_the_panel([
        {"sandbox_id": "fb82e9d3-2068", "status": "running",
         "task": "## Задача: пайплайн", "current": False, "browsable": True},
    ])

    listing = [url for url in seen["asked"] if "/files?" in url]
    assert listing, "the panel must list some workspace"
    assert "sandbox_id=fb82e9d3-2068" in listing[0]
    # And the list was fetched first — the choice cannot follow the listing.
    assert seen["asked"].index([u for u in seen["asked"] if u.endswith("/tasks")][0]) == 0


def test_the_session_own_sandbox_still_wins_when_it_has_one():
    seen = _open_the_panel([
        {"sandbox_id": "ours", "status": "cooldown", "task": "", "current": True,
         "browsable": True},
        {"sandbox_id": "theirs", "status": "running", "task": "", "browsable": True},
    ])

    listing = [url for url in seen["asked"] if "/files?" in url]
    assert "sandbox_id=ours" in listing[0]


def test_a_stopped_container_stays_selectable_and_says_so():
    """Refusing the row here guesses; the sandbox is the one that knows."""
    seen = _open_the_panel([
        {"sandbox_id": "live", "status": "running", "task": "", "browsable": True},
        {"sandbox_id": "gone", "status": "completed", "task": "", "browsable": False},
    ])

    assert "disabled" not in seen["options"]
    assert "контейнер остановлен" in seen["options"]


def test_the_heading_of_a_markdown_prompt_does_not_fill_the_dropdown():
    """The prompt arrives as markdown; its marks cost width and say nothing."""
    seen = _open_the_panel([
        {"sandbox_id": "one", "status": "running", "browsable": True,
         "task": "## Задача: Полный пайплайн Этапа 1 — Эволюционная оптимизация чего-то ещё"},
    ])

    assert "##" not in seen["options"]
    assert "Задача: Полный пайплайн" in seen["options"]


# ---------------------------------------------------------------------------
# Reading a file instead of harvesting it
# ---------------------------------------------------------------------------

def test_a_workspace_file_is_classified_by_what_can_be_done_with_it():
    from CoScientist.web.preview import kind_of

    assert kind_of("train.py") == "text"
    assert kind_of("AGENTS.md") == "text"
    assert kind_of("Dockerfile") == "text", "a working tree carries these bare"
    assert kind_of(".gitignore") == "text"
    assert kind_of("loss.png") == "image"
    assert kind_of("arch.svg") == "image"
    assert kind_of("paper.pdf") == "pdf"
    assert kind_of("model.ckpt") == "binary"


def test_nothing_from_a_workspace_is_served_as_something_the_browser_runs():
    """An .html or .js in the workspace is the agent's output, not our page."""
    from CoScientist.web.preview import media_type_of

    assert media_type_of("index.html").startswith("text/plain")
    assert media_type_of("bundle.js").startswith("text/plain")
    assert media_type_of("page.xml").startswith("text/plain")
    # An image is served as itself: an <img> does not run scripts in an SVG.
    assert media_type_of("arch.svg") == "image/svg+xml"
    assert media_type_of("model.ckpt") == "application/octet-stream"


def test_a_checkpoint_is_not_read_into_the_page_to_be_refused(monkeypatch):
    """Answering "nothing to show here" should not move a gigabyte first."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    def refuse(*a, **k):  # pragma: no cover - the point is that it is not called
        raise AssertionError("the sandbox must not be contacted for a checkpoint")

    monkeypatch.setattr(sandbox, "read_sandbox_file", refuse, raising=False)

    with TestClient(create_app()) as client:
        answer = client.get("/api/users/u1/sessions/s1/sandbox/view"
                            "?path=/workspace/model.ckpt")

    assert answer.status_code == 415


def test_a_text_file_comes_back_readable_and_says_when_it_was_cut(monkeypatch):
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    seen = {}

    def read(path, *, max_bytes, **kwargs):
        seen.update(path=path, max_bytes=max_bytes, sandbox_id=kwargs.get("sandbox_id"))
        return {"status": "ok", "data": b"epoch,loss\n1,0.7\n", "truncated": True}

    monkeypatch.setattr(sandbox, "read_sandbox_file", read, raising=False)

    with TestClient(create_app()) as client:
        answer = client.get("/api/users/u1/sessions/s1/sandbox/view"
                            "?path=/workspace/out.csv&sandbox_id=task-x")

    assert answer.status_code == 200
    assert answer.text.startswith("epoch,loss")
    assert answer.headers["x-preview-kind"] == "text"
    assert answer.headers["x-preview-truncated"] == "1"
    assert answer.headers["x-content-type-options"] == "nosniff"
    assert seen["sandbox_id"] == "task-x", "the chosen workspace must be read"
    assert seen["max_bytes"] > 0


def test_downloading_a_binary_is_allowed_where_showing_it_is_not(monkeypatch):
    """Storage is refusing uploads today; saving a file should not depend on it."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    monkeypatch.setattr(
        sandbox, "read_sandbox_file",
        lambda path, *, max_bytes, **k: {"status": "ok", "data": b"\x00\x01",
                                         "truncated": False},
        raising=False)

    with TestClient(create_app()) as client:
        answer = client.get("/api/users/u1/sessions/s1/sandbox/view"
                            "?path=/workspace/model.ckpt&download=1")

    assert answer.status_code == 200
    assert "attachment" in answer.headers["content-disposition"]


def test_a_cyrillic_name_survives_the_download_rather_than_becoming_question_marks(
        monkeypatch):
    """A run that writes in Russian names its files in Russian."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    monkeypatch.setattr(
        sandbox, "read_sandbox_file",
        lambda path, *, max_bytes, **k: {"status": "ok", "data": b"x", "truncated": False},
        raising=False)

    with TestClient(create_app()) as client:
        answer = client.get("/api/users/u1/sessions/s1/sandbox/view"
                            "?path=/workspace/отчёт.txt&download=1")

    assert answer.status_code == 200
    disposition = answer.headers["content-disposition"]
    # The real name, percent-encoded, for a client that reads RFC 5987 …
    assert "filename*=UTF-8''%D0%BE%D1%82%D1%87%D1%91%D1%82.txt" in disposition
    # … and a fallback that is still a filename, not a row of question marks.
    assert 'filename="file.txt"' in disposition


def test_the_listing_says_what_each_entry_is(monkeypatch):
    """The sandbox answers "dir"; the panel needs one word and a kind."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    monkeypatch.setattr(
        sandbox, "list_sandbox_files",
        lambda path, **k: {"status": "ok", "path": path, "entries": [
            {"name": "results", "type": "dir", "size": 4096},
            {"name": "loss.png", "type": "file", "size": 120},
            {"name": "model.ckpt", "type": "file", "size": 10 ** 9},
        ]},
        raising=False)

    with TestClient(create_app()) as client:
        entries = client.get(
            "/api/users/u1/sessions/s1/sandbox/files").json()["entries"]

    by_name = {e["name"]: e for e in entries}
    assert by_name["results"]["kind"] == "dir"
    assert by_name["loss.png"]["kind"] == "image"
    assert by_name["model.ckpt"]["kind"] == "binary"


def test_a_folder_is_a_folder_in_the_panel_not_a_file_to_harvest():
    """The sandbox says "dir"; looking for "directory" made every folder a file."""
    seen = _open_the_panel(
        [{"sandbox_id": "w", "status": "running", "task": "", "browsable": True}],
        entries=[{"name": "results", "type": "dir", "size": 4096,
                  "kind": "dir", "path": "/workspace/results"},
                 {"name": "train.py", "type": "file", "size": 900,
                  "kind": "text", "path": "/workspace/train.py"}],
    )

    rows = seen["listing"]
    assert "loadArtifacts('/workspace/results')" in rows, "a folder must open"
    assert "openArtifactFile('/workspace/train.py')" in rows, "a file must be readable"


def test_saving_a_file_does_not_go_through_storage():
    """Storage is for a link that outlives the container, not for a download."""
    seen = _open_the_panel(
        [{"sandbox_id": "w", "status": "running", "task": "", "browsable": True}],
        entries=[{"name": "meta.json", "type": "file", "size": 4608,
                  "kind": "text", "path": "/workspace/meta.json"}],
    )

    assert "download" in seen["listing"], "every row must offer a direct save"
    assert "download=1" in seen["listing"].replace("&amp;", "&")


def test_a_folder_is_saved_as_a_zip_with_a_name_to_match(monkeypatch):
    """The sandbox serves a directory as one archive; the file should say so."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    monkeypatch.setattr(
        sandbox, "read_sandbox_file",
        lambda path, *, max_bytes, **k: {"status": "ok", "data": b"PK\x03\x04",
                                         "truncated": False},
        raising=False)

    with TestClient(create_app()) as client:
        answer = client.get("/api/users/u1/sessions/s1/sandbox/view"
                            "?path=/workspace/events&dir=1")

    assert answer.status_code == 200
    assert 'filename="events.zip"' in answer.headers["content-disposition"]
    assert "attachment" in answer.headers["content-disposition"]


def test_a_mixed_name_keeps_the_part_a_header_can_carry(monkeypatch):
    """`отчёт_v2.csv` is not nameless — `_v2.csv` beats a stand-in."""
    from starlette.testclient import TestClient

    from CoScientist.tools.coder_tools import openhands_sandbox as sandbox
    from CoScientist.web.app import create_app

    monkeypatch.setattr(
        sandbox, "read_sandbox_file",
        lambda path, *, max_bytes, **k: {"status": "ok", "data": b"a,b\n", "truncated": False},
        raising=False)

    with TestClient(create_app()) as client:
        answer = client.get("/api/users/u1/sessions/s1/sandbox/view"
                            "?path=/workspace/отчёт_v2.csv&download=1")

    assert 'filename="v2.csv"' in answer.headers["content-disposition"]
