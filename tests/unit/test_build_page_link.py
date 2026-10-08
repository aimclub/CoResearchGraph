"""What a build hands back about itself.

The agent acts on these fields, so they must not point at a page that is not
there or invite it to look for the build somewhere else on the machine.
"""

import pytest

from CoScientist.tools import alembic_tools


@pytest.fixture
def job(tmp_path):
    return {
        "job_id": "gget-938c68",
        "repo_url": "https://github.com/pachterlab/gget",
        "status": "running",
        "started_at": 0.0,
        "log_file": str(tmp_path / "build.log"),
    }


def test_no_web_ui_configured_means_no_link_to_one(job, monkeypatch):
    """The build page exists only while the web UI runs. A link that does not
    open sends the agent hunting for the build elsewhere."""
    monkeypatch.setattr(alembic_tools, "_WEB_BASE_URL", "")

    out = alembic_tools._snapshot(job)

    assert "progress_url" not in out
    assert out["progress_page"] == "/alembic/builds/gget-938c68"  # the web layer resolves it


def test_a_configured_web_ui_is_linked_absolutely(job, monkeypatch):
    monkeypatch.setattr(alembic_tools, "_WEB_BASE_URL", "http://box:8000")

    out = alembic_tools._snapshot(job)

    assert out["progress_url"] == "http://box:8000/alembic/builds/gget-938c68"


def test_a_running_build_is_told_where_its_result_comes_from(job):
    """A live run watched an agent validate a leftover container from an earlier
    build and call the task done. The note says not to."""
    note = alembic_tools._snapshot(job)["note"]

    assert "check_mcp_build('gget-938c68')" in note
    assert "only source" in note


def test_the_tool_image_is_read_from_the_serve_summary(job, tmp_path):
    """The log mentions the base image first ("building base image: docker
    build ..."). Reading that line put "docker" into the image field."""
    (tmp_path / "build.log").write_text(
        "[alembic] building base image: docker build -t alembic-base:latest .\n"
        "[start-chain] MCP server up.\n"
        "  image     : alembic-tool:gget\n"
        "  container : alembic-serve-gget-b29990\n"
        "  url       : http://localhost:20162/mcp\n",
        encoding="utf-8",
    )

    alembic_tools._finalize(job, 0)

    assert job["image"] == "alembic-tool:gget"
    assert job["container"] == "alembic-serve-gget-b29990"


_VALIDATOR_DONE = (
    'ALEMBIC_EVENT {"type": "stage", "stage": "validator", "status": "done", "counts": '
    '{"tools_total": 4, "tools_passed": 3, "tools_perfect": 3, "tests_passed": 9, "tests_total": 12}}\n'
)


def test_a_finished_build_reports_how_many_tools_passed_validation(job, tmp_path):
    (tmp_path / "build.log").write_text("[start-chain] building\n" + _VALIDATOR_DONE, encoding="utf-8")
    job.update(status="done", finished_at=1.0)

    counts = alembic_tools._snapshot(job)["tool_counts"]

    assert counts["tools_passed"] == 3
    assert counts["tools_total"] == 4


def test_a_build_known_only_from_disk_reports_its_tool_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(alembic_tools, "LOG_DIR", tmp_path)
    monkeypatch.setattr(alembic_tools, "_JOBS", {})
    (tmp_path / "gget-938c68.log").write_text(
        _VALIDATOR_DONE + "  url       : http://localhost:20162/mcp\n", encoding="utf-8")

    snap = alembic_tools.web_build_snapshot("gget-938c68")

    assert snap["tool_counts"]["tools_total"] == 4


def test_a_build_without_a_validator_result_has_no_tool_counts(job, tmp_path):
    (tmp_path / "build.log").write_text("[start-chain] pipeline failed\n", encoding="utf-8")
    job.update(status="failed", finished_at=1.0)

    assert "tool_counts" not in alembic_tools._snapshot(job)
