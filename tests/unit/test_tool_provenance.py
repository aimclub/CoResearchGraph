"""The report section on a built tool: where it came from and how it was built."""
from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from CoScientist.reporting import tool_provenance as tp

REPO = "https://github.com/Owner/Survival-Toolkit"
RUN_ID = "EXRUN-abc"
URL = "http://172.17.0.1:21754/mcp"


def _write_build(root: Path, job_id: str = "Survival-Toolkit-aa11", *, status: str = "done",
                 run_id: str = RUN_ID, scopes=None) -> dict:
    workdir = root / job_id / "workdir"
    reports = workdir / "Survival-Toolkit" / "reports"
    reports.mkdir(parents=True)
    (reports / "stage_status.json").write_text(json.dumps({
        "explorer": {"status": "passed", "gate": {"layout": "two-venv", "verified": 2, "dropped": 0}},
        "environment": {"status": "passed", "gate": {"hard": [], "soft": []}},
        "coder": {"status": "passed", "gate": {"tools": ["calc_km", "calc_naive"]}},
        "validator": {"status": "passed" if status == "done" else "failed",
                      "counts": {"tests_passed": 5, "tests_total": 6, "invoc_passed": 3, "invoc_total": 4},
                      "debugger_rounds": 1},
        "wrapper": {"status": "passed", "gate": {"tools_wrapped": ["calc_km", "calc_naive"], "skipped": []}},
    }), encoding="utf-8")
    (reports / "metrics.json").write_text(json.dumps({
        "durations_per_stage": {"explorer": 40.0, "environment": 3.0, "coder": 100.0,
                                "validator": 30.0, "wrapper": 5.0},
        "total_cost_usd": 0.1234, "total_tokens": 450000,
    }), encoding="utf-8")
    (reports / "plan.json").write_text(json.dumps({
        "env": {"layout": "two-venv", "server_python": "3.11", "dependencies": ["numpy==1.26.4"]},
        "tools": [
            {"name": "calc_km", "target": "toolkit:calc_km", "purpose": "Kaplan-Meier estimate | with SE"},
            {"name": "calc_naive", "target": "toolkit:calc_naive", "purpose": "Naive estimate"},
        ],
    }), encoding="utf-8")
    (reports / "validation.json").write_text(json.dumps({
        "tools": [
            {"name": "calc_km", "tests_passed": 3, "tests_total": 3, "invoc_passed": 2,
             "invoc_total": 2, "status": "perfect"},
            {"name": "calc_naive", "tests_passed": 2, "tests_total": 3, "invoc_passed": 1,
             "invoc_total": 2, "status": "failed"},
        ],
        "counts": {"tests_passed": 5, "tests_total": 6, "invoc_passed": 3, "invoc_total": 4},
        "debugger_rounds": 1,
    }), encoding="utf-8")
    record = {
        "job_id": job_id, "repo_url": REPO, "status": status, "run_id": run_id,
        "started_at": 1_790_000_000.0, "finished_at": 1_790_000_240.0,
        "workdir": str(workdir), "image": f"alembic-tool:{job_id}",
        "mcp_url": URL if status == "done" else None, "scopes": scopes or [],
    }
    (root / f"{job_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return record


@pytest.fixture
def builds_dir(tmp_path, monkeypatch):
    from CoScientist.tools import alembic_tools as at

    monkeypatch.setattr(at, "LOG_DIR", tmp_path)
    monkeypatch.setattr(at, "JOB_METADATA_DIR", tmp_path / "jobs")
    return tmp_path


def _state(lang: str = "ru") -> dict:
    server = {"url": URL, "source": "alembic", "tools": [{"name": "calc_km"}]}
    return {
        "report_language": lang,
        "experiment_runtime": {
            "run_id": RUN_ID,
            "task_order": ["EXP-1", "EXP-2", "EXP-3"],
            "tasks": {
                "EXP-1": {"status": "done_with_warnings", "current_route": "react_tools",
                          "route_history": [{"route": "alembic_build"}, {"route": "react_tools",
                                                                         "reason": "alembic_post_build"}],
                          "task": {"id": "EXP-1", "name": "Adapter", "repo_url": REPO,
                                   "mcp_servers": [server]}},
                "EXP-2": {"status": "done", "current_route": "coder", "route_history": [],
                          "task": {"id": "EXP-2", "name": "Sweep", "repo_url": REPO,
                                   "mcp_servers": [server]}},
                "EXP-3": {"status": "done", "current_route": "coder", "route_history": [],
                          "task": {"id": "EXP-3", "name": "Unrelated", "mcp_servers": []}},
            },
        },
    }


REPORT = "# Report\n\n## Результаты\n\nNumbers.\n\n## Обсуждение\n\nText.\n"


def _svg_of(markdown: str) -> ET.Element:
    found = re.search(r"data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)", markdown)
    assert found, "the section embeds its diagram"
    return ET.fromstring(base64.b64decode(found.group(1)))


def test_section_names_repository_stages_tools_and_tasks(builds_dir):
    _write_build(builds_dir)
    out = tp.with_tool_provenance(REPORT, _state())

    section = out[out.index(tp.START_MARK):out.index(tp.END_MARK)]
    assert f"[Owner/Survival-Toolkit]({REPO})" in section
    assert "MCP-сервер с 2 тулами за 4 мин 0 с" in section
    assert "Проверку прошли 1 из 2 (тесты 5/6)" in section
    assert "| Проверка | ✅ | 30 с | тесты 5/6, вызовы 3/4, раундов отладки: 1 |" in section
    # The wall clock the stages do not account for is the image and serve step.
    assert "| Образ Docker и запуск | ✅ | 1 мин 2 с |" in section
    assert "Код тулов взят из `toolkit.py`." in section
    assert "| `calc_km` | `calc_km` | 3/3 · 2/2 | ✅ все проверки |" in section
    assert "| `calc_naive` | `calc_naive` | 2/3 · 1/2 | ❌ не прошёл |" in section
    assert "- `calc_km`: Kaplan-Meier estimate | with SE" in section
    assert "Зависимости окружения: `numpy==1.26.4`." in section
    assert "- ⚠️ **EXP-1** Adapter: собрал сервер, агент проверил тулы, `react_tools`" in section
    assert "- ✅ **EXP-2** Sweep: скрипт кодера через MCP-клиент, `coder`" in section
    assert "EXP-3" not in section
    assert "0,12 $" in section
    assert "/alembic/builds/Survival-Toolkit-aa11" in section
    _svg_of(section)


def test_section_goes_before_discussion_and_replaces_its_old_copy(builds_dir):
    _write_build(builds_dir)
    once = tp.with_tool_provenance(REPORT, _state())
    twice = tp.with_tool_provenance(once, _state())

    assert twice.count(tp.START_MARK) == 1
    assert twice.index(tp.END_MARK) < twice.index("## Обсуждение")
    assert twice.index("## Результаты") < twice.index(tp.START_MARK)


def test_report_without_a_build_is_unchanged(builds_dir):
    state = _state()
    state["experiment_runtime"]["run_id"] = "EXRUN-other"
    state["experiment_runtime"]["tasks"] = {}
    _write_build(builds_dir)

    assert tp.with_tool_provenance(REPORT, state) == REPORT
    assert tp.with_tool_provenance(REPORT, {}) == REPORT


def test_build_found_by_session_scope_without_runtime(builds_dir):
    _write_build(builds_dir, run_id="", scopes=[["user_1", "session_1"]])

    out = tp.with_tool_provenance(REPORT, {"report_language": "en"}, ("user_1", "session_1"))
    assert "Alembic built an MCP server with 2 tools" in out
    assert "No plan task had the server attached." in out


def test_failed_build_is_reported_with_its_status(builds_dir):
    _write_build(builds_dir, status="failed")

    out = tp.with_tool_provenance(REPORT, _state())
    assert "сборка завершилась со статусом «failed»" in out
    assert "| Проверка | ❌ |" in out
    svg = _svg_of(out)
    assert svg.tag.endswith("svg")


def test_english_section(builds_dir):
    _write_build(builds_dir)
    out = tp.with_tool_provenance("# R\n\n## Discussion\n\nx\n", _state("en"))

    assert "## Tool from a repository: where it came from and how it was built" in out
    assert "| Validation | ✅ | 30 s |" in out
    assert "a Coder script through the MCP client" in out
    assert out.index(tp.END_MARK) < out.index("## Discussion")


def test_diagram_is_valid_svg_with_every_tool(builds_dir):
    record = _write_build(builds_dir)
    record["used_by"] = []
    svg = ET.fromstring(tp.render_svg(record, tp.build_details(record), "ru"))
    texts = " ".join(t.text or "" for t in svg.iter() if t.tag.endswith("text"))

    assert "calc_km" in texts and "calc_naive" in texts
    assert "5/6 тестов" in texts and ":21754/mcp" in texts


def test_inline_images_are_dropped_for_latex():
    md = "a\n\n![x](data:image/svg+xml;base64,QUJD)\n\nb ![y](figures/y.png)"
    assert tp.without_inline_images(md) == "a\n\nb ![y](figures/y.png)"


def test_finalize_writes_the_section_into_report_md(builds_dir, tmp_path, monkeypatch):
    from CoScientist.config.report import ReportConfig
    from CoScientist.reporting import finalize

    _write_build(builds_dir)
    monkeypatch.setattr(finalize, "_publish_to_research_graph", lambda *a, **k: None)
    monkeypatch.setattr(finalize, "_promote_sources", lambda *a, **k: {})
    config = ReportConfig.from_mapping({"reports_root": str(tmp_path / "reports")})

    result = finalize.finalize_report("session_1", REPORT, config, _state())
    written = (result.report_dir / "report.md").read_text(encoding="utf-8")

    assert tp.START_MARK in written and tp.START_MARK in result.markdown
    assert "calc_km" in written
