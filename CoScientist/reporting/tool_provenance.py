"""Where a tool the run built came from, and how Alembic built it.

The final report told the reader that a result came "through the MCP server
built from the authors' repository" and left it there. The build itself — the
repository, the stages, the tools that passed validation and the tasks that
called them — sat on the build page of one server and in a JSON dump the
aggregator pasted as a table. This module renders that as one section of the
report, from the build records and the experiment runtime, with no model in
between: the numbers in it are the ones the build wrote.

``finalize_report`` calls :func:`with_tool_provenance` on the aggregator's
markdown. A run that built nothing gets its report back unchanged.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

START_MARK = "<!-- coscientist:tool-provenance:start -->"
END_MARK = "<!-- coscientist:tool-provenance:end -->"
_BLOCK_RE = re.compile(re.escape(START_MARK) + r".*?" + re.escape(END_MARK) + r"\n*", re.S)
# The section goes in front of the first of these, else at the end.
_TAIL_HEADING_RE = re.compile(
    r"^##\s+(Обсуждение|Выводы|Заключение|Ограничения|Discussion|Conclusions?|Limitations)\b",
    re.M | re.I,
)
_DATA_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(data:image/[^)]+\)\n*")

# Stage keys in the order the pipeline runs them (stage_status.json).
_STAGES = ("explorer", "environment", "coder", "validator", "wrapper")

_WORDS: Dict[str, Dict[str, str]] = {
    "ru": {
        "heading": "Инструмент из репозитория: откуда взят и как собран",
        "lead": "Alembic собрал из репозитория {repo} MCP-сервер с {n_tools} за {duration}. "
                "Проверку прошли {passed} из {total}{tests}. {usage}",
        "lead_failed": "Alembic пытался собрать MCP-сервер из репозитория {repo} ({duration}), "
                       "сборка завершилась со статусом «{status}».",
        "tests": " (тесты {tp}/{tt})",
        "usage": "Сервер подключён к задачам {tasks}.",
        "usage_none": "Ни одна задача плана сервер не подключала.",
        "tools_word": ("тулом", "тулами", "тулами"),
        "figure": "Схема сборки",
        "field": "Параметр", "value": "Значение",
        "repo": "Репозиторий", "job": "Сборка", "build_page": "страница сборки",
        "when": "Время сборки", "env": "Окружение", "image": "Образ",
        "server": "Сервер", "cost": "Стоимость сборки",
        "tokens": "токенов", "two_venv": "два venv (сервер и репозиторий)",
        "one_venv": "один venv", "python": "Python",
        "stages": "Этапы сборки", "stage": "Этап", "result": "Итог",
        "time": "Время", "done": "Что сделано",
        "stage_names": {
            "explorer": "Разведка репозитория", "environment": "Окружение",
            "coder": "Код обёрток", "validator": "Проверка",
            "wrapper": "MCP-обёртка", "image": "Образ Docker и запуск",
        },
        "explorer_done": "проверено функций: {verified}",
        "explorer_dropped": ", отброшено: {dropped}",
        "environment_done": "{layout}",
        "environment_issues": "; замечаний: {n}",
        "coder_done": "обёрток: {n}",
        "validator_done": "тесты {tp}/{tt}, вызовы {ip}/{it}, раундов отладки: {rounds}",
        "wrapper_done": "в MCP обёрнуто: {n}",
        "wrapper_skipped": ", пропущено: {skipped}",
        "image_done": "сервер поднят: {url}",
        "tools": "Тулы: {passed} из {total} прошли проверку",
        "tool": "Тул", "target": "Функция", "purpose": "Назначение",
        "module": "Код тулов взят из {module}.",
        "deps": "Зависимости окружения: {deps}.",
        "checks": "Тесты · вызовы", "calls": "Пробные вызовы", "verdict": "Вердикт",
        "verdicts": {"perfect": "✅ все проверки", "passed": "☑️ прошёл",
                     "failed": "❌ не прошёл", "untested": "⚪ не проверен"},
        "used": "Где использован в исследовании",
        "task": "Задача", "route": "Маршрут", "how": "Как подключён", "status": "Статус",
        "how_builder": "собрал сервер, агент проверил тулы",
        "how_agent": "тулы сервера у агента",
        "how_coder": "скрипт кодера через MCP-клиент",
        "status_names": {"done": "✅ выполнена", "done_with_warnings": "⚠️ с замечаниями",
                         "failed": "❌ не выполнена", "blocked": "⛔ заблокирована",
                         "skipped": "⏭️ пропущена"},
        "min": "мин", "sec": "с",
        "svg_title": "{repo} → MCP-сервер · {n} · {duration}",
        "svg_repo": "Репозиторий GitHub",
        "svg_server": "MCP-сервер",
        "svg_tools": "Тулы ({passed}/{total} прошли проверку)",
        "svg_used": "Подключён к задачам",
        "svg_tools_n": ("тул", "тула", "тулов"),
        "svg_stage": {"explorer": "Разведка", "environment": "Окружение", "coder": "Обёртки",
                      "validator": "Проверка", "wrapper": "MCP", "image": "Docker"},
        "svg_detail": {"explorer": ("функция", "функции", "функций"),
                       "coder": ("обёртка", "обёртки", "обёрток"),
                       "validator": "{tp}/{tt} тестов", "wrapper": ("тул", "тула", "тулов")},
        "svg_how": {"how_builder": "сборка и агент", "how_agent": "агент",
                    "how_coder": "кодер, MCP-клиент"},
        "failed_word": "ошибка",
    },
    "en": {
        "heading": "Tool from a repository: where it came from and how it was built",
        "lead": "Alembic built an MCP server with {n_tools} from {repo} in {duration}. "
                "{passed} of {total} passed validation{tests}. {usage}",
        "lead_failed": "Alembic tried to build an MCP server from {repo} ({duration}); "
                       "the build ended with status \"{status}\".",
        "tests": " (tests {tp}/{tt})",
        "usage": "The server was attached to tasks {tasks}.",
        "usage_none": "No plan task had the server attached.",
        "tools_word": ("tool", "tools", "tools"),
        "figure": "Build diagram",
        "field": "Field", "value": "Value",
        "repo": "Repository", "job": "Build", "build_page": "build page",
        "when": "Build time", "env": "Environment", "image": "Image",
        "server": "Server", "cost": "Build cost",
        "tokens": "tokens", "two_venv": "two venvs (server and repository)",
        "one_venv": "one venv", "python": "Python",
        "stages": "Build stages", "stage": "Stage", "result": "Result",
        "time": "Time", "done": "What it did",
        "stage_names": {
            "explorer": "Repository exploration", "environment": "Environment",
            "coder": "Wrapper code", "validator": "Validation",
            "wrapper": "MCP wrapper", "image": "Docker image and serve",
        },
        "explorer_done": "functions verified: {verified}",
        "explorer_dropped": ", dropped: {dropped}",
        "environment_done": "{layout}",
        "environment_issues": "; issues: {n}",
        "coder_done": "wrappers: {n}",
        "validator_done": "tests {tp}/{tt}, calls {ip}/{it}, debugger rounds: {rounds}",
        "wrapper_done": "wrapped into MCP: {n}",
        "wrapper_skipped": ", skipped: {skipped}",
        "image_done": "server up: {url}",
        "tools": "Tools: {passed} of {total} passed validation",
        "tool": "Tool", "target": "Function", "purpose": "Purpose",
        "module": "The tools wrap functions of {module}.",
        "deps": "Environment dependencies: {deps}.",
        "checks": "Tests · calls", "calls": "Trial calls", "verdict": "Verdict",
        "verdicts": {"perfect": "✅ all checks", "passed": "☑️ passed",
                     "failed": "❌ failed", "untested": "⚪ untested"},
        "used": "Where the study used it",
        "task": "Task", "route": "Route", "how": "How it was attached", "status": "Status",
        "how_builder": "built the server, the agent checked its tools",
        "how_agent": "the server's tools in the agent's toolset",
        "how_coder": "a Coder script through the MCP client",
        "status_names": {"done": "✅ done", "done_with_warnings": "⚠️ with warnings",
                         "failed": "❌ failed", "blocked": "⛔ blocked",
                         "skipped": "⏭️ skipped"},
        "min": "min", "sec": "s",
        "svg_title": "{repo} → MCP server · {n} · {duration}",
        "svg_repo": "GitHub repository",
        "svg_server": "MCP server",
        "svg_tools": "Tools ({passed}/{total} passed validation)",
        "svg_used": "Attached to tasks",
        "svg_tools_n": ("tool", "tools", "tools"),
        "svg_stage": {"explorer": "Explore", "environment": "Environment", "coder": "Wrappers",
                      "validator": "Validate", "wrapper": "MCP", "image": "Docker"},
        "svg_detail": {"explorer": ("function", "functions", "functions"),
                       "coder": ("wrapper", "wrappers", "wrappers"),
                       "validator": "{tp}/{tt} tests", "wrapper": ("tool", "tools", "tools")},
        "svg_how": {"how_builder": "build and agent", "how_agent": "agent",
                    "how_coder": "Coder, MCP client"},
        "failed_word": "error",
    },
}


# ── data ────────────────────────────────────────────────────────────────────
def _norm_repo(url: Any) -> str:
    return str(url or "").strip().rstrip("/").removesuffix(".git").lower()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _build_records() -> List[Dict[str, Any]]:
    """Every build record on this host: the durable job file merged over the
    log-side meta file, keyed by job id."""
    from CoScientist.tools import alembic_tools as at

    merged: Dict[str, Dict[str, Any]] = {}
    for folder in (at.LOG_DIR, at.JOB_METADATA_DIR):
        if not folder.exists():
            continue
        for path in folder.glob("*.json"):
            record = _read_json(path)
            if isinstance(record, dict) and record.get("job_id"):
                merged.setdefault(record["job_id"], {}).update(record)
    return list(merged.values())


def _session_usage(state: Mapping[str, Any]) -> Dict[str, Any]:
    """The run id, the served URLs and, per URL, the tasks that used it."""
    runtime = state.get("experiment_runtime") if isinstance(state, Mapping) else None
    runtime = runtime if isinstance(runtime, Mapping) else {}
    by_url: Dict[str, List[Dict[str, Any]]] = {}
    repos: set = set()
    order = list(runtime.get("task_order") or []) or list((runtime.get("tasks") or {}).keys())
    tasks = runtime.get("tasks") or {}
    for tid in order:
        entry = tasks.get(tid)
        if not isinstance(entry, Mapping):
            continue
        task = entry.get("task") if isinstance(entry.get("task"), Mapping) else {}
        history = [h for h in entry.get("route_history") or [] if isinstance(h, Mapping)]
        built = any(str(h.get("route") or "") == "alembic_build" for h in history)
        if built:
            repos.add(_norm_repo(task.get("repo_url")))
        for server in task.get("mcp_servers") or []:
            if not isinstance(server, Mapping) or str(server.get("source") or "") != "alembic":
                continue
            url = str(server.get("url") or "").strip()
            if not url.startswith("http"):
                continue
            route = str(entry.get("current_route") or task.get("route") or "")
            how = "how_builder" if built else ("how_coder" if route == "coder" else "how_agent")
            by_url.setdefault(url, []).append({
                "id": str(task.get("id") or tid),
                "name": str(task.get("name") or ""),
                "route": route,
                "how": how,
                "status": str(entry.get("status") or ""),
            })
    return {"run_id": str(runtime.get("run_id") or ""), "by_url": by_url, "repos": repos}


def _in_scope(record: Mapping[str, Any], scope: tuple) -> bool:
    user_id, session_id = scope
    for item in record.get("scopes") or []:
        if isinstance(item, (list, tuple)) and len(item) == 2 and str(item[1]) == session_id:
            return not user_id or str(item[0]) == user_id
    return False


def session_builds(state: Mapping[str, Any], scope: tuple = ("", "")) -> List[Dict[str, Any]]:
    """The Alembic builds this session ran or used, oldest first, each with
    the tasks that called its server under ``used_by``."""
    usage = _session_usage(state)
    run_id = usage["run_id"]
    urls = usage["by_url"]
    picked: List[Dict[str, Any]] = []
    for record in _build_records():
        mine = bool(run_id) and str(record.get("run_id") or "") == run_id
        mine = mine or (bool(scope[1]) and _in_scope(record, scope))
        url = str(record.get("mcp_url") or "").strip()
        if not mine and not (url and url in urls and str(record.get("status")) == "done"):
            continue
        entry = dict(record)
        entry["used_by"] = urls.get(url, []) if str(record.get("status")) == "done" else []
        picked.append(entry)
    # A reused server can match two builds of one repository by URL. Keep the
    # one this run started, else the newest.
    by_url: Dict[str, Dict[str, Any]] = {}
    rest: List[Dict[str, Any]] = []
    for entry in sorted(picked, key=lambda r: r.get("started_at") or 0):
        url = str(entry.get("mcp_url") or "")
        if not url or str(entry.get("status")) != "done":
            rest.append(entry)
            continue
        kept = by_url.get(url)
        if kept is None or str(kept.get("run_id") or "") != run_id:
            by_url[url] = entry
    return sorted(rest + list(by_url.values()), key=lambda r: r.get("started_at") or 0)


def build_details(record: Mapping[str, Any]) -> Dict[str, Any]:
    """What the build's reports say: stages, tools, environment, cost."""
    out: Dict[str, Any] = {"stages": {}, "tools": [], "metrics": {}, "env": {}, "counts": {}}
    workdir, repo_url = record.get("workdir"), record.get("repo_url")
    if not workdir or not repo_url:
        return out
    from CoScientist.alembic.web import artifacts

    reports = artifacts.reports_dir(Path(workdir), str(repo_url))
    stages = _read_json(reports / "stage_status.json")
    metrics = _read_json(reports / "metrics.json")
    plan = _read_json(reports / "plan.json")
    validation = _read_json(reports / "validation.json")
    out["stages"] = stages if isinstance(stages, dict) else {}
    out["metrics"] = metrics if isinstance(metrics, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    out["env"] = plan.get("env") if isinstance(plan.get("env"), dict) else {}
    planned = {t.get("name"): t for t in plan.get("tools") or [] if isinstance(t, dict)}
    checked = {}
    if isinstance(validation, dict):
        checked = {t.get("name"): t for t in validation.get("tools") or [] if isinstance(t, dict)}
        out["counts"] = validation.get("counts") if isinstance(validation.get("counts"), dict) else {}
        out["debugger_rounds"] = validation.get("debugger_rounds")
    names = list(checked) or list(planned)
    if not names:
        names = list(((out["stages"].get("wrapper") or {}).get("gate") or {}).get("tools_wrapped") or [])
    for name in names:
        if not name:
            continue
        p, v = planned.get(name) or {}, checked.get(name) or {}
        out["tools"].append({
            "name": name,
            "target": str(p.get("target") or ""),
            "purpose": str(p.get("purpose") or ""),
            "tests": _ratio(v.get("tests_passed"), v.get("tests_total")),
            "calls": _ratio(v.get("invoc_passed"), v.get("invoc_total")),
            "status": str(v.get("status") or ("untested" if not v else "")) or "untested",
        })
    return out


def _ratio(done: Any, total: Any) -> str:
    return f"{done or 0}/{total}" if isinstance(total, int) and total else ""


# ── formatting ──────────────────────────────────────────────────────────────
def _plural(n: int, forms: tuple) -> str:
    """Russian plural forms (one, few, many); English uses the first two."""
    n = abs(int(n))
    if forms[0].isascii():
        return forms[0] if n == 1 else forms[1]
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        return forms[1]
    return forms[2]


def _duration(seconds: Any, w: Mapping[str, Any]) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "—"
    minutes, sec = divmod(total, 60)
    if minutes:
        return f"{minutes} {w['min']} {sec} {w['sec']}"
    return f"{sec} {w['sec']}"


def _money(value: Any, lang: str) -> str:
    try:
        text = f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""
    return (text.replace(".", ",") if lang == "ru" else text) + " $"


def _thousands(value: Any, lang: str) -> str:
    try:
        text = f"{int(value):,}"
    except (TypeError, ValueError):
        return ""
    return text.replace(",", "\u00a0" if lang == "ru" else ",")


def _cell(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def _repo_label(repo_url: str) -> str:
    parts = str(repo_url or "").rstrip("/").removesuffix(".git").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(repo_url or "")


def _stage_rows(record: Mapping[str, Any], details: Mapping[str, Any], w: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """One row per pipeline stage, plus the image/serve step the stages do not
    time themselves (the rest of the wall clock)."""
    stages = details.get("stages") or {}
    durations = (details.get("metrics") or {}).get("durations_per_stage") or {}
    counts = details.get("counts") or {}
    rows: List[Dict[str, Any]] = []
    for key in _STAGES:
        stage = stages.get(key)
        if not isinstance(stage, dict):
            continue
        gate = stage.get("gate") if isinstance(stage.get("gate"), dict) else {}
        if key == "explorer":
            done = w["explorer_done"].format(verified=gate.get("verified", "—"))
            if gate.get("dropped"):
                done += w["explorer_dropped"].format(dropped=gate["dropped"])
        elif key == "environment":
            env = details.get("env") or {}
            layout = w["two_venv"] if (env.get("layout") or gate.get("layout")) == "two-venv" else w["one_venv"]
            if env.get("server_python"):
                layout += f", {w['python']} {env['server_python']}"
            done = w["environment_done"].format(layout=layout)
            issues = len(gate.get("hard") or []) + len(gate.get("soft") or [])
            if issues:
                done += w["environment_issues"].format(n=issues)
        elif key == "coder":
            done = w["coder_done"].format(n=len(gate.get("tools") or []))
        elif key == "validator":
            c = stage.get("counts") if isinstance(stage.get("counts"), dict) else counts
            done = w["validator_done"].format(
                tp=c.get("tests_passed", 0), tt=c.get("tests_total", 0),
                ip=c.get("invoc_passed", 0), it=c.get("invoc_total", 0),
                rounds=stage.get("debugger_rounds", details.get("debugger_rounds", 0)) or 0)
        else:
            done = w["wrapper_done"].format(n=len(gate.get("tools_wrapped") or []))
            if gate.get("skipped"):
                done += w["wrapper_skipped"].format(skipped=len(gate["skipped"]))
        if stage.get("error"):
            done += f"; {w['failed_word']}: {_cell(str(stage['error'])[:160])}"
        rows.append({"key": key, "name": w["stage_names"][key],
                     "status": str(stage.get("status") or ""),
                     "seconds": durations.get(key), "done": done,
                     "detail": _stage_detail(key, stage, gate, details, w)})
    elapsed = _elapsed(record)
    timed = sum(float(r["seconds"]) for r in rows if isinstance(r["seconds"], (int, float)))
    if rows and elapsed and elapsed - timed > 1 and str(record.get("status")) == "done":
        rows.append({"key": "image", "name": w["stage_names"]["image"], "status": "passed",
                     "seconds": elapsed - timed,
                     "done": w["image_done"].format(url=f"`{_port(record)}/mcp`" if _port(record) else "—"),
                     "detail": _port(record)})
    return rows


def _stage_detail(key: str, stage: Mapping[str, Any], gate: Mapping[str, Any],
                  details: Mapping[str, Any], w: Mapping[str, Any]) -> str:
    """The one number a stage box in the picture shows."""
    template = w["svg_detail"].get(key)
    if key == "environment":
        python = (details.get("env") or {}).get("server_python")
        return f"Python {python}" if python else ""
    if not template:
        return ""
    if key == "validator":
        c = stage.get("counts") if isinstance(stage.get("counts"), dict) else details.get("counts") or {}
        return template.format(tp=c.get("tests_passed", 0), tt=c.get("tests_total", 0)) if c.get("tests_total") else ""
    listed = {"explorer": None, "coder": "tools", "wrapper": "tools_wrapped"}[key]
    n = gate.get("verified") if listed is None else len(gate.get(listed) or [])
    return f"{n} {_plural(n, template)}" if isinstance(n, int) and n else ""


def _port(record: Mapping[str, Any]) -> str:
    found = re.search(r":(\d+)/", str(record.get("mcp_url") or ""))
    return f":{found.group(1)}" if found else ""


def _elapsed(record: Mapping[str, Any]) -> Optional[float]:
    try:
        return float(record["finished_at"]) - float(record["started_at"])
    except (KeyError, TypeError, ValueError):
        return None


def _stage_mark(status: str) -> str:
    return {"passed": "✅", "failed": "❌", "skipped": "⏭️", "running": "⏳"}.get(status, "⚪")


def _passed_tools(tools: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for t in tools if t.get("status") in ("perfect", "passed"))


# ── the diagram ─────────────────────────────────────────────────────────────
_C = {
    "bg": "#ffffff", "border": "#dbe2ea", "ink": "#0f172a", "muted": "#64748b",
    "ok_fill": "#ecfdf5", "ok_line": "#10b981", "ok_ink": "#047857",
    "warn_fill": "#fffbeb", "warn_line": "#f59e0b", "warn_ink": "#b45309",
    "bad_fill": "#fef2f2", "bad_line": "#ef4444", "bad_ink": "#b91c1c",
    "idle_fill": "#f1f5f9", "idle_line": "#94a3b8", "idle_ink": "#475569",
    "repo_fill": "#eef2ff", "repo_line": "#6366f1", "repo_ink": "#3730a3",
    "srv_fill": "#f0f9ff", "srv_line": "#0284c7", "srv_ink": "#075985",
}
_FONT = "Inter, 'Segoe UI', Roboto, Arial, sans-serif"


def _tone(status: str) -> str:
    if status in ("passed", "perfect", "done"):
        return "ok"
    if status in ("done_with_warnings",):
        return "warn"
    if status in ("failed", "blocked", "error"):
        return "bad"
    return "idle"


def _clip(text: str, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def _text(x: float, y: float, body: str, size: int = 12, color: str = "", weight: int = 400,
          anchor: str = "middle") -> str:
    return (f'<text x="{x:.0f}" y="{y:.0f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{color or _C["ink"]}" text-anchor="{anchor}">{escape(body)}</text>')


def _box(x: float, y: float, w: float, h: float, fill: str, line: str, radius: int = 10) -> str:
    return (f'<rect x="{x:.0f}" y="{y:.0f}" width="{w:.0f}" height="{h:.0f}" rx="{radius}" '
            f'fill="{fill}" stroke="{line}" stroke-width="1.5"/>')


def _chips(items: List[Dict[str, str]], x0: float, y0: float, width: float) -> tuple:
    """Rounded chips laid out left to right, wrapping at ``width``. Returns
    the SVG and the y below the last row."""
    parts: List[str] = []
    x, y, row_h, gap = x0, y0, 30, 8
    for item in items:
        label = _clip(item["label"], 40)
        chip_w = min(width, 28 + 7.6 * len(label))
        if x > x0 and x + chip_w > x0 + width:
            x, y = x0, y + row_h + gap
        tone = item.get("tone") or "idle"
        parts.append(_box(x, y, chip_w, row_h, _C[f"{tone}_fill"], _C[f"{tone}_line"], radius=15))
        parts.append(f'<circle cx="{x + 14:.0f}" cy="{y + row_h / 2:.0f}" r="4" fill="{_C[f"{tone}_line"]}"/>')
        parts.append(_text(x + 24, y + 20, label, 13, _C[f"{tone}_ink"], 500, "start"))
        x += chip_w + gap
    return "".join(parts), y + row_h


def _down_arrow(x: float, y1: float, y2: float) -> str:
    return (f'<line x1="{x:.0f}" y1="{y1:.0f}" x2="{x:.0f}" y2="{y2 - 6:.0f}" stroke="{_C["idle_line"]}" '
            f'stroke-width="1.5"/><path d="M{x - 4:.0f},{y2 - 7:.0f} L{x:.0f},{y2:.0f} L{x + 4:.0f},{y2 - 7:.0f} z" '
            f'fill="{_C["idle_line"]}"/>')


def render_svg(record: Mapping[str, Any], details: Mapping[str, Any], lang: str = "ru") -> str:
    """A self-contained picture of one build, top to bottom: the repository,
    the stages, the served server, its tools with their verdicts, and the
    tasks it was attached to. Sized for the report panel, about 600 px wide."""
    w = _WORDS[lang if lang in _WORDS else "en"]
    width, pad = 660, 22
    inner = width - 2 * pad
    rows = _stage_rows(record, details, w)
    tools = details.get("tools") or []
    used = record.get("used_by") or []
    repo = _repo_label(str(record.get("repo_url") or ""))
    n_tools = len(tools)
    tools_n = f"{n_tools} {_plural(n_tools, w['svg_tools_n'])}"
    parts: List[str] = []

    title = w["svg_title"].format(repo=repo.split("/")[-1], n=tools_n,
                                  duration=_duration(_elapsed(record), w))
    parts.append(_text(pad, 34, _clip(title, 72), 15, _C["ink"], 700, "start"))

    # The repository.
    y, h = 50, 52
    parts.append(_box(pad, y, inner, h, _C["repo_fill"], _C["repo_line"]))
    parts.append(_text(pad + 16, y + 21, w["svg_repo"], 11, _C["repo_ink"], 600, "start"))
    parts.append(_text(pad + 16, y + 40, _clip(repo, 70), 14, _C["repo_ink"], 700, "start"))
    y += h
    parts.append(_down_arrow(width / 2, y + 2, y + 20))
    y += 20

    # The stages, one box each.
    n = max(len(rows), 1)
    gap, h = 8, 96
    stage_w = (inner - gap * (n - 1)) / n
    x = pad
    for row in rows:
        tone = _tone(row["status"])
        cx = x + stage_w / 2
        parts.append(_box(x, y, stage_w, h, _C[f"{tone}_fill"], _C[f"{tone}_line"]))
        mark = {"ok": "✓", "bad": "✗", "warn": "!"}.get(tone, "•")
        parts.append(_text(cx, y + 24, mark, 17, _C[f"{tone}_ink"], 700))
        limit = max(6, int(stage_w / 7.5))
        parts.append(_text(cx, y + 46, _clip(w["svg_stage"][row["key"]], limit), 12, _C["ink"], 700))
        if row.get("detail"):
            parts.append(_text(cx, y + 64, _clip(row["detail"], limit + 2), 11, _C[f"{tone}_ink"], 500))
        parts.append(_text(cx, y + 82, _duration(row["seconds"], w), 11, _C["muted"]))
        x += stage_w + gap
    y += h
    parts.append(_down_arrow(width / 2, y + 2, y + 20))
    y += 20

    # The server.
    h = 52
    done = str(record.get("status")) == "done"
    tone = "srv" if done else "bad"
    parts.append(_box(pad, y, inner, h, _C[f"{tone}_fill"], _C[f"{tone}_line"]))
    parts.append(_text(pad + 16, y + 21, w["svg_server"], 11, _C[f"{tone}_ink"], 600, "start"))
    port = _port(record)
    headline = f"{port}/mcp · {tools_n}" if done and port else str(record.get("status") or "")
    parts.append(_text(pad + 16, y + 40, headline, 14, _C[f"{tone}_ink"], 700, "start"))
    y += h

    # Tools.
    if tools:
        y += 30
        passed = _passed_tools(tools)
        parts.append(_text(pad, y, w["svg_tools"].format(passed=passed, total=n_tools), 13, _C["ink"], 700, "start"))
        chips = [{"label": t["name"] + (f"  {t['tests']}" if t.get("tests") else ""),
                  "tone": _tone(t.get("status") or "")} for t in tools]
        svg, y = _chips(chips, pad, y + 10, inner)
        parts.append(svg)

    # Tasks.
    if used:
        y += 30
        parts.append(_text(pad, y, w["svg_used"], 13, _C["ink"], 700, "start"))
        chips = [{"label": f"{u['id']} · {w['svg_how'][u['how']]}", "tone": _tone(u.get("status") or "")}
                 for u in used]
        svg, y = _chips(chips, pad, y + 10, inner)
        parts.append(svg)

    height = int(y + 22)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="{_FONT}">'
            f'<rect x="0.75" y="0.75" width="{width - 1.5}" height="{height - 1.5}" rx="14" '
            f'fill="{_C["bg"]}" stroke="{_C["border"]}" stroke-width="1.5"/>'
            + "".join(parts) + "</svg>")


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# ── the section ─────────────────────────────────────────────────────────────
def _usage_line(used: List[Mapping[str, Any]], w: Mapping[str, Any]) -> str:
    if not used:
        return w["usage_none"]
    return w["usage"].format(tasks=", ".join(u["id"] for u in used))


def render_build(record: Mapping[str, Any], details: Mapping[str, Any], lang: str = "ru") -> str:
    """Markdown for one build: lead, diagram, facts, stages, tools, tasks."""
    w = _WORDS[lang if lang in _WORDS else "en"]
    repo_url = str(record.get("repo_url") or "")
    repo = f"[{_repo_label(repo_url)}]({repo_url})" if repo_url else "—"
    tools = details.get("tools") or []
    used = list(record.get("used_by") or [])
    counts = details.get("counts") or {}
    duration = _duration(_elapsed(record), w)
    out: List[str] = []

    if str(record.get("status")) == "done":
        tests = w["tests"].format(tp=counts["tests_passed"], tt=counts["tests_total"]) \
            if counts.get("tests_total") else ""
        out.append(w["lead"].format(
            repo=repo, n_tools=f"{len(tools)} {_plural(len(tools), w['tools_word'])}",
            duration=duration, passed=_passed_tools(tools), total=len(tools),
            tests=tests, usage=_usage_line(used, w)))
    else:
        out.append(w["lead_failed"].format(repo=repo, duration=duration, status=record.get("status")))

    out.append(f"![{w['figure']}: {_repo_label(repo_url)}]({_data_uri(render_svg(record, details, lang))})")

    facts = [(w["repo"], repo)]
    job_id = str(record.get("job_id") or "")
    if job_id:
        facts.append((w["job"], f"`{job_id}` · [{w['build_page']}](/alembic/builds/{job_id})"))
    started, finished = record.get("started_at"), record.get("finished_at")
    if isinstance(started, (int, float)):
        when = datetime.fromtimestamp(started).strftime("%d.%m.%Y %H:%M")
        if isinstance(finished, (int, float)):
            when += "–" + datetime.fromtimestamp(finished).strftime("%H:%M")
        facts.append((w["when"], f"{duration} ({when})"))
    if record.get("image"):
        facts.append((w["image"], f"`{record['image']}`"))
    if record.get("mcp_url"):
        facts.append((w["server"], f"`{record['mcp_url']}`"))
    metrics = details.get("metrics") or {}
    if metrics.get("total_cost_usd") is not None:
        cost = _money(metrics["total_cost_usd"], lang)
        if metrics.get("total_tokens"):
            cost += f" ({_thousands(metrics['total_tokens'], lang)} {w['tokens']})"
        facts.append((w["cost"], cost))
    out.append("\n".join([f"| {w['field']} | {w['value']} |", "| --- | --- |"]
                         + [f"| {k} | {_cell(v)} |" for k, v in facts]))

    # The report panel scrolls a wide table sideways rather than wrap it, so
    # the tables keep short cells and the prose goes into lists below them.
    rows = _stage_rows(record, details, w)
    if rows:
        table = [f"#### {w['stages']}", "",
                 f"| {w['stage']} | {w['result']} | {w['time']} | {w['done']} |",
                 "| --- | :---: | ---: | --- |"]
        table += [f"| {r['name']} | {_stage_mark(r['status'])} | {_duration(r['seconds'], w)} | {_cell(r['done'])} |"
                  for r in rows]
        out.append("\n".join(table))
        deps = [str(d) for d in (details.get("env") or {}).get("dependencies") or []]
        if deps:
            out.append(w["deps"].format(deps=", ".join(f"`{d}`" for d in deps)))

    if tools:
        modules = {t["target"].split(":", 1)[0] for t in tools if ":" in (t.get("target") or "")}
        single = len(modules) == 1 and all(":" in (t.get("target") or "") for t in tools)
        table = [f"#### {w['tools'].format(passed=_passed_tools(tools), total=len(tools))}", ""]
        if single:
            table += [w["module"].format(module=f"`{next(iter(modules)).replace('.', '/')}.py`"), ""]
        table += [f"| {w['tool']} | {w['target']} | {w['checks']} | {w['verdict']} |",
                  "| --- | --- | :---: | --- |"]
        for t in tools:
            target = t.get("target") or ""
            target = target.split(":", 1)[1] if single else target
            verdict = w["verdicts"].get(t.get("status") or "untested", t.get("status") or "")
            checks = " · ".join(x for x in (t.get("tests"), t.get("calls")) if x) or "—"
            table.append(f"| `{t['name']}` | {('`' + target + '`') if target else '—'} | {checks} | {verdict} |")
        out.append("\n".join(table))
        described = [t for t in tools if t.get("purpose")]
        if described:
            out.append("\n".join(f"- `{t['name']}`: {_clip(t['purpose'], 220)}" for t in described))

    if used:
        lines = [f"#### {w['used']}", ""]
        for u in used:
            mark = w["status_names"].get(u.get("status") or "", "").split(" ", 1)[0] or "•"
            name = f" {u['name']}" if u.get("name") else ""
            lines.append(f"- {mark} **{u['id']}**{name}: {w[u['how']]}, `{u['route']}`")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def render_section(builds: List[Mapping[str, Any]], lang: str = "ru") -> str:
    """The whole section, between markers, or '' when there is no build."""
    if not builds:
        return ""
    w = _WORDS[lang if lang in _WORDS else "en"]
    parts = [START_MARK, f"## {w['heading']}"]
    for record in builds:
        try:
            details = build_details(record)
        except Exception as exc:  # noqa: BLE001 - a build without reports still has a record
            logger.warning("tool provenance: reports of %s unreadable (%s)", record.get("job_id"), exc)
            details = {}
        if len(builds) > 1:
            parts.append(f"### {_repo_label(str(record.get('repo_url') or ''))}")
        parts.append(render_build(record, details, lang))
    parts.append(END_MARK)
    return "\n\n".join(parts) + "\n"


def insert_section(markdown: str, section: str) -> str:
    """Put the section in front of the discussion, replacing an earlier copy."""
    text = _BLOCK_RE.sub("", markdown or "")
    if not section:
        return text
    match = _TAIL_HEADING_RE.search(text)
    if match:
        return text[: match.start()] + section + "\n" + text[match.start():]
    return text.rstrip() + "\n\n" + section


def with_tool_provenance(markdown: str, state: Optional[Mapping[str, Any]], scope: tuple = ("", "")) -> str:
    """The report with its tool-provenance section. Never raises."""
    try:
        from CoScientist.agents.callbacks.report_language import session_report_language

        builds = session_builds(state or {}, scope)
        if not builds:
            return markdown
        return insert_section(markdown, render_section(builds, session_report_language(state or {})))
    except Exception as exc:  # noqa: BLE001 - the report outranks this section
        logger.warning("tool provenance: section skipped (%s)", exc)
        return markdown


def without_inline_images(markdown: str) -> str:
    """The markdown without data-URI images, for renderers that need a file."""
    return _DATA_IMAGE_RE.sub("", markdown or "")


__all__ = [
    "build_details",
    "insert_section",
    "render_build",
    "render_section",
    "render_svg",
    "session_builds",
    "with_tool_provenance",
    "without_inline_images",
]
