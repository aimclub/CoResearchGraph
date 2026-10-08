#!/usr/bin/env python3
"""Benchmark the alembic pipeline against a list of repos in parallel.

The base image (``alembic-base:latest``) is built once up-front, then N
workers run ``start_chain.py --no-serve`` concurrently. Every metric is
extracted from the structured run data the pipeline writes as it goes
(stage_status.json / validation.json / metrics.json) — no report parsing (R2).
Final artefacts (tools/, tests/, server.py, setup.sh, tmbench/) are copied out
of each committed image into the run folder (R3).

Usage:
    python benchmarks/alembic/run_benchmark.py \\
        --repos https://github.com/Roestlab/massformer https://github.com/whitead/synspace

    python benchmarks/alembic/run_benchmark.py --repos-file repos.txt --parallel 8

    # TM-Bench mode: tasks grouped by repo (two tasks on one repo = one dual run);
    # data dir is bind-mounted; images additionally tagged
    # toolmaker-runtime:installed-<task> for scoring in ToolMaker's harness.
    python benchmarks/alembic/run_benchmark.py \\
        --tasks tasks/stamp_extract_features.yaml tasks/stamp_train_classification_model.yaml \\
        --mount-dir /path/to/ToolMaker/benchmark/data
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import yaml

# benchmarks/alembic/run_benchmark.py → project root is 2 levels up
PROJECT_ROOT    = Path(__file__).resolve().parents[2]
COSCIENTIST_DIR = PROJECT_ROOT / "CoScientist"
RUNS_DIR        = Path(__file__).resolve().parent / "runs"

sys.path.insert(0, str(COSCIENTIST_DIR))

from alembic.common import get_repo_name, ensure_base_image

START_CHAIN = COSCIENTIST_DIR / "alembic" / "start_chain.py"
DOCKERFILE  = PROJECT_ROOT / "docker" / "alembic" / "Dockerfile"

AVAILABILITY_TIMEOUT = 15  # seconds — cheap network check, no clone
PIPELINE_STAGES = ("explorer", "environment", "coder", "validator", "wrapper")


def check_repo_available(repo_url: str, timeout: int = AVAILABILITY_TIMEOUT) -> tuple[bool, str]:
    """True if ``repo_url`` is reachable and has at least one ref, via
    ``git ls-remote`` — a dead/private/empty repo is caught in seconds instead
    of burning a full pipeline run."""
    try:
        r = subprocess.run(
            ["git", "ls-remote", "--exit-code", "--heads", repo_url],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if r.returncode == 0:
        return True, ""
    lines = [l.strip() for l in (r.stderr or r.stdout).splitlines() if l.strip()]
    return False, (lines[0] if lines else f"git ls-remote exit {r.returncode}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-repo job execution
# ══════════════════════════════════════════════════════════════════════════════
def run_one(job: dict, extra_args: list[str], log_dir: Path, out_dir: Path,
            idx: int, total: int) -> dict:
    """Invoke start_chain.py --no-serve for one repo; stream logs to a file;
    extract run data + artefacts from the committed image."""
    import os
    name     = job["name"]
    log_path = log_dir / f"{name}.log"
    print(f"[bench] ↑ start  {name}  ({idx}/{total})  log: {log_path.name}", flush=True)

    env = os.environ.copy()
    if job.get("tasks"):
        env["ALEMBIC_TASKS"] = json.dumps(job["tasks"])

    started = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        cmd = [sys.executable, str(START_CHAIN), job["url"], "--no-serve", *extra_args]
        logf.write(f"$ {' '.join(cmd)}\n\n")
        logf.flush()
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started

    record: dict = {
        "repo":        name,
        "url":         job["url"],
        "tasks":       [t.get("name") for t in job.get("tasks", [])],
        "elapsed_sec": round(elapsed, 1),
        "exit_code":   r.returncode,
        "log":         str(log_path),
        "error_tail":  None,
    }
    if r.returncode == 0:
        record.update(extract_run_data(name))
        export_artefacts(name, out_dir / name)
        for task_name in record["tasks"]:
            tag = f"toolmaker-runtime:installed-{task_name}"
            subprocess.run(["docker", "tag", f"alembic-tool:{name}", tag],
                           capture_output=True)
            print(f"[bench]   tagged {tag}", flush=True)
    else:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            record["error_tail"] = "\n".join(lines[-60:])
        except OSError:
            pass

    status = "ok" if r.returncode == 0 else f"exit={r.returncode}"
    print(f"[bench] ↓ done   {name}  ({elapsed:.0f}s, {status})", flush=True)
    return record


def _image_json(image: str, path: str) -> dict | None:
    r = subprocess.run(["docker", "run", "--rm", "--entrypoint", "cat", image, path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def extract_run_data(repo: str) -> dict:
    """Pull the structured run data out of the committed image (R2)."""
    image = f"alembic-tool:{repo}"
    base  = f"/work/.alembic/{repo}/reports"
    return {
        "stage_status":     _image_json(image, f"{base}/stage_status.json"),
        "validation":       _image_json(image, f"{base}/validation.json"),
        "pipeline_metrics": _image_json(image, f"{base}/metrics.json"),
        "pipeline_error":   _image_json(image, f"{base}/error.json"),
    }


_ARTEFACTS = ["output/server.py", "output/setup.sh", "output/tools",
              "output/tests", "output/helpers", "output/tmbench", "reports"]


def export_artefacts(repo: str, dest: Path) -> None:
    """Copy the run's final artefacts (NOT the venvs) out of the image (R3)."""
    image = f"alembic-tool:{repo}"
    c = subprocess.run(["docker", "create", image], capture_output=True, text=True)
    if c.returncode != 0:
        return
    cid = c.stdout.strip()
    try:
        dest.mkdir(parents=True, exist_ok=True)
        for item in _ARTEFACTS:
            subprocess.run(
                ["docker", "cp", f"{cid}:/work/.alembic/{repo}/{item}", str(dest / Path(item).name)],
                capture_output=True)
    finally:
        subprocess.run(["docker", "rm", cid], capture_output=True)


# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
def _stage_reached(record: dict) -> str:
    ss = record.get("stage_status") or {}
    reached = "-"
    for s in PIPELINE_STAGES:
        if s in ss:
            mark = "✓" if ss[s].get("status") == "passed" else "✗"
            resets = ss[s].get("resets", 0)
            reached = f"{s} {mark}" + (f" (r{resets})" if resets else "")
    return reached


def _frac(passed, total) -> str:
    return "-" if not total else f"{passed or 0}/{total}"


def _row(record: dict) -> str:
    c = (record.get("validation") or {}).get("counts") or {}
    not_run = record["exit_code"] is None
    if not_run:
        overall = record.get("error", "not run")
        return (f"| {record['repo']} | 0s | — | {overall} | - | - | - | - |")
    tools = (f"{c.get('tools_passed', 0)}/{c.get('tools_perfect', 0)}"
             f"/{c.get('tools_total', 0)}") if c else "-"
    return (f"| {record['repo']} "
            f"| {record['elapsed_sec']:.0f}s "
            f"| {record['exit_code']} "
            f"| {_stage_reached(record)} "
            f"| {tools} "
            f"| {_frac(c.get('tests_passed'), c.get('tests_total'))} "
            f"| {_frac(c.get('exec_ok'), c.get('exec_attempted'))} "
            f"| {_frac(c.get('invoc_passed'), c.get('invoc_total'))} |")


def aggregate_metrics(records: list[dict]) -> dict:
    """Roll per-repo run data into cross-repo stage pass-rates, tool counts,
    and the failure-taxonomy table."""
    stage_attempted = {s: 0 for s in PIPELINE_STAGES}
    stage_passed    = {s: 0 for s in PIPELINE_STAGES}
    failures_by_class: dict[str, int] = {}
    totals = {"tools_total": 0, "tools_passed": 0, "tools_perfect": 0,
              "tests_passed": 0, "tests_total": 0,
              "invoc_passed": 0, "invoc_total": 0,
              "exec_ok": 0, "exec_attempted": 0}
    resets_total = 0
    repos_with_data = 0

    for r in records:
        ss = r.get("stage_status") or {}
        if ss:
            repos_with_data += 1
        for s in PIPELINE_STAGES:
            if s in ss:
                stage_attempted[s] += 1
                if ss[s].get("status") == "passed":
                    stage_passed[s] += 1
                resets_total += ss[s].get("resets", 0)
        c = (r.get("validation") or {}).get("counts") or {}
        for k in totals:
            totals[k] += c.get(k) or 0
        pm = r.get("pipeline_metrics") or {}
        for label, count in pm.get("failures_by_class", {}).items():
            failures_by_class[label] = failures_by_class.get(label, 0) + count

    return {
        "repos_with_data": repos_with_data,
        "stage_completion": {s: f"{stage_passed[s]}/{stage_attempted[s]}"
                             for s in PIPELINE_STAGES},
        "stage_resets_total": resets_total,
        "tool_totals": totals,
        "failures_by_class": dict(sorted(failures_by_class.items(), key=lambda kv: -kv[1])),
    }


def write_summary(records: list[dict], out: Path) -> None:
    """Rewrite the markdown summary (called after every finished worker)."""
    lines = [
        f"# Alembic benchmark — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        f"Repos processed: {len(records)}",
        "",
        "Tools column is passed/perfect/total (a tool is *passed* when all its",
        "tests are green and it never crashed; *perfect* when it also passed all",
        "evidence-based invocation tests).",
        "",
        "| Repo | Time | Exit | Stage reached | Tools p/pf/t | Tests | Exec | Invoc |",
        "|---|---:|---:|---|---|---|---|---|",
    ]
    for r in sorted(records, key=lambda x: x["repo"]):
        lines.append(_row(r))

    lines += ["", "## Per-repo details"]
    for r in sorted(records, key=lambda x: x["repo"]):
        lines += ["", f"### {r['repo']}", f"- URL: {r['url']}",
                  f"- Duration: {r['elapsed_sec']}s",
                  f"- Exit code: {'N/A' if r['exit_code'] is None else r['exit_code']}",
                  f"- Log: {r.get('log') or '—'}"]
        if r.get("tasks"):
            lines.append(f"- Target tasks: {', '.join(r['tasks'])}")
        for t in (r.get("validation") or {}).get("tools", []):
            ex = ("ok" if t.get("exec_ok") else
                  "not invoked" if t.get("exec_ok") is None else "CRASHED")
            lines.append(f"  - {t['name']}: {t['status']} "
                         f"(tests {_frac(t.get('tests_passed'), t.get('tests_total'))}, "
                         f"exec {ex}, invoc {_frac(t.get('invoc_passed'), t.get('invoc_total'))})")
        err = r.get("pipeline_error")
        if err:
            lines.append(f"  - pipeline error: {err.get('exception')}: {err.get('message', '')[:200]}")

    agg = aggregate_metrics(records)
    if agg["repos_with_data"]:
        lines += ["", "## Aggregate", "",
                  f"Repos with run data: {agg['repos_with_data']}/{len(records)}",
                  "", "**Stage completion (gate passed / attempted):**", ""]
        for s in PIPELINE_STAGES:
            lines.append(f"- {s}: {agg['stage_completion'][s]}")
        t = agg["tool_totals"]
        lines += ["", "**Tool totals:**", "",
                  f"- tools passed {t['tools_passed']}/{t['tools_total']}, perfect {t['tools_perfect']}",
                  f"- tests passed {t['tests_passed']}/{t['tests_total']}",
                  f"- invocations exec-ok {t['exec_ok']}/{t['exec_attempted']}, "
                  f"correctness {t['invoc_passed']}/{t['invoc_total']}",
                  f"- stage resets used: {agg['stage_resets_total']}",
                  "", "**Failure taxonomy:**", ""]
        if agg["failures_by_class"]:
            lines += [f"- {label}: {count}" for label, count in agg["failures_by_class"].items()]
        else:
            lines.append("- (none)")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# Job construction (--repos / --repos-file / --tasks)
# ══════════════════════════════════════════════════════════════════════════════
def build_jobs(ns: argparse.Namespace) -> list[dict]:
    """A job = one pipeline run: {url, name, tasks}. TM-Bench tasks that share
    a repo are grouped into ONE job (the STAMP dual-task case, R4)."""
    if ns.tasks:
        by_repo: dict[str, dict] = {}
        for p in ns.tasks:
            task = yaml.safe_load(Path(p).read_text(encoding="utf-8"))
            url = ((task.get("repo") or {}).get("url") or "").strip().strip('"')
            if not url:
                sys.exit(f"[bench] task file {p} has no repo.url")
            job = by_repo.setdefault(url, {"url": url, "name": get_repo_name(url), "tasks": []})
            job["tasks"].append(task)
        return list(by_repo.values())

    if ns.repos:
        urls = ns.repos
    else:
        urls = [line.strip() for line in ns.repos_file.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")]
    return [{"url": u, "name": get_repo_name(u), "tasks": []} for u in urls]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run alembic on N repos in parallel, summarise outcomes.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repos", nargs="+",
                     help="Explicit list of repo URLs to benchmark.")
    src.add_argument("--repos-file", type=Path,
                     help="File with one URL per line ('#' starts a comment).")
    src.add_argument("--tasks", nargs="+",
                     help="TM-Bench task YAML files; tasks sharing a repo run "
                          "as one multi-task pipeline.")

    ap.add_argument("--parallel", type=int, default=4,
                    help="How many pipelines to run concurrently (default 4).")
    ap.add_argument("--mount-dir", type=Path, default=None,
                    help="Host data dir bind-mounted ro at /mount/data "
                         "(TM-Bench input data); forwarded to start_chain.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Markdown summary path (default: runs/<timestamp>/summary.md).")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="Per-repo log dir (default: runs/<timestamp>/logs).")
    ap.add_argument("--json-output", type=Path, default=None,
                    help="JSON dump of all per-repo records (default: "
                         "runs/<timestamp>/summary.json).")
    ap.add_argument("--artefact-dir", type=Path, default=None,
                    help="Where final artefacts are copied (default: "
                         "runs/<timestamp>/output).")
    ap.add_argument("--rebuild-base", action="store_true",
                    help="Force rebuild of alembic-base:latest before workers start.")
    ap.add_argument("--platform", default=None,
                    help="Pass-through to docker --platform (build + run).")
    ap.add_argument("--until", default=None, choices=PIPELINE_STAGES,
                    help="Stop each repo's pipeline after completing this stage.")
    ap.add_argument("--skip-availability-check", action="store_true",
                    help="Skip the pre-flight 'git ls-remote' reachability check.")
    return ap.parse_args()


def main() -> None:
    ns = parse_args()

    run_dir = RUNS_DIR / datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ns.output       = ns.output or run_dir / "summary.md"
    ns.log_dir      = ns.log_dir or run_dir / "logs"
    ns.json_output  = ns.json_output or run_dir / "summary.json"
    ns.artefact_dir = ns.artefact_dir or run_dir / "output"

    jobs = build_jobs(ns)
    if not jobs:
        sys.exit("[bench] no repos to run")

    ns.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[bench] {len(jobs)} jobs, {ns.parallel} parallel workers")
    print(f"[bench] logs   → {ns.log_dir}")
    print(f"[bench] summary→ {ns.output}")

    records: list[dict] = []
    lock = threading.Lock()

    if ns.skip_availability_check:
        available = jobs
    else:
        print(f"[bench] checking reachability of {len(jobs)} repos "
              f"(git ls-remote, {AVAILABILITY_TIMEOUT}s timeout each)...")
        available = []
        with ThreadPoolExecutor(max_workers=ns.parallel) as pool:
            checks = {pool.submit(check_repo_available, j["url"]): j for j in jobs}
            for fut in as_completed(checks):
                job = checks[fut]
                ok, reason = fut.result()
                if ok:
                    available.append(job)
                else:
                    print(f"[bench] ✗ skip    {job['name']}  — unreachable: {reason}", flush=True)
                    records.append({"repo": job["name"], "url": job["url"], "tasks": [],
                                    "elapsed_sec": 0, "exit_code": None, "log": None,
                                    "error": f"repo unreachable: {reason}"})
        if not available:
            write_summary(records, ns.output)
            sys.exit("[bench] no reachable repos — nothing to run")
        print(f"[bench] {len(available)}/{len(jobs)} repos reachable")

    ensure_base_image(DOCKERFILE, PROJECT_ROOT, platform=ns.platform, rebuild=ns.rebuild_base)

    extra: list[str] = []
    if ns.platform:
        extra += ["--platform", ns.platform]
    if ns.until:
        extra += ["--until", ns.until]
    if ns.mount_dir:
        extra += ["--mount-dir", str(ns.mount_dir)]

    total = len(available)

    def flush_outputs() -> None:
        with lock:
            write_summary(records, ns.output)
            if ns.json_output:
                ns.json_output.write_text(
                    json.dumps({"repos": records, "aggregate": aggregate_metrics(records)},
                               indent=2, ensure_ascii=False),
                    encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=ns.parallel) as pool:
            futures = {
                pool.submit(run_one, job, extra, ns.log_dir, ns.artefact_dir, i + 1, total): job
                for i, job in enumerate(available)
            }
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    job = futures[fut]
                    rec = {"repo": job["name"], "url": job["url"], "tasks": [],
                           "elapsed_sec": 0, "exit_code": -1, "log": None,
                           "error": f"worker raised {type(e).__name__}: {e}"}
                with lock:
                    records.append(rec)
                flush_outputs()
    except KeyboardInterrupt:
        print("\n[bench] interrupted by user — partial results saved", file=sys.stderr)

    flush_outputs()
    print(f"\n[bench] done. summary → {ns.output}")
    print(f"[bench]       json    → {ns.json_output}")
    print(f"[bench]       artefacts → {ns.artefact_dir}")


if __name__ == "__main__":
    main()
