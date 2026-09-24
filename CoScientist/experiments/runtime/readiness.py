"""Task readiness: depends_on, missing upstream artifacts, soft evidence deps."""
from __future__ import annotations

from typing import Any

from CoScientist.experiments.runtime.artifacts import find_artifact
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.schemas import ExperimentTask

TERMINAL_TASK_STATES = frozenset({"done", "done_with_warnings", "failed", "skipped", "blocked"})
SUCCESS_DEPENDENCY_STATES = frozenset({"done", "done_with_warnings", "skipped"})


def missing_task_artifacts(runtime: dict[str, Any], task_dump: dict[str, Any]) -> list[str]:
    """Required task_artifact inputs that cannot be resolved yet, as ``task:artifact``."""
    task = ExperimentTask.model_validate(task_dump)
    missing: list[str] = []
    for data_ref in task.input_data:
        if not data_ref.required or data_ref.kind != "task_artifact":
            continue
        try:
            find_artifact(
                runtime,
                str(data_ref.source_artifact_id or ""),
                source_task_id=str(data_ref.source_task_id) if data_ref.source_task_id else None,
            )
        except ExperimentRuntimeError:
            src = str(data_ref.source_task_id or "?")
            missing.append(f"{src}:{data_ref.source_artifact_id or ''}")
    return missing


def required_task_artifacts_missing(runtime: dict[str, Any], task_dump: dict[str, Any]) -> bool:
    """True when a required task_artifact input cannot be resolved yet."""
    return bool(missing_task_artifacts(runtime, task_dump))


def artifact_producers_terminal(runtime: dict[str, Any], task_id: str, task_dump: dict[str, Any]) -> bool:
    """Wait for named producers (or every other task) before treating a miss as blocked."""
    tasks = runtime["tasks"]
    sources: list[str] = []
    task = ExperimentTask.model_validate(task_dump)
    for data_ref in task.input_data:
        if not data_ref.required or data_ref.kind != "task_artifact":
            continue
        src = str(data_ref.source_task_id or "").strip()
        if src:
            sources.append(src)
    if sources:
        return all(
            sid not in tasks or tasks[sid]["status"] in TERMINAL_TASK_STATES
            for sid in sources
        )
    return all(
        tid == task_id or tasks[tid]["status"] in TERMINAL_TASK_STATES
        for tid in runtime["task_order"]
    )


def dep_is_soft_evidence(runtime: dict[str, Any], dep_id: str, consumer: dict[str, Any]) -> bool:
    """Failed research/medical does not block compute unless a task_artifact is required."""
    dep = runtime["tasks"].get(dep_id) or {}
    route = str(dep.get("planned_route") or (dep.get("task") or {}).get("route") or "")
    if route not in {"research", "medical"}:
        return False
    try:
        model = ExperimentTask.model_validate(consumer)
    except Exception:  # noqa: BLE001
        return True
    for data_ref in model.input_data:
        if not data_ref.required or data_ref.kind != "task_artifact":
            continue
        if str(data_ref.source_task_id or "").strip() == dep_id:
            return False
    return True


def refresh_readiness(runtime: dict[str, Any]) -> None:
    tasks = runtime["tasks"]
    for task_id in runtime["task_order"]:
        task = tasks[task_id]
        if task["status"] != "pending":
            continue
        dep_ids = list(task["task"]["depends_on"])
        deps = [tasks[dep]["status"] for dep in dep_ids]
        hard_fail = any(
            status in {"failed", "blocked"}
            and not dep_is_soft_evidence(runtime, dep_id, task["task"])
            for dep_id, status in zip(dep_ids, deps)
        )
        if hard_fail:
            task["status"], task["last_message"] = "blocked", "A required dependency failed."
        elif all(
            status in SUCCESS_DEPENDENCY_STATES
            or (
                status in {"failed", "blocked"}
                and dep_is_soft_evidence(runtime, dep_id, task["task"])
            )
            for dep_id, status in zip(dep_ids, deps)
        ):
            dumped = task["task"]
            if missing := missing_task_artifacts(runtime, dumped):
                if artifact_producers_terminal(runtime, task_id, dumped):
                    task["status"] = "blocked"
                    task["last_message"] = (
                        "Required upstream artifact is missing: " + ", ".join(missing)
                    )
            else:
                task["status"] = "ready"
