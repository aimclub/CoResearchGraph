"""Task readiness: depends_on, missing upstream artifacts, soft evidence deps."""
from __future__ import annotations

from typing import Any

from CoScientist.experiments.runtime.artifacts import find_artifact
from CoScientist.experiments.runtime.errors import ExperimentRuntimeError
from CoScientist.experiments.runtime.outputs import REPAIRABLE_BLOCK_CODES, is_repairable_block
from CoScientist.experiments.schemas import ExperimentTask

TERMINAL_TASK_STATES = frozenset({"done", "done_with_warnings", "failed", "skipped", "blocked"})
SUCCESS_DEPENDENCY_STATES = frozenset({"done", "done_with_warnings", "skipped"})


def _producer_settled(task_runtime: dict[str, Any]) -> bool:
    status = str(task_runtime.get("status") or "")
    if status not in TERMINAL_TASK_STATES:
        return False
    return not is_repairable_block(task_runtime)


def required_task_artifacts_missing(runtime: dict[str, Any], task_dump: dict[str, Any]) -> bool:
    """True when a required task_artifact input cannot be resolved yet."""
    task = ExperimentTask.model_validate(task_dump)
    for data_ref in task.input_data:
        if not data_ref.required or data_ref.kind != "task_artifact":
            continue
        try:
            find_artifact(
                runtime,
                str(data_ref.source_artifact_id or data_ref.source_output_id or ""),
                source_task_id=str(data_ref.source_task_id) if data_ref.source_task_id else None,
                source_output_id=data_ref.source_output_id,
            )
        except ExperimentRuntimeError:
            return True
    return False


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
            sid not in tasks or _producer_settled(tasks[sid])
            for sid in sources
        )
    return all(
        tid == task_id or _producer_settled(tasks[tid])
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


def _revisit(task: dict[str, Any]) -> bool:
    if task["status"] == "pending":
        return True
    reason = str((task.get("blocked_reason") or {}).get("code") or "")
    return task["status"] == "blocked" and reason in REPAIRABLE_BLOCK_CODES


def refresh_readiness(runtime: dict[str, Any]) -> None:
    tasks = runtime["tasks"]
    # Producers first, so a repaired output unblocks its consumers in this pass.
    order = list(runtime["task_order"])
    for task_id in order:
        task = tasks[task_id]
        if not _revisit(task):
            continue
        dep_ids = list(task["task"]["depends_on"])
        deps = [tasks[dep]["status"] for dep in dep_ids]
        hard = [
            (dep_id, status)
            for dep_id, status in zip(dep_ids, deps)
            if status in {"failed", "blocked"}
            and not dep_is_soft_evidence(runtime, dep_id, task["task"])
        ]
        repairable_deps = [
            (dep_id, status)
            for dep_id, status in hard
            if status == "blocked" and is_repairable_block(tasks[dep_id])
        ]
        fatal_deps = [pair for pair in hard if pair not in repairable_deps]
        if fatal_deps:
            failed = [f"{dep_id}:{status}" for dep_id, status in fatal_deps]
            task["status"] = "blocked"
            task["blocked_reason"] = {
                "code": "required_dependency_failed",
                "dependencies": failed,
            }
            task["last_message"] = (
                "Required dependency failed: " + ", ".join(failed)
            )
        elif repairable_deps:
            waiting = [f"{dep_id}:{status}" for dep_id, status in repairable_deps]
            task["status"] = "blocked"
            task["blocked_reason"] = {
                "code": "repairable_dependency_blocked",
                "dependencies": waiting,
            }
            task["last_message"] = (
                "Required dependency is blocked for a repairable output: "
                + ", ".join(waiting)
            )
        elif all(
            status in SUCCESS_DEPENDENCY_STATES
            or (
                status in {"failed", "blocked"}
                and dep_is_soft_evidence(runtime, dep_id, task["task"])
            )
            for dep_id, status in zip(dep_ids, deps)
        ):
            dumped = task["task"]
            if required_task_artifacts_missing(runtime, dumped):
                if artifact_producers_terminal(runtime, task_id, dumped):
                    task["status"] = "blocked"
                    missing = [
                        {
                            "source_task_id": str(ref.source_task_id or ""),
                            "source_artifact_id": str(ref.source_artifact_id or ""),
                            "source_output_id": str(ref.source_output_id or ""),
                        }
                        for ref in ExperimentTask.model_validate(dumped).input_data
                        if ref.required and ref.kind == "task_artifact"
                    ]
                    task["blocked_reason"] = {
                        "code": "required_upstream_artifact_missing",
                        "artifacts": missing,
                    }
                    task["last_message"] = (
                        "Required upstream artifact is missing after its producer became terminal."
                    )
            else:
                task["status"] = "ready"
                task.pop("blocked_reason", None)
        elif task["status"] == "blocked":
            # The repairable cause is gone, and the dependency has not finished.
            task["status"] = "pending"
            task.pop("blocked_reason", None)
