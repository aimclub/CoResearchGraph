"""Task coverage-blob policy shared by plan repair and deterministic critique.

Prefer the frame operation statement so a rewritten task (name/description
stuffed with leftover tool names) cannot fake coverage.
"""
from __future__ import annotations

from typing import Any, Mapping


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def operation_statement(task: Any, ops_index: Mapping[str, Mapping[str, str]] | None) -> str:
    """Frame slot text for the task's ``operation_ref``, or ``""``."""
    design = _field(task, "design")
    ref = str(_field(design, "operation_ref") or _field(task, "operation_ref") or "").strip().upper()
    if ops_index and ref and ref in ops_index:
        return str((ops_index[ref] or {}).get("statement") or "")
    return ""


def task_ask_blob(task: Any) -> str:
    """Ask text only — bound leftover tool names must not fake coverage."""
    design = _field(task, "design")
    parts = [
        str(_field(task, "name") or ""),
        str(_field(task, "description") or ""),
        str(_field(task, "rationale") or ""),
        str(_field(design, "experiment_question") or ""),
    ]
    for art in _field(design, "analysis_artifacts") or []:
        parts.append(str(_field(art, "path_or_tool") or ""))
        parts.append(str(_field(art, "name") or ""))
    return " ".join(parts)


def task_coverage_blob(task: Any, ops_index: Mapping[str, Mapping[str, str]] | None = None) -> str:
    statement = operation_statement(task, ops_index)
    if statement:
        return statement
    return task_ask_blob(task)


__all__ = ["operation_statement", "task_ask_blob", "task_coverage_blob"]
