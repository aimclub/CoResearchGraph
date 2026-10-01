"""Experiment Module runtime errors — imported by helpers."""
from __future__ import annotations

from typing import Any


class ExperimentRuntimeError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        next_actions: list[dict[str, Any]] | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.next_actions = list(next_actions or [])
        self.details = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": "error",
            "error_code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.next_actions:
            out["next_actions"] = self.next_actions
        if self.details:
            out["details"] = self.details
        return out


__all__ = ["ExperimentRuntimeError"]
