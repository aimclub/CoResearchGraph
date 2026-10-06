"""
Audit trail logger for the Hypothesis Agent subsystem.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from logging import Logger
from typing import Optional

from CoScientist.hypothesis_subsystem.models import (
    CriticReview,
    Hypothesis,
    HypothesisQuery,
    HypothesisStatus,
    ToolResult,
)


class HypothesisAuditLogger:
    """Structured audit logger for hypothesis generation lifecycle."""

    def __init__(self, logger: Optional[Logger] = None):
        if logger is None:
            import logging
            self._logger = logging.getLogger("hypothesis_subsystem")
        else:
            self._logger = logger
        self._generation_counter = 0

    def log_tool_invocation(
        self, strategy_type: str, query: HypothesisQuery, result: ToolResult, duration_ms: float = 0.0
    ) -> None:
        self._generation_counter += 1
        hyp_count = len(result.hypotheses)
        status = "SUCCESS" if result.success else "FAILURE"
        self._logger.info(
            "[hypothesis] TOOL #%d | strategy=%s | q='%s' | hyps=%d | %.0fms | %s",
            self._generation_counter, strategy_type, query.research_question[:120],
            hyp_count, duration_ms, status)
        if not result.success:
            self._logger.error("[hypothesis] TOOL_ERROR #%d | %s | %s",
                               self._generation_counter, strategy_type, result.error_message or "?")

    def log_critic_iteration(self, iteration: int, hypothesis_claim: str, review: CriticReview) -> None:
        self._logger.info("[hypothesis] CRITIC %d | verdict=%s | claim='%s' | suggestions=%d | fields=%s",
                          iteration, review.verdict.value, hypothesis_claim[:120],
                          len(review.suggestions), review.fields_to_revise)

    def log_status_change(self, hypothesis_claim: str, old: HypothesisStatus, new: HypothesisStatus, reason: str = "") -> None:
        self._logger.info("[hypothesis] STATUS | '%s' | %s→%s | %s",
                          hypothesis_claim[:120], old.value, new.value, reason)

    def log_revision(self, hypothesis_claim: str, iteration: int) -> None:
        self._logger.info("[hypothesis] REVISION | '%s' | iter=%d", hypothesis_claim[:120], iteration)

    def log_generation_start(self, research_question: str, strategy: str) -> float:
        self._logger.info("[hypothesis] GEN_START | strategy=%s | q='%s'", strategy, research_question[:200])
        return time.monotonic()

    def log_generation_end(self, research_question: str, hypothesis_count: int, start_timestamp: float) -> None:
        duration_ms = (time.monotonic() - start_timestamp) * 1000.0
        self._logger.info("[hypothesis] GEN_END | q='%s' | hyps=%d | %.0fms",
                          research_question[:200], hypothesis_count, duration_ms)

    def log_error(self, stage: str, error: str, hypothesis_claim: Optional[str] = None) -> None:
        self._logger.error("[hypothesis] ERROR | stage=%s | claim='%s' | %s",
                           stage, hypothesis_claim[:120] if hypothesis_claim else "N/A", error)
