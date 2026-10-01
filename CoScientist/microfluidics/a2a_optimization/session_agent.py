"""Session agents that cannot finish while their external A2A task is still running."""
from __future__ import annotations

from typing import ClassVar, Optional

from google.adk.agents.invocation_context import InvocationContext

from CoScientist.hitl.session_agent import SessionAgent
from CoScientist.microfluidics.a2a_optimization import campaign
from CoScientist.microfluidics.a2a_optimization.adapter import ACTIVE_KEY, RESULT_KEY


class OptimizationSessionAgent(SessionAgent):
    """Feed an early final answer back until A2A reaches a stable boundary."""

    unfinished_max_rounds: int = 12
    # The channel's state keys and the prefix of its tools (<prefix>_start, …).
    active_key: ClassVar[str] = ACTIVE_KEY
    input_error_key: ClassVar[str] = RESULT_KEY
    tool_prefix: ClassVar[str] = "optimization"

    def _unfinished_feedback(self, ctx: InvocationContext) -> Optional[str]:
        prefix = self.tool_prefix
        task = ctx.session.state.get(self.active_key)
        if not isinstance(task, dict):
            result = ctx.session.state.get(self.input_error_key)
            if isinstance(result, dict) and result.get("state") == "invalid_input":
                if result.get("economics_ranking_required"):
                    # <prefix>_start already asked the operator for the
                    # route costs in its own form (or no operator is attached):
                    # another pass of the model cannot add the missing numbers.
                    return None
                return (
                    f"{prefix}_start отклонил локальные входные данные. Не закрывай "
                    "сессию: прочитай точную ошибку и запроси у человека через HITL "
                    "недостающие сведения. Не придумывай данные и не повторяй "
                    f"{prefix}_start, пока неисправленные данные остаются в состоянии."
                )
            return None
        state = str(task.get("state") or "")
        if state in {"submitting", "sending_input", "submitted", "working"}:
            return (
                "Внешняя A2A-задача ещё не завершена. Не закрывай стадию: вызови "
                f"sleep_tool(minutes=0.1), затем {prefix}_get_status и продолжай "
                "тот же task_id. Завершить можно только на terminal state или когда "
                "действительно требуется новый ввод пользователя."
            )
        if state == "input_required" and task.get("phase") == "approval":
            return (
                "Внешняя система ждёт утверждения плана. Проверь план и вызови "
                f"{prefix}_approve; сам инструмент запросит решение человека."
            )
        return None


class CampaignSessionAgent(OptimizationSessionAgent):
    """The same guard for the rig campaign task that precedes optimization."""

    active_key: ClassVar[str] = campaign.ACTIVE_KEY
    input_error_key: ClassVar[str] = campaign.INPUT_ERROR_KEY
    tool_prefix: ClassVar[str] = "campaign"


__all__ = ["CampaignSessionAgent", "OptimizationSessionAgent"]
