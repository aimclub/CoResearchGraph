"""Non-Web entry-point adapter for the shared provider budget."""
from __future__ import annotations

import asyncio
from typing import Any

from CoScientist.execution_control import RunController, RunStopped, bind_run


_BUDGET_CONTINUE = "Continue +100"
_FINISH_LIMITS = "Finish with documented limits"
_INSPECTED_RETRY = "I inspected the remote outcome; authorize retry"
_STOP = "Stop"


async def _watch_run_control(handle, controller, handler, *, poll_interval: float = 0.25):
    """Resolve durable CLI pauses only from explicit human decisions.

    Causes are handled one at a time. Budget is first because granting it is
    independent of (and must not clear) semantic or external-outcome pauses.
    """
    from CoScientist.hitl.models import HITLRequest, HITLAction, HITLDecisionSource

    async def ask(*, message: str, options: list[str], context: dict[str, Any]):
        return await handler.handle_request(HITLRequest(
            agent_name="RunController",
            action_type=HITLAction.SELECT,
            message=message,
            options=options,
            requires_human=True,
            invoked_via="internal_loop",
            trigger="run_control_pause",
            context={"run_id": handle.run_id, **context},
        ))

    while True:
        status = handle.status()
        if status.state in {"stopped", "completed"}:
            return
        causes = set(status.pause_causes)

        if "budget_exhausted" in causes:
            answer = await ask(
                message=(
                    f"Used {status.attempts_used} controlled model calls. "
                    "Continue with 100 more or stop? Remote agents are outside this quota."
                ),
                options=[_BUDGET_CONTINUE, _STOP],
                context={"kind": "budget_exhausted", "budget": status.to_dict()["budget"]},
            )
            if answer.decision_source == HITLDecisionSource.HUMAN:
                if answer.selected_option == _BUDGET_CONTINUE:
                    controller.grant(
                        handle.run_id,
                        f"budget:{handle.run_id}:{status.budget_total}",
                        metadata={"source": "human"},
                    )
                elif answer.selected_option == _STOP:
                    controller.stop(handle.run_id, "operator_stopped_at_budget")
                    return
            await asyncio.sleep(poll_interval)
            continue

        if "unknown_completion" in causes:
            # A crash after the journals but before resume is completed from
            # the durable human decisions without asking or replaying again.
            from CoScientist.agents.run_control_plugin import unresolved_actions

            unknown = unresolved_actions(controller.journal_entries(handle.run_id))
            if not unknown:
                controller.resume(handle.run_id, "unknown_completion")
                await asyncio.sleep(poll_interval)
                continue
            actions = [
                {
                    "action_id": row["action_id"],
                    "tool": (row.get("data") or {}).get("tool"),
                    "job_id": (row.get("data") or {}).get("job_id"),
                }
                for row in unknown
            ]
            answer = await ask(
                message=(
                    "An external operation may have been accepted before its response was lost. "
                    "Inspect the remote outcome before authorizing any repeat, or stop."
                ),
                options=[_INSPECTED_RETRY, _STOP],
                context={"kind": "unknown_completion", "actions": actions},
            )
            if answer.decision_source == HITLDecisionSource.HUMAN:
                if answer.selected_option == _INSPECTED_RETRY:
                    notes = (
                        answer.instructions or answer.free_input
                        or "CLI operator confirmed external outcome inspection."
                    )
                    for row in unknown:
                        controller.journal(
                            handle.run_id,
                            "tool_retry_authorized",
                            action_id=row["action_id"],
                            data={
                                "source": "human",
                                "notes": notes,
                                "tool": (row.get("data") or {}).get("tool"),
                            },
                        )
                    if not unresolved_actions(
                        controller.journal_entries(handle.run_id)
                    ):
                        controller.resume(handle.run_id, "unknown_completion")
                elif answer.selected_option == _STOP:
                    controller.stop(handle.run_id, "operator_stopped_at_unknown_outcome")
                    return
            await asyncio.sleep(poll_interval)
            continue

        if "execution_error" in causes:
            pending = status.pending_decisions.get("execution_error") or {}
            resolution = status.metadata.get("control_resolution") or {}
            if (
                isinstance(resolution, dict)
                and resolution.get("decision") == "finish_with_limits"
                and resolution.get("source") == "human"
            ):
                # Finish a decision whose durable write succeeded immediately
                # before a process/task interruption.
                controller.resume(handle.run_id, "execution_error")
                await asyncio.sleep(poll_interval)
                continue
            semantic = pending.get("kind") == "semantic_control_loop_guard"
            options = [_FINISH_LIMITS, _STOP] if semantic else [_STOP]
            answer = await ask(
                message=(
                    "Automatic experiment control stopped after making no progress. "
                    "Finish with the recorded results and documented limitations, or stop."
                    if semantic else
                    "Execution is paused by an unrecoverable control error. Stop the run."
                ),
                options=options,
                context={"kind": "execution_error", "pending_decision": pending},
            )
            if answer.decision_source == HITLDecisionSource.HUMAN:
                if semantic and answer.selected_option == _FINISH_LIMITS:
                    decision = {
                        "decision": "finish_with_limits",
                        "source": "human",
                        "reason": pending.get("error_code") or "control_no_progress",
                    }
                    controller.update_metadata(
                        handle.run_id, {"control_resolution": decision}
                    )
                    controller.journal(
                        handle.run_id,
                        "control_resolution_requested",
                        action_id=(
                            f"semantic:{handle.run_id}:"
                            f"{pending.get('state_revision') or 'unknown'}"
                        ),
                        data=decision,
                    )
                    controller.resume(handle.run_id, "execution_error")
                elif answer.selected_option == _STOP:
                    controller.stop(handle.run_id, "operator_stopped_at_execution_error")
                    return
            await asyncio.sleep(poll_interval)
            continue

        await asyncio.sleep(poll_interval)


async def run_with_control(manager, query, *, verbose=True, report_config=None):
    from CoScientist.config import get_settings
    from CoScientist.hitl import ConsoleHITLHandler
    from CoScientist.reporting import RunResult

    controller = RunController()
    handle = controller.create_run(initial_budget=get_settings().orchestrator.max_llm_calls,
        metadata={"user_id": manager.user_id, "session_id": manager.session_id,
                  "root_query": query, "entrypoint": "cli"})
    handler = manager._hitl_handler or ConsoleHITLHandler()

    with bind_run(handle):
        watcher = asyncio.create_task(
            _watch_run_control(handle, controller, handler)
        )
        try:
            result = await manager._run(query, verbose=verbose, report_config=report_config)
            if handle.status().state == "running":
                controller.complete(handle.run_id)
            return result
        except RunStopped:
            return RunResult(markdown="Research stopped by the operator. Saved results were preserved.")
        except BaseException:
            if handle.status().state not in {"stopped", "completed"}:
                controller.request_pause(handle.run_id, "interrupted")
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
