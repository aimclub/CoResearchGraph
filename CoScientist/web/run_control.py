"""Web-facing execution controls, separate from destructive checkpoint rollback."""
from __future__ import annotations

import asyncio
import copy
from datetime import datetime, timezone
from typing import Any

from pydantic_core import to_jsonable_python

from CoScientist.execution_control import RunController, RunStopped, bind_run
from CoScientist.agents.run_control_plugin import RECOVERY_STATE_KEY, unresolved_actions
from CoScientist.experiments.outcome.reconciliation import (
    DispositionKind,
    reconcile_scientific_outcome,
    roadmap_digest,
)


_RESUME_PENDING_STATE_KEY = "_execution_resume_pending"
_PLAN_RECOVERY_STATE_KEY = "experiment_plan_recovery_requested"


class ExecutionControlMixin:
    def init_execution_control(self) -> None:
        self.execution_controller = RunController(notifier=self._execution_changed)
        self.execution_handles = {}
        self._execution_public_versions = {}

    def execution_handle(self, key):
        handle = self.execution_handles.get(key)
        if handle is None:
            handle = self.execution_controller.latest_run(user_id=key[0], session_id=key[1])
            if handle is not None:
                self.execution_handles[key] = handle
        return handle

    def execution_snapshot(self, key) -> dict[str, Any] | None:
        handle = self.execution_handle(key)
        if handle is None:
            return None
        status = handle.status()
        # A process restart never means permission to resume autonomous work.
        owner = self.active_runs.get(key)
        if (status.state in {"running", "pause_requested"}
                and (owner is None or owner.done())
                and status.metadata.get("boot_id") != self.boot_id):
            self.execution_controller.request_pause(handle.run_id, "server_restart")
            status = handle.status()
        return self._public_execution(status)

    @staticmethod
    def _public_execution(status) -> dict[str, Any]:
        saved = (status.metadata.get("continuation") or {}).get("state") or {}
        runtime = saved.get("experiment_runtime") or {}
        tasks = runtime.get("tasks") or {}
        disposition = status.metadata.get("final_disposition") or {}
        return {
            "run_id": status.run_id,
            "state": status.state,
            "started_at": datetime.fromtimestamp(status.created_at, timezone.utc).isoformat(),
            "revision": getattr(status, "control_revision", 0),
            "pause_causes": list(status.pause_causes) if status.state not in {"stopped", "completed"} else [],
            "pending_decisions": status.pending_decisions if status.state not in {"stopped", "completed"} else {},
            "budget": {
                "used": status.attempts_used, "limit": status.budget_total,
                "remaining": status.remaining, "grant_size": 100,
                "scope": "controlled_provider_attempts",
                "remote_agents_included": False,
            },
            "active_task_id": runtime.get("active_task_id"),
            "active_attempt_id": runtime.get("active_attempt_id"),
            "tasks": [{"id": tid, "name": (row.get("task") or {}).get("name", tid),
                       "status": row.get("status"), "message": row.get("last_message", "")}
                      for tid, row in tasks.items()],
            "can_restore": bool(status.metadata.get("continuation"))
                           and not status.metadata.get("continuation_error"),
            "stop_reason": status.stop_reason,
            "disposition": disposition,
        }

    async def _execution_changed(self, handle, _status) -> None:
        # Notifications may have queued while another decision was being made.
        # Always send the current state, never a stale notification snapshot.
        status = handle.status()
        meta = status.metadata
        if not meta.get("user_id") or not meta.get("session_id"):
            return
        key = (meta["user_id"], meta["session_id"])
        current = self.execution_handles.get(key)
        if current is not None and current.run_id != handle.run_id:
            return
        public = self._public_execution(status)
        signature = (public["revision"], status.state, status.attempts_used,
                     tuple(status.pause_causes), public["active_attempt_id"])
        if self._execution_public_versions.get(key) == signature:
            return
        self._execution_public_versions[key] = signature
        payload = {"type": "run_control", **public}
        try:
            await self.send(key, payload)
            mapped = "paused" if public["pause_causes"] else (
                "processing" if status.state == "running" else status.state)
            self.registry.touch_session(*key, status=mapped)
        except KeyError:
            pass

    async def run_controlled_chat(self, key, data) -> None:
        from CoScientist.web.app import _handle_chat
        handle = self.execution_handles[key]
        try:
            with bind_run(handle):
                await _handle_chat(self, key, data)
            # A normal ADK return is only a scheduling fact.  Re-read the
            # durable session after the runner has published every state delta,
            # save that exact continuation, and only then decide terminality.
            state = await self._fresh_execution_state(key, handle)
            disposition = reconcile_scientific_outcome(
                state, current_run_id=handle.run_id,
            )
            handle.controller.update_metadata(handle.run_id, {
                "continuation": {
                    "state": state,
                    "boundary": "chat_return_reconciled",
                    "agent": "OrchestratorAgent",
                },
                "continuation_error": None,
                "final_disposition": disposition.to_dict(),
            })
            status = handle.status()
            if status.state in {"stopped", "completed"}:
                return
            if disposition.terminal:
                # Independent pause causes win.  In particular a budget or
                # unresolved HITL decision cannot be erased by a final answer.
                if not status.pause_causes:
                    self.execution_controller.complete(handle.run_id)
                return
            handle.controller.request_pause(
                handle.run_id,
                disposition.pause_cause or "scientific_work_incomplete",
                pending_decision=disposition.pending_decision,
            )
        except RunStopped:
            pass
        except asyncio.CancelledError:
            # Shutdown and explicit Stop have different semantics: on shutdown
            # retain a resumable record; stop_run already marked it stopped.
            if handle.status().state not in {"stopped", "completed"}:
                self.execution_controller.request_pause(handle.run_id, "server_restart")
            raise
        except Exception as exc:
            if handle.status().state not in {"stopped", "completed"}:
                handle.controller.update_metadata(handle.run_id, {
                    "continuation_error": f"{type(exc).__name__}: {exc}",
                })
                handle.controller.request_pause(
                    handle.run_id,
                    "execution_error",
                    pending_decision={
                        "kind": "outcome_reconciliation_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            raise

    async def _fresh_execution_state(self, key, handle) -> dict[str, Any]:
        """Load the newest ADK state, falling back only when no session exists."""
        session = await self.session_service.get_session(
            app_name="coscientist_app", user_id=key[0], session_id=key[1],
        )
        raw = getattr(session, "state", None) if session is not None else None
        if raw is None:
            raw = ((handle.status().metadata.get("continuation") or {}).get("state"))
        if raw is None:
            return {}
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        if not isinstance(raw, dict):
            raw = dict(raw)
        # Do not stringify executable state.  A value that cannot be preserved
        # must pause through the ordinary execution_error path, never complete.
        return to_jsonable_python({
            name: value for name, value in raw.items()
            if not str(name).startswith("temp:")
        })

    def prepare_execution(self, key, data, settings):
        recovering = data.get("_execution_run_id")
        if recovering:
            handle = self.execution_controller.get_handle(str(recovering))
            meta = handle.status().metadata
            if (meta.get("user_id"), meta.get("session_id")) != key:
                raise ValueError("Run does not belong to this session")
            self.execution_controller.update_metadata(handle.run_id, {"boot_id": self.boot_id})
        else:
            previous = self.execution_handle(key)
            if previous and previous.status().state not in {"stopped", "completed"}:
                raise ValueError("Research is paused or running; continue it or explicitly stop it first")
            handle = self.execution_controller.create_run(initial_budget=settings.orchestrator.max_llm_calls,
                metadata={"user_id": key[0], "session_id": key[1], "boot_id": self.boot_id,
                          "root_query": data.get("message", ""),
                          "report_language": data.get("report_language")})
        self.execution_handles[key] = handle
        return handle

    def _require_execution(self, key, run_id, expected_revision=None):
        handle = self.execution_handle(key)
        if handle is None or handle.run_id != run_id:
            raise KeyError("Unknown current run for this session")
        status = handle.status()
        if expected_revision is not None and int(expected_revision) != getattr(status, "control_revision", 0):
            raise ValueError("Execution state changed; refresh before sending this decision")
        return handle

    async def pause_execution(self, key, run_id, *, expected_revision=None):
        async with self.control_lock(key):
            handle = self._require_execution(key, run_id, expected_revision)
            self.execution_controller.request_pause(handle.run_id, "manual")
        return self.execution_snapshot(key)

    def resolve_durable_hitl(self, key, request_id, data) -> bool:
        """Accept a saved approval after the owning Python Future was lost."""
        handle = self.execution_handle(key)
        cause = f"hitl:{request_id}"
        if handle is None or cause not in handle.status().pause_causes:
            return False
        decision = handle.status().pending_decisions.get(cause) or {}
        response = {name: data.get(name) for name in (
            "action", "approved", "selected_option", "selected_task_ids", "instructions", "free_input", "form_values",
        )}
        response["decision_source"] = "human"
        handle.controller.journal(handle.run_id, "hitl_decision", action_id=request_id, data=response)
        if decision.get("kind") == "request_input":
            handle.controller.update_metadata(handle.run_id, {
                "request_input_response": {"interrupt_id": request_id, "response": response},
            })
        handle.controller.resume(handle.run_id, cause)
        self.record_event(key, {"type": "hitl_response", "request_id": request_id, **response})
        owner = self.active_runs.get(key)
        if owner is None or owner.done():
            async def resume_saved():
                try:
                    await self.resume_execution(key, handle.run_id)
                except ValueError:
                    # Other pending causes or an ambiguous dispatched action
                    # deliberately keep the run paused and visible.
                    await self._execution_changed(handle, handle.status())
            asyncio.create_task(resume_saved())
        return True

    async def resume_execution(self, key, run_id, *, expected_revision=None):
        async with self.control_lock(key):
            handle = self._require_execution(key, run_id, expected_revision)
            status = handle.status()
            if status.state in {"stopped", "completed"}:
                raise ValueError("This run is terminal, not paused")
            recoverable_planning = {
                cause for cause in status.pause_causes
                if (status.pending_decisions.get(cause) or {}).get("kind")
                == "experiment_plan_recovery"
            }
            remaining_causes = set(status.pause_causes) - {
                "manual", "server_restart", "execution_error", *recoverable_planning,
            }
            if remaining_causes:
                raise ValueError("Resolve the pending decision first: " + ", ".join(sorted(remaining_causes)))
            owner = self.active_runs.get(key)
            alive = owner is not None and not owner.done()
            if not alive:
                if recoverable_planning:
                    self._prepare_saved_plan_recovery(handle, recoverable_planning)
                await self._restore_execution(key, handle)
            self.execution_controller.resume(run_id, "manual")
            self.execution_controller.resume(run_id, "server_restart")
            self.execution_controller.resume(run_id, "execution_error")
            for cause in recoverable_planning:
                self.execution_controller.resume(run_id, cause)
        if not alive:
            current_meta = handle.status().metadata
            resolution = current_meta.get("outcome_resolution") or {}
            if not await self.start_run(key, {
                "message": status.metadata.get("root_query") or "Continue the paused research.",
                "_execution_run_id": run_id, "_execution_resume": True,
                "_execution_resume_instruction": resolution.get("instruction"),
                "report_language": status.metadata.get("report_language"),
            }):
                raise ValueError("Another invocation already owns the session")
        return self.execution_snapshot(key)

    async def decide_scientific_outcome(self, key, run_id, data):
        """Resolve a bounded capacity/root-roadmap pause with a human decision."""
        decision = str(data.get("decision") or "").strip()
        decision_id = str(data.get("decision_id") or "").strip()
        if not decision_id or data.get("confirmed") is not True:
            raise ValueError("decision_id and explicit confirmation are required")
        if decision not in {"continue", "accept_limited", "approve_capacity"}:
            raise ValueError("unsupported scientific outcome decision")

        async with self.control_lock(key):
            handle = self._require_execution(key, run_id)
            seen = next((
                row for row in handle.controller.journal_entries(run_id)
                if row.get("event") == "scientific_outcome_decided"
                and row.get("action_id") == decision_id
            ), None)
            if seen:
                if (seen.get("data") or {}).get("decision") != decision:
                    raise ValueError("This decision_id was already used for another decision")
                return self.execution_snapshot(key)
            self._require_execution(key, run_id, data.get("expected_revision"))
            status = handle.status()
            pending_pair = next((
                (cause, pending)
                for cause, pending in status.pending_decisions.items()
                if (pending or {}).get("kind") in {
                    "scientific_outcome", "plan_capacity_conflict",
                }
            ), None)
            if pending_pair is None:
                raise ValueError("No supported scientific outcome decision is pending")
            cause, pending = pending_pair
            state = copy.deepcopy(
                (status.metadata.get("continuation") or {}).get("state")
            )
            if not isinstance(state, dict):
                raise ValueError("No safe continuation is available")
            disposition = reconcile_scientific_outcome(
                state, current_run_id=run_id,
            )

            if pending.get("kind") == "plan_capacity_conflict":
                if decision != "approve_capacity":
                    raise ValueError(
                        "Capacity pauses support approve_capacity only; "
                        "otherwise stop and revise or split the requested scope"
                    )
                outcome = state.get("experiment_module_outcome") or {}
                if outcome.get("can_raise_limit") is not True:
                    raise ValueError("This plan exceeds the hard 20-task schema limit")
                requested = outcome.get("requested_max_plan_tasks")
                if type(requested) is not int or not 1 <= requested <= 20:
                    raise ValueError("The saved capacity request is invalid")
                context = state.get("experiment_context") or {}
                from CoScientist.experiments.plan_policy import operations_digest

                state["experiment_plan_capacity_override"] = {
                    "run_id": context.get("experiment_run_id"),
                    "operations_digest": operations_digest(context.get("operations") or []),
                    "max_plan_tasks": requested,
                    "decision_source": "human",
                }
                state["experiment_capacity_recovery_requested"] = {
                    "source": "run_control", "reason": "plan_capacity_conflict",
                }
                # Keep the capacity pause typed until the module context builder
                # has reused the saved run and the planner preflight has verified
                # this exact-run/exact-operations grant.  Clearing it here makes
                # the builder treat the continuation as a new experiment, which
                # discards the scoped grant and recreates the same conflict.
                state["experiment_plan_review_paused"] = True
                state["experiment_review_pause_reason"] = "plan_capacity_conflict"
                state["experiment_module_outcome"] = {
                    **outcome,
                    "status": "running",
                    "reason": "plan_capacity_approved",
                    "accepted": True,
                }
                instruction = (
                    "Continue the same saved experiment after the explicit capacity "
                    "approval. Do not reset planning revisions, root tasks, or budgets."
                )
            else:
                if disposition.reason not in {
                    "root_roadmap_incomplete", "root_roadmap_not_successful",
                }:
                    raise ValueError("This outcome must be resolved by its owning review")
                if decision == "approve_capacity":
                    raise ValueError("approve_capacity does not apply to the root roadmap")
                if decision == "accept_limited":
                    accepted_ids = tuple(dict.fromkeys(
                        str(item) for item in (data.get("accepted_item_ids") or [])
                    ))
                    expected_ids = disposition.unfinished_task_ids
                    if not expected_ids or set(accepted_ids) != set(expected_ids):
                        raise ValueError(
                            "accepted_item_ids must name every exact outstanding roadmap item"
                        )
                    state["scientific_limited_scope_acceptance"] = {
                        "accepted": True,
                        "decision_source": "human",
                        "decision_id": decision_id,
                        "run_id": run_id,
                        "roadmap_digest": roadmap_digest(state),
                        "accepted_task_ids": list(accepted_ids),
                    }
                    instruction = (
                        "The operator explicitly accepted a limited outcome for these "
                        f"outstanding roadmap items: {list(accepted_ids)}. Do not execute "
                        "or retry them. Produce an honest, clearly partial final report "
                        "using only work that actually completed."
                    )
                else:
                    state.pop("scientific_limited_scope_acceptance", None)
                    instruction = (
                        "Continue only the outstanding saved root-roadmap work. Preserve "
                        "completed tasks and do not reset or replay the scientific run."
                    )

            state[_RESUME_PENDING_STATE_KEY] = True
            resolution = {
                "decision": decision,
                "decision_id": decision_id,
                "source": "human",
                "instruction": instruction,
            }
            handle.controller.update_metadata(run_id, {
                "continuation": {
                    "state": state,
                    "boundary": "scientific_outcome_decision",
                    "agent": "OrchestratorAgent",
                },
                "continuation_error": None,
                "outcome_resolution": resolution,
            })
            handle.controller.journal(
                run_id, "scientific_outcome_decided", action_id=decision_id,
                data={"decision": decision, "source": "human"},
            )
            handle.controller.resume(run_id, cause)

        owner = self.active_runs.get(key)
        if not handle.status().pause_causes and (owner is None or owner.done()):
            await self.resume_execution(key, run_id)
        return self.execution_snapshot(key)

    @staticmethod
    def _prepare_saved_plan_recovery(handle, causes: set[str]) -> None:
        """Request review of the saved candidate without resetting the run."""
        status = handle.status()
        saved = copy.deepcopy((status.metadata.get("continuation") or {}).get("state"))
        if not isinstance(saved, dict):
            raise ValueError("No saved plan state is available for recovery")
        pending = next(
            (status.pending_decisions.get(cause) or {} for cause in causes), {}
        )
        reason = str(pending.get("reason") or "max_plan_revisions")
        saved[_PLAN_RECOVERY_STATE_KEY] = {
            "source": "run_control", "reason": reason,
        }
        saved["experiment_plan_review_paused"] = False
        saved["experiment_plan_fallback_pending"] = False
        saved[_RESUME_PENDING_STATE_KEY] = True
        handle.controller.update_metadata(handle.run_id, {
            "continuation": {
                "state": saved,
                "boundary": "explicit_plan_recovery",
                "agent": "ExperimentReviewSessionAgent",
            },
            "continuation_error": None,
        })

    async def _restore_execution(self, key, handle):
        """Restore the latest continuation, not an old rollback checkpoint."""
        status = handle.status()
        entries = self.execution_controller.journal_entries(handle.run_id)
        unknown = unresolved_actions(entries)
        if unknown:
            self.execution_controller.request_pause(handle.run_id, "unknown_completion",
                pending_decision={"actions": [{"action_id": row["action_id"],
                                              "tool": (row.get("data") or {}).get("tool")}
                                             for row in unknown]})
            raise ValueError("An external action has an unknown outcome; inspect it before retrying")
        saved = status.metadata.get("continuation") or {}
        if status.metadata.get("continuation_error") or not isinstance(saved.get("state"), dict):
            raise ValueError("No safe continuation is available; use an explicit checkpoint/research restart")
        manager = await self.get_manager(*key)
        state = copy.deepcopy(saved["state"])
        state[RECOVERY_STATE_KEY] = True
        state[_RESUME_PENDING_STATE_KEY] = True
        if status.metadata.get("stage_index") is not None:
            state["_checkpoint_resume"] = {"stage_index": status.metadata["stage_index"]}
        for name, value in state.items():
            if not str(name).startswith("temp:"):
                await manager._set_state(name, value)

    async def decide_execution_budget(self, key, run_id, data):
        decision = data.get("decision")
        if decision not in {"continue", "stop"}:
            raise ValueError("decision must be continue or stop")
        decision_id = str(data.get("decision_id") or "").strip()
        if not decision_id:
            raise ValueError("decision_id is required")
        async with self.control_lock(key):
            handle = self._require_execution(key, run_id)
            # A repeated HTTP request returns the existing decision, including
            # if its expected revision predates the first successful request.
            seen = next((row for row in self.execution_controller.journal_entries(run_id)
                         if row.get("action_id") == decision_id
                         and row.get("event") in {"budget_granted", "budget_stopped"}), None)
            if seen:
                original = "continue" if seen["event"] == "budget_granted" else "stop"
                if decision != original:
                    raise ValueError("This decision_id has already been used for another decision")
                return self.execution_snapshot(key)
            if not seen:
                self._require_execution(key, run_id, data.get("expected_revision"))
                if "budget_exhausted" not in handle.status().pause_causes:
                    raise ValueError("There is no pending budget request")
            if decision == "continue":
                self.execution_controller.grant(run_id, decision_id, metadata={"source": "human"})
            else:
                self.execution_controller.stop(run_id, "operator_stopped_at_budget")
                self.execution_controller.journal(run_id, "budget_stopped", action_id=decision_id,
                                                  data={"source": "human"})
        if decision == "stop":
            await self.stop_run(key)
        elif not handle.status().pause_causes:
            owner = self.active_runs.get(key)
            if owner is None or owner.done():
                await self.resume_execution(key, run_id)
        return self.execution_snapshot(key)

    async def authorize_unknown_retry(self, key, run_id, data):
        """Only a recorded operator decision may release an ambiguous dispatch.

        This does not assert success or reset attempts/budget. The operator
        must first inspect the remote job; a running job should not be retried.
        """
        action_id = str(data.get("action_id") or "").strip()
        notes = str(data.get("notes") or "").strip()
        if not action_id or not notes or data.get("confirmed") is not True:
            raise ValueError("action_id, inspection notes and explicit confirmation are required")
        async with self.control_lock(key):
            handle = self._require_execution(key, run_id)
            entries = handle.controller.journal_entries(run_id)
            latest = next((row for row in reversed(entries)
                           if row.get("action_id") == action_id
                           and str(row.get("event", "")).startswith("tool_")), None)
            if latest and latest.get("event") == "tool_retry_authorized":
                return self.execution_snapshot(key)
            self._require_execution(key, run_id, data.get("expected_revision"))
            unknown = unresolved_actions(entries)
            if not any(row.get("action_id") == action_id for row in unknown):
                raise ValueError("No ambiguous dispatch with this action_id")
            handle.controller.journal(run_id, "tool_retry_authorized", action_id=action_id,
                                      data={"source": "human", "notes": notes})
            # Do not restart as a side effect of acknowledging a diagnostic.
            handle.controller.request_pause(run_id, "manual")
            remaining = unresolved_actions(handle.controller.journal_entries(run_id))
            if remaining:
                handle.controller.request_pause(run_id, "unknown_completion", pending_decision={
                    "actions": [{"action_id": row["action_id"], "tool": (row.get("data") or {}).get("tool")}
                                for row in remaining]})
            else:
                handle.controller.resume(run_id, "unknown_completion")
        return self.execution_snapshot(key)

    async def finish_stalled_execution(self, key, run_id, data):
        """Leave a control-loop dead end with an honest, reviewable result."""
        if data.get("confirmed") is not True:
            raise ValueError("Explicit confirmation is required")
        async with self.control_lock(key):
            handle = self._require_execution(key, run_id, data.get("expected_revision"))
            pending = handle.status().pending_decisions.get("execution_error") or {}
            if pending.get("kind") != "semantic_control_loop_guard":
                raise ValueError("No stalled control-loop decision is pending")
            handle.controller.update_metadata(run_id, {"control_resolution": {
                "decision": "finish_with_limits", "source": "human",
                "reason": pending.get("error_code") or "control_no_progress",
            }})
            handle.controller.journal(run_id, "control_resolution_requested", data={
                "decision": "finish_with_limits", "source": "human"})
            # The live callback consumes the decision at a cooperative boundary.
            # Other pause causes (budget/HITL/manual) are intentionally retained.
            handle.controller.resume(run_id, "execution_error")
        owner = self.active_runs.get(key)
        if not handle.status().pause_causes and (owner is None or owner.done()):
            await self.resume_execution(key, run_id)
        return self.execution_snapshot(key)


def register_execution_routes(app, runtime):
    from fastapi import HTTPException

    prefix = "/api/users/{user_id}/sessions/{session_id}"

    def validate(user_id, session_id):
        try:
            runtime.registry.require_session(user_id, session_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return user_id, session_id

    def http_error(exc):
        return HTTPException(404 if isinstance(exc, KeyError) else 409, str(exc))

    @app.get(prefix + "/run-control")
    async def current_execution(user_id: str, session_id: str):
        return {"run": runtime.execution_snapshot(validate(user_id, session_id))}

    @app.get(prefix + "/runs/{run_id}")
    async def execution(user_id: str, session_id: str, run_id: str):
        key = validate(user_id, session_id)
        try:
            runtime._require_execution(key, run_id)
            return {"run": runtime.execution_snapshot(key)}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc

    @app.post(prefix + "/runs/{run_id}/pause")
    async def pause(user_id: str, session_id: str, run_id: str, data: dict):
        try:
            return {"run": await runtime.pause_execution(validate(user_id, session_id), run_id,
                                                          expected_revision=data.get("expected_revision"))}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc

    @app.post(prefix + "/runs/{run_id}/outcome/decision")
    async def scientific_outcome_decision(
        user_id: str, session_id: str, run_id: str, data: dict,
    ):
        try:
            return {"run": await runtime.decide_scientific_outcome(
                validate(user_id, session_id), run_id, data,
            )}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc

    @app.post(prefix + "/runs/{run_id}/resume")
    async def resume(user_id: str, session_id: str, run_id: str, data: dict):
        try:
            return {"run": await runtime.resume_execution(validate(user_id, session_id), run_id,
                                                           expected_revision=data.get("expected_revision"))}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc

    @app.post(prefix + "/runs/{run_id}/budget/decision")
    async def budget_decision(user_id: str, session_id: str, run_id: str, data: dict):
        try:
            return {"run": await runtime.decide_execution_budget(validate(user_id, session_id), run_id, data)}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc

    @app.post(prefix + "/runs/{run_id}/recovery/authorize-retry")
    async def authorize_retry(user_id: str, session_id: str, run_id: str, data: dict):
        try:
            return {"run": await runtime.authorize_unknown_retry(validate(user_id, session_id), run_id, data)}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc

    @app.post(prefix + "/runs/{run_id}/recovery/finish-with-limits")
    async def finish_stalled(user_id: str, session_id: str, run_id: str, data: dict):
        try:
            return {"run": await runtime.finish_stalled_execution(validate(user_id, session_id), run_id, data)}
        except (KeyError, ValueError) as exc:
            raise http_error(exc) from exc
