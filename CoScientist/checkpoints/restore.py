"""Restore: rehydrate a NEW run from a checkpoint bundle.

Principle (SynapseNmas §6.6.1): restore never mutates a live run — it creates
a fresh session under new ids and fills it from the snapshot. Sequence:

1. compatibility gate (pins: profile/prompts/model fingerprints);
2. mint a new contextId; because ADK maps A2A ``contextId`` -> ``session_id``
   verbatim and unauthenticated ``user_id`` -> ``f"A2A_USER_{contextId}"``
   (google/adk/a2a/converters/request_converter.py:66-118), choosing the
   session id IS choosing the A2A contextId;
3. ``create_session`` + ``append_event`` loop over the stored events
   (ids/timestamps preserved verbatim — function_call/response id pairs are
   required by the LiteLLM path);
4. one synthetic event force-applies the AUTHORITATIVE state blob — this is a
   load-bearing step, not defensive: direct-mutation keys (deployed_mcps,
   planner edits) exist in no state_delta and would otherwise be lost;
5. module stores are imported back into their singletons;
6. the caller (web UI / A2A client) continues with a normal message/send using
   the returned contextId — ADK finds the rehydrated session naturally.

Trailing unanswered function_calls (the un-run approval of a T0 bundle, the
call that raised in a failed turn) are truncated: replaying a dangling tool
call breaks the LiteLLM/OpenAI path, and re-issuing the un-executed action is
the honest resume semantics.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions

from CoScientist.checkpoints.capture import collect_pins
from CoScientist.checkpoints.model import CheckpointManifest
from CoScientist.checkpoints.store import LocalZipStore, get_default_store

logger = logging.getLogger(__name__)


class CompatibilityError(RuntimeError):
    """Raised in strict mode when the live process does not match the pins."""

    def __init__(self, mismatches: List[str]):
        super().__init__("checkpoint pins do not match the live process: " + "; ".join(mismatches))
        self.mismatches = mismatches


def _pin_mismatches(manifest: CheckpointManifest) -> List[str]:
    saved, live = manifest.pins or {}, collect_pins()
    out: List[str] = []
    for key in ("profile", "profile_sha256", "prompts_sha256"):
        if saved.get(key) and live.get(key) and saved[key] != live[key]:
            out.append(f"{key}: saved={saved[key]!r} live={live[key]!r}")
    saved_models = saved.get("models") or {}
    live_models = live.get("models") or {}
    if saved_models.get("main") and live_models.get("main") and saved_models["main"] != live_models["main"]:
        out.append(f"models.main: saved={saved_models['main']!r} live={live_models['main']!r}")
    return out


def _truncate_dangling_calls(events: List[dict]) -> List[dict]:
    """Drop TRAILING events whose function_calls never got a function_response.

    Applies to any tool, not only HITL: a T0 bundle ends with the un-run
    approval call, and a T5 bundle of a failed turn can end with the call that
    raised. Replaying an unanswered assistant tool_call breaks the
    LiteLLM/OpenAI request path; truncating it makes the orchestrator simply
    re-issue the call — the honest resume semantics for an un-executed action.
    Mid-history calls are untouched (their responses follow later).
    """
    answered: set = set()
    for ev in events:
        for part in (ev.get("content") or {}).get("parts") or []:
            fr = part.get("function_response")
            if fr and fr.get("id"):
                answered.add(fr["id"])

    def _dangling(ev: dict) -> bool:
        parts = (ev.get("content") or {}).get("parts") or []
        calls = [p.get("function_call") for p in parts if p.get("function_call")]
        if not calls:
            return False
        return all(c.get("id") not in answered for c in calls)

    while events and _dangling(events[-1]):
        dropped = events.pop()
        logger.info("restore: truncated dangling function_call event %s", dropped.get("id"))
    return events


def _import_stores(parts: Dict[str, bytes], warnings: List[str]) -> None:
    """Load module-store blobs back into the process singletons. Best-effort:
    each failure is reported as a warning, never fatal."""
    if "task_tracker" in parts:
        try:
            from CoScientist.tools.task_tracker import task_tracker_instance
            task_tracker_instance.tasks = json.loads(parts["task_tracker"])
            task_tracker_instance._save()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"task tracker import failed: {exc}")

    if "research_graph" in parts:
        try:
            from CoScientist.graph.research.store import research_graph
            research_graph.reset(archive=True)  # never silently destroy the current graph
            with research_graph._lock:
                research_graph._path.parent.mkdir(parents=True, exist_ok=True)
                research_graph._path.write_bytes(parts["research_graph"])
                research_graph._load()
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"research graph import failed: {exc}")

    if "execution_graph" in parts:
        try:
            import networkx as nx
            from CoScientist.graph.memory import knowledge_graph
            data = json.loads(parts["execution_graph"])
            g = nx.DiGraph()
            for node in data.get("nodes", []):
                g.add_node(node["id"], **node)
            for edge in data.get("edges", []):
                g.add_edge(edge["src"], edge["dst"], type=edge.get("type"))
            graph_store = knowledge_graph._store
            with graph_store._lock:
                graph_store._graphs[knowledge_graph.run_id] = g
                graph_store._snapshot(knowledge_graph.run_id)
            knowledge_graph._seeded = True
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"execution graph import failed: {exc}")

    # knowledge_memory is CROSS-RUN accumulated memory: restoring an old copy
    # would erase entries learned since the snapshot, so it is deliberately
    # NOT imported (the bundle keeps a copy for inspection/portability).


def _external_warnings(manifest: CheckpointManifest) -> List[str]:
    out: List[str] = []
    ext = manifest.external or {}
    if ext.get("fedot_jobs"):
        out.append("FEDOT work at snapshot time is detached; re-run the experiment step if needed")
    if ext.get("deployed_mcps"):
        out.append("deployed_mcps endpoints were not health-checked; ToolPreparer may need a re-run")
    if ext.get("mcp_server_ids"):
        out.append("filtered_tools reference MCP registry server_ids; they re-resolve on next use")
    return out


async def restore_checkpoint(
    checkpoint_id: str,
    *,
    session_service,
    app_name: Optional[str] = None,
    compat: str = "relaxed",
    store: Optional[LocalZipStore] = None,
    import_stores: bool = True,
) -> Dict[str, Any]:
    """Rehydrate a checkpoint into a NEW session; returns everything a client
    needs to continue the run over A2A (new contextId first of all)."""
    store = store or get_default_store()
    manifest, parts = store.load(checkpoint_id)
    warnings: List[str] = []

    mismatches = _pin_mismatches(manifest)
    if mismatches:
        if compat == "strict":
            raise CompatibilityError(mismatches)
        warnings.extend(f"pin mismatch: {m}" for m in mismatches)

    target_app = app_name or manifest.session.app_name
    if app_name and app_name != manifest.session.app_name:
        warnings.append(
            f"restoring into app_name={app_name!r} (snapshot came from "
            f"{manifest.session.app_name!r})"
        )

    new_context_id = f"restore-{checkpoint_id[-12:]}-{uuid.uuid4().hex[:8]}"
    # MUST match ADK's unauthenticated fallback (request_converter.py:66-76),
    # or the client's follow-up message/send will miss the session.
    new_user_id = f"A2A_USER_{new_context_id}"

    events: List[dict] = json.loads(parts["session_events"])
    state: Dict[str, Any] = json.loads(parts["session_state"])
    events = _truncate_dangling_calls(events)

    profile = (manifest.pins or {}).get("profile")
    if profile and profile != "system":
        warnings.append(
            f"profile {profile!r} has a sequential-root pipeline: a new invocation "
            "re-enters it from the first child; restored state makes the re-run "
            "idempotent, but expect repeated module entries"
        )

    session = await session_service.create_session(
        app_name=target_app, user_id=new_user_id, session_id=new_context_id, state={},
    )
    replayed = 0
    for ev_json in events:
        try:
            event = Event.model_validate(ev_json)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"event {ev_json.get('id')} failed validation and was skipped: {exc}")
            continue
        await session_service.append_event(session, event)
        replayed += 1

    # Authoritative state overlay — carries the direct-mutation keys that no
    # state_delta replay can reproduce (design §1, fact 2).
    await session_service.append_event(
        session,
        Event(
            author="checkpoint_restore",
            invocation_id=f"restore-{checkpoint_id[-12:]}",
            actions=EventActions(state_delta=state),
        ),
    )

    if import_stores:
        _import_stores(parts, warnings)
    warnings.extend(_external_warnings(manifest))

    resume_hint = (
        "re-present the pending human review; the operator's decision resumes the run"
        if manifest.hitl_pending
        else f"restored at {manifest.label}; next action: "
             f"{(manifest.resume_from or {}).get('next_action', 'continue')}"
    )
    logger.info(
        "Checkpoint %s restored into session %s (%d events replayed, %d warnings)",
        checkpoint_id, new_context_id, replayed, len(warnings),
    )
    return {
        "checkpoint_id": checkpoint_id,
        "label": manifest.label,
        "context_id": new_context_id,
        "session_id": new_context_id,
        "user_id": new_user_id,
        "app_name": target_app,
        "event_count": replayed,
        "resume_hint": resume_hint,
        "resume_from": manifest.resume_from,
        "hitl_pending": manifest.hitl_pending.model_dump() if manifest.hitl_pending else None,
        "warnings": warnings,
    }
