"""Snapshot capture: assemble a checkpoint bundle from a LIVE invocation session.

Two facts of ADK 2.3.0 / this codebase shape everything here (see
CHECKPOINT_DESIGN.md §1):

1. Plugin ``on_event_callback`` fires BEFORE the event is persisted
   (google/adk/runners.py:781, 1382-1384) — so the triggering event must be
   merged into the export by hand, or the checkpoint misses exactly the module
   result it exists to capture.
2. Some state is written by DIRECT MUTATION outside the state_delta protocol
   (``ctx.session.state['deployed_mcps'] = …``, planner edits in
   hitl/session_agent.py) — such keys exist only on the live invocation
   session object. Therefore we export from the live ``session`` we are handed
   (never a ``get_session()`` re-fetch) and the state blob is AUTHORITATIVE on
   restore.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from CoScientist.checkpoints.model import CheckpointManifest, HitlPending, SessionRef
from CoScientist.checkpoints.store import LocalZipStore, get_default_store, new_checkpoint_id

logger = logging.getLogger(__name__)

# Key-name redaction for STATE values. Unlike a2a/server.py:_redact (substring
# match over settings dumps, where any "key" hit really is a secret), session
# state has legitimate keys like "uploaded_paper_s3_keys" — so hints match
# whole TOKENS of the key name ("api_key" → {"api","key"} is redacted,
# "downloaded_paper_s3_keys" → {"downloaded","paper","s3","keys"} is not).
# Free-text VALUES are not scanned (M1 scope, design §10).
_SECRET_HINTS = {"key", "apikey", "password", "secret", "token", "login", "credential", "credentials"}
_REDACTED = "***redacted***"
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def _secret_key(key: Any) -> bool:
    return bool(_SECRET_HINTS.intersection(_TOKEN_SPLIT.split(str(key).lower())))

RESUME_FROM = {
    "T0_before_hitl": {"module": "plan", "next_action": "re_present_review"},
    "T0a_after_tz": {"module": "tz", "next_action": "plan"},
    "T1_after_literature_review": {"module": "literature", "next_action": "hypotheses"},
    "T2_after_hypotheses": {"module": "hypotheses", "next_action": "plan_experiment"},
    "T3_before_experiment": {"module": "experiment", "next_action": "run_experiment"},
    "T4_after_experiment": {"module": "experiment", "next_action": "report"},
    "T5_invocation_end": {"module": "turn_end", "next_action": "await_user"},
}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (_REDACTED if _secret_key(k) else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=3,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 — pins are best-effort
        return None


def collect_pins() -> Dict[str, Any]:
    """Reproducibility fingerprint: everything that shapes the agent tree.

    Restore compares these against the live process — instructions are baked
    at assembly time, so the same events under a different profile would run a
    different system.
    """
    from CoScientist.config import get_settings

    settings = get_settings()
    pkg_dir = Path(__file__).parent.parent          # CoScientist/
    profile = os.getenv("COSCIENTIST_CONFIG") or "system"
    profile_path = pkg_dir / "agents" / f"{profile}.yaml"
    if not profile_path.exists():
        profile_path = pkg_dir / "agents" / "system.yaml"

    try:
        import importlib.metadata as md
        adk_version = md.version("google-adk")
    except Exception:  # noqa: BLE001
        adk_version = None

    return {
        "profile": profile,
        "profile_file": str(profile_path.name),
        "profile_sha256": _sha256_file(profile_path),
        "prompts_sha256": _sha256_file(pkg_dir / "agents" / "prompts" / "templates.py"),
        "git_commit": _git_commit(pkg_dir.parent),
        "models": {
            "main": settings.llm.main_model,
            "coder": settings.llm.coder_model,
        },
        "env": {
            "HITL__ENABLED": settings.hitl.enabled,
            "RESEARCH_GRAPH__ENABLED": settings.research_graph.enabled,
            "ORCHESTRATOR__USE_PLANNER": settings.orchestrator.use_planner,
            "COSCIENTIST_CONFIG": profile,
        },
        "adk_version": adk_version,
    }


def _external_refs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Durable references to the outside world — never payloads.

    Presigned URLs are stripped (they expire); FEDOT in-flight work has no
    external id (in-process coroutine) so the only honest policy is detach.
    """
    fedot_artifacts = []
    for art in state.get("fedot_artifacts") or []:
        if isinstance(art, dict):
            fedot_artifacts.append({k: v for k, v in art.items() if k != "url"})
    mcp_server_ids = [
        t.get("server_id")
        for t in state.get("filtered_tools") or []
        if isinstance(t, dict) and t.get("server_id") is not None
    ]
    return {
        "s3_artifacts": fedot_artifacts,
        "uploaded_paper_s3_keys": state.get("uploaded_paper_s3_keys") or [],
        "downloaded_paper_s3_keys": state.get("downloaded_paper_s3_keys") or [],
        "mcp_server_ids": mcp_server_ids,
        "deployed_mcps": state.get("deployed_mcps") or [],
        "fedot_jobs": [{"kind": "fedot_mas", "status": "detached"}] if state.get("fedot_artifacts") else [],
        "coder_workspace_id": state.get("coder_workspace_id"),
    }


def _collect_store_parts() -> Dict[str, bytes]:
    """Serialize the module stores from their in-process singletons.

    Each store already writes atomically to disk after every mutation, but the
    in-process object is the source of truth — we serialize it directly.
    All best-effort: a missing/broken store never fails the snapshot.
    """
    parts: Dict[str, bytes] = {}

    try:
        from CoScientist.tools.task_tracker import task_tracker_instance
        task_tracker_instance._load()
        parts["task_tracker"] = _json_bytes(task_tracker_instance.tasks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpoint: task tracker capture failed: %s", exc)

    try:
        from CoScientist.graph.research.store import research_graph
        with research_graph._lock:
            parts["research_graph"] = _json_bytes(research_graph._serialize())
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpoint: research graph capture failed: %s", exc)

    try:
        from CoScientist.graph.memory import knowledge_graph
        parts["execution_graph"] = _json_bytes(knowledge_graph.full())
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpoint: execution graph capture failed: %s", exc)

    try:
        from CoScientist.graph.memory_store import knowledge_memory
        parts["knowledge_memory"] = _json_bytes(knowledge_memory.full())
    except Exception as exc:  # noqa: BLE001
        logger.warning("checkpoint: knowledge memory capture failed: %s", exc)

    return parts


def _validator_pending() -> bool:
    """Are BackgroundValidatorPlugin judgments still in flight? Their verdicts
    land after the snapshot; restore should re-fire validation when true."""
    try:
        from CoScientist.graph.research import validator
        return bool(getattr(validator, "_TASKS", None))
    except Exception:  # noqa: BLE001
        return False


def run_key(session) -> str:
    return f"{session.app_name}__{session.id}"


async def capture_checkpoint(
    *,
    session,
    label: str,
    reason: str = "module_boundary",
    trigger_event=None,
    hitl_pending: Optional[HitlPending] = None,
    parent_checkpoint_id: Optional[str] = None,
    store: Optional[LocalZipStore] = None,
    validator_pending: Optional[bool] = None,
) -> Optional[CheckpointManifest]:
    """Assemble and persist one checkpoint bundle from a live session object.

    ``trigger_event`` is the event in hand inside ``on_event_callback`` (fires
    pre-persist — fact #1 above) or a buffered final event (SessionAgent HITL);
    it is merged into both the event list and the state.
    Never raises: a checkpoint failure must not fail the science run.
    """
    store = store or get_default_store()
    try:
        events = [e.model_dump(mode="json", exclude_none=True) for e in session.events]
        state: Dict[str, Any] = dict(session.state)

        if trigger_event is not None:
            already = any(e.get("id") == trigger_event.id for e in events if trigger_event.id)
            if not already:
                events.append(trigger_event.model_dump(mode="json", exclude_none=True))
            delta = (trigger_event.actions.state_delta or {}) if trigger_event.actions else {}
            for k, v in delta.items():
                if not str(k).startswith("temp:"):
                    state[k] = v

        raw_state = {k: v for k, v in state.items() if not str(k).startswith("temp:")}
        # external refs are computed from the RAW state: redaction may mask a
        # legitimate key and external holds durable ids, never secret values
        external = _external_refs(raw_state)
        state = _redact(raw_state)

        from CoScientist.checkpoints import synapse
        from CoScientist.checkpoints.trace_context import trusted_active_run_id
        rid = synapse.run_id_for(session.id) or trusted_active_run_id() or run_key(session)
        manifest = CheckpointManifest(
            checkpoint_id=new_checkpoint_id(label),
            label=label,
            run_id=rid,
            parent_checkpoint_id=parent_checkpoint_id or store.latest(rid),
            created_at=datetime.now(timezone.utc).isoformat(),
            reason=reason,
            resume_from=RESUME_FROM.get(label, {}),
            session=SessionRef(
                app_name=session.app_name,
                user_id=session.user_id,
                session_id=session.id,
                event_count=len(events),
            ),
            hitl_pending=hitl_pending,
            external=external,
            validator_pending=_validator_pending() if validator_pending is None else validator_pending,
            pins=collect_pins(),
        )
        manifest.snapshot_ref = synapse.snapshot_ref_for(manifest.checkpoint_id)

        parts = {
            "session_events": _json_bytes(events),
            "session_state": _json_bytes(state),
        }
        parts.update(_collect_store_parts())

        saved = store.save(manifest, parts)
        synapse.notify_snapshot_saved(saved)
        return saved
    except Exception:  # noqa: BLE001 — never break the run because of a snapshot
        logger.exception("checkpoint capture failed (label=%s); run continues", label)
        return None
