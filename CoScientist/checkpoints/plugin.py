"""CheckpointPlugin — automatic snapshot triggers at module boundaries (T0–T5).

Attaches via the plugin lists in main.py / a2a/server.py — no agent, prompt or
YAML changes. Trigger map (CHECKPOINT_DESIGN.md §5):

| Label                     | Trigger                                            |
|---------------------------|----------------------------------------------------|
| T0_before_hitl            | before_tool on request_approval/request_selection  |
|                           | (+ explicit hook in hitl/session_agent.py for the  |
|                           | buffered-final-event path no plugin hook can see)  |
| T1/T2/T4 (in-process)     | on_event: state_delta carries the module output_key|
| T1/T2/T4 (A2A remote)     | on_event: function_response named after the        |
|                           | delegated sub-agent (deltas stay on the sub-agent  |
|                           | server in remote mode — design §1)                 |
| T3_before_experiment      | before_agent on ExperimentAgent (the deployed_mcps |
|                           | state_delta DOES NOT EXIST — direct mutation)      |
| T5_invocation_end         | after_run (turn boundary, everything committed)    |

Quiescence: on_event fires when the producing step is complete and history up
to N-1 is committed (producers block on _enqueue_event); the in-hand event N is
merged by capture. The plugin refuses boundary snapshots while a delegation /
fedot tool call is still in flight on another branch.

Scoping rules learned the hard way (adversarial review):

* AgentTool spawns a CHILD Runner per delegation with the parent's plugins and
  a fresh random session — snapshots there would capture only the delegation
  sub-conversation. Child runs are detected by session_service identity (the
  root Runner's service is stable; AgentTool builds a throwaway one) and are
  fully ignored.
* ADK does NOT run ``after_tool_callback`` when a tool raises — the in-flight
  counter is also released in ``on_tool_error_callback`` and force-reset at the
  turn boundary, so one failed FEDOT call can never suppress checkpoints
  forever.
* Use CheckpointRunner: it holds the busy gate until the invocation iterator
  closes, including errors where ADK skips ``after_run_callback``. Elapsed
  time never proves quiescence.
* ``any_busy()`` is process-wide (all plugin instances): run_all serves six
  A2A apps in one process and restore mutates process-wide store singletons.
"""
from __future__ import annotations

import logging
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Dict, Optional, Set, Tuple

from google.adk.plugins.base_plugin import BasePlugin

from CoScientist.checkpoints.capture import capture_checkpoint, run_key
from CoScientist.checkpoints.model import HitlPending
from CoScientist.checkpoints.store import LocalZipStore, get_default_store

logger = logging.getLogger(__name__)

STATE_DELTA_TRIGGERS: Dict[str, str] = {
    "search_results": "T1_after_literature_review",
    "hypotheses": "T2_after_hypotheses",
    "fedot_results": "T4_after_experiment",
    "structured_tz": "T0a_after_tz",
}
FUNC_RESPONSE_TRIGGERS: Dict[str, str] = {
    "ResearchAgent": "T1_after_literature_review",
    "HypothesesAgent": "T2_after_hypotheses",
    "TaskExecutorAgent": "T4_after_experiment",
}
BEFORE_AGENT_TRIGGERS: Dict[str, str] = {
    "ExperimentAgent": "T3_before_experiment",
}
HITL_TOOL_NAMES = {"request_approval", "request_selection"}
# Tool calls whose in-flight presence blocks a boundary snapshot on OTHER
# branches (long external work: FEDOT, delegations to sub-agents).
LONG_TOOL_NAMES = {"fedot_tool"} | set(FUNC_RESPONSE_TRIGGERS)


class CheckpointPlugin(BasePlugin):
    # process-wide roster: restore must refuse while ANY runner is active,
    # because it mutates process-wide store singletons.
    _instances: "weakref.WeakSet[CheckpointPlugin]" = weakref.WeakSet()

    def __init__(self, store: Optional[LocalZipStore] = None) -> None:
        super().__init__(name="checkpoint")
        self._store = store  # resolved lazily so settings load once
        self._fired: Set[Tuple[str, str, str]] = set()    # (invocation, label, event)
        self._inflight_tools: Dict[str, int] = {}          # run_key -> count
        self._active: Set[object] = set()                 # live Runner scopes
        self._root_service = None                          # the root Runner's session service
        self._child_runs: Set[str] = set()                 # AgentTool-spawned run_keys
        CheckpointPlugin._instances.add(self)

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def store(self) -> LocalZipStore:
        if self._store is None:
            self._store = get_default_store()
        return self._store

    def is_busy(self) -> bool:
        """Any invocation whose Runner iterator has not finished closing?"""
        return bool(self._active)

    @contextmanager
    def track_run(self) -> Iterator[None]:
        """Hold restore until actual Runner completion, not a plugin callback."""
        token = object()
        self._active.add(token)
        try:
            yield
        finally:
            self._active.discard(token)

    @classmethod
    def any_busy(cls) -> bool:
        return any(p.is_busy() for p in cls._instances)

    def _is_child(self, session) -> bool:
        return run_key(session) in self._child_runs

    async def _save(self, session, label: str, *, trigger_event=None,
                    hitl_pending: Optional[HitlPending] = None,
                    reason: str = "module_boundary") -> None:
        await capture_checkpoint(
            session=session,
            label=label,
            reason=reason,
            trigger_event=trigger_event,
            hitl_pending=hitl_pending,
            store=self.store,
        )

    def _dedup(self, invocation_id: str, label: str, event_id: str) -> bool:
        key = (invocation_id or "?", label, event_id or "?")
        if key in self._fired:
            return True
        self._fired.add(key)
        if len(self._fired) > 4096:  # bounded memory over long processes
            self._fired.clear()
        return False

    # ── run lifecycle (child detection + T5) ─────────────────────────────────
    async def before_run_callback(self, *, invocation_context) -> None:
        session = invocation_context.session
        svc = getattr(invocation_context, "session_service", None)
        if self._root_service is None and svc is not None:
            # the first run on this plugin instance is always the root Runner's:
            # child runners only ever spawn from inside a root invocation
            self._root_service = svc
        if svc is not None and svc is not self._root_service:
            # AgentTool-spawned child runner: ephemeral random session holding
            # only the delegation sub-conversation — never checkpoint it
            self._child_runs.add(run_key(session))
            return None
        return None

    async def after_run_callback(self, *, invocation_context) -> None:
        session = invocation_context.session
        rid = run_key(session)
        if rid in self._child_runs:
            self._child_runs.discard(rid)
            return None
        try:
            await self._save(session, "T5_invocation_end", reason="turn_boundary")
        finally:
            # safety net: a raising tool can leak the in-flight counter (no
            # after_tool on error paths ADK misses); the turn boundary is by
            # definition quiescent, so force-reset
            self._inflight_tools.pop(rid, None)
        return None

    # ── module boundaries via committed events ──────────────────────────────
    async def on_event_callback(self, *, invocation_context, event):
        try:
            if event.partial:
                return None
            session = invocation_context.session
            if self._is_child(session):
                return None
            label = self._match_event(event)
            if label is None:
                return None
            if self._dedup(invocation_context.invocation_id, label, event.id):
                return None
            if self._inflight_tools.get(run_key(session), 0) > 0:
                # a long tool/delegation is mid-flight on another branch — not
                # a point of rest; the T5 turn-boundary snapshot still covers us
                logger.debug("checkpoint %s skipped: long tool in flight", label)
                return None
            await self._save(session, label, trigger_event=event)
        except Exception:  # noqa: BLE001
            logger.exception("checkpoint on_event trigger failed; run continues")
        return None

    def _match_event(self, event) -> Optional[str]:
        # in-process boundary: the module's output_key lands as a state_delta
        delta = (event.actions.state_delta or {}) if event.actions else {}
        for key, label in STATE_DELTA_TRIGGERS.items():
            if key in delta:
                return label
        # A2A remote boundary: the delegation returns as a function_response
        content = getattr(event, "content", None)
        for part in (content.parts if content and content.parts else []):
            fr = getattr(part, "function_response", None)
            if fr is not None and fr.name in FUNC_RESPONSE_TRIGGERS:
                return FUNC_RESPONSE_TRIGGERS[fr.name]
        return None

    # ── T3 (before_agent) ───────────────────────────────────────────────────
    async def before_agent_callback(self, *, agent, callback_context):
        try:
            label = BEFORE_AGENT_TRIGGERS.get(agent.name)
            if label is None:
                return None
            ic = callback_context._invocation_context
            if self._is_child(ic.session):
                return None
            if self._dedup(ic.invocation_id, label, f"agent:{agent.name}"):
                return None
            await self._save(ic.session, label)
        except Exception:  # noqa: BLE001
            logger.exception("checkpoint before_agent trigger failed; run continues")
        return None

    # ── T0 (function-tool HITL) + in-flight bookkeeping ─────────────────────
    async def before_tool_callback(self, *, tool, tool_args, tool_context):
        try:
            ic = tool_context._invocation_context
            if self._is_child(ic.session):
                return None
            if tool.name in LONG_TOOL_NAMES:
                rid = run_key(ic.session)
                self._inflight_tools[rid] = self._inflight_tools.get(rid, 0) + 1
            if tool.name in HITL_TOOL_NAMES:
                # the function_call event is already committed here (producers
                # block on enqueue), so the export is consistent without merging
                if not self._dedup(ic.invocation_id, "T0_before_hitl",
                                   f"tool:{tool.name}:{len(ic.session.events)}"):
                    pending = HitlPending(
                        agent=str(tool_args.get("agent_name") or tool_context.agent_name),
                        kind="approval" if tool.name == "request_approval" else "selection",
                        payload={k: v for k, v in tool_args.items()},
                    )
                    await self._save(ic.session, "T0_before_hitl",
                                     hitl_pending=pending, reason="pre_approval")
        except Exception:  # noqa: BLE001
            logger.exception("checkpoint before_tool trigger failed; run continues")
        return None

    def _release_tool(self, tool, tool_context) -> None:
        if tool.name in LONG_TOOL_NAMES:
            rid = run_key(tool_context._invocation_context.session)
            self._inflight_tools[rid] = max(0, self._inflight_tools.get(rid, 1) - 1)

    async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
        try:
            self._release_tool(tool, tool_context)
        except Exception:  # noqa: BLE001
            pass
        return None

    async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
        # ADK re-raises BEFORE after_tool_callback when a tool fails — without
        # this hook one failed fedot/delegation call would leak the in-flight
        # counter and silently suppress all later boundary checkpoints.
        try:
            self._release_tool(tool, tool_context)
        except Exception:  # noqa: BLE001
            pass
        return None  # never swallow the tool error
