"""Replay a recorded session into the live UI, for a demonstration.

A study that took six hours cannot be shown at a booth, and re-running it on
stage is not an option either. This replays what was actually recorded — the
agent events of one session and the growth of its research graph — into a fresh
session of the running server, at a chosen speed. What the screen shows is the
real run; only the clock is different.

Nothing here fabricates content. Events are the recorded events; the graph is
rebuilt from the final snapshot by inserting each node and edge at its own
recorded ``created_at``, so the shape at any moment is the shape the study
actually had at that moment.

Started through ``POST /api/demo/replay``; the caller gets the new session id and
watches it like any other.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("CoScientist.web.replay")

#: Marker put on every replayed event, so a recording is never mistaken for a
#: live run by anything reading the transcript later.
REPLAY_FLAG = "replayed_from"

#: Event types the browser renders in the chat column, as opposed to the tool
#: panel. These are the ones a viewer reads, so they get their own spacing.
_CHAT_TYPES = ("agent_event", "user_message", "hitl_request")

#: Tools that are a question to the operator, not a computation.
_HITL_TOOLS = ("request_selection", "request_approval")


def _load_events(path: Path) -> List[Dict[str, Any]]:
    """Recorded UI events, oldest first. Accepts a JSONL or a JSON array."""
    if not path.exists():
        raise FileNotFoundError(f"no event transcript at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        events = json.loads(text)
    else:
        events = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [e for e in events if isinstance(e, dict)]


def _event_time(event: Dict[str, Any]) -> Optional[float]:
    """Seconds since epoch for an event, from whichever field carries its time."""
    for key in ("ts", "timestamp", "time", "created_at"):
        value = event.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)):
            return float(value)
        try:
            from datetime import datetime
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
    return None


def _graph_steps(graph: Dict[str, Any]) -> List[Tuple[float, str, Dict[str, Any]]]:
    """The graph's growth, as (time, kind, item) sorted by recorded creation.

    Edges inherit the later of their endpoints' times when they carry none, so an
    edge never appears before the nodes it connects.
    """
    nodes = graph.get("nodes") or []
    born = {n["id"]: float(n.get("created_at") or 0) for n in nodes if isinstance(n, dict)}
    steps: List[Tuple[float, str, Dict[str, Any]]] = []
    for n in nodes:
        if not isinstance(n, dict):
            continue
        history = [h for h in (n.get("status_history") or []) if isinstance(h, dict)]
        # A node enters the record at the status it was created with, not at the
        # one it ended on: replaying the final status makes a hypothesis appear
        # already confirmed. Later transitions arrive as their own steps.
        first = dict(n)
        if history:
            first["status"] = history[0].get("to") or n.get("status")
        steps.append((born.get(n["id"], 0.0), "node", first))
        for change in history[1:]:
            when = change.get("at")
            steps.append((float(when) if when else born.get(n["id"], 0.0), "status",
                          {"id": n["id"], "status": change.get("to"),
                           "source": change.get("source"),
                           "reason": change.get("reason")}))
    for e in graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        t = e.get("created_at")
        t = float(t) if t else max(born.get(e.get("from"), 0.0), born.get(e.get("to"), 0.0))
        steps.append((t, "edge", e))
    # At one instant a node must land before an edge that references it, and a
    # verdict must land after the conclusion that carries it: the validator wrote
    # both in the same commit, and a hypothesis turning green before its
    # conclusion exists reads as the graph knowing the answer in advance.
    rank = {"node": 0, "edge": 1, "status": 2}
    steps.sort(key=lambda s: (s[0], rank.get(s[1], 3)))
    # A status the study passed through in the same instant is churn on screen;
    # keep only the last of a same-instant run for one node.
    collapsed: List[Tuple[float, str, Dict[str, Any]]] = []
    for step in steps:
        if (step[1] == "status" and collapsed
                and collapsed[-1][1] == "status"
                and collapsed[-1][2]["id"] == step[2]["id"]
                and collapsed[-1][0] == step[0]):
            collapsed[-1] = step
            continue
        collapsed.append(step)
    return collapsed


class ReplaySession:
    """One replay in flight. Cancelled by dropping the reference."""

    def __init__(self, runtime, user_id: str, session_id: str, *,
                 events: List[Dict[str, Any]], graph: Dict[str, Any],
                 speed: float, max_gap: float, source: str,
                 min_gap: float = 0.4, warmup: float = 25.0,
                 chat_gap: float = 10.0, thoughts: bool = True) -> None:
        self.runtime = runtime
        self.key = (user_id, session_id)
        self.events = events
        self.graph = graph
        self.speed = max(1.0, float(speed))
        self.max_gap = max(0.0, float(max_gap))
        # A floor, not only a cap. The recording is bursty -- dozens of events
        # share a second, then nothing for forty minutes -- and replaying that
        # faithfully dumps the burst in one frame. The floor spaces the burst
        # out without reordering it.
        self.min_gap = max(0.0, float(min_gap))
        self.warmup = max(0.0, float(warmup))
        # Chat is paced separately from the tool panel. A tool row is a line and
        # can flow; a message is a paragraph the viewer has to read, and 150
        # tool calls should not be slowed to its speed.
        self.chat_gap = max(0.0, float(chat_gap))
        self.thoughts = bool(thoughts)
        self.source = source
        self.cancelled = False

    # ── timing ───────────────────────────────────────────────────────────────
    def _sleep_for(self, previous: Optional[float], current: Optional[float]) -> float:
        """Wall-clock pause before the next item, compressed and capped.

        The cap matters more than the ratio: a recorded run has forty-minute
        waits in it, and at any honest speed those are still dead air on stage.
        """
        if previous is None or current is None or current < previous:
            return self.min_gap
        return max(self.min_gap,
                   min(self.max_gap, (current - previous) / self.speed))

    # ── the run ──────────────────────────────────────────────────────────────
    async def run(self) -> None:
        started = time.time()
        try:
            if self.warmup:
                await asyncio.sleep(self.warmup)
            await self._emit({"type": "status", "status": "processing",
                              "message": f"Replaying a recorded session at "
                                         f"{self.speed:g}x. Every event below was "
                                         f"produced by the real run."})
            await asyncio.gather(self._play_events(), self._grow_graph())
            await self._emit({"type": "status", "status": "idle",
                              "message": "Replay complete."})
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — a demo must not take the server down
            logger.exception("replay failed")
            await self._emit({"type": "error", "message": f"replay failed: {exc}"})
        finally:
            logger.info("replay of %s finished in %.1fs", self.source,
                        time.time() - started)

    async def _emit(self, event: Dict[str, Any]) -> None:
        event = {**event, REPLAY_FLAG: self.source}
        self.runtime.record_event(self.key, event)
        # Recording alone leaves an open tab showing the snapshot it connected
        # with and nothing after it; the fan-out is what makes a replay live.
        send = getattr(self.runtime, "send", None)
        if callable(send):
            try:
                await send(self.key, event)
            except Exception:  # noqa: BLE001 — a dead socket must not stop playback
                logger.debug("replay could not deliver an event", exc_info=True)

    #: The recorded commit payloads carry the study's own Russian text. Half
    #: translating a JSON blob reads worse than not showing it, so a payload
    #: with Cyrillic in it is displayed as its size instead. The bundle keeps
    #: the payload itself.
    _CYRILLIC = re.compile("[\u0410-\u044f\u0401\u0451]")

    @classmethod
    def _payload(cls, value: Any) -> Any:
        if isinstance(value, str) and cls._CYRILLIC.search(value):
            return f"[{len(value)} chars, not shown in the English recording]"
        return value

    @classmethod
    def _to_ui(cls, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Recorded agent event -> the shape the browser dispatches on.

        The transcript is written by the agent-event logger, which keys on
        ``kind``; the UI switches on ``type`` and would silently drop anything
        else. Nothing is invented here: author, text, tool name, arguments and
        result are the recorded ones.
        """
        if event.get("type"):                     # already a UI event
            return dict(event)
        kind = event.get("kind")
        author = event.get("agent") or "system"
        stamp = event.get("ts")
        if kind in ("text", "message", "agent_text", "thought", "final_response"):
            text = (event.get("text") or "").strip()
            if not text:
                return None
            return {"type": "agent_event", "author": author, "content": text,
                    "timestamp": stamp, "is_final": kind == "final_response"}
        if kind in ("user_input", "user", "user_message", "prompt"):
            text = (event.get("text") or event.get("message") or "").strip()
            return {"type": "user_message", "message": text,
                    "timestamp": stamp} if text else None
        if kind == "tool_error":
            return {"type": "tool_activity", "phase": "error", "author": author,
                    "tool": event.get("tool"),
                    "error": cls._payload(event.get("error")),
                    "call_id": event.get("call_id") or event.get("id"),
                    "timestamp": stamp}
        # The two human-in-the-loop calls are the operator's decision points, and
        # the browser draws them as a card with buttons rather than as a row in
        # the tool panel. Message and options are the recorded ones.
        if kind == "tool_call" and event.get("tool") in _HITL_TOOLS:
            payload = event.get("args")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except ValueError:
                    payload = {}
            payload = payload if isinstance(payload, dict) else {}
            return {"type": "hitl_request",
                    "request_id": f"replay-hitl-{event.get('ts')}",
                    "agent_name": payload.get("agent_name") or author,
                    "action_type": ("select_option"
                                    if event["tool"] == "request_selection" else "approve"),
                    "message": payload.get("message") or "",
                    "description": payload.get("message") or "",
                    "options": payload.get("options") or [],
                    "timestamp": stamp}
        if kind == "tool_result" and event.get("tool") in _HITL_TOOLS:
            answer = event.get("result")
            if isinstance(answer, str):
                try:
                    answer = json.loads(answer)
                except ValueError:
                    answer = {}
            answer = answer if isinstance(answer, dict) else {}
            said = (answer.get("feedback") or answer.get("selected")
                    or ("approved" if answer.get("approved") else "declined"))
            return {"type": "agent_event", "author": "human",
                    "content": f"Operator: {said}", "timestamp": stamp,
                    "is_final": False}
        if kind in ("tool_call", "tool_start"):
            return {"type": "tool_activity", "phase": "call", "author": author,
                    "tool": event.get("tool"),
                    "args": cls._payload(event.get("args")),
                    "call_id": event.get("call_id") or event.get("id"),
                    "timestamp": stamp}
        if kind in ("tool_result", "tool_end"):
            failed = str(event.get("status") or "").lower() in ("error", "failed")
            out = {"type": "tool_activity", "author": author,
                   "tool": event.get("tool"),
                   "call_id": event.get("call_id") or event.get("id"),
                   "timestamp": stamp}
            if failed:
                out["phase"] = "error"
                out["error"] = cls._payload(event.get("result"))
            else:
                out["phase"] = "result"
                out["result"] = cls._payload(event.get("result"))
            return out
        return None

    @staticmethod
    def _pair_tool_calls(events: List[Dict[str, Any]]) -> None:
        """Give each call and its result a shared id, in place.

        The transcript carries no call id, and the tool panel pairs a response
        to its call by one. Matching the next result for the same agent and
        tool reproduces the pairing the run actually had.
        """
        pending: Dict[Tuple[str, str], str] = {}
        counter = 0
        for event in events:
            key = (str(event.get("agent")), str(event.get("tool")))
            kind = event.get("kind")
            if kind == "tool_call":
                counter += 1
                pending[key] = f"replay-{counter}"
                event["call_id"] = pending[key]
            elif kind in ("tool_result", "tool_end", "tool_error"):
                event["call_id"] = pending.pop(key, None)

    async def _play_events(self) -> None:
        self._pair_tool_calls(self.events)
        previous = None
        last_chat = 0.0
        for event in self.events:
            if self.cancelled:
                return
            if event.get("kind") == "tool_start":   # duplicate of tool_call
                continue
            if not self.thoughts and event.get("kind") == "thought":
                continue
            current = _event_time(event)
            await asyncio.sleep(self._sleep_for(previous, current))
            previous = current if current is not None else previous
            translated = self._to_ui(event)
            if translated is None:
                continue
            if translated["type"] in _CHAT_TYPES and self.chat_gap:
                waited = time.monotonic() - last_chat
                if last_chat and waited < self.chat_gap:
                    await asyncio.sleep(self.chat_gap - waited)
                last_chat = time.monotonic()
            await self._emit(translated)

    def _event_wall_clock(self) -> float:
        """Seconds the event stream will take, under the current pacing."""
        total, previous, since_chat = 0.0, None, None
        for event in self.events:
            if event.get("kind") == "tool_start":
                continue
            if not self.thoughts and event.get("kind") == "thought":
                continue
            current = _event_time(event)
            step = self._sleep_for(previous, current)
            previous = current if current is not None else previous
            translated = self._to_ui(event)
            if translated is None:
                total += step
                continue
            if translated["type"] in _CHAT_TYPES and self.chat_gap:
                if since_chat is None:
                    since_chat = 0.0
                else:
                    step = max(step, self.chat_gap - since_chat)
                since_chat = 0.0
            elif since_chat is not None:
                since_chat += step
            total += step
        return total

    async def _grow_graph(self) -> None:
        """Insert nodes and edges into the session's live graph, in recorded order."""
        if self.warmup:
            await asyncio.sleep(self.warmup)
        from CoScientist.graph.research.store import get_research_graph
        store = get_research_graph(user_id=self.key[0], session_id=self.key[1])
        steps = _graph_steps(self.graph)
        if not steps:
            return
        # The recorded graph was written in a handful of bursts, so replaying
        # its own intervals makes it appear almost at once. Spread the
        # insertions evenly over the projected length of the event stream, in
        # the recorded order, so the shape grows while the transcript runs.
        span = self._event_wall_clock()
        step_delay = max(self.min_gap, span / len(steps)) if span else self.min_gap
        for _when, kind, item in steps:
            if self.cancelled:
                return
            await asyncio.sleep(step_delay)
            try:
                self._insert(store, kind, item)
            except Exception:  # noqa: BLE001 — one bad item must not stop the show
                logger.debug("replay could not insert %s %s", kind, item.get("id"),
                             exc_info=True)

    @staticmethod
    def _insert(store, kind: str, item: Dict[str, Any]) -> None:
        """Write one recorded item straight into the graph.

        Two deliberate bypasses. The commit path is skipped because it re-checks
        role permissions and status transitions this content already passed when
        the study ran, and a replay is not entitled to re-assert them. And the
        write goes to the store's own graph rather than to ``full_graph()``,
        which hands out a copy — inserting there changes nothing anyone can see.
        """
        with store._lock:                       # noqa: SLF001 — demo playback
            graph = store._g                    # noqa: SLF001
            ReplaySession._write(graph, kind, item)
        save = getattr(store, "_save", None)
        if callable(save):
            save()

    @staticmethod
    def _write(graph, kind: str, item: Dict[str, Any]) -> None:
        if kind == "status":
            node = graph.nodes.get(item["id"]) if item["id"] in graph.nodes else None
            if node is not None and item.get("status"):
                node["status"] = item["status"]
            return
        if kind == "node":
            # The whole record goes in as node data, `id` included: the store
            # serialises a node from its attribute dict alone, so an id kept only
            # as the networkx key would vanish on the way to the renderer.
            graph.add_node(item["id"], **dict(item))
        else:
            u, v = item.get("from"), item.get("to")
            if u in graph.nodes and v in graph.nodes:
                graph.add_edge(u, v, key=item.get("type"), type=item.get("type"),
                               attrs=item.get("attrs") or {},
                               source=item.get("source"))


def load_recording(bundle: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Events and final graph of a recorded session.

    ``bundle`` is either a directory produced by ``scripts/collect_session.py``
    or a session directory under ``graph_runs/sessions``; the event transcript is
    looked up next to it under the web-state directory when the bundle has none.
    """
    root = Path(bundle)
    if not root.exists():
        raise FileNotFoundError(f"no recording at {root}")

    graph_path = next((p for p in (root / "graphs" / "research_active.json",
                                   root / "research_active.json")
                       if p.exists()), None)
    if graph_path is None:
        raise FileNotFoundError(f"no research_active.json under {root}")
    graph = json.loads(graph_path.read_text(encoding="utf-8"))

    # Deterministic, and ``events.jsonl`` wins: a bundle often keeps a second
    # transcript beside it (an untrimmed original, a variant), and glob order is
    # filesystem order, so picking the first match silently played the wrong file.
    candidates = sorted((root / "events").glob("*.jsonl")) if (root / "events").is_dir() else []
    candidates += sorted(root.glob("*.jsonl"))
    candidates.sort(key=lambda p: (p.name != "events.jsonl", p.name))
    if not candidates:
        # A session directory under graph_runs/sessions holds the graph; its UI
        # transcript lives beside it under the web-state tree, keyed the same way.
        import os
        state = Path(os.getenv("WEB_STATE_DIR", "graph_runs/web_state"))
        guess = state / "sessions" / root.parent.name / f"{root.name}.jsonl"
        if guess.exists():
            candidates = [guess]
    events = _load_events(candidates[0]) if candidates else []
    return events, graph
