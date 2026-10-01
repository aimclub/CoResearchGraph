"""Durable run control and model-attempt accounting.

The SQLite database is the authority.  In-memory events are only an
optimisation, so pause, stop and budget decisions survive a process restart and
are observed by another worker using the same database.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, Optional


DEFAULT_INITIAL_BUDGET = 100
GRANT_SIZE = 100
BUDGET_PAUSE_CAUSE = "budget_exhausted"
_TERMINAL_STATES = frozenset({"stopped", "completed"})


class ExecutionControlSignal(BaseException):
    """A control-flow signal which ordinary ``except Exception`` retries ignore."""


class RunStopped(ExecutionControlSignal):
    """Raised at a cooperative gate after a run has been stopped."""


class RunCompleted(ExecutionControlSignal):
    """Raised when new work is attempted for a completed run."""


class DuplicateAttempt(ExecutionControlSignal):
    """An idempotency key was already reserved; replay is not permitted."""


class ConcurrentControlUpdate(RuntimeError):
    """The caller made a decision against a stale control revision."""


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    revision: int
    control_revision: int
    state: str
    initial_budget: int
    budget_total: int
    attempts_used: int
    remaining: int
    pause_causes: tuple[str, ...] = ()
    pending_decisions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    stop_reason: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable status payload for CLI and Web APIs."""
        value = asdict(self)
        value["pause_causes"] = list(self.pause_causes)
        value["budget"] = {
            "used": self.attempts_used,
            "limit": self.budget_total,
            "remaining": self.remaining,
        }
        return value


@dataclass(frozen=True)
class AttemptReservation:
    run_id: str
    attempt_no: int
    action: str
    action_id: str
    reserved_at: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunHandle:
    """Small durable reference safe to place in a ContextVar."""

    controller: "RunController" = field(compare=False, repr=False)
    run_id: str

    def status(self) -> RunStatus:
        return self.controller.status(self.run_id)

    async def before_model_attempt(
        self,
        action: str,
        *,
        action_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AttemptReservation:
        return await self.controller.before_model_attempt(
            self.run_id, action, action_id=action_id, metadata=metadata
        )

    async def before_tool_action(
        self, action: str, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        await self.controller.before_tool_action(
            self.run_id, action, metadata=metadata
        )


Notifier = Callable[[RunHandle, RunStatus], Optional[Awaitable[None]]]


def default_database_path() -> Path:
    """Database in the application's durable state directory.

    ``COSCIENTIST_EXECUTION_DB`` is the precise override.  Otherwise this uses
    the same root as the Web session store without importing the application
    (which would eagerly configure agents and telemetry during tests).
    """
    explicit = os.getenv("COSCIENTIST_EXECUTION_DB")
    if explicit:
        return Path(explicit)
    state_root = os.getenv("WEB_STATE_DIR")
    if state_root:
        return Path(state_root) / "execution_control.sqlite3"
    graph_root = Path(os.getenv("RESEARCH_GRAPH_DIR", "./graph_runs"))
    return graph_root / "web_state" / "execution_control.sqlite3"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _object(raw: Optional[str]) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


class RunController:
    """SQLite-backed controller shared by all agents and nested tasks in a run."""

    def __init__(
        self,
        db_path: str | os.PathLike[str] | None = None,
        *,
        poll_interval: float = 0.1,
        notifier: Optional[Notifier] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_database_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.poll_interval = max(0.01, float(poll_interval))
        self._notifier = notifier
        self._init_lock = threading.Lock()
        self._initialised = False
        self._events_lock = threading.Lock()
        self._events: dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Event]] = {}
        self._initialise()

    def set_notifier(self, notifier: Optional[Notifier]) -> None:
        self._notifier = notifier

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path, timeout=30.0, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialise(self) -> None:
        if self._initialised:
            return
        with self._init_lock:
            if self._initialised:
                return
            with self._connect() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS runs (
                        run_id TEXT PRIMARY KEY,
                        revision INTEGER NOT NULL DEFAULT 0,
                        control_revision INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL,
                        initial_budget INTEGER NOT NULL,
                        budget_total INTEGER NOT NULL,
                        attempts_used INTEGER NOT NULL DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        pending_decisions_json TEXT NOT NULL DEFAULT '{}',
                        stop_reason TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS pause_causes (
                        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                        cause TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        PRIMARY KEY (run_id, cause)
                    );
                    CREATE TABLE IF NOT EXISTS grants (
                        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                        decision_id TEXT NOT NULL,
                        amount INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        PRIMARY KEY (run_id, decision_id)
                    );
                    CREATE TABLE IF NOT EXISTS model_attempts (
                        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                        attempt_no INTEGER NOT NULL,
                        action TEXT NOT NULL,
                        action_id TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        result_status TEXT,
                        result_json TEXT,
                        reserved_at REAL NOT NULL,
                        finished_at REAL,
                        PRIMARY KEY (run_id, attempt_no),
                        UNIQUE (run_id, action_id)
                    );
                    CREATE TABLE IF NOT EXISTS journal (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
                        event TEXT NOT NULL,
                        action_id TEXT,
                        data_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS ix_journal_run_id
                        ON journal(run_id, id);
                    """
                )
                columns = {
                    row["name"]
                    for row in db.execute("PRAGMA table_info(runs)").fetchall()
                }
                if "revision" not in columns:
                    db.execute(
                        "ALTER TABLE runs ADD COLUMN revision INTEGER NOT NULL DEFAULT 0"
                    )
                if "control_revision" not in columns:
                    db.execute(
                        "ALTER TABLE runs ADD COLUMN control_revision INTEGER NOT NULL DEFAULT 0"
                    )
                db.executescript(
                    """
                    CREATE TRIGGER IF NOT EXISTS bump_run_revision
                    AFTER UPDATE ON runs
                    WHEN NEW.revision = OLD.revision
                    BEGIN
                        UPDATE runs SET revision = OLD.revision + 1
                        WHERE run_id = OLD.run_id;
                    END;
                    """
                )
            self._initialised = True

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        value = str(run_id or "").strip()
        if not value:
            raise ValueError("run_id must be non-empty")
        return value

    @staticmethod
    def _validate_cause(cause: str) -> str:
        value = str(cause or "").strip()
        if not value:
            raise ValueError("pause cause must be non-empty")
        return value

    def _event(self, run_id: str) -> asyncio.Event:
        # Events are loop-local accelerators only.  A gate still polls SQLite,
        # which is what makes another process' state changes visible.
        running_loop = asyncio.get_running_loop()
        with self._events_lock:
            entry = self._events.get(run_id)
            if entry is None or entry[0] is not running_loop:
                event = asyncio.Event()
                self._events[run_id] = (running_loop, event)
                return event
            return entry[1]

    def _wake(self, run_id: str) -> None:
        with self._events_lock:
            entry = self._events.get(run_id)
        if entry is not None:
            loop, event = entry
            if loop.is_running():
                loop.call_soon_threadsafe(event.set)

    async def _wait_for_change(self, run_id: str) -> None:
        event = self._event(run_id)
        event.clear()
        try:
            await asyncio.wait_for(event.wait(), timeout=self.poll_interval)
        except asyncio.TimeoutError:
            pass

    def _notify(self, run_id: str) -> None:
        self._wake(run_id)
        callback = self._notifier
        if callback is None:
            return
        handle = RunHandle(self, run_id)
        status = self.status(run_id)
        result: Any = None
        try:
            result = callback(handle, status)
            if inspect.isawaitable(result):
                task = asyncio.get_running_loop().create_task(result)
                task.add_done_callback(self._consume_notification_error)
        except RuntimeError:
            # No running event loop: persistence is already complete.  A Web
            # process can observe the state by polling after restart.
            if inspect.iscoroutine(result):
                result.close()
        except Exception:
            # Delivery is advisory; it must never roll back durable control.
            return

    @staticmethod
    def _consume_notification_error(task: asyncio.Task[Any]) -> None:
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass

    @staticmethod
    def _journal_tx(
        db: sqlite3.Connection,
        run_id: str,
        event: str,
        *,
        action_id: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
        now: Optional[float] = None,
    ) -> None:
        db.execute(
            "INSERT INTO journal(run_id,event,action_id,data_json,created_at) "
            "VALUES(?,?,?,?,?)",
            (run_id, event, action_id, _json(dict(data or {})), now or time.time()),
        )

    def create_run(
        self,
        run_id: Optional[str] = None,
        *,
        initial_budget: int = DEFAULT_INITIAL_BUDGET,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RunHandle:
        run_id = self._validate_run_id(run_id or uuid.uuid4().hex)
        budget = int(initial_budget)
        if not 1 <= budget <= DEFAULT_INITIAL_BUDGET:
            raise ValueError("initial_budget must be between 1 and 100")
        now = time.time()
        created = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            cursor = db.execute(
                "INSERT OR IGNORE INTO runs("
                "run_id,state,initial_budget,budget_total,attempts_used,metadata_json,"
                "pending_decisions_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (run_id, "running", budget, budget, 0, _json(dict(metadata or {})), "{}", now, now),
            )
            created = cursor.rowcount == 1
            if created:
                self._journal_tx(
                    db, run_id, "run_created", data={"initial_budget": budget}, now=now
                )
            db.commit()
        if created:
            self._notify(run_id)
        return RunHandle(self, run_id)

    def get_handle(self, run_id: str) -> RunHandle:
        run_id = self._validate_run_id(run_id)
        self.status(run_id)  # fail early if it does not exist
        return RunHandle(self, run_id)

    def _status_tx(self, db: sqlite3.Connection, run_id: str) -> RunStatus:
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run_id: {run_id}")
        causes = tuple(
            item["cause"]
            for item in db.execute(
                "SELECT cause FROM pause_causes WHERE run_id=? ORDER BY created_at,cause",
                (run_id,),
            )
        )
        total, used = int(row["budget_total"]), int(row["attempts_used"])
        return RunStatus(
            run_id=run_id,
            revision=int(row["revision"]),
            control_revision=int(row["control_revision"]),
            state=row["state"],
            initial_budget=int(row["initial_budget"]),
            budget_total=total,
            attempts_used=used,
            remaining=max(0, total - used),
            pause_causes=causes,
            pending_decisions=_object(row["pending_decisions_json"]),
            metadata=_object(row["metadata_json"]),
            stop_reason=row["stop_reason"],
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def status(self, run_id: str) -> RunStatus:
        run_id = self._validate_run_id(run_id)
        with self._connect() as db:
            return self._status_tx(db, run_id)

    def list_runs(
        self,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        *,
        limit: Optional[int] = None,
    ) -> list[RunStatus]:
        sql = "SELECT run_id FROM runs ORDER BY updated_at DESC"
        with self._connect() as db:
            statuses = [self._status_tx(db, row["run_id"]) for row in db.execute(sql)]
        wanted = dict(metadata_filter or {})
        if wanted:
            statuses = [
                status
                for status in statuses
                if all(status.metadata.get(key) == value for key, value in wanted.items())
            ]
        if limit is not None:
            if int(limit) < 1:
                return []
            statuses = statuses[: int(limit)]
        return statuses

    def latest_run(
        self,
        metadata_filter: Optional[Mapping[str, Any]] = None,
        **metadata: Any,
    ) -> Optional[RunHandle]:
        """Return the most recently changed run matching durable metadata."""
        wanted = dict(metadata_filter or {})
        wanted.update(metadata)
        matches = self.list_runs(wanted, limit=1)
        return self.get_handle(matches[0].run_id) if matches else None

    def update_metadata(
        self,
        run_id: str,
        values: Mapping[str, Any],
        *,
        replace: bool = False,
    ) -> RunStatus:
        run_id = self._validate_run_id(run_id)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = self._status_tx(db, run_id)
            metadata = {} if replace else dict(status.metadata)
            metadata.update(dict(values))
            db.execute(
                "UPDATE runs SET metadata_json=?,updated_at=? WHERE run_id=?",
                (_json(metadata), now, run_id),
            )
            self._journal_tx(db, run_id, "metadata_updated", data={"keys": list(values)}, now=now)
            db.commit()
        self._notify(run_id)
        return self.status(run_id)

    def request_pause(
        self,
        run_id: str,
        cause: str,
        *,
        pending_decision: Optional[Mapping[str, Any]] = None,
        expected_revision: Optional[int] = None,
    ) -> RunStatus:
        run_id, cause = self._validate_run_id(run_id), self._validate_cause(cause)
        now = time.time()
        changed = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = self._status_tx(db, run_id)
            self._expect_control_revision(status, expected_revision)
            if status.state not in _TERMINAL_STATES:
                inserted = db.execute(
                    "INSERT OR IGNORE INTO pause_causes(run_id,cause,created_at) VALUES(?,?,?)",
                    (run_id, cause, now),
                ).rowcount == 1
                pending = dict(status.pending_decisions)
                if pending_decision is not None:
                    pending[cause] = dict(pending_decision)
                state = "paused" if status.state == "paused" else "pause_requested"
                changed = inserted or state != status.state or pending != status.pending_decisions
                if changed:
                    db.execute(
                        "UPDATE runs SET state=?,pending_decisions_json=?,updated_at=?,"
                        "control_revision=control_revision+1 WHERE run_id=?",
                        (state, _json(pending), now, run_id),
                    )
                    self._journal_tx(
                        db, run_id, "pause_requested", data={"cause": cause}, now=now
                    )
            db.commit()
        if changed:
            self._notify(run_id)
        return self.status(run_id)

    def resume(
        self, run_id: str, cause: str, *, expected_revision: Optional[int] = None
    ) -> RunStatus:
        run_id, cause = self._validate_run_id(run_id), self._validate_cause(cause)
        if cause == BUDGET_PAUSE_CAUSE:
            raise ValueError("budget exhaustion can only be resumed by an explicit grant")
        now = time.time()
        changed = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = self._status_tx(db, run_id)
            self._expect_control_revision(status, expected_revision)
            if status.state not in _TERMINAL_STATES:
                removed = db.execute(
                    "DELETE FROM pause_causes WHERE run_id=? AND cause=?",
                    (run_id, cause),
                ).rowcount == 1
                pending = dict(status.pending_decisions)
                pending.pop(cause, None)
                left = db.execute(
                    "SELECT COUNT(*) FROM pause_causes WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                state = "running" if left == 0 else "paused"
                changed = removed or state != status.state or pending != status.pending_decisions
                if changed:
                    db.execute(
                        "UPDATE runs SET state=?,pending_decisions_json=?,updated_at=?,"
                        "control_revision=control_revision+1 WHERE run_id=?",
                        (state, _json(pending), now, run_id),
                    )
                    self._journal_tx(
                        db, run_id, "pause_cause_resolved", data={"cause": cause}, now=now
                    )
            db.commit()
        if changed:
            self._notify(run_id)
        return self.status(run_id)

    def grant(
        self,
        run_id: str,
        decision_id: str,
        *,
        amount: int = GRANT_SIZE,
        metadata: Optional[Mapping[str, Any]] = None,
        expected_revision: Optional[int] = None,
    ) -> RunStatus:
        run_id = self._validate_run_id(run_id)
        decision_id = str(decision_id or "").strip()
        if not decision_id:
            raise ValueError("decision_id is required for an explicit user grant")
        if int(amount) != GRANT_SIZE:
            raise ValueError("each explicit budget renewal grants exactly 100 attempts")
        now = time.time()
        inserted = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = self._status_tx(db, run_id)
            # HTTP/user-decision replay is idempotent even when it carries the
            # revision from before the first successful grant.  Check the
            # durable decision id before optimistic-concurrency validation.
            existing = db.execute(
                "SELECT 1 FROM grants WHERE run_id=? AND decision_id=?",
                (run_id, decision_id),
            ).fetchone()
            if existing is not None:
                db.rollback()
                return status
            self._expect_control_revision(status, expected_revision)
            if status.state in _TERMINAL_STATES:
                db.rollback()
                return status
            inserted = db.execute(
                "INSERT OR IGNORE INTO grants(run_id,decision_id,amount,metadata_json,created_at) "
                "VALUES(?,?,?,?,?)",
                (run_id, decision_id, GRANT_SIZE, _json(dict(metadata or {})), now),
            ).rowcount == 1
            if inserted:
                db.execute(
                    "DELETE FROM pause_causes WHERE run_id=? AND cause=?",
                    (run_id, BUDGET_PAUSE_CAUSE),
                )
                pending = dict(status.pending_decisions)
                pending.pop(BUDGET_PAUSE_CAUSE, None)
                left = db.execute(
                    "SELECT COUNT(*) FROM pause_causes WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                state = "running" if left == 0 else "paused"
                db.execute(
                    "UPDATE runs SET budget_total=budget_total+?,state=?,"
                    "pending_decisions_json=?,updated_at=?,"
                    "control_revision=control_revision+1 WHERE run_id=?",
                    (GRANT_SIZE, state, _json(pending), now, run_id),
                )
                self._journal_tx(
                    db,
                    run_id,
                    "budget_granted",
                    action_id=decision_id,
                    data={"amount": GRANT_SIZE},
                    now=now,
                )
            db.commit()
        if inserted:
            self._notify(run_id)
        return self.status(run_id)

    def stop(
        self,
        run_id: str,
        reason: Optional[str] = None,
        *,
        expected_revision: Optional[int] = None,
    ) -> RunStatus:
        return self._terminal(run_id, "stopped", reason, expected_revision)

    def complete(
        self, run_id: str, *, expected_revision: Optional[int] = None
    ) -> RunStatus:
        return self._terminal(run_id, "completed", None, expected_revision)

    @staticmethod
    def _expect_control_revision(
        status: RunStatus, expected_revision: Optional[int]
    ) -> None:
        if expected_revision is not None and int(expected_revision) != status.control_revision:
            raise ConcurrentControlUpdate(
                f"stale control revision {expected_revision}; "
                f"current revision is {status.control_revision}"
            )

    def _terminal(
        self,
        run_id: str,
        state: str,
        reason: Optional[str],
        expected_revision: Optional[int],
    ) -> RunStatus:
        if state not in _TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {state}")
        run_id = self._validate_run_id(run_id)
        now = time.time()
        changed = False
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            status = self._status_tx(db, run_id)
            self._expect_control_revision(status, expected_revision)
            if status.state not in _TERMINAL_STATES:
                db.execute(
                    "UPDATE runs SET state=?,stop_reason=?,updated_at=?,"
                    "control_revision=control_revision+1 WHERE run_id=?",
                    (state, reason if state == "stopped" else None, now, run_id),
                )
                self._journal_tx(
                    db, run_id, f"run_{state}", data={"reason": reason}, now=now
                )
                changed = True
            db.commit()
        if changed:
            self._notify(run_id)
        return self.status(run_id)

    def journal(
        self,
        run_id: str,
        event: str,
        *,
        action_id: Optional[str] = None,
        data: Optional[Mapping[str, Any]] = None,
    ) -> int:
        run_id = self._validate_run_id(run_id)
        event = str(event or "").strip()
        if not event:
            raise ValueError("journal event must be non-empty")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._status_tx(db, run_id)
            cursor = db.execute(
                "INSERT INTO journal(run_id,event,action_id,data_json,created_at) VALUES(?,?,?,?,?)",
                (run_id, event, action_id, _json(dict(data or {})), time.time()),
            )
            entry_id = int(cursor.lastrowid)
            db.commit()
        return entry_id

    def journal_entries(
        self, run_id: str, *, after_id: int = 0, limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Read the journal in chronological order.

        Internal recovery consumers need the complete history so the latest
        completion/authorization cannot disappear beyond an arbitrary first
        page.  API/UI callers may still request an explicitly bounded page.
        """
        run_id = self._validate_run_id(run_id)
        sql = (
            "SELECT id,event,action_id,data_json,created_at FROM journal "
            "WHERE run_id=? AND id>? ORDER BY id"
        )
        params: tuple[Any, ...] = (run_id, max(0, int(after_id)))
        if limit is not None:
            bounded = max(1, min(int(limit), 10_000))
            sql += " LIMIT ?"
            params += (bounded,)
        with self._connect() as db:
            self._status_tx(db, run_id)
            rows = db.execute(sql, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "event": row["event"],
                "action_id": row["action_id"],
                "data": _object(row["data_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    async def before_tool_action(
        self,
        run_id: str,
        action: str,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        run_id = self._validate_run_id(run_id)
        while True:
            notify = False
            allowed = False
            terminal: Optional[ExecutionControlSignal] = None
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                status = self._status_tx(db, run_id)
                if status.state == "stopped":
                    db.rollback()
                    terminal = RunStopped(
                        status.stop_reason or f"run {run_id} was stopped"
                    )
                elif status.state == "completed":
                    db.rollback()
                    terminal = RunCompleted(f"run {run_id} is already completed")
                elif status.pause_causes:
                    if status.state != "paused":
                        now = time.time()
                        db.execute(
                            "UPDATE runs SET state='paused',updated_at=?,"
                            "control_revision=control_revision+1 WHERE run_id=?",
                            (now, run_id),
                        )
                        self._journal_tx(db, run_id, "run_paused", now=now)
                        notify = True
                    db.commit()
                else:
                    now = time.time()
                    if status.state != "running":
                        db.execute(
                            "UPDATE runs SET state='running',updated_at=?,"
                            "control_revision=control_revision+1 WHERE run_id=?",
                            (now, run_id),
                        )
                        notify = True
                    self._journal_tx(
                        db,
                        run_id,
                        "tool_action_started",
                        data={"action": action, **dict(metadata or {})},
                        now=now,
                    )
                    db.commit()
                    allowed = True
            if terminal is not None:
                raise terminal
            if notify:
                self._notify(run_id)
            if allowed:
                return
            await self._wait_for_change(run_id)

    async def before_model_attempt(
        self,
        run_id: str,
        action: str,
        *,
        action_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AttemptReservation:
        """Wait for permission and atomically reserve one actual provider attempt."""
        run_id = self._validate_run_id(run_id)
        action = str(action or "model_call").strip() or "model_call"
        action_id = str(action_id or uuid.uuid4().hex)
        while True:
            notify = False
            reservation: Optional[AttemptReservation] = None
            terminal: Optional[ExecutionControlSignal] = None
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                status = self._status_tx(db, run_id)
                duplicate = db.execute(
                    "SELECT attempt_no FROM model_attempts WHERE run_id=? AND action_id=?",
                    (run_id, action_id),
                ).fetchone()
                if duplicate is not None:
                    db.rollback()
                    raise DuplicateAttempt(
                        f"model attempt {action_id!r} was already reserved as "
                        f"#{duplicate['attempt_no']}; refusing provider replay"
                    )
                if status.state == "stopped":
                    db.rollback()
                    terminal = RunStopped(status.stop_reason or f"run {run_id} was stopped")
                elif status.state == "completed":
                    db.rollback()
                    terminal = RunCompleted(f"run {run_id} is already completed")
                elif status.pause_causes:
                    if status.state != "paused":
                        now = time.time()
                        db.execute(
                            "UPDATE runs SET state='paused',updated_at=?,"
                            "control_revision=control_revision+1 WHERE run_id=?",
                            (now, run_id),
                        )
                        self._journal_tx(db, run_id, "run_paused", now=now)
                        notify = True
                    db.commit()
                elif status.attempts_used >= status.budget_total:
                    now = time.time()
                    db.execute(
                        "INSERT OR IGNORE INTO pause_causes(run_id,cause,created_at) VALUES(?,?,?)",
                        (run_id, BUDGET_PAUSE_CAUSE, now),
                    )
                    pending = dict(status.pending_decisions)
                    pending[BUDGET_PAUSE_CAUSE] = {
                        "kind": "budget_renewal",
                        "grant_size": GRANT_SIZE,
                        "attempts_used": status.attempts_used,
                    }
                    db.execute(
                        "UPDATE runs SET state='paused',pending_decisions_json=?,updated_at=?,"
                        "control_revision=control_revision+1 "
                        "WHERE run_id=?",
                        (_json(pending), now, run_id),
                    )
                    self._journal_tx(
                        db,
                        run_id,
                        "budget_exhausted",
                        data={"attempts_used": status.attempts_used},
                        now=now,
                    )
                    db.commit()
                    notify = True
                else:
                    now = time.time()
                    attempt_no = status.attempts_used + 1
                    db.execute(
                        "UPDATE runs SET attempts_used=?,state='running',updated_at=? WHERE run_id=?",
                        (attempt_no, now, run_id),
                    )
                    db.execute(
                        "INSERT INTO model_attempts(run_id,attempt_no,action,action_id,metadata_json,reserved_at) "
                        "VALUES(?,?,?,?,?,?)",
                        (run_id, attempt_no, action, action_id, _json(dict(metadata or {})), now),
                    )
                    self._journal_tx(
                        db,
                        run_id,
                        "model_attempt_reserved",
                        action_id=action_id,
                        data={"attempt_no": attempt_no, "action": action},
                        now=now,
                    )
                    db.commit()
                    reservation = AttemptReservation(
                        run_id=run_id,
                        attempt_no=attempt_no,
                        action=action,
                        action_id=action_id,
                        reserved_at=now,
                    )
                    notify = True
            if terminal is not None:
                raise terminal
            if notify:
                self._notify(run_id)
            if reservation is not None:
                return reservation
            await self._wait_for_change(run_id)

    def finish_model_attempt(
        self,
        run_id: str,
        action_id: str,
        status: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Attach a result to a reservation without affecting accounting."""
        run_id = self._validate_run_id(run_id)
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            updated = db.execute(
                "UPDATE model_attempts SET result_status=?,result_json=?,finished_at=? "
                "WHERE run_id=? AND action_id=? AND finished_at IS NULL",
                (str(status), _json(dict(data or {})), now, run_id, action_id),
            ).rowcount
            if not updated:
                db.rollback()
                raise KeyError(f"unknown or already finished attempt: {action_id}")
            self._journal_tx(
                db,
                run_id,
                "model_attempt_finished",
                action_id=action_id,
                data={"status": str(status), **dict(data or {})},
                now=now,
            )
            db.commit()


__all__ = [
    "AttemptReservation",
    "BUDGET_PAUSE_CAUSE",
    "ConcurrentControlUpdate",
    "DEFAULT_INITIAL_BUDGET",
    "DuplicateAttempt",
    "ExecutionControlSignal",
    "GRANT_SIZE",
    "RunCompleted",
    "RunController",
    "RunHandle",
    "RunStatus",
    "RunStopped",
    "default_database_path",
]
