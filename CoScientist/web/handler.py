"""Session-aware Human-in-the-Loop handler for the local Web UI."""

import asyncio
import logging
import time
import uuid
from datetime import datetime

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLDecisionSource, HITLRequest, HITLResponse
from CoScientist.hitl.resolver import resolve_auto, resolve_timeout


#: Stamped on both transcript events of a decision the MODE took, so a reader —
#: and the chat card that replays them — can tell it from a human's click.
AUTO_MARK = "mode=auto"


def _auto_approves() -> bool:
    """Whether this run answers its own confirmations. Read per request, so the
    Approvals tab applies to the next card rather than the next restart."""
    try:
        from CoScientist.hitl.mode import auto_approves

        return auto_approves()
    except Exception:  # noqa: BLE001 — an unreadable mode still asks the human
        return False

logger = logging.getLogger("CoScientist.web.hitl")
SessionKey = tuple[str, str]


def hitl_response_event(request_id: str, response_data: dict) -> dict:
    """The operator's answer to one HITL card, as a transcript entry."""
    return {
        "type": "hitl_response",
        "request_id": request_id,
        "action": response_data.get("action"),
        "approved": response_data.get("approved"),
        "selected_option": response_data.get("selected_option"),
        "instructions": response_data.get("instructions"),
        "free_input": response_data.get("free_input"),
        "form_values": response_data.get("form_values"),
        "decision_source": response_data.get("decision_source", "human"),
        "system_reason": response_data.get("system_reason"),
        "timestamp": datetime.now().isoformat(),
    }


class WebHITLHandler(AbstractHITLHandler):
    """Route HITL requests only to tabs displaying the owning session."""

    def __init__(self):
        # request_id -> future/payload/session metadata
        self._pending: dict[str, dict] = {}
        # Two resumed invocations can reach the same logical review before
        # either one has published its card.  Reserving the request under this
        # lock keeps them on one future instead of replacing each other.
        self._request_lock = asyncio.Lock()
        # (user_id, session_id) -> connected browser sockets. ``None`` is kept
        # solely for backwards-compatible callers that have no session context.
        self._sockets: dict[SessionKey | None, list] = {}
        self._event_log: list[dict] = []
        self._sender = None
        # Writes HITL cards and their answers into the session transcript, so a
        # reload, an export or an import still shows decisions already taken.
        self._recorder = None
        # Optional durable checkpoint hook.  It is called before every HITL
        # wait, including callback/tool HITLs that never produce an ADK
        # RequestInput interrupt.
        self._checkpoint = None

    @property
    def hitl_timeout_seconds(self) -> float:
        try:
            from CoScientist.hitl.mode import wait_seconds

            window = wait_seconds()
            # `None` means "wait for the human". Callers of this property expect
            # a number and read `<= 0` as exactly that, so it is spoken here in
            # the vocabulary they already have.
            return -1.0 if window is None else float(window)
        except Exception:
            return 300

    @hitl_timeout_seconds.setter
    def hitl_timeout_seconds(self, value: float) -> None:
        # Compatibility-only: callers may still assign the legacy field, but
        # the global HITL mode is the sole source of timing behaviour.
        self._legacy_hitl_timeout_seconds = float(value)

    @property
    def HITL_TIMEOUT_SECONDS(self) -> int:
        return self.hitl_timeout_seconds

    def __deepcopy__(self, memo):
        return self

    @property
    def _websocket(self):
        """Backwards-compatible access to the most recently attached socket."""
        sockets = [socket for group in self._sockets.values() for socket in group]
        return sockets[-1] if sockets else None

    def set_websocket(self, ws):
        """Legacy setter: register one unscoped socket, or clear all sockets."""
        self._sockets = {} if ws is None else {None: [ws]}

    def set_sender(self, sender) -> None:
        """Use the Web runtime's serialized socket writer when available."""
        self._sender = sender

    def set_recorder(self, recorder) -> None:
        """Persist HITL events through ``recorder(session_key, event)``."""
        self._recorder = recorder

    def set_checkpoint_sink(self, checkpoint) -> None:
        """Persist a resumable boundary before a human decision is awaited."""
        self._checkpoint = checkpoint

    def _record(self, session_key: SessionKey | None, event: dict) -> None:
        if self._recorder is None or session_key is None:
            return
        try:
            # A copy: a hold later mutates the live payload for redelivery.
            self._recorder(session_key, dict(event))
        except Exception as exc:  # noqa: BLE001 — persistence must never break a run
            logger.warning("HITL event could not be recorded: %s", exc)

    async def _send_json(self, ws, payload: dict, session_key) -> None:
        if self._sender is not None:
            await self._sender(ws, payload, session_key)
        else:
            await ws.send_json(payload)

    def connection_count(self) -> int:
        return sum(len(group) for group in self._sockets.values())

    def has_connections(self, session_key: SessionKey | None = None) -> bool:
        if session_key is None:
            return self.connection_count() > 0
        return bool(self._sockets.get(session_key))

    async def attach_websocket(
        self,
        ws,
        session_key: SessionKey | None = None,
    ) -> None:
        """Attach a tab and redeliver its unresolved session requests."""
        sockets = self._sockets.setdefault(session_key, [])
        if ws not in sockets:
            sockets.append(ws)
        logger.info("HITL websocket attached (%d connection(s))", self.connection_count())

        for request_id, entry in list(self._pending.items()):
            future = entry.get("future")
            if future is None or future.done():
                self._pending.pop(request_id, None)
                continue
            if entry.get("session_key") not in (None, session_key):
                continue
            try:
                await self._send_json(ws, entry["payload"], session_key)
                logger.info("HITL request %s redelivered", request_id[:8])
            except Exception as exc:  # noqa: BLE001
                logger.warning("HITL redelivery of %s failed: %s", request_id[:8], exc)

    def detach_websocket(
        self,
        ws,
        session_key: SessionKey | None = None,
    ) -> None:
        sockets = self._sockets.get(session_key, [])
        try:
            sockets.remove(ws)
        except ValueError:
            pass
        if not sockets:
            self._sockets.pop(session_key, None)
        logger.info("HITL websocket detached (%d connection(s) left)", self.connection_count())

    async def _broadcast(
        self,
        payload: dict,
        session_key: SessionKey | None,
    ) -> int:
        """Send a request to one session, or all tabs for legacy unscoped calls."""
        if session_key is None:
            targets = [socket for group in self._sockets.values() for socket in group]
        else:
            targets = list(self._sockets.get(session_key, []))

        delivered = 0
        for ws in targets:
            try:
                await self._send_json(ws, payload, session_key)
                delivered += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("HITL delivery failed (%s); pruning socket", exc)
                for key, sockets in list(self._sockets.items()):
                    if ws in sockets:
                        self.detach_websocket(ws, key)
        return delivered

    def pending_summary(self) -> list[dict]:
        now = time.time()
        return [
            {
                "request_id": request_id[:8],
                "agent_name": entry["payload"].get("agent_name"),
                "session_key": entry.get("session_key"),
                "age_seconds": round(now - entry["created"], 1),
            }
            for request_id, entry in self._pending.items()
        ]

    @staticmethod
    def _request_session_key(request: HITLRequest) -> SessionKey | None:
        session = (request.context or {}).get("_session")
        if not isinstance(session, dict):
            return None
        user_id = session.get("user_id")
        session_id = session.get("session_id")
        if not user_id or not session_id:
            return None
        return str(user_id), str(session_id)

    @staticmethod
    def _document_source(payload: dict) -> tuple[str, str, str]:
        """`(markdown, kind, summary)` for one request, or empty strings.

        Every review already carries its own rendered body in `context.output` —
        the plan matrix, the work order, the frame, the agent's proposed result.
        That body is the document; this only decides what to call it and, where
        the request knows better words than the text does, what to say about it
        in the feed.
        """
        context = payload.get("context") or {}
        markdown = str(context.get("output") or "").strip()
        if not markdown:
            return "", "", ""

        from CoScientist.reporting.documents import short_summary

        trigger = str(payload.get("trigger") or "")
        if context.get("experiment_plan"):
            goal = (context["experiment_plan"] or {}).get("goal")
            return markdown, "plan", short_summary(goal)
        if trigger == "work_report":
            # The agent wrote this one itself: `submit_work_report` asks for the
            # result in two to four sentences.
            report = context.get("work_report") or {}
            return markdown, "work_report", short_summary(report.get("summary"))
        if trigger.startswith("work_order"):
            order = context.get("work_order") or {}
            return markdown, "work_order", short_summary(order.get("goal"))
        if payload.get("form"):
            return markdown, "frame", ""
        return markdown, "review", ""

    async def _attach_document(self, payload: dict, session_key: SessionKey | None) -> None:
        """Publish the request's body as a document and point the card at it.

        Silent on failure by design: a card with no document shows its body the
        way it always did, which is worse to read but never missing.
        """
        if not session_key:
            return
        markdown, kind, summary = self._document_source(payload)
        if not markdown:
            return
        try:
            from CoScientist.reporting.documents import publish_document

            document = await asyncio.to_thread(
                publish_document,
                session_key,
                markdown=markdown,
                kind=kind,
                summary=summary,
                agent=str(payload.get("agent_name") or ""),
            )
        except Exception as exc:  # noqa: BLE001 — never lose the request itself
            logger.warning("HITL document not published: %s", exc)
            return
        if document:
            payload["document"] = document.as_payload()
            payload["summary"] = document.summary

    @staticmethod
    def _is_experiment_review(request: HITLRequest) -> bool:
        kind = (request.context or {}).get("experiment_review_kind")
        return kind in {"plan", "result"}

    @staticmethod
    def _request_id(
        request: HITLRequest,
        session_key: SessionKey | None,
    ) -> str:
        """Return a stable id when the caller supplies a logical review id.

        Random ids remain the default for ordinary confirmations.  A durable
        experiment review includes its plan digest in ``experiment_review_id``;
        replaying that exact checkpoint therefore redraws the same card rather
        than creating a second independently actionable request.
        """
        review_id = (request.context or {}).get("experiment_review_id")
        if not review_id:
            return str(uuid.uuid4())
        scope = ":".join(session_key or ("unscoped", request.agent_name))
        return str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"coscientist:hitl:{scope}:{review_id}",
        ))

    @staticmethod
    def _response(response_data: dict) -> HITLResponse:
        action = HITLAction(response_data.get("action", "approve"))
        return HITLResponse(
            action=action,
            approved=response_data.get("approved", False),
            selected_option=response_data.get("selected_option"),
            instructions=response_data.get("instructions"),
            free_input=response_data.get("free_input"),
            form_values=response_data.get("form_values"),
            timed_out=response_data.get("timed_out", False),
            decision_source=response_data.get(
                "decision_source", HITLDecisionSource.HUMAN.value),
            system_reason=response_data.get("system_reason"),
        )

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        session_key = self._request_session_key(request)
        request_id = self._request_id(request, session_key)
        public_context = dict(request.context or {})
        public_context.pop("_session", None)

        # `auto`: nobody is asked, so no card is drawn and nothing waits. Both
        # events are still written to the transcript — a reload or an export
        # must show WHAT was asked as well as that the mode answered it, and a
        # response with no request reads as a gap in the record.
        if _auto_approves() and not request.requires_human:
            response = resolve_auto(request)
            asked = {
                "type": "hitl_request", "request_id": request_id,
                "agent_name": request.agent_name,
                "action_type": request.action_type.value,
                "message": request.message, "options": request.options,
                "default_option": request.default_option,
                "context": public_context, "form": request.form,
                "invoked_via": request.invoked_via, "trigger": request.trigger,
                "timeout_seconds": 0, "auto": AUTO_MARK,
                "timestamp": datetime.now().isoformat(),
            }
            self._record(session_key, asked)
            self._event_log.append({**asked, "_session_key": session_key})
            taken = {
                "type": "hitl_response", "request_id": request_id,
                "agent_name": request.agent_name,
                **response.model_dump(mode="json"),
                "auto": AUTO_MARK, "timestamp": datetime.now().isoformat(),
            }
            self._record(session_key, taken)
            self._event_log.append({**taken, "_session_key": session_key})
            logger.info("HITL %s answered by the mode, not a human "
                        "(HITL__MODE=auto): %s%s", request_id[:8],
                        request.agent_name,
                        f" -> {response.selected_option!r}"
                        if response.selected_option is not None else "")
            return response

        # One global mode owns every review window. Per-request and legacy
        # settings remain serializable for old sessions but no longer override
        # auto/basic/debug.
        from CoScientist.hitl.mode import wait_seconds

        timeout_sec = (
            request.timeout_seconds
            if request.requires_human and request.timeout_seconds is not None
            else wait_seconds()
        )
        payload = {
            "type": "hitl_request",
            "request_id": request_id,
            "agent_name": request.agent_name,
            "action_type": request.action_type.value,
            "message": request.message,
            "options": request.options,
            "default_option": request.default_option,
            "context": public_context,
            "form": request.form,
            "invoked_via": request.invoked_via,
            "trigger": request.trigger,
            "requires_human": request.requires_human,
            "timeout_seconds": timeout_sec,
            "timestamp": datetime.now().isoformat(),
        }
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        entry = {
            "future": future,
            "payload": payload,
            "created": time.time(),
            "session_key": session_key,
            # Auto-approve moment (loop time); None waits for the human. A hold
            # from the browser clears it mid-wait (see hold_request).
            "deadline": loop.time() + timeout_sec
            if timeout_sec is not None and timeout_sec > 0 else None,
        }
        async with self._request_lock:
            existing = self._pending.get(request_id)
            if existing is None:
                self._pending[request_id] = entry

        if existing is not None:
            # The owner records, publishes and handles the timeout.  Every
            # concurrent replay observes the same human decision without
            # drawing another card or appending duplicate transcript events.
            response_data = await asyncio.shield(existing["future"])
            return self._response(response_data)

        try:
            # Before `_record`: the transcript must carry the document
            # reference too, or a reopened session would show a card with no
            # way in.
            await self._attach_document(payload, session_key)
            self._record(session_key, payload)
            if self._checkpoint is not None and session_key is not None:
                try:
                    await self._checkpoint(session_key, payload)
                except Exception as exc:  # noqa: BLE001 - observer-only
                    logger.warning("HITL checkpoint could not be created: %s", exc)

            log_payload = dict(payload)
            log_payload["_session_key"] = session_key
            self._event_log.append(log_payload)
            delivered = await self._broadcast(payload, session_key)
        except BaseException:
            # Reservation happens before the first await to close the duplicate
            # race.  If publication itself is cancelled, release that
            # reservation and wake any coalesced waiter instead of leaving a
            # card that can be redelivered forever.
            if self._pending.get(request_id) is entry:
                self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()
            raise

        if delivered:
            logger.info("HITL request %s sent to %d tab(s)", request_id[:8], delivered)
        else:
            if timeout_sec is not None and timeout_sec > 0:
                logger.warning(
                    "HITL request %s has no live tab; waiting %ss for reconnect",
                    request_id[:8],
                    timeout_sec,
                )
            else:
                logger.warning(
                    "HITL request %s has no live tab; waiting indefinitely for reconnect",
                    request_id[:8],
                )

        try:
            response_data = await self._await_response(entry)
        except asyncio.TimeoutError:
            response_data = resolve_timeout(
                reason=("experiment_review_timeout"
                        if self._is_experiment_review(request)
                        else "review_window_elapsed")
            ).model_dump(mode="json")
            # Wake any coalesced invocation with the same fail-closed result.
            if not future.done():
                future.set_result(response_data)
            timeout_event = {
                "type": "hitl_timeout",
                "request_id": request_id,
                "agent_name": request.agent_name,
                "timeout_seconds": timeout_sec,
                # Always true now: nothing is approved by silence, so every
                # expiry leaves the work where it was. In transcripts recorded
                # before this, `false` is the signature of an auto-approval.
                "paused": True,
                "timestamp": datetime.now().isoformat(),
            }
            self._record(session_key, timeout_event)
            await self._broadcast(timeout_event, session_key)
        except asyncio.CancelledError:
            cancelled_event = {
                "type": "hitl_cancelled",
                "request_id": request_id,
                "agent_name": request.agent_name,
                "timestamp": datetime.now().isoformat(),
            }
            self._record(session_key, cancelled_event)
            await self._broadcast(cancelled_event, session_key)
            raise
        finally:
            # ``shield`` deliberately keeps the response future alive when the
            # caller is cancelled. Remove and cancel it explicitly so a stopped
            # run cannot be redelivered as a stale request on reconnect.
            if self._pending.get(request_id) is entry:
                self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

        return self._response(response_data)

    @staticmethod
    async def _await_response(entry: dict) -> dict:
        """Wait for the human's answer until the entry's (movable) deadline.

        ``asyncio.wait`` never cancels the future it watches, so — like the
        ``shield`` it replaces — a cancelled caller leaves the response future
        to the ``finally`` in ``handle_request``. The deadline is re-read on
        every wake-up: a hold clears it while we sleep, and the next wake-up
        then waits for the human with no limit.
        """
        future = entry["future"]
        loop = asyncio.get_running_loop()
        while True:
            deadline = entry.get("deadline")
            if deadline is None:
                return await asyncio.shield(future)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait({future}, timeout=remaining)
            if future.done():
                return future.result()

    async def hold_request(
        self,
        request_id: str,
        session_key: SessionKey | None = None,
    ) -> bool:
        """Stop a request's auto-approve countdown ("Pause" on a veto window)."""
        entry = self._pending.get(request_id)
        if not entry or entry["future"].done():
            return False
        if session_key is not None and entry.get("session_key") not in (None, session_key):
            logger.warning("Ignoring HITL hold %s from wrong session", request_id[:8])
            return False
        entry["deadline"] = None
        # A tab that reconnects must not restart a countdown that was paused.
        entry["payload"]["timeout_seconds"] = 0
        entry["payload"]["held"] = True
        await self._broadcast(
            {"type": "hitl_hold", "request_id": request_id},
            entry.get("session_key"),
        )
        logger.info("HITL request %s put on hold", request_id[:8])
        return True

    async def notify(self, payload: dict) -> None:
        """Deliver a non-blocking Work Order notice to the owning session."""
        body = dict(payload)
        session = body.pop("_session", None)
        session_key = None
        if isinstance(session, dict) and session.get("user_id") and session.get("session_id"):
            session_key = (str(session["user_id"]), str(session["session_id"]))
        message = {"type": "work_order_notice", "timestamp": time.time(), **body}
        self._record(session_key, message)
        log_payload = dict(message)
        log_payload["_session_key"] = session_key
        self._event_log.append(log_payload)
        await self._broadcast(message, session_key)

    def resolve_request(
        self,
        request_id: str,
        response_data: dict,
        session_key: SessionKey | None = None,
    ) -> bool:
        """Resolve a request, rejecting responses from a different session."""
        entry = self._pending.get(request_id)
        if entry and session_key is not None and entry.get("session_key") not in (None, session_key):
            logger.warning("Ignoring HITL response %s from wrong session", request_id[:8])
            return False
        entry = self._pending.get(request_id)
        if entry and not entry["future"].done():
            entry["future"].set_result(response_data)
            self._record(entry.get("session_key"), hitl_response_event(request_id, response_data))
            return True
        return False

    def reset(self, session_key: SessionKey | None = None) -> None:
        """Cancel unresolved requests globally or only for one session."""
        for request_id, entry in list(self._pending.items()):
            if session_key is not None and entry.get("session_key") != session_key:
                continue
            if not entry["future"].done():
                entry["future"].cancel()
            self._pending.pop(request_id, None)
        self.clear_event_log(session_key)

    def get_event_log(self, session_key: SessionKey | None = None) -> list[dict]:
        events = self._event_log
        if session_key is not None:
            events = [event for event in events if event.get("_session_key") == session_key]
        return [
            {key: value for key, value in event.items() if key != "_session_key"}
            for event in events
        ]

    def clear_event_log(self, session_key: SessionKey | None = None) -> None:
        if session_key is None:
            self._event_log.clear()
        else:
            self._event_log = [
                event for event in self._event_log
                if event.get("_session_key") != session_key
            ]
