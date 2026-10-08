"""Session-aware Human-in-the-Loop handler for the local Web UI."""

import asyncio
import logging
import time
import uuid

from CoScientist.hitl.handler import AbstractHITLHandler
from CoScientist.hitl.models import HITLAction, HITLRequest, HITLResponse

logger = logging.getLogger("CoScientist.web.hitl")
SessionKey = tuple[str, str]


class WebHITLHandler(AbstractHITLHandler):
    """Route HITL requests only to tabs displaying the owning session."""

    def __init__(self):
        # request_id -> future/payload/session metadata
        self._pending: dict[str, dict] = {}
        # (user_id, session_id) -> connected browser sockets. ``None`` is kept
        # solely for backwards-compatible callers that have no session context.
        self._sockets: dict[SessionKey | None, list] = {}
        self._event_log: list[dict] = []
        self._sender = None

    @property
    def hitl_timeout_seconds(self) -> float:
        if hasattr(self, "_hitl_timeout_seconds"):
            return self._hitl_timeout_seconds
        try:
            from CoScientist.config import get_settings
            return get_settings().web.hitl_auto_approve_timeout
        except Exception:
            return 300

    @hitl_timeout_seconds.setter
    def hitl_timeout_seconds(self, value: float) -> None:
        self._hitl_timeout_seconds = float(value)

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

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        request_id = str(uuid.uuid4())
        session_key = self._request_session_key(request)
        public_context = dict(request.context or {})
        public_context.pop("_session", None)

        timeout_sec = self.hitl_timeout_seconds
        payload = {
            "type": "hitl_request",
            "request_id": request_id,
            "agent_name": request.agent_name,
            "action_type": request.action_type.value,
            "message": request.message,
            "options": request.options,
            "context": public_context,
            "form": request.form,
            "invoked_via": request.invoked_via,
            "timeout_seconds": timeout_sec,
        }

        log_payload = dict(payload)
        log_payload["_session_key"] = session_key
        self._event_log.append(log_payload)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[request_id] = {
            "future": future,
            "payload": payload,
            "created": time.time(),
            "session_key": session_key,
        }

        delivered = await self._broadcast(payload, session_key)
        if delivered:
            logger.info("HITL request %s sent to %d tab(s)", request_id[:8], delivered)
        else:
            if timeout_sec > 0:
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
            if timeout_sec > 0:
                response_data = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=timeout_sec,
                )
            else:
                response_data = await asyncio.shield(future)
        except asyncio.TimeoutError:
            response_data = {"action": "approve", "approved": True}
            await self._broadcast(
                {
                    "type": "hitl_timeout",
                    "request_id": request_id,
                    "agent_name": request.agent_name,
                    "timeout_seconds": timeout_sec,
                },
                session_key,
            )
        except asyncio.CancelledError:
            await self._broadcast(
                {
                    "type": "hitl_cancelled",
                    "request_id": request_id,
                    "agent_name": request.agent_name,
                },
                session_key,
            )
            raise
        finally:
            # ``shield`` deliberately keeps the response future alive when the
            # caller is cancelled. Remove and cancel it explicitly so a stopped
            # run cannot be redelivered as a stale request on reconnect.
            self._pending.pop(request_id, None)
            if not future.done():
                future.cancel()

        action = HITLAction(response_data.get("action", "approve"))
        return HITLResponse(
            action=action,
            approved=response_data.get("approved", False),
            selected_option=response_data.get("selected_option"),
            instructions=response_data.get("instructions"),
            free_input=response_data.get("free_input"),
            form_values=response_data.get("form_values"),
        )

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
        entry = self._pending.pop(request_id, None)
        if entry and not entry["future"].done():
            entry["future"].set_result(response_data)
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
