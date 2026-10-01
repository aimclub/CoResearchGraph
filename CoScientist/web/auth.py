"""Shared-password gate for the web UI.

The deployment runs on the public internet. Every route — the 40-odd REST
endpoints, the ``/ws`` socket that starts agent runs and answers HITL prompts,
and the mounted ``/alembic`` sub-app that clones and builds arbitrary git repos
— used to answer anyone who asked. This module closes all of them at once.

The design is deliberately small:

* One password for the whole deployment. The ``.env`` file holds a salted
  PBKDF2 digest of it in ``AUTH__PASSWORD_HASH``, never the password itself,
  so a leaked file yields nothing a reader can type into the login form. No
  user records, no per-user isolation. Everyone who logs in sees everything.
* A session cookie carrying nothing but its own expiry, signed with HMAC-SHA256.
  There is no user data to leak and no payload to tamper with beyond the
  deadline, which the MAC covers.
* One deny-by-default ASGI middleware instead of per-route dependencies. The app
  registers 58+ routes across two apps, two of which are already shadowed dead
  code, and mounts a sub-app; a guard that has to be remembered per route is a
  guard that will be forgotten.

No new dependency: the deploy runs ``uv sync --frozen``, so a library would mean
regenerating the lock in the deploy path. ``hmac`` + ``hashlib`` is enough here
and mirrors the one auth check the repo already has
(``integrations/codesynapse/control_api.py``).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import logging
import secrets
import sys
import time
from typing import Iterable, Optional

from CoScientist.config import get_settings

_logger = logging.getLogger("CoScientist.web.auth")

COOKIE_NAME = "cs_session"

#: Paths served without a session. Matched exactly, never by prefix — a prefix
#: match on "/login" would also open "/login/../api/users" style surprises.
EXEMPT_PATHS = frozenset({"/healthz", "/login", "/auth/login"})

#: Close code for a socket refused for want of a session. 1008 is "policy
#: violation"; the frontend maps it to a redirect instead of its usual
#: reconnect loop.
WS_CLOSE_POLICY = 1008

#: Close code for a socket refused because of its Origin. Kept apart from
#: WS_CLOSE_POLICY on purpose: the caller here is already logged in, so sending
#: it to /login produces a loop that no correct password can escape.
#:
#: Neither code reaches a browser on the shipped server. uvicorn turns a close
#: sent before the accept into HTTP 403 on the upgrade, and the browser then
#: reports 1006 — measured in Chrome, not inferred. The codes still matter to
#: the test client, which reports them faithfully, and to any ASGI server that
#: closes cleanly. What an operator relies on instead is warn_origin_refused
#: below, which names the refused origin on stderr.
WS_CLOSE_ORIGIN = 4403

# A secret minted once per process, used when AUTH__SECRET_KEY is unset. Every
# restart then invalidates outstanding cookies. That matches the app: the ADK
# session service is in-memory (web/app.py) and a restart already wipes state.
_EPHEMERAL_SECRET = secrets.token_bytes(32)


# ---------------------------------------------------------------------------
# Password and token
# ---------------------------------------------------------------------------
#: Stored-credential format: ``pbkdf2_sha256:<rounds>:<salt>:<digest>``, the
#: last two base64url without padding. Separated by ":" and not the usual "$":
#: a "$" makes the value expand to nothing the moment somebody pastes the line
#: into a shell, and the result still looks like a hash.
_HASH_SCHEME = "pbkdf2_sha256"

#: Work factor. Deliberately below the figure recommended for human-chosen
#: passwords: the README already requires 20 or more random characters, and
#: against that much entropy stretching buys almost nothing. The digest is here
#: so a leaked .env carries no password, not to survive an offline attack on a
#: weak one.
_HASH_ROUNDS = 200_000


def hash_password(password: str, *, rounds: int = _HASH_ROUNDS,
                  salt: Optional[bytes] = None) -> str:
    """Build the value an operator puts in ``AUTH__PASSWORD_HASH``."""
    salt = salt if salt is not None else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return ":".join((_HASH_SCHEME, str(rounds), _b64(salt), _b64(digest)))


def _parse_hash(stored: str) -> Optional[tuple[int, bytes, bytes]]:
    """Split a stored digest, or return None when it is not one.

    Parsing is lazy, never a pydantic validator. ``Settings()`` runs at module
    import, so a typo checked there would stop the server before any code can
    report why. A None here makes ``is_configured`` false, which routes the
    problem through the 503 banner the operator can actually read.
    """
    parts = stored.strip().split(":")
    if len(parts) != 4 or parts[0] != _HASH_SCHEME:
        return None
    try:
        rounds = int(parts[1])
        salt = _unb64(parts[2])
        digest = _unb64(parts[3])
    except (ValueError, binascii.Error):
        return None
    if rounds < 1 or not salt or not digest:
        return None
    return rounds, salt, digest


def _stored_hash() -> str:
    return (get_settings().auth.password_hash or "").strip()


def _secret() -> bytes:
    """The HMAC key for session cookies, bound to the current credential.

    The token payload is only an expiry, so nothing in it changes when the
    password changes. Mixing a digest of the credential into the key is what
    makes a rotation revoke outstanding cookies: after the change, every
    cookie signed with the old key fails its MAC check. Without this, an
    operator who rotates a leaked password keeps the thief logged in for up to
    ``session_max_age``.

    What goes into the mix is the stored string, not the password. For a
    digest that is the whole ``AUTH__PASSWORD_HASH`` value, which costs one
    SHA-256 per request — the PBKDF2 rounds run at login only. It also means a
    fresh salt over the same password still logs everybody out, because the
    stored string changed.
    """
    auth = get_settings().auth
    configured = (auth.secret_key or "").strip()
    base = configured.encode("utf-8") if configured else _EPHEMERAL_SECRET
    material = _stored_hash() or (auth.password or "").strip()
    credential_digest = hashlib.sha256(material.encode("utf-8")).digest()
    return hmac.new(base, b"cs-session-v1:" + credential_digest, hashlib.sha256).digest()


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def is_configured() -> bool:
    """True when the gate can actually open.

    A malformed ``AUTH__PASSWORD_HASH`` counts as unconfigured. Falling back
    to ``AUTH__PASSWORD`` instead would turn a typo in the digest into a
    quietly weaker gate, and refusing to start would hide the reason.
    """
    auth = get_settings().auth
    stored = _stored_hash()
    if stored:
        return _parse_hash(stored) is not None
    return bool((auth.password or "").strip())


def check_password(given: str) -> bool:
    """Constant-time check against the stored credential.

    ``AUTH__PASSWORD_HASH`` wins when both are set. Both branches compare
    fixed-length digests rather than raw strings, so neither leaks the
    password's length through timing.

    This runs the PBKDF2 rounds, so it blocks for a measurable time. Callers
    on the event loop must hand it to a thread — see ``login_submit``.
    """
    stored = _stored_hash()
    if stored:
        parsed = _parse_hash(stored)
        if parsed is None:
            return False
        rounds, salt, expected_digest = parsed
        given_digest = hashlib.pbkdf2_hmac("sha256", given.encode("utf-8"), salt, rounds)
        return hmac.compare_digest(given_digest, expected_digest)

    expected = (get_settings().auth.password or "").strip()
    if not expected:
        return False
    return hmac.compare_digest(
        hashlib.sha256(given.encode("utf-8")).digest(),
        hashlib.sha256(expected.encode("utf-8")).digest(),
    )


def issue_token(now: Optional[float] = None) -> str:
    """Mint a cookie value that expires ``session_max_age`` seconds from now."""
    auth = get_settings().auth
    expiry = int((now if now is not None else time.time()) + auth.session_max_age)
    payload = str(expiry).encode("ascii")
    mac = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return f"{_b64(payload)}.{_b64(mac)}"


def verify_token(token: Optional[str], now: Optional[float] = None) -> bool:
    """True when *token* carries an intact signature and has not expired."""
    if not token or "." not in token:
        return False
    encoded_payload, _, encoded_mac = token.partition(".")
    try:
        payload = _unb64(encoded_payload)
        mac = _unb64(encoded_mac)
    except Exception:  # noqa: BLE001 — any malformed cookie is simply invalid
        return False

    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        return False

    # Only past the MAC check is the payload trustworthy enough to parse.
    try:
        expiry = int(payload.decode("ascii"))
    except ValueError:
        return False
    return expiry > (now if now is not None else time.time())


# ---------------------------------------------------------------------------
# Login rate limit
# ---------------------------------------------------------------------------
class LoginRateLimiter:
    """Fixed-window failed-login counter, keyed by client IP.

    One shared password on the public internet makes online guessing the main
    residual risk. In-memory and per-process, which is enough for a single
    uvicorn worker — it is a brake on guessing, not an audit log.
    """

    def __init__(self) -> None:
        self._windows: dict[str, tuple[float, int]] = {}

    def _prune(self, now: float, window: int) -> None:
        stale = [ip for ip, (start, _) in self._windows.items() if now - start >= window]
        for ip in stale:
            del self._windows[ip]

    def is_limited(self, ip: str, now: Optional[float] = None) -> bool:
        auth = get_settings().auth
        now = now if now is not None else time.time()
        self._prune(now, auth.login_window_seconds)
        start, count = self._windows.get(ip, (now, 0))
        if now - start >= auth.login_window_seconds:
            return False
        return count >= auth.max_login_attempts

    def record_failure(self, ip: str, now: Optional[float] = None) -> None:
        auth = get_settings().auth
        now = now if now is not None else time.time()
        start, count = self._windows.get(ip, (now, 0))
        if now - start >= auth.login_window_seconds:
            start, count = now, 0
        self._windows[ip] = (start, count + 1)

    def reset(self, ip: str) -> None:
        self._windows.pop(ip, None)


#: Process-wide limiter. The middleware does not use it; the login route does.
login_limiter = LoginRateLimiter()


def client_ip(scope: dict) -> str:
    """Best-effort client address for rate limiting.

    Reads ``scope["client"]`` only. When uvicorn runs with ``--proxy-headers``
    and a trusted ``forwarded_allow_ips`` it has already rewritten this from
    ``X-Forwarded-For``. Parsing that header here instead would let any caller
    rotate it freely and walk straight past the limit.
    """
    client = scope.get("client")
    return client[0] if client else "unknown"


# ---------------------------------------------------------------------------
# Origin checking
# ---------------------------------------------------------------------------
def _header(scope: dict, name: bytes) -> Optional[str]:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def origin_allowed(scope: dict) -> bool:
    """True when a WebSocket handshake comes from an approved origin.

    ``SameSite=Lax`` does not reliably cover a WebSocket handshake, so without
    this any page on any site could open ``/ws`` in a logged-in user's browser
    and send ``chat_message`` or ``hitl_response``. An absent Origin is refused
    too: browsers always send one, and a non-browser client has no business on
    this socket.
    """
    origin = _header(scope, b"origin")
    if not origin:
        return False
    origin = origin.strip().rstrip("/").lower()

    allowed = get_settings().auth.origin_list
    if allowed:
        return origin in allowed

    # Nothing configured: fall back to same origin rather than refusing every
    # socket. A page on another site sends its own Origin, which cannot match
    # this server's Host, so the cross-site handshake this check exists to stop
    # is still refused. The session cookie is still checked after this.
    #
    # The comparison is authority against authority, and scheme-blind on
    # purpose: Origin carries a scheme and Host does not, so behind a
    # TLS-terminating proxy the browser sends "https://host" while Host is
    # bare "host". Comparing full origins instead would refuse every socket on
    # exactly the TLS deployments this fallback exists to serve. The cost is
    # that http://host also matches an https deployment, which needs control
    # of the hostname to reach anyway.
    host = (_header(scope, b"host") or "").strip().lower()
    if not host:
        return False
    return origin.partition("://")[2] == host


#: Origins already reported to stderr, so a refusal loop cannot flood the
#: journal. Capped, because the Origin header comes from the caller and an
#: attacker could otherwise send an unbounded number of distinct values.
_reported_origins: set[str] = set()
_REPORTED_ORIGINS_CAP = 20


def warn_origin_refused(scope: dict) -> None:
    """Name the refused Origin where the operator will find it.

    Without this the failure is silent and undiagnosable: uvicorn answers a
    refused handshake with HTTP 403, which reaches the browser as an ordinary
    abnormal close, so the UI reports "Disconnected" and retries for ever with
    no reason given.

    The message goes to stderr as well as the log, and once per origin. The
    ``CoScientist`` logger sets ``propagate = False`` and writes to app.log
    only, so a plain warning never reaches journalctl — which is where somebody
    debugging a service looks first.
    """
    origin = _header(scope, b"origin")
    allowed = get_settings().auth.allowed_origins
    message = (
        f"WebSocket refused: origin {origin!r} is not allowed. Set "
        f"AUTH__ALLOWED_ORIGINS to the origin browsers use (currently {allowed!r})."
    )
    _logger.warning(message)

    key = origin or ""
    if key not in _reported_origins and len(_reported_origins) < _REPORTED_ORIGINS_CAP:
        _reported_origins.add(key)
        print(f"[CoScientist Web] {message}", file=sys.stderr, flush=True)


def cookie_from_scope(scope: dict) -> Optional[str]:
    raw = _header(scope, b"cookie")
    if not raw:
        return None
    for part in raw.split(";"):
        name, _, value = part.strip().partition("=")
        if name == COOKIE_NAME:
            return value
    return None


# ---------------------------------------------------------------------------
# Telling the operator what went wrong
# ---------------------------------------------------------------------------
_MISCONFIGURED_BANNER = (
    "AUTH__PASSWORD_HASH is missing or malformed. The web UI answers 503 to "
    "every request, including /healthz, so the deploy health check will fail. "
    "Run 'python3 deploy/make_password_hash.py' and put the line it prints in "
    "the .env file, or set AUTH__ENABLED=false to serve without a gate."
)

_PLAINTEXT_BANNER = (
    "AUTH__PASSWORD holds the password in clear text. Anyone who reads the "
    ".env file can sign in. Run 'python3 deploy/make_password_hash.py', put the "
    "result in AUTH__PASSWORD_HASH, and delete AUTH__PASSWORD. The switch "
    "logs everybody out once, because the cookie signing key follows the "
    "stored credential."
)

_warned_unconfigured = False


def warn_unconfigured() -> None:
    """Say it once, somewhere the operator will actually look.

    The ``CoScientist`` logger sets ``propagate = False`` and owns a file
    handler, so a plain ``logger.error`` never reaches journalctl — which is
    exactly where somebody debugging a failed deploy looks, and the deploy
    workflow dumps on failure. stderr goes to the service journal. Once only:
    this runs per request, and a flood would bury the line it is meant to
    surface.
    """
    global _warned_unconfigured
    if not _warned_unconfigured:
        _warned_unconfigured = True
        _logger.error(_MISCONFIGURED_BANNER)
        print(f"[CoScientist Web] {_MISCONFIGURED_BANNER}", file=sys.stderr, flush=True)


def check_configuration() -> None:
    """Startup check, so a misconfigured server says so before the first request."""
    auth = get_settings().auth
    if auth.enabled and not is_configured():
        warn_unconfigured()
        return
    if auth.enabled and not _stored_hash():
        _logger.warning(_PLAINTEXT_BANNER)
        print(f"[CoScientist Web] {_PLAINTEXT_BANNER}", file=sys.stderr, flush=True)
    if auth.enabled and not auth.origin_list:
        message = (
            "AUTH__ALLOWED_ORIGINS is empty. The WebSocket gate falls back to "
            "same-origin, which works when the proxy forwards the Host header "
            "unchanged. Set it to the origin browsers use, e.g. "
            "https://cosci.example.org, to stop depending on that."
        )
        _logger.warning(message)
        print(f"[CoScientist Web] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class RequireAuth:
    """Deny-by-default ASGI middleware for HTTP and WebSocket scopes.

    Raw ASGI on purpose. Starlette's ``BaseHTTPMiddleware`` never sees a
    ``websocket`` scope, so the same logic written that way would leave ``/ws``
    — the endpoint that starts runs and approves HITL prompts — wide open while
    every REST test still passed.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        auth = get_settings().auth
        if not auth.enabled:
            await self.app(scope, receive, send)
            return

        # Strip root_path, so the exemptions still match if the app is ever
        # served under a sub-path behind a proxy. Without this, /login arrives
        # as /cosci/login, misses the exemption, and the login page redirects
        # to itself forever.
        path = scope.get("path", "")
        root = scope.get("root_path") or ""
        if root and path.startswith(root):
            path = path[len(root):] or "/"

        # Fail closed. An unset password must never mean an open server, so
        # this check comes before the exemptions: without a password even the
        # login page has nothing to check against.
        if not is_configured():
            warn_unconfigured()
            await self._deny(
                scope, receive, send,
                status=503,
                detail="Authentication is enabled but AUTH__PASSWORD is not set.",
            )
            return

        if path in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket" and not origin_allowed(scope):
            warn_origin_refused(scope)
            await self._deny(scope, receive, send, status=403, detail="Origin not allowed")
            return

        if not verify_token(cookie_from_scope(scope)):
            await self._deny(scope, receive, send, status=401, detail="Not authenticated")
            return

        await self.app(scope, receive, send)

    async def _deny(self, scope, receive, send, *, status: int, detail: str) -> None:
        if scope["type"] == "websocket":
            # Consume the handshake before refusing it, as ASGI expects, then
            # close without ever accepting. The endpoint is never reached.
            #
            # What the client sees depends on the server. uvicorn turns a
            # pre-accept close into an outright HTTP 403 on the upgrade, so a
            # browser gets an abnormal close (1006), not this code. Starlette's
            # TestClient reports the code faithfully, which is what the tests
            # assert. The frontend handles both — see ws.js. Do not "fix" this
            # to accept-then-close: refusing the handshake is the stronger
            # denial, and the code was never the contract.
            try:
                await receive()
            except Exception:  # noqa: BLE001 — the client may already be gone
                pass
            code = WS_CLOSE_ORIGIN if status == 403 else WS_CLOSE_POLICY
            await send({"type": "websocket.close", "code": code})
            return

        # A browser navigation should land on the login form; an API call
        # should get a status its caller can act on.
        if status == 401 and _wants_html(scope):
            # Prefix root_path, so the redirect still points at the app when a
            # proxy mounts it under a sub-path. Note that this is the only
            # root_path-aware URL in the feature: the form actions in
            # login.html and index.html stay root-absolute, so a sub-path
            # deployment needs those changed too. Serving at the root, which
            # is what deploy/ does, is the supported shape.
            root = (scope.get("root_path") or "").rstrip("/")
            location = f"{root}/login".encode("latin-1")
            await _send_response(
                send, 302, b"", extra_headers=[(b"location", location)]
            )
            return

        body = json.dumps({"detail": detail}).encode("utf-8")
        await _send_response(send, status, body, content_type=b"application/json")


def _wants_html(scope: dict) -> bool:
    accept = (_header(scope, b"accept") or "").lower()
    return "text/html" in accept


async def _send_response(
    send,
    status: int,
    body: bytes,
    *,
    content_type: bytes = b"text/plain; charset=utf-8",
    extra_headers: Optional[Iterable[tuple[bytes, bytes]]] = None,
) -> None:
    headers = [
        (b"content-type", content_type),
        (b"content-length", str(len(body)).encode("ascii")),
        (b"cache-control", b"no-store"),
    ]
    headers.extend(extra_headers or ())
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


# ---------------------------------------------------------------------------
# Cookie helpers for the login/logout routes
# ---------------------------------------------------------------------------
def request_is_https(scope: dict) -> bool:
    """True when the browser reached this server over HTTPS.

    Reads ``X-Forwarded-Proto`` first. uvicorn rewrites ``scope["scheme"]``
    only with ``--proxy-headers``, which the shipped unit does not pass, so
    behind a TLS proxy the scheme says "http" even though the browser used
    HTTPS. A forged header only wins the sender a Secure cookie on its own
    session, which then fails over plain HTTP, so there is nothing to gain by
    lying and no trusted-proxy list is needed.
    """
    forwarded = (_header(scope, b"x-forwarded-proto") or "").split(",")[0].strip().lower()
    if forwarded:
        return forwarded == "https"
    return (scope.get("scheme") or "").lower() in ("https", "wss")


def use_secure_cookie(scope: dict) -> bool:
    """Whether to mark the session cookie ``Secure``.

    ``AUTH__COOKIE_SECURE`` pins the answer when it is set. Unset means decide
    per request, which is the only setting that is right for both deployments:
    a browser never sends a Secure cookie back over plain HTTP, so pinning it
    true without TLS produces a silent login loop — the password is accepted,
    the cookie is dropped, and ``/`` redirects to ``/login`` again.
    """
    configured = get_settings().auth.cookie_secure
    if configured is not None:
        return configured
    return request_is_https(scope)


def cookie_kwargs(scope: dict) -> dict:
    """Keyword arguments for ``Response.set_cookie``."""
    auth = get_settings().auth
    return {
        "key": COOKIE_NAME,
        "httponly": True,
        "samesite": "lax",
        "secure": use_secure_cookie(scope),
        "path": "/",
        "max_age": auth.session_max_age,
    }
