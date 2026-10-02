"""The web UI must answer nobody without a session.

The deployment binds 0.0.0.0 on the public internet, and before this gate every
route answered anyone: the REST API, the ``/ws`` socket that starts agent runs
and approves HITL prompts, and the mounted ``/alembic`` sub-app that clones and
builds arbitrary git repositories.

The load-bearing test here is ``test_every_route_is_closed``. It walks the
router rather than listing paths, so a route added later — or one of the two
already-shadowed duplicate registrations — cannot quietly arrive unprotected.
"""
import time

import pytest
from fastapi.routing import APIRoute, APIWebSocketRoute
from fastapi.testclient import TestClient
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.websockets import WebSocketDisconnect

from CoScientist.config import get_settings
from CoScientist.web import auth as web_auth
from CoScientist.web.app import create_app

PASSWORD = "correct-horse-battery-staple"
#: Rounds for test digests. The shipped value is 200000, which is the point of
#: it — far too slow to run in a fixture that every test here uses.
TEST_ROUNDS = 1000
ORIGIN = "https://cosci.example.org"


@pytest.fixture(scope="module")
def app():
    """One app for the module — create_app() wires the whole agent system."""
    return create_app()


@pytest.fixture(autouse=True)
def _gate_open_for_tests(monkeypatch):
    """A configured gate, and a clean rate limiter, for every test."""
    auth = get_settings().auth
    monkeypatch.setattr(auth, "enabled", True)
    # The digest is the path a deployment runs, so it is the path these tests
    # exercise. Low rounds on purpose: the throttle tests below make about 45
    # login calls, and 200000 rounds each would dominate the run.
    # test_the_digest_rounds_are_read_from_the_stored_line pins that the count
    # comes from the stored line, which is what makes this safe.
    monkeypatch.setattr(
        auth, "password_hash", web_auth.hash_password(PASSWORD, rounds=TEST_ROUNDS)
    )
    monkeypatch.setattr(auth, "secret_key", "test-secret")
    monkeypatch.setattr(auth, "cookie_secure", False)
    monkeypatch.setattr(auth, "allowed_origins", ORIGIN)
    web_auth.login_limiter = web_auth.LoginRateLimiter()
    web_auth._reported_origins.clear()


@pytest.fixture
def client(app):
    return TestClient(app)


def refused_code(client, url="/ws?user_id=x&session_id=y", **headers) -> int:
    """Open *url*, expect the gate to refuse it, and return the close code.

    Every socket test needs the same two nested context managers to get at one
    integer, which buried the assertion. Note the code only reaches a caller
    that speaks ASGI directly, as TestClient does — see WS_CLOSE_ORIGIN in
    web/auth.py for what a browser sees instead.
    """
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect(url, headers=dict(headers)):
            pass
    return excinfo.value.code


@pytest.fixture
def session_cookie():
    return {web_auth.COOKIE_NAME: web_auth.issue_token()}


# ---------------------------------------------------------------------------
# The shipped example file must actually load
# ---------------------------------------------------------------------------
def test_the_example_env_file_parses(monkeypatch):
    """Settings() runs at import, so a bad AUTH line stops the server dead.

    It stops before the fail-closed banner can say why, which is worse than
    anything the gate itself does. An empty value is the trap: it is fine for
    a string field, and a ValidationError for a bool.
    """
    from pathlib import Path as _Path

    from CoScientist.config.settings import Settings

    example = _Path(__file__).resolve().parents[2] / "CoScientist/examples/example_config.env"
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("AUTH__"):
            continue
        key, _, value = line.partition("=")
        monkeypatch.setenv(key, value)

    assert Settings().auth is not None


def test_every_auth_setting_is_documented():
    """Three descriptions of one model drift, and two of them already had.

    AUTH__LOGIN_WINDOW_SECONDS was in neither document, so an operator could
    not discover it. The README said "15 minutes" in prose instead of naming the
    key. A field nobody can find is a field nobody can set.
    """
    from pathlib import Path as _Path

    from CoScientist.config.settings import AuthSettings

    root = _Path(__file__).resolve().parents[2]
    readme = (root / "deploy/README.md").read_text(encoding="utf-8")
    example = (root / "CoScientist/examples/example_config.env").read_text(encoding="utf-8")
    for field in AuthSettings.model_fields:
        key = f"AUTH__{field.upper()}"
        assert key in readme, f"{key} is missing from deploy/README.md"
        assert key in example, f"{key} is missing from example_config.env"


# ---------------------------------------------------------------------------
# Deny by default
# ---------------------------------------------------------------------------
def _walk(routes, prefix=""):
    """Yield (path, methods) for every route, descending into mounted apps."""
    for route in routes:
        if isinstance(route, Mount):
            sub = getattr(route.app, "routes", None)
            if sub:
                yield from _walk(sub, prefix + route.path)
            else:  # a StaticFiles mount has no routes of its own
                yield prefix + route.path + "/probe.txt", {"GET"}
        elif isinstance(route, (APIWebSocketRoute, WebSocketRoute)):
            yield prefix + route.path, {"WEBSOCKET"}
        elif isinstance(route, (APIRoute, Route)):
            yield prefix + route.path, set(route.methods or {"GET"}) - {"HEAD", "OPTIONS"}


def _concrete(path: str) -> str:
    """Fill path parameters so the URL routes to something real."""
    out = []
    for segment in path.split("/"):
        if segment.startswith("{") and segment.endswith("}"):
            out.append("probe")
        else:
            out.append(segment)
    return "/".join(out)


def test_route_walk_finds_the_dangerous_surface(app):
    """Guard the guard: if the walk stops seeing routes, the sweep below passes vacuously."""
    paths = {path for path, _ in _walk(app.routes)}
    assert "/ws" in paths
    assert "/api/users" in paths
    assert "/api/settings" in paths
    # The mounted sub-app must be reached through the Mount, not listed by hand.
    assert any(p.startswith("/alembic/") for p in paths), sorted(paths)
    assert len(paths) > 40, sorted(paths)


def test_every_route_is_closed(app, client):
    """No cookie, no answer — for every route the router actually has."""
    escaped = []
    for path, methods in _walk(app.routes):
        if path in web_auth.EXEMPT_PATHS:
            continue
        url = _concrete(path)
        for method in sorted(methods):
            if method == "WEBSOCKET":
                with pytest.raises(WebSocketDisconnect) as excinfo:
                    with client.websocket_connect(url):
                        pass
                refusals = (web_auth.WS_CLOSE_POLICY, web_auth.WS_CLOSE_ORIGIN)
                if excinfo.value.code not in refusals:
                    escaped.append((method, url, excinfo.value.code))
                continue
            response = client.request(method, url, follow_redirects=False)
            # Only 302 and 401 count as "the gate refused". 403 is excluded on
            # purpose: alembic's _require_controls() returns 403 for its own
            # feature flag, and accepting it here would let an ungated route
            # pass the sweep.
            if response.status_code not in (302, 401):
                escaped.append((method, url, response.status_code))
    assert not escaped, f"routes answered without a session: {escaped}"


@pytest.mark.parametrize("path", ["/", "/docs", "/openapi.json", "/static/js/state.js"])
def test_the_self_documenting_surface_is_closed(client, path):
    """/docs and /openapi.json hand over a map of everything else."""
    assert client.get(path, follow_redirects=False).status_code in (302, 401)


def test_alembic_build_endpoint_is_closed(client):
    """POST /alembic/api/builds clones and docker-builds an arbitrary git URL."""
    response = client.post(
        "/alembic/api/builds",
        json={"repo_url": "https://github.com/attacker/payload"},
        follow_redirects=False,
    )
    assert response.status_code == 401


def test_a_browser_navigation_is_redirected_to_the_login_page(client):
    response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"] == "/login"


def test_an_api_call_gets_401_not_a_redirect(client):
    response = client.get("/api/users", follow_redirects=False)
    assert response.status_code == 401
    assert response.json()["detail"]


def test_healthz_answers_without_a_session(client):
    """The deploy workflow polls this; auth on it would break every deploy."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_the_login_page_is_reachable_without_a_session(client):
    assert client.get("/login").status_code == 200


# ---------------------------------------------------------------------------
# The WebSocket — where BaseHTTPMiddleware would have silently failed
# ---------------------------------------------------------------------------
def test_the_socket_is_refused_without_a_session(client):
    """An allowed origin, so the missing cookie is what refuses this one."""
    assert refused_code(client, origin=ORIGIN) == web_auth.WS_CLOSE_POLICY


def test_the_socket_is_refused_from_a_foreign_origin(client, session_cookie):
    """A valid cookie is not enough: SameSite=Lax does not cover a handshake.

    The code differs from the no-session refusal on purpose. A logged-in caller
    refused for their Origin must not be sent to /login: the cookie is still
    valid, so /login bounces them back and the loop never ends.
    """
    client.cookies.update(session_cookie)
    code = refused_code(client, origin="https://evil.example")
    assert code == web_auth.WS_CLOSE_ORIGIN
    assert code != web_auth.WS_CLOSE_POLICY


def test_the_socket_is_refused_when_the_origin_header_is_absent(client, session_cookie):
    client.cookies.update(session_cookie)
    assert refused_code(client) == web_auth.WS_CLOSE_ORIGIN


def test_an_origin_refusal_is_reported_once_per_origin(client, session_cookie, capsys):
    """The reason has to reach stderr, and must not flood it.

    The CoScientist logger sets propagate = False, so a plain warning lands in
    app.log and never in journalctl. The Origin header is caller-controlled, so
    reporting every refusal would let anyone bury the journal.
    """
    client.cookies.update(session_cookie)
    for _ in range(3):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://evil.example"}):
                pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws", headers={"origin": "https://other.example"}):
            pass

    reported = [
        line for line in capsys.readouterr().err.splitlines()
        if "WebSocket refused" in line
    ]
    assert len(reported) == 2, reported
    assert "AUTH__ALLOWED_ORIGINS" in reported[0]
    assert "evil.example" in reported[0]


def test_the_origin_report_cannot_be_used_to_flood(client, session_cookie, capsys):
    """A caller sending endless distinct origins must not fill the journal."""
    client.cookies.update(session_cookie)
    for i in range(web_auth._REPORTED_ORIGINS_CAP + 10):
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": f"https://s{i}.example"}):
                pass

    reported = [
        line for line in capsys.readouterr().err.splitlines()
        if "WebSocket refused" in line
    ]
    assert len(reported) == web_auth._REPORTED_ORIGINS_CAP


def test_the_socket_opens_with_a_cookie_and_an_allowed_origin(client, session_cookie):
    """Past the gate the socket reaches the endpoint, which rejects the unknown user."""
    client.cookies.update(session_cookie)
    with client.websocket_connect(
        "/ws?user_id=nobody&session_id=nothing",
        headers={"origin": ORIGIN},
    ) as socket:
        message = socket.receive_json()
    # The endpoint's own "unknown user or session" answer, which only the
    # endpoint can produce — so the gate let the handshake through.
    assert message["type"] == "error"


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def test_the_right_password_mints_a_usable_session(client):
    response = client.post("/auth/login", data={"password": PASSWORD}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert web_auth.COOKIE_NAME in response.cookies

    client.cookies.update({web_auth.COOKIE_NAME: response.cookies[web_auth.COOKIE_NAME]})
    assert client.get("/api/users").status_code == 200


def test_the_wrong_password_mints_nothing(client):
    response = client.post("/auth/login", data={"password": "wrong"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login?error=bad"
    assert web_auth.COOKIE_NAME not in response.cookies


def test_the_cookie_carries_the_protective_flags(client):
    response = client.post("/auth/login", data={"password": PASSWORD}, follow_redirects=False)
    header = response.headers["set-cookie"].lower()
    assert "httponly" in header
    assert "samesite=lax" in header
    assert "path=/" in header


def test_changing_the_credential_revokes_outstanding_cookies(client, monkeypatch):
    """Rotation after a leak must actually lock the thief out.

    The token carries only an expiry, so nothing in it changes with the
    password. The signing key mixes the stored credential in for exactly this
    reason.
    """
    stolen = web_auth.issue_token()
    assert web_auth.verify_token(stolen)

    monkeypatch.setattr(
        get_settings().auth, "password_hash",
        web_auth.hash_password(PASSWORD + "-rotated", rounds=TEST_ROUNDS),
    )
    assert not web_auth.verify_token(stolen)

    client.cookies.update({web_auth.COOKIE_NAME: stolen})
    assert client.get("/api/users", follow_redirects=False).status_code == 401


@pytest.mark.parametrize("configured,forwarded,expected", [
    # Pinned true demands HTTPS whatever the request looks like.
    (True, None, True),
    # Unset means decide per request. uvicorn rewrites the scope scheme only
    # with --proxy-headers, which the shipped unit does not pass, so the
    # forwarded header is the only signal available.
    (None, "https", True),
    # A Secure cookie over plain HTTP is dropped by httpx and by a browser
    # alike, which reads as a login loop. That is why unset does not mean true.
    (None, None, False),
    # Pinned false wins over the forwarded header.
    (False, "https", False),
])
def test_the_secure_flag_follows_the_setting_then_the_scheme(
    client, monkeypatch, configured, forwarded, expected
):
    monkeypatch.setattr(get_settings().auth, "cookie_secure", configured)
    response = client.post(
        "/auth/login",
        data={"password": PASSWORD},
        headers={"x-forwarded-proto": forwarded} if forwarded else {},
        follow_redirects=False,
    )
    assert ("secure" in response.headers["set-cookie"].lower()) is expected


def test_logout_clears_the_session(client):
    client.post("/auth/login", data={"password": PASSWORD}, follow_redirects=False)
    response = client.post("/auth/logout", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    client.cookies.clear()
    assert client.get("/api/users", follow_redirects=False).status_code == 401


def test_repeated_failures_from_one_address_are_throttled(client):
    limit = get_settings().auth.max_login_attempts
    for _ in range(limit):
        client.post("/auth/login", data={"password": "wrong"}, follow_redirects=False)

    response = client.post("/auth/login", data={"password": "wrong"}, follow_redirects=False)
    assert response.headers["location"] == "/login?error=rate"


def test_the_throttle_never_refuses_the_right_password(client):
    """Inverted on purpose: the earlier "the brake holds against the right
    password" behavior was a remote lockout of the whole team.

    uvicorn runs without --proxy-headers, so behind a reverse proxy every
    client shares one counter keyed on the proxy address. Ten wrong guesses
    from anywhere would then refuse the correct password for everybody, for the
    whole window, over and over. A caller who knows the password gets in.
    """
    for _ in range(get_settings().auth.max_login_attempts + 5):
        client.post("/auth/login", data={"password": "wrong"}, follow_redirects=False)

    response = client.post("/auth/login", data={"password": PASSWORD}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert web_auth.COOKIE_NAME in response.cookies


def test_the_throttle_window_expires(monkeypatch):
    limiter = web_auth.LoginRateLimiter()
    auth = get_settings().auth
    now = time.time()
    for _ in range(auth.max_login_attempts):
        limiter.record_failure("1.2.3.4", now=now)
    assert limiter.is_limited("1.2.3.4", now=now)
    assert not limiter.is_limited("1.2.3.4", now=now + auth.login_window_seconds + 1)


def test_a_successful_login_clears_the_counter(client):
    for _ in range(get_settings().auth.max_login_attempts - 1):
        client.post("/auth/login", data={"password": "wrong"}, follow_redirects=False)
    client.post("/auth/login", data={"password": PASSWORD}, follow_redirects=False)

    for _ in range(get_settings().auth.max_login_attempts - 1):
        response = client.post("/auth/login", data={"password": "wrong"}, follow_redirects=False)
    assert response.headers["location"] == "/login?error=bad"


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------
def test_a_valid_token_verifies():
    assert web_auth.verify_token(web_auth.issue_token())


@pytest.mark.parametrize(
    "token",
    ["", "garbage", "no-dot-here", "YWJj.YWJj", "....", "YWJj."],
)
def test_a_malformed_token_is_rejected(token):
    assert not web_auth.verify_token(token)


def test_a_tampered_signature_is_rejected():
    payload, _, mac = web_auth.issue_token().partition(".")
    forged = "A" if mac[0] != "A" else "B"
    assert not web_auth.verify_token(f"{payload}.{forged}{mac[1:]}")


def test_a_tampered_expiry_is_rejected():
    """Extending the deadline requires the key, which is the point of the MAC."""
    _, _, mac = web_auth.issue_token().partition(".")
    far_future = web_auth._b64(str(int(time.time()) + 10**9).encode("ascii"))
    assert not web_auth.verify_token(f"{far_future}.{mac}")


def test_an_expired_token_is_rejected():
    token = web_auth.issue_token(now=time.time() - get_settings().auth.session_max_age - 10)
    assert not web_auth.verify_token(token)


def test_an_expired_cookie_closes_the_door(client):
    stale = web_auth.issue_token(now=time.time() - get_settings().auth.session_max_age - 10)
    client.cookies.update({web_auth.COOKIE_NAME: stale})
    assert client.get("/api/users", follow_redirects=False).status_code == 401


def test_a_token_signed_with_another_key_is_rejected(monkeypatch):
    token = web_auth.issue_token()
    monkeypatch.setattr(get_settings().auth, "secret_key", "a-different-key")
    assert not web_auth.verify_token(token)


def test_password_check_rejects_the_empty_password(monkeypatch):
    """An unset credential must never make an empty submission succeed."""
    monkeypatch.setattr(get_settings().auth, "password_hash", "")
    assert not web_auth.check_password("")
    assert not web_auth.check_password(PASSWORD)


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------
# password_hash is a str, so these are the blank values it can actually hold.
# Whitespace counts as unset: _stored_hash strips before it decides.
@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_an_unset_credential_closes_everything(client, monkeypatch, blank):
    """Misconfiguration must never degrade into an open server."""
    monkeypatch.setattr(get_settings().auth, "password_hash", blank)
    for path in ("/", "/api/users", "/login", "/healthz", "/alembic/api/builds"):
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 503, path


def test_an_unset_credential_also_closes_the_socket(client, monkeypatch, session_cookie):
    monkeypatch.setattr(get_settings().auth, "password_hash", "")
    client.cookies.update(session_cookie)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws", headers={"origin": ORIGIN}):
            pass
    assert excinfo.value.code == web_auth.WS_CLOSE_POLICY


def test_an_empty_allowlist_still_refuses_a_foreign_origin(client, monkeypatch, session_cookie):
    """With nothing configured the check falls back to same-origin.

    ORIGIN is not this server's host, so it is refused exactly as a real
    cross-site page would be.
    """
    monkeypatch.setattr(get_settings().auth, "allowed_origins", "")
    client.cookies.update(session_cookie)
    with pytest.raises(WebSocketDisconnect) as excinfo:
        with client.websocket_connect("/ws", headers={"origin": ORIGIN}):
            pass
    assert excinfo.value.code == web_auth.WS_CLOSE_ORIGIN


def test_an_empty_allowlist_accepts_the_same_origin(client, monkeypatch, session_cookie):
    """Otherwise a correctly configured password still leaves the UI offline."""
    monkeypatch.setattr(get_settings().auth, "allowed_origins", "")
    client.cookies.update(session_cookie)
    with client.websocket_connect(
        "/ws?user_id=nobody&session_id=nothing",
        headers={"origin": "http://testserver"},
    ) as socket:
        message = socket.receive_json()
    assert message["type"] == "error"


def test_the_origin_match_ignores_case(client, monkeypatch, session_cookie):
    """Browsers lowercase the host whatever the operator typed into the .env."""
    monkeypatch.setattr(get_settings().auth, "allowed_origins", "https://CoSci.Example.ORG")
    client.cookies.update(session_cookie)
    with client.websocket_connect(
        "/ws?user_id=nobody&session_id=nothing",
        headers={"origin": ORIGIN},
    ) as socket:
        message = socket.receive_json()
    assert message["type"] == "error"


def test_the_gate_can_be_turned_off_deliberately(client, monkeypatch):
    """AUTH__ENABLED=false is the documented escape hatch, e.g. on localhost."""
    monkeypatch.setattr(get_settings().auth, "enabled", False)
    assert client.get("/healthz").status_code == 200
    assert client.get("/api/users").status_code == 200


def test_exemptions_survive_a_sub_path_deployment(app):
    """Behind a proxy that mounts the app under a prefix, /login is still /login."""
    client = TestClient(app, root_path="/cosci")
    assert client.get("/cosci/healthz").status_code == 200
    assert client.get("/cosci/login").status_code == 200
    assert client.get("/cosci/api/users", follow_redirects=False).status_code == 401


def test_the_sub_path_redirect_points_back_into_the_app(app):
    """A status code alone cannot tell a working redirect from a broken one.

    Without the prefix the browser is sent to /login, which the proxy does not
    route to this app, so the user gets a 404 instead of the login form.
    """
    client = TestClient(app, root_path="/cosci")
    response = client.get(
        "/cosci/", headers={"accept": "text/html"}, follow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == "/cosci/login"


def test_the_root_deployment_redirect_has_no_prefix(client):
    response = client.get("/", headers={"accept": "text/html"}, follow_redirects=False)
    assert response.headers["location"] == "/login"


@pytest.mark.parametrize(
    "error",
    ["<script>alert(1)</script>", "bad'\"><img src=x onerror=alert(1)>", "rate.."],
)
def test_the_login_page_never_echoes_its_query_parameter(client, error):
    """The only caller-controlled input on the unauthenticated surface.

    The handler maps a fixed set of codes to fixed messages, so nothing from
    the URL reaches the HTML. Pinned here so a later refactor to
    ``message = error`` cannot land unnoticed.
    """
    body = client.get("/login", params={"error": error}).text
    assert "<script>" not in body
    assert "onerror" not in body
    assert error not in body


# ---------------------------------------------------------------------------
# The stored credential is a salted digest, not the password
# ---------------------------------------------------------------------------
def test_the_digest_round_trips(monkeypatch):
    """What the generator prints is what check_password accepts."""
    auth = get_settings().auth
    monkeypatch.setattr(auth, "password_hash", web_auth.hash_password(PASSWORD))

    assert web_auth.check_password(PASSWORD)
    assert not web_auth.check_password(PASSWORD + "x")
    assert not web_auth.check_password("")


def test_the_stored_line_never_contains_the_password():
    """The point of the whole change: a reader of .env learns nothing."""
    stored = web_auth.hash_password(PASSWORD)
    assert PASSWORD not in stored
    scheme, rounds, salt, digest = stored.split(":")
    assert scheme == "pbkdf2_sha256"
    assert int(rounds) >= 1
    assert salt and digest and salt != digest


def test_a_fresh_salt_every_time():
    """Two runs over one password must not produce the same line."""
    assert web_auth.hash_password(PASSWORD) != web_auth.hash_password(PASSWORD)


def test_the_digest_survives_a_dotenv_round_trip(tmp_path, monkeypatch):
    """The separator must not be eaten on the way in.

    python-dotenv expands ``${VAR}`` but not a bare ``$VAR``, so "$" would
    survive here — it does not survive a paste into a shell. This test pins
    the value that actually reaches the settings object.
    """
    from dotenv import dotenv_values

    stored = web_auth.hash_password(PASSWORD)
    env_file = tmp_path / ".env"
    env_file.write_text(f"AUTH__PASSWORD_HASH={stored}\n", encoding="utf-8")

    assert dotenv_values(env_file)["AUTH__PASSWORD_HASH"] == stored

    auth = get_settings().auth
    monkeypatch.setattr(auth, "password_hash", dotenv_values(env_file)["AUTH__PASSWORD_HASH"])
    assert web_auth.check_password(PASSWORD)


@pytest.mark.parametrize("broken", [
    "not-a-hash",
    "pbkdf2_sha256:200000:onlythree",
    "pbkdf2_sha256:notanumber:c2FsdA:ZGlnZXN0",
    "pbkdf2_sha256:0:c2FsdA:ZGlnZXN0",
    "scrypt:200000:c2FsdA:ZGlnZXN0",
    "pbkdf2_sha256:200000::ZGlnZXN0",
])
def test_a_malformed_digest_closes_the_gate(monkeypatch, client, broken):
    """A typo must fail closed and say so, never fall back to the plaintext.

    Falling back would turn one bad character into a quietly weaker gate.
    Raising at import would stop the server before it could report why — the
    503 path is the one an operator can read.
    """
    auth = get_settings().auth
    monkeypatch.setattr(auth, "password_hash", broken)
    monkeypatch.setattr(web_auth, "_warned_unconfigured", False)

    assert web_auth._parse_hash(broken) is None
    assert not web_auth.is_configured()
    assert not web_auth.check_password(PASSWORD)
    assert client.get("/healthz").status_code == 503


def test_re_salting_the_same_password_also_revokes(monkeypatch):
    """The key follows the stored string, so a new salt is a new key."""
    auth = get_settings().auth
    monkeypatch.setattr(auth, "password_hash", web_auth.hash_password(PASSWORD))
    token = web_auth.issue_token()

    monkeypatch.setattr(auth, "password_hash", web_auth.hash_password(PASSWORD))
    assert not web_auth.verify_token(token)


def test_the_login_form_accepts_a_digest_password(client, monkeypatch):
    """End to end: the form path, not just check_password."""
    auth = get_settings().auth
    monkeypatch.setattr(auth, "password_hash", web_auth.hash_password(PASSWORD))

    response = client.post("/auth/login", data={"password": PASSWORD},
                           follow_redirects=False)
    assert response.status_code == 303
    assert web_auth.COOKIE_NAME in response.cookies
    assert client.get("/api/users").status_code == 200


def test_the_digest_rounds_are_read_from_the_stored_line(monkeypatch):
    """Raising the work factor must not invalidate existing lines."""
    auth = get_settings().auth
    cheap = web_auth.hash_password(PASSWORD, rounds=1000)
    assert ":1000:" in cheap
    monkeypatch.setattr(auth, "password_hash", cheap)
    assert web_auth.check_password(PASSWORD)


def test_the_generator_script_loads_the_format_without_the_package(monkeypatch):
    """The script must reach the format without importing CoScientist.

    That import builds the whole agent system, so it needs a complete LLM
    configuration — which an operator writing the .env file does not have. The
    script therefore path-loads one leaf module, and this test pins both halves
    of that: the load works with no package import, and what it produces is what
    the server accepts.
    """
    import importlib.util
    from pathlib import Path as _Path

    script = _Path(__file__).resolve().parents[2] / "deploy/make_password_hash.py"
    spec = importlib.util.spec_from_file_location("make_password_hash", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fmt = module._load_format()
    # The same source file the server imports, reached a second way. This is
    # what rules out drift: there is one copy of the format, not two.
    from CoScientist.web import password_hash as served
    assert fmt.__file__ == served.__file__
    assert fmt.SCHEME == served.SCHEME and fmt.ROUNDS == served.ROUNDS
    assert fmt.ENV_KEY == "AUTH__PASSWORD_HASH"

    auth = get_settings().auth
    monkeypatch.setattr(auth, "password_hash", fmt.hash_password(PASSWORD, rounds=1000))
    assert web_auth.check_password(PASSWORD)
    assert not web_auth.check_password(PASSWORD + "x")


