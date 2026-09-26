"""helpers/s3_transfer.py — the S3 pass-through the generated server.py loads.

Loaded standalone by file path (it is copied verbatim into
output/helpers/s3_transfer.py and is deliberately self-contained: stdlib only,
boto3 imported lazily inside functions), so no CoScientist package needs to be
on sys.path to test it — same reasoning as tests/unit/_codegen_loader.py.
"""

import email.message
import importlib.util
import json
import os
import re
import sys
import time
import types
from pathlib import Path

_S3_TRANSFER = (
    Path(__file__).resolve().parents[2]
    / "CoScientist"
    / "alembic"
    / "tools"
    / "scripts"
    / "s3_transfer.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("alembic_s3_transfer_under_test", _S3_TRANSFER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


s3t = _load()

_ALL_ENV = s3t.S3_ENV + tuple(f"S3__{name}" for name in s3t.S3_ENV) + (
    "S3_PRESIGN_EXPIRATION", "S3_REGION", "S3_HTTP_TIMEOUT", "S3_HTTP_MAX_BYTES",
    "S3_UPLOAD_MAX_BYTES", "S3__EXTERNAL_ENDPOINT_URL",
)


def _clear_env(monkeypatch):
    for name in _ALL_ENV:
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch, **overrides):
    values = {
        "ENDPOINT_URL": "https://s3.example.com",
        "ACCESS_KEY": "AKIA",
        "SECRET_KEY": "secret",
        "BUCKET_NAME": "bucket",
    }
    values.update(overrides)
    for k, v in values.items():
        monkeypatch.setenv(k, v)


class _FakeUploadClient:
    """Records every upload_file/generate_presigned_url call it sees."""

    def __init__(self):
        self.uploads = []
        self.presign_calls = []

    def upload_file(self, local_path, bucket, key):
        self.uploads.append({"local_path": local_path, "bucket": bucket, "key": key})

    def generate_presigned_url(self, method, Params, ExpiresIn):
        self.presign_calls.append(ExpiresIn)
        return f"https://signed/{Params['Key']}?exp={ExpiresIn}"


# ── s3_enabled ────────────────────────────────────────────────────────────

def test_s3_enabled_false_without_any_env(monkeypatch):
    _clear_env(monkeypatch)
    assert s3t.s3_enabled() is False


def test_s3_enabled_false_when_one_var_missing(monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.delenv("BUCKET_NAME", raising=False)
    assert s3t.s3_enabled() is False


def test_s3_enabled_false_when_a_var_is_empty(monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch, BUCKET_NAME="")
    assert s3t.s3_enabled() is False


def test_s3_enabled_true_with_all_four(monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    assert s3t.s3_enabled() is True


def test_s3_enabled_true_with_only_the_nested_settings_spelling(monkeypatch):
    _clear_env(monkeypatch)
    for name in s3t.S3_ENV:
        monkeypatch.setenv(f"S3__{name}", f"aliased-{name.lower()}")
    assert s3t.s3_enabled() is True


def test_s3_enabled_true_with_a_mix_of_bare_and_aliased_names(monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.delenv("BUCKET_NAME", raising=False)
    monkeypatch.setenv("S3__BUCKET_NAME", "aliased-bucket")
    assert s3t.s3_enabled() is True


def test_env_prefers_the_s3_prefixed_name_over_the_legacy_bare_name(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("BUCKET_NAME", "bare-bucket")
    monkeypatch.setenv("S3__BUCKET_NAME", "aliased-bucket")
    assert s3t._env("BUCKET_NAME") == "aliased-bucket"


def test_maybe_upload_uses_the_aliased_bucket_name(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    for name in s3t.S3_ENV:
        monkeypatch.setenv(f"S3__{name}", f"aliased-{name.lower()}")
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    src = tmp_path / "data.csv"
    src.write_text("x", encoding="utf-8")
    uploaded = s3t.maybe_upload(str(src), "prefix", "input_path")

    assert uploaded is not None
    assert client.uploads[0]["bucket"] == "aliased-bucket_name"


# ── is_file_param ─────────────────────────────────────────────────────────

def test_is_file_param_matches_path_and_file_suffixes():
    assert s3t.is_file_param("input_path")
    assert s3t.is_file_param("output_file")
    assert s3t.is_file_param("Weights_Path")   # case-insensitive
    assert not s3t.is_file_param("path_input")
    assert not s3t.is_file_param("n_cells")
    assert not s3t.is_file_param("device")


# ── scope_from_headers / call_prefix ────────────────────────────────────────

def test_scope_from_headers_falls_back_without_headers():
    assert s3t.scope_from_headers(None) == ("local", "default")
    assert s3t.scope_from_headers({}) == ("local", "default")


def test_scope_from_headers_falls_back_when_one_header_missing():
    assert s3t.scope_from_headers({"X-Coscientist-User": "alice"}) == ("local", "default")


def test_scope_from_headers_reads_case_insensitively():
    headers = {"x-coscientist-user": "alice", "X-COSCIENTIST-SESSION": "sess-1"}
    assert s3t.scope_from_headers(headers) == ("alice", "sess-1")


def test_scope_from_headers_sanitizes_unsafe_characters():
    headers = {"X-Coscientist-User": "alice/bob ?!", "X-Coscientist-Session": "s e s"}
    user, session = s3t.scope_from_headers(headers)
    assert user == "alice_bob"
    assert session == "s_e_s"


def test_call_prefix_shape():
    prefix = s3t.call_prefix(("alice", "sess-1"), "massformer", "predict")
    parts = prefix.split("/")
    assert parts[:5] == ["ephemeral", "alice", "sess-1", "alembic_massformer", "predict"]
    assert len(parts) == 6
    assert len(parts[5]) == 8  # uuid4().hex[:8]


def test_call_prefix_is_unique_per_call():
    a = s3t.call_prefix(("local", "default"), "repo", "tool")
    b = s3t.call_prefix(("local", "default"), "repo", "tool")
    assert a != b


# ── prepare_kwargs / publish_result pass-through with no S3 env ────────────

def test_prepare_kwargs_leaves_local_paths_untouched(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    kwargs = {"input_path": "/some/local/data.csv", "n": 3}

    out = s3t.prepare_kwargs(kwargs, tmp_path / "scratch")

    assert out == kwargs


def test_publish_result_is_a_noop_without_s3_env(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    real_file = tmp_path / "out.csv"
    real_file.write_text("x", encoding="utf-8")
    result = {"output_path": str(real_file), "score": 0.9}

    out = s3t.publish_result(result, "prefix", tmp_path / "repos")

    assert out == result
    assert "output_path_s3" not in out


# ── resolve_input ────────────────────────────────────────────────────────

def test_resolve_input_leaves_a_local_path_alone(tmp_path):
    assert s3t.resolve_input("/some/local/data.csv", tmp_path) == "/some/local/data.csv"


def test_resolve_input_leaves_non_string_values_alone(tmp_path):
    assert s3t.resolve_input(42, tmp_path) == 42
    assert s3t.resolve_input(["a", "b"], tmp_path) == ["a", "b"]
    assert s3t.resolve_input(None, tmp_path) is None


def test_resolve_input_downloads_an_s3_uri(tmp_path, monkeypatch):
    downloaded = {}

    class _FakeClient:
        def download_file(self, bucket, key, local_path):
            downloaded["bucket"] = bucket
            downloaded["key"] = key
            Path(local_path).write_text("content", encoding="utf-8")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _FakeClient())

    scratch = tmp_path / "scratch"
    result = s3t.resolve_input("s3://my-bucket/some/dir/input.csv", scratch)

    assert downloaded == {"bucket": "my-bucket", "key": "some/dir/input.csv"}
    assert Path(result).name == "input.csv"
    assert Path(result).parent.parent == scratch   # its own scratch subdirectory
    assert Path(result).read_text(encoding="utf-8") == "content"


def test_resolve_input_matches_s3_scheme_case_insensitively(tmp_path, monkeypatch):
    monkeypatch.setattr(
        s3t, "_client_factory",
        lambda: type("C", (), {"download_file": lambda self, b, k, p: Path(p).write_text("x")})(),
    )

    result = s3t.resolve_input("S3://Bucket/Key/input.csv", tmp_path / "scratch")

    assert Path(result).name == "input.csv"


def test_resolve_input_downloads_an_http_uri(tmp_path, monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"payload"

    captured = {}

    def _urlopen(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(s3t.urllib.request, "urlopen", _urlopen)

    scratch = tmp_path / "scratch"
    result = s3t.resolve_input("https://example.com/data/input.png", scratch)

    assert Path(result).name == "input.png"
    assert Path(result).parent.parent == scratch
    assert Path(result).read_bytes() == b"payload"
    assert captured["timeout"] == s3t._DEFAULT_HTTP_TIMEOUT


def test_resolve_input_matches_http_scheme_case_insensitively(tmp_path, monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"x"

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    result = s3t.resolve_input("HTTPS://example.com/input.png", tmp_path / "scratch")

    assert Path(result).name == "input.png"


def test_prepare_kwargs_resolves_only_file_params(tmp_path, monkeypatch):
    monkeypatch.setattr(
        s3t, "_client_factory",
        lambda: type("C", (), {"download_file": lambda self, b, k, p: Path(p).write_text("x")})(),
    )
    scratch = tmp_path / "scratch"

    out = s3t.prepare_kwargs(
        {"input_path": "s3://bucket/key/file.csv", "note": "s3://not-a-file-param"},
        scratch,
    )

    assert Path(out["input_path"]).name == "file.csv"
    assert Path(out["input_path"]).parent.parent == scratch
    assert out["note"] == "s3://not-a-file-param"   # untouched: not a *_path/*_file key


# ── BLOCKER 1 regression: two inputs sharing a basename never collide ──────

def test_prepare_kwargs_two_s3_inputs_with_the_same_basename_do_not_collide(tmp_path, monkeypatch):
    written = {}

    class _FakeClient:
        def download_file(self, bucket, key, local_path):
            written[(bucket, key)] = local_path
            Path(local_path).write_text(f"{bucket}/{key}", encoding="utf-8")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _FakeClient())
    scratch = tmp_path / "scratch"

    out = s3t.prepare_kwargs(
        {
            "a_path": "s3://bucket-1/dir1/data.csv",
            "b_path": "s3://bucket-2/dir2/data.csv",
        },
        scratch,
    )

    assert out["a_path"] != out["b_path"]
    assert Path(out["a_path"]).name == "data.csv"
    assert Path(out["b_path"]).name == "data.csv"
    assert Path(out["a_path"]).read_text(encoding="utf-8") == "bucket-1/dir1/data.csv"
    assert Path(out["b_path"]).read_text(encoding="utf-8") == "bucket-2/dir2/data.csv"


def test_resolve_input_two_calls_with_the_same_basename_get_isolated_subdirs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        s3t, "_client_factory",
        lambda: type("C", (), {"download_file": lambda self, b, k, p: Path(p).write_text("x")})(),
    )
    scratch = tmp_path / "scratch"

    first = s3t.resolve_input("s3://b/dir1/data.csv", scratch)
    second = s3t.resolve_input("s3://b/dir2/data.csv", scratch)

    assert first != second
    assert Path(first).exists()
    assert Path(second).exists()   # first download was not overwritten in place


# ── MAJOR 3 regression: bounded, streamed HTTP download ─────────────────────

def test_download_http_enforces_a_configurable_timeout(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("S3_HTTP_TIMEOUT", "7")
    captured = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return b""

    def _urlopen(url, timeout=None):
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(s3t.urllib.request, "urlopen", _urlopen)

    s3t.resolve_input("http://example.com/empty.bin", tmp_path / "scratch")

    assert captured["timeout"] == 7


def test_download_http_raises_and_cleans_up_over_the_size_cap(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("S3_HTTP_MAX_BYTES", "10")

    class _FakeResponse:
        def __init__(self):
            self._chunks = [b"0123456789", b"more-bytes-over-the-cap"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return self._chunks.pop(0) if self._chunks else b""

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    scratch = tmp_path / "scratch"
    try:
        s3t.resolve_input("http://example.com/big.bin", scratch)
        assert False, "expected a size-limit error"
    except ValueError as exc:
        assert "S3_HTTP_MAX_BYTES" in str(exc)

    # nothing left behind in the scratch tree
    assert list(scratch.rglob("*.bin")) == []


# ── MAJOR 2 (round 2) regression: non-ASCII basenames keep their extension ──

def test_safe_filename_preserves_non_ascii_names():
    assert s3t._safe_filename("данные.csv") == "данные.csv"


def test_safe_filename_replaces_path_separators_and_nul():
    assert s3t._safe_filename("a/b\\c\x00d.csv") == "a_b_c_d.csv"


def test_safe_filename_falls_back_for_empty_or_dot_only_names():
    assert s3t._safe_filename("") == "download"
    assert s3t._safe_filename(".") == "download"
    assert s3t._safe_filename("..") == "download"


def test_resolve_input_preserves_a_non_ascii_basename_and_extension(tmp_path, monkeypatch):
    """safe_component('данные.csv') collapses to 'csv' (non-ASCII stem
    replaced by re.sub then eaten by .strip('._')) — a tool dispatching on
    suffix (pandas/PIL/torch) would then read the wrong file under a
    misleadingly-named path. _download_s3/_download_http must not do that."""

    class _FakeClient:
        def download_file(self, bucket, key, local_path):
            Path(local_path).write_text("x", encoding="utf-8")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _FakeClient())

    result = s3t.resolve_input("s3://bucket/dir/данные.csv", tmp_path / "scratch")

    assert Path(result).name == "данные.csv"
    assert Path(result).suffix == ".csv"


# ── minor: _download_s3 wraps the underlying error with the URI ────────────

def test_download_s3_wraps_the_underlying_error_with_the_uri(tmp_path, monkeypatch):
    class _BrokenClient:
        def download_file(self, bucket, key, local_path):
            raise RuntimeError("access denied")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _BrokenClient())

    try:
        s3t.resolve_input("s3://bucket/key/data.csv", tmp_path / "scratch")
        assert False, "expected a RuntimeError"
    except RuntimeError as exc:
        assert "s3://bucket/key/data.csv" in str(exc)
        assert "access denied" in str(exc)


# ── minor: HTTP downloads reject an over-cap Content-Length up front ───────

def test_download_http_rejects_up_front_on_content_length_over_the_cap(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("S3_HTTP_MAX_BYTES", "10")

    class _FakeResponse:
        headers = {"Content-Length": "1000"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            raise AssertionError("must not start reading once Content-Length rejects it")

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    try:
        s3t.resolve_input("http://example.com/big.bin", tmp_path / "scratch")
        assert False, "expected a size-limit error"
    except ValueError as exc:
        assert "S3_HTTP_MAX_BYTES" in str(exc)


def test_download_http_tolerates_a_missing_or_malformed_content_length(tmp_path, monkeypatch):
    class _FakeResponse:
        headers = {"Content-Length": "not-a-number"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"x"

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    result = s3t.resolve_input("http://example.com/data.bin", tmp_path / "scratch")

    assert Path(result).read_bytes() == b"x"


# ── MINOR 1 (round 3) regression: a response with ZERO headers must not crash ─

def test_check_content_length_tolerates_a_response_with_zero_headers():
    """A real urllib response with no headers at all gives an
    email.message.Message() that defines __len__ — falsy, but NOT None. The
    old `getattr(response, "headers", None) and response.headers.get(...)`
    idiom short-circuited `and` to that falsy Message object itself instead
    of None, and int(Message) raised an uncaught TypeError."""
    empty_headers = email.message.Message()
    assert not empty_headers   # sanity: falsy...
    assert empty_headers is not None   # ...but not None

    class _Resp:
        headers = empty_headers

    s3t._check_content_length("http://x", _Resp(), max_bytes=10)   # must not raise


def test_download_http_tolerates_a_response_with_zero_headers(tmp_path, monkeypatch):
    class _FakeResponse:
        headers = email.message.Message()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            if getattr(self, "_done", False):
                return b""
            self._done = True
            return b"payload"

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    result = s3t.resolve_input("http://example.com/data.bin", tmp_path / "scratch")

    assert Path(result).read_bytes() == b"payload"


# ── MINOR 2 (round 3) regression: a very long basename must not crash ───────

def test_safe_filename_truncates_a_very_long_name_but_keeps_the_suffix():
    long_name = ("x" * 300) + ".csv"

    result = s3t._safe_filename(long_name)

    assert len(result) < 255
    assert result.endswith(".csv")


def test_resolve_input_downloads_a_file_with_a_300_char_basename(tmp_path, monkeypatch):
    """An S3 key can run up to 1024 bytes — well past the ~255-byte filename
    limit most filesystems enforce (OSError: File name too long)."""

    class _FakeClient:
        def download_file(self, bucket, key, local_path):
            Path(local_path).write_text("x", encoding="utf-8")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _FakeClient())

    long_name = "d" * 300 + ".csv"
    result = s3t.resolve_input(f"s3://bucket/dir/{long_name}", tmp_path / "scratch")

    assert Path(result).exists()
    assert Path(result).read_text(encoding="utf-8") == "x"
    assert Path(result).suffix == ".csv"
    assert len(Path(result).name) < 255


# ── nit: a bare socket timeout is wrapped with the URI like any other error ──

def test_download_http_wraps_a_bare_socket_timeout_with_the_uri(tmp_path, monkeypatch):
    """A raw socket.timeout/TimeoutError from urllib itself (a stalled
    connection, NOT our own overall-deadline check) used to propagate bare
    ('timed out', no context) — it must be wrapped like every other download
    failure."""
    def _urlopen(url, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(s3t.urllib.request, "urlopen", _urlopen)

    try:
        s3t.resolve_input("http://example.com/data.bin", tmp_path / "scratch")
        assert False, "expected a RuntimeError"
    except RuntimeError as exc:
        assert "http://example.com/data.bin" in str(exc)
        assert "timed out" in str(exc)


def test_download_http_own_deadline_error_is_not_double_wrapped(tmp_path, monkeypatch):
    """The overall-deadline TimeoutError already carries the URI in its own
    message — it must propagate AS a TimeoutError, not get re-wrapped into a
    RuntimeError (which would also nest the message)."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("S3_HTTP_TIMEOUT", "1")

    clock = {"t": 0.0}
    monkeypatch.setattr(s3t.time, "monotonic", lambda: clock["t"])

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            clock["t"] += 10
            return b"x"

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    try:
        s3t.resolve_input("http://example.com/slow.bin", tmp_path / "scratch")
        assert False, "expected a TimeoutError"
    except RuntimeError:
        assert False, "the deadline error must not be wrapped into RuntimeError"
    except TimeoutError as exc:
        assert str(exc).count("http://example.com/slow.bin") == 1


# ── minor: an overall wall-clock deadline, not just a per-read socket timeout ─

def test_download_http_enforces_an_overall_wall_clock_deadline(tmp_path, monkeypatch):
    """A peer trickling data slowly enough to keep dodging the per-read
    socket timeout must still be bounded by the same S3_HTTP_TIMEOUT budget,
    checked via a wall-clock deadline across the whole read loop."""
    _clear_env(monkeypatch)
    monkeypatch.setenv("S3_HTTP_TIMEOUT", "5")

    clock = {"t": 0.0}
    monkeypatch.setattr(s3t.time, "monotonic", lambda: clock["t"])

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            clock["t"] += 10   # jump well past the 5s deadline on every read
            return b"x"

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    scratch = tmp_path / "scratch"
    try:
        s3t.resolve_input("http://example.com/slow.bin", scratch)
        assert False, "expected a deadline TimeoutError"
    except TimeoutError as exc:
        assert "S3_HTTP_TIMEOUT" in str(exc)

    assert list(scratch.rglob("*.bin")) == []   # partial file cleaned up


# ── maybe_upload / publish_result: the vault-compatible nested contract ─────

def test_maybe_upload_returns_bucket_alongside_the_key_and_url(tmp_path, monkeypatch):
    """s3_refs._walk (the framework's artifact walker) only recognises a
    durable reference from a dict carrying BOTH bucket and s3_key — so
    maybe_upload must hand bucket back too, not just s3_key/presigned_url."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    src = tmp_path / "data.csv"
    src.write_text("x", encoding="utf-8")
    uploaded = s3t.maybe_upload(str(src), "prefix", "output_path")

    assert uploaded == {
        "bucket": "bucket",
        "s3_key": "prefix/output_path/data.csv",
        "presigned_url": uploaded["presigned_url"],
        "expires_in": 3600,
    }


# ── publish_result: upload gating and shape ─────────────────────────────────

def test_publish_result_uploads_existing_file_outside_deny_root(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    out_file = tmp_path / "output" / "result.csv"
    out_file.parent.mkdir(parents=True)
    out_file.write_text("data", encoding="utf-8")
    deny_root = tmp_path / "repos"

    result = s3t.publish_result({"result_path": str(out_file)}, "prefix/abc", deny_root)

    assert result["result_path"] == str(out_file)   # original untouched
    assert result["result_path_s3"]["bucket"] == "bucket"
    assert result["result_path_s3"]["s3_key"] == "prefix/abc/result_path/result.csv"
    assert result["result_path_s3"]["presigned_url"].startswith("https://signed/")
    assert "result_path_s3_key" not in result
    assert "result_path_presigned_url" not in result
    assert client.uploads[0]["bucket"] == "bucket"


def test_publish_result_ignores_a_file_inside_deny_root(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.setattr(s3t, "_client_factory", lambda: (_ for _ in ()).throw(
        AssertionError("must not upload a file inside deny_root")))

    deny_root = tmp_path / "repos"
    repo_file = deny_root / "data" / "input.csv"
    repo_file.parent.mkdir(parents=True)
    repo_file.write_text("data", encoding="utf-8")

    result = s3t.publish_result({"weights_path": str(repo_file)}, "prefix", deny_root)

    assert result == {"weights_path": str(repo_file)}


def test_publish_result_ignores_a_file_inside_any_of_several_deny_roots(tmp_path, monkeypatch):
    """A tool that echoes its downloaded input_path back (a common pattern)
    must not get that scratch file re-uploaded — the local copy is deleted
    right after the call, so the sibling keys would describe a dangling
    path."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    repos = tmp_path / "repos"
    scratch = tmp_path / ".scratch" / "abc123"
    echoed_input = scratch / "d1" / "input.csv"
    echoed_input.parent.mkdir(parents=True)
    echoed_input.write_text("in", encoding="utf-8")
    out_file = tmp_path / "output" / "result.csv"
    out_file.parent.mkdir(parents=True)
    out_file.write_text("out", encoding="utf-8")

    result = s3t.publish_result(
        {"input_path": str(echoed_input), "result_path": str(out_file)},
        "prefix",
        (repos, scratch),
    )

    assert "input_path_s3" not in result
    assert result["result_path_s3"]["s3_key"] == "prefix/result_path/result.csv"
    assert [u["local_path"] for u in client.uploads] == [str(out_file)]


def test_publish_result_ignores_a_missing_or_nonfile_path(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)

    result = s3t.publish_result({"output_path": str(tmp_path / "does-not-exist.csv")},
                                 "prefix", tmp_path / "repos")

    assert result == {"output_path": str(tmp_path / "does-not-exist.csv")}


def test_publish_result_recurses_into_nested_dict_and_list(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    f1 = tmp_path / "a.csv"
    f1.write_text("1", encoding="utf-8")
    f2 = tmp_path / "b.csv"
    f2.write_text("2", encoding="utf-8")
    deny_root = tmp_path / "repos"

    result = s3t.publish_result(
        {
            "nested": {"output_path": str(f1)},
            "items": [{"output_path": str(f2)}, {"score": 1}],
        },
        "prefix",
        deny_root,
    )

    assert result["nested"]["output_path_s3"]["s3_key"] == "prefix/output_path/a.csv"
    assert result["items"][0]["output_path_s3"]["s3_key"] == "prefix/output_path/b.csv"
    assert result["items"][1] == {"score": 1}


def test_publish_result_returns_none_from_maybe_upload_on_client_error(tmp_path, monkeypatch, capsys):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)

    class _BrokenClient:
        def upload_file(self, *a, **kw):
            raise RuntimeError("network down")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _BrokenClient())

    out_file = tmp_path / "result.csv"
    out_file.write_text("data", encoding="utf-8")

    result = s3t.publish_result({"result_path": str(out_file)}, "prefix", tmp_path / "repos")

    assert result == {"result_path": str(out_file)}
    assert "result_path_s3" not in result


# ── MAJOR 4 regression: upload failures are logged, not swallowed silently ──

def test_maybe_upload_logs_the_failure_reason_to_stderr(tmp_path, monkeypatch, capsys):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)

    class _BrokenClient:
        def upload_file(self, *a, **kw):
            raise RuntimeError("network down")

    monkeypatch.setattr(s3t, "_client_factory", lambda: _BrokenClient())
    src = tmp_path / "result.csv"
    src.write_text("x", encoding="utf-8")

    out = s3t.maybe_upload(str(src), "prefix", "result_path")

    assert out is None
    err = capsys.readouterr().err
    assert "[s3] upload failed" in err
    assert str(src) in err
    assert "RuntimeError" in err   # exception type is logged...
    assert "network down" in err   # ...alongside a (truncated) message


def test_maybe_upload_truncates_a_very_long_error_message(tmp_path, monkeypatch, capsys):
    """A botocore error can embed request internals (e.g. presigned-URL query
    params with AWSAccessKeyId=...); the logged message must be bounded."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)

    class _BrokenClient:
        def upload_file(self, *a, **kw):
            raise RuntimeError("x" * 5000)

    monkeypatch.setattr(s3t, "_client_factory", lambda: _BrokenClient())

    s3t.maybe_upload("/tmp/some/result.csv", "prefix", "result_path")

    err = capsys.readouterr().err
    assert len(err) < 1000


# ── MAJOR 2 regression: two outputs with the same basename get distinct keys ─

def test_publish_result_two_outputs_with_the_same_basename_get_distinct_keys(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    f1 = tmp_path / "out1" / "result.csv"
    f1.parent.mkdir(parents=True)
    f1.write_text("1", encoding="utf-8")
    f2 = tmp_path / "out2" / "result.csv"
    f2.parent.mkdir(parents=True)
    f2.write_text("2", encoding="utf-8")

    result = s3t.publish_result(
        {"a_path": str(f1), "b_path": str(f2)}, "prefix", tmp_path / "repos",
    )

    assert result["a_path_s3"]["s3_key"] != result["b_path_s3"]["s3_key"]
    assert result["a_path_s3"]["presigned_url"] != result["b_path_s3"]["presigned_url"]
    keys = {u["key"] for u in client.uploads}
    assert len(keys) == 2   # both actually landed under distinct keys


# ── MAJOR 2 (round 2) regression: two fully non-ASCII basenames that both ──
# ── collapse to "download" under _key_safe_filename must still get distinct
# ── keys, not silently overwrite one another ────────────────────────────────

def test_publish_result_two_non_ascii_basenames_collapsing_to_download_get_distinct_keys(
    tmp_path, monkeypatch
):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    f1 = tmp_path / "out1" / "данные а.csv"
    f1.parent.mkdir(parents=True)
    f1.write_text("1", encoding="utf-8")
    f2 = tmp_path / "out2" / "данные б.csv"
    f2.parent.mkdir(parents=True)
    f2.write_text("2", encoding="utf-8")

    # sanity: both basenames really do collapse to the same sanitized name
    assert s3t._key_safe_filename(f1.name) == s3t._key_safe_filename(f2.name) == "download.csv"

    result = s3t.publish_result(
        {"items": [{"output_path": str(f1)}, {"output_path": str(f2)}]},
        "ephemeral/local/default/alembic_demo/predict/abcd1234",
        tmp_path / "repos",
    )

    key_a = result["items"][0]["output_path_s3"]["s3_key"]
    key_b = result["items"][1]["output_path_s3"]["s3_key"]
    assert key_a != key_b
    for key in (key_a, key_b):
        assert _VAULT_KEY_RE.match(key), key


# ── minor: presign expiration is clamped ─────────────────────────────────────

def test_presign_expiration_is_clamped_to_a_sane_range(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)
    out_file = tmp_path / "result.csv"
    out_file.write_text("x", encoding="utf-8")

    monkeypatch.setenv("S3_PRESIGN_EXPIRATION", "99999999")
    s3t.publish_result({"result_path": str(out_file)}, "prefix", tmp_path / "repos")
    assert client.presign_calls[-1] == s3t._MAX_PRESIGN_EXPIRATION

    monkeypatch.setenv("S3_PRESIGN_EXPIRATION", "0")
    s3t.publish_result({"result_path": str(out_file)}, "prefix", tmp_path / "repos")
    assert client.presign_calls[-1] == s3t._MIN_PRESIGN_EXPIRATION

    monkeypatch.setenv("S3_PRESIGN_EXPIRATION", "not-a-number")
    s3t.publish_result({"result_path": str(out_file)}, "prefix", tmp_path / "repos")
    assert client.presign_calls[-1] == s3t._DEFAULT_PRESIGN_EXPIRATION


# ── minor: region_name is always passed to the boto3 client ─────────────────

def test_client_factory_passes_a_region_default_and_override(monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    calls = []

    class _FakeConfig:
        def __init__(self, signature_version, **bounds):
            self.signature_version = signature_version

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda service, **kwargs: calls.append(kwargs) or object()
    fake_botocore = types.ModuleType("botocore")
    fake_botocore_client = types.ModuleType("botocore.client")
    fake_botocore_client.Config = _FakeConfig

    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setitem(sys.modules, "botocore", fake_botocore)
    monkeypatch.setitem(sys.modules, "botocore.client", fake_botocore_client)

    s3t._client_factory()
    assert calls[-1]["region_name"] == "us-east-1"

    monkeypatch.setenv("S3_REGION", "eu-west-1")
    s3t._client_factory()
    assert calls[-1]["region_name"] == "eu-west-1"

    # The committed image carries a blanked ENV S3_REGION=.
    monkeypatch.setenv("S3_REGION", "")
    s3t._client_factory()
    assert calls[-1]["region_name"] == "us-east-1"


# ── M1/M2 regression: the built key/id must clear the vault's own validation ─

# Mirrors mcp-servers/vault-mcp-server/vault_server.py:56 `_KEY_RE` exactly —
# duplicated (not imported) because that server is a separate, independently
# deployed process this module has no import relationship with.
_VAULT_KEY_RE = re.compile(
    r"^(ephemeral|permanent)/[a-zA-Z0-9_-]{1,64}/[a-zA-Z0-9_-]{1,64}/[a-zA-Z0-9_/.-]+$"
)


def test_maybe_upload_key_matches_the_vault_key_regex_for_a_nasty_input(tmp_path, monkeypatch):
    """M1 (basename) + M2 (user/session) regression, combined: a call_prefix
    built from a user/session with dots and non-ASCII characters, plus a
    maybe_upload basename with a space, a '+', and non-ASCII characters, must
    still produce a key the vault server's own _KEY_RE accepts — not just one
    this test suite happens to accept by hand."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    prefix = s3t.call_prefix(("ali.ceé", "sess.1é"), "demo_repo", "predict")

    src = tmp_path / "данные а+б.csv"
    src.write_text("x", encoding="utf-8")
    uploaded = s3t.maybe_upload(str(src), prefix, "output_path")

    assert uploaded is not None
    assert _VAULT_KEY_RE.match(uploaded["s3_key"]), uploaded["s3_key"]


# ── MINOR (round 4) regression: safe_id ─────────────────────────────────────

def test_safe_id_falls_back_to_default_for_none():
    """MINOR 1: `str(None)` used to sanitize to the literal id `"None"`
    instead of falling back to `default`, mirroring
    mcp-servers/chemical-mcp-server/server/utils/vault.py:safe_id, which
    treats `None`/empty the same via `str(value or "")`."""
    assert s3t.safe_id(None, "local") == "local"


def test_safe_id_falls_back_to_default_for_empty_string():
    assert s3t.safe_id("", "default") == "default"


def test_safe_id_replaces_dots_and_unicode_with_underscore():
    assert s3t.safe_id("ali.ceé", "local") == "ali_ce"


def test_safe_id_truncates_to_64_chars():
    long_id = "a" * 200
    result = s3t.safe_id(long_id, "local")
    assert result == "a" * 64
    assert len(result) == 64


# ── MINOR (round 4) regression: _key_safe_filename ──────────────────────────

def test_key_safe_filename_collapses_non_ascii_to_download_plus_suffix():
    assert s3t._key_safe_filename("данные.csv") == "download.csv"


def test_key_safe_filename_falls_back_for_empty_or_dot_only_names():
    assert s3t._key_safe_filename("") == "download"
    assert s3t._key_safe_filename(".") == "download"
    assert s3t._key_safe_filename("..") == "download"


def test_key_safe_filename_truncates_stem_and_suffix_separately():
    long_name = ("a" * 300) + ("." + "b" * 100)
    result = s3t._key_safe_filename(long_name)
    stem, suffix = result.split(".", 1)
    assert len(stem) == s3t._MAX_KEY_FILENAME_STEM
    assert len(suffix) == s3t._MAX_KEY_FILENAME_SUFFIX - 1  # the leading "." is part of the suffix cap


def test_key_safe_filename_preserves_an_already_ascii_name_as_is():
    assert s3t._key_safe_filename("data-1_v2.csv") == "data-1_v2.csv"


# ── integration: publish_result output is what the framework's own artifact
#    walker (CoScientist.utils.s3_refs.find_s3_artifacts) actually expects ──

def test_publish_result_output_is_found_by_the_real_artifact_walker(tmp_path, monkeypatch):
    """publish_result's nested {"bucket", "s3_key", "presigned_url"} dict is
    only useful if CoScientist.utils.s3_refs.find_s3_artifacts (the
    framework's real artifact walker, not a hand-rolled stand-in) actually
    recognises it. Imported locally, not at module level: this file is
    otherwise stand-alone loadable with no CoScientist package on sys.path
    (see the module docstring) — this one integration test is a deliberate
    exception, and pyproject's `pythonpath = ["."]` makes the import resolve
    when pytest runs from the repo root."""
    from CoScientist.utils.s3_refs import find_s3_artifacts

    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda: client)

    out_file = tmp_path / "result.csv"
    out_file.write_text("data", encoding="utf-8")

    result = s3t.publish_result({"output_path": str(out_file)}, "prefix", tmp_path / "repos")

    artifacts = find_s3_artifacts(result)

    assert len(artifacts) == 1
    assert artifacts[0]["s3_uri"] == "s3://bucket/prefix/output_path/result.csv"


# ── bounds, external endpoint, upload cap ─────────────────────────────────

def test_client_factory_bounds_every_call_and_takes_an_endpoint_override(monkeypatch):
    """Same bounds as paper_parser/s3_connection.py (#349): an endpoint that
    drops packets must fail in seconds, not hold the tool call for minutes."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    import boto3

    seen = {}
    monkeypatch.setattr(boto3, "client", lambda service, **kw: seen.update(kw) or object())

    s3t._client_factory()
    config = seen["config"]
    assert (config.connect_timeout, config.read_timeout) == (5, 30)
    assert config.retries["max_attempts"] == 3
    assert seen["endpoint_url"] == "https://s3.example.com"

    s3t._client_factory("http://localhost:9000")
    assert seen["endpoint_url"] == "http://localhost:9000"


def test_maybe_upload_presigns_for_the_external_endpoint(tmp_path, monkeypatch):
    """The serve container uploads through its own endpoint, while the link is
    signed for the host the caller opens it from."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.setenv("S3__EXTERNAL_ENDPOINT_URL", "http://localhost:9000")
    inner, outer = _FakeUploadClient(), _FakeUploadClient()
    monkeypatch.setattr(
        s3t, "_client_factory", lambda endpoint_url=None: outer if endpoint_url else inner)

    src = tmp_path / "a.csv"
    src.write_text("x", encoding="utf-8")
    assert s3t.maybe_upload(str(src), "prefix", "out_path") is not None

    assert len(inner.uploads) == 1 and not outer.uploads
    assert outer.presign_calls and not inner.presign_calls


def test_maybe_upload_presigns_with_the_same_client_without_an_external_endpoint(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.setenv("S3__EXTERNAL_ENDPOINT_URL", "https://s3.example.com")
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda endpoint_url=None: client)

    src = tmp_path / "a.csv"
    src.write_text("x", encoding="utf-8")
    s3t.maybe_upload(str(src), "prefix", "out_path")

    assert len(client.uploads) == 1 and len(client.presign_calls) == 1


def test_maybe_upload_skips_a_file_over_the_upload_cap(tmp_path, monkeypatch, capsys):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.setenv("S3_UPLOAD_MAX_BYTES", "4")
    monkeypatch.setattr(s3t, "_client_factory", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not upload a file over the cap")))

    src = tmp_path / "checkpoint.pt"
    src.write_bytes(b"0123456789")

    assert s3t.maybe_upload(str(src), "prefix", "model_path") is None
    assert "S3_UPLOAD_MAX_BYTES" in capsys.readouterr().err


def test_download_http_decodes_a_percent_encoded_filename(tmp_path, monkeypatch):
    _clear_env(monkeypatch)

    class _FakeResponse:
        headers = None

        def __init__(self):
            self._chunks = [b"a,b\n"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n=-1):
            return self._chunks.pop() if self._chunks else b""

    monkeypatch.setattr(s3t.urllib.request, "urlopen", lambda url, timeout=None: _FakeResponse())

    local = s3t.resolve_input(
        "http://example.com/in/%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D0%B5.csv?X-Amz-Signature=1",
        tmp_path / "scratch")

    assert Path(local).name == "данные.csv"


def test_publish_result_uploads_what_the_call_wrote_into_the_repo(tmp_path, monkeypatch):
    """Generated tools resolve a relative out path against the cloned repo, so
    an output often lands there. A repo file the call did not write stays out."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakeUploadClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda *a, **k: client)
    repo = tmp_path / "repos"
    (repo / "Tests").mkdir(parents=True)
    source = repo / "Tests" / "rosemary.pro"
    source.write_text(">x\nMK\n", encoding="utf-8")
    os.utime(source, (time.time() - 3600, time.time() - 3600))
    started = time.time()
    written = repo / "out.fasta"
    written.write_text(">x\nMK\n", encoding="utf-8")

    result = s3t.publish_result({"in_file": str(source), "out_file": str(written)},
                                "prefix", (), repo, started)

    assert "out_file_s3" in result
    assert "in_file_s3" not in result
    assert [u["key"] for u in client.uploads] == ["prefix/out_file/out.fasta"]


def test_publish_result_without_a_call_start_keeps_the_whole_repo_out(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    monkeypatch.setattr(s3t, "_client_factory", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not upload a repo file")))
    repo = tmp_path / "repos"
    repo.mkdir()
    written = repo / "out.fasta"
    written.write_text("x", encoding="utf-8")

    assert "out_file_s3" not in s3t.publish_result({"out_file": str(written)}, "prefix", (), repo)


# ── shrink_result: a big result goes to S3 whole, the caller gets its shape ──

class _FakePutClient(_FakeUploadClient):
    """_FakeUploadClient that also records put_object (JSON documents)."""

    def __init__(self):
        super().__init__()
        self.puts = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.puts.append({"bucket": Bucket, "key": Key, "body": Body})


def _descriptor_result(n=1613):
    return {"smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "num_descriptors": n,
            "descriptor_names": [f"D{i}" for i in range(n)],
            "values": [i * 0.5 for i in range(n)]}


def test_a_big_result_is_stored_whole_and_the_caller_keeps_its_structure(monkeypatch):
    """mordred's calc_descriptors returned 1613 names and then their values; a
    client cutting the text for the model's context left the model only names."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakePutClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda *a: client)
    result = _descriptor_result()
    prefix = "ephemeral/u/s/alembic_mordred/calc_descriptors/abcd1234"

    short = s3t.shrink_result(result, prefix)

    [put] = client.puts
    assert put["key"] == f"{prefix}/result/result.json"
    assert json.loads(put["body"]) == result
    assert set(short) == set(result) | {"result_truncated", "result_s3"}
    assert short["num_descriptors"] == 1613
    assert short["values"] and short["values"] == result["values"][:len(short["values"])]
    assert short["result_truncated"]["shortened"]["values"]["items"] == 1613
    assert short["result_s3"]["s3_key"] == put["key"]
    assert len(json.dumps(short)) <= s3t._DEFAULT_RESULT_MAX_CHARS


def test_a_result_that_fits_or_cannot_be_stored_comes_back_unchanged(monkeypatch):
    """Without the full copy in S3 nothing would lead to what was cut away."""
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakePutClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda *a: client)
    small = {"num_descriptors": 3, "values": [1, 2, 3]}
    assert s3t.shrink_result(small, "p") is small
    assert client.puts == []

    class _Failing(_FakePutClient):
        def put_object(self, **kw):
            raise RuntimeError("endpoint down")

    monkeypatch.setattr(s3t, "_client_factory", lambda *a: _Failing())
    big = _descriptor_result()
    assert s3t.shrink_result(big, "p") is big

    _clear_env(monkeypatch)
    assert s3t.shrink_result(big, "p") is big


def test_file_references_survive_the_shortening(monkeypatch, tmp_path):
    _clear_env(monkeypatch)
    _set_env(monkeypatch)
    client = _FakePutClient()
    monkeypatch.setattr(s3t, "_client_factory", lambda *a: client)
    table = tmp_path / "table.csv"
    table.write_text("x", encoding="utf-8")
    result = {"rows": [{"i": i, "label": "x" * 50} for i in range(500)], "out_file": str(table)}

    short = s3t.publish_result(result, "p", tmp_path / "scratch")

    assert short["out_file_s3"]["s3_key"] == "p/out_file/table.csv"
    assert "result_s3" in short and len(short["rows"]) < 500
    assert "out_file_s3" in json.loads(client.puts[0]["body"])
