"""Checkpoint management is reserved for the trusted instance administrator."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from CoScientist.checkpoints import api, synapse
from CoScientist.checkpoints.model import CheckpointManifest, SessionRef
from CoScientist.checkpoints.store import LocalZipStore

TOKEN = "test-only-platform-admin-credential"
ROUTES = [
    ("GET", "/api/checkpoints", None),
    ("GET", "/api/checkpoints/ckpt_private", None),
    ("GET", "/api/checkpoints/ckpt_private/bundle", None),
    ("POST", "/api/checkpoints/ckpt_private/restore", {}),
    ("POST", "/api/checkpoints/runs", {"context_id": "ctx", "run_id": "run"}),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    import CoScientist.config

    monkeypatch.setattr(
        CoScientist.config,
        "get_settings",
        lambda: SimpleNamespace(
            checkpoints=SimpleNamespace(api_token=SecretStr(TOKEN))
        ),
    )
    store = LocalZipStore(str(tmp_path))
    store.save(
        CheckpointManifest(
            checkpoint_id="ckpt_private",
            run_id="private-run",
            label="T0",
            created_at="t",
            session=SessionRef(app_name="app", user_id="owner", session_id="private"),
        ),
        {"session_state": b'{"private":"data"}', "session_events": b"[]"},
    )
    effects = []
    for method in ("list", "load", "bundle_path"):
        original = getattr(store, method)

        def tracked(*args, _method=method, _original=original, **kwargs):
            effects.append(_method)
            return _original(*args, **kwargs)

        monkeypatch.setattr(store, method, tracked)

    async def restore(*args, **kwargs):
        effects.append("restore")
        return {"context_id": "restored"}

    monkeypatch.setattr(api, "restore_checkpoint", restore)
    monkeypatch.setattr(synapse, "register_run", lambda *a: effects.append("register"))
    app = FastAPI()
    app.include_router(
        api.make_checkpoint_router(
            session_service=object(), app_name="app", store=store
        )
    )
    with TestClient(app) as c:
        yield c, effects


@pytest.mark.parametrize("method,path,body", ROUTES)
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic ignored"])
def test_no_checkpoint_operation_without_admin(
    client, method, path, body, authorization
):
    c, effects = client
    headers = {"Authorization": authorization} if authorization else {}
    response = c.request(method, path, json=body, headers=headers)
    assert response.status_code == 401
    assert effects == []
    assert "private" not in response.text


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_admin_can_manage_checkpoint(client, method, path, body):
    c, _effects = client
    response = c.request(
        method, path, json=body, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("method,path,body", ROUTES)
def test_missing_admin_configuration_fails_closed(
    client, monkeypatch, method, path, body
):
    import CoScientist.config

    monkeypatch.setattr(
        CoScientist.config,
        "get_settings",
        lambda: SimpleNamespace(checkpoints=SimpleNamespace(api_token=None)),
    )
    c, effects = client
    response = c.request(
        method, path, json=body, headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 503
    assert effects == []


def test_admin_secret_settings(monkeypatch):
    from CoScientist.config.settings import CheckpointSettings, Settings

    monkeypatch.setenv("CHECKPOINTS__API_TOKEN", TOKEN)
    assert Settings().checkpoints.api_token.get_secret_value() == TOKEN
    assert TOKEN not in repr(Settings().checkpoints)
    with pytest.raises(ValueError):
        CheckpointSettings(api_token="")
