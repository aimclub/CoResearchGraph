"""Checkpoint snapshots must preserve resumable JSON types exactly."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic_core import PydanticSerializationError

from CoScientist.web import checkpoints


def test_checkpoint_rejects_an_unknown_state_leaf_without_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "state_dir", lambda: tmp_path)

    with pytest.raises(PydanticSerializationError):
        checkpoints.save_checkpoint(
            "user",
            "session",
            {"state": {"diagnostic": ValueError("not resumable")}},
        )

    assert not checkpoints.checkpoint_dir("user", "session").exists()


def test_checkpoint_keeps_supported_values_json_native(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoints, "state_dir", lambda: tmp_path)
    created = datetime(2026, 9, 26, 12, 30, tzinfo=timezone.utc)

    public = checkpoints.save_checkpoint(
        "user",
        "session",
        {"id": "cp-safe", "created_at": created, "state": {"ok": True}},
    )
    stored = checkpoints.load_checkpoint("user", "session", "cp-safe")

    assert public["id"] == "cp-safe"
    assert stored["created_at"] == "2026-09-26T12:30:00Z"
    assert stored["state"] == {"ok": True}
