"""A restored A2A run receives the checkpoint's scoped research graph."""

from __future__ import annotations

from uuid import uuid4

import pytest
from google.adk.sessions import InMemorySessionService

from CoScientist.checkpoints.capture import capture_checkpoint
from CoScientist.checkpoints.restore import restore_checkpoint
from CoScientist.checkpoints.store import LocalZipStore
from CoScientist.graph.research.store import get_research_graph


@pytest.mark.asyncio
async def test_checkpoint_restores_scoped_graph_under_new_run(tmp_path, monkeypatch):
    from CoScientist.config import settings

    monkeypatch.setattr(settings.research_graph, "dir", str(tmp_path / "graphs"))
    context_id = f"ctx-{uuid4().hex}"
    user_id = f"A2A_USER_{context_id}"
    sessions = InMemorySessionService()
    session = await sessions.create_session(
        app_name="orchestrator",
        user_id=user_id,
        session_id=context_id,
    )
    graph = get_research_graph(user_id=user_id, session_id=context_id)
    assert graph.init_research(source="OrchestratorAgent", question="Does X bind Y?")[
        "ok"
    ]
    assert graph.commit(
        source="HypothesesAgent",
        nodes=[{"type": "Hypothesis", "attrs": {"formulation": "X binds Y"}}],
    ).ok

    store = LocalZipStore(str(tmp_path / "checkpoints"))
    saved = await capture_checkpoint(
        session=session,
        label="T2_after_hypotheses",
        store=store,
        validator_pending=False,
    )
    assert saved is not None
    restored = await restore_checkpoint(
        saved.checkpoint_id,
        session_service=InMemorySessionService(),
        store=store,
    )
    assert restored["context_id"] != context_id
    resumed_graph = get_research_graph(
        user_id=restored["user_id"], session_id=restored["context_id"]
    )
    assert "error" not in resumed_graph.get_context_slice("Q1")
    assert "error" not in resumed_graph.get_context_slice("H1")
