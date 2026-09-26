"""Side REST API for checkpoint management (list / inspect / download / restore).

Mounted on the SAME FastAPI apps that already serve the system: the A2A
server (``A2AFastAPIApplication.build()`` returns plain FastAPI) and the web
UI. Control commands must not pass through LLM interpretation — per the
Synapse contract, A2A stays the dialogue channel and snapshots are managed by
a service API ("первичен API", SynapseNmas §6.8). Every route requires the
trusted instance-administrator credential (see docs/checkpoint_control.md).

The busy gate is PROCESS-wide (``CheckpointPlugin.any_busy()``): restore
mutates process-wide store singletons (task tracker, research graph), and in
``run_all`` mode the A2A servers share one process — a per-router check would
guard only its own runner.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from CoScientist.checkpoints.auth import require_checkpoint_admin
from CoScientist.checkpoints.plugin import CheckpointPlugin
from CoScientist.checkpoints.restore import CompatibilityError, restore_checkpoint
from CoScientist.checkpoints.store import LocalZipStore, get_default_store

logger = logging.getLogger(__name__)

# async () -> (session_service, app_name): lets the web app resolve its
# lazily-created manager at request time instead of at mount time.
SessionResolver = Callable[[], Awaitable[Tuple[object, str]]]


class RestoreRequest(BaseModel):
    compat: str = "relaxed"           # "relaxed" | "strict"
    import_stores: bool = True


class RunRegisterRequest(BaseModel):
    context_id: str
    run_id: str
    traceparent: Optional[str] = None


def make_checkpoint_router(
    *,
    session_service=None,
    app_name: Optional[str] = None,
    session_resolver: Optional[SessionResolver] = None,
    store: Optional[LocalZipStore] = None,
    on_restored: Optional[Callable[[dict], Awaitable[None]]] = None,
) -> APIRouter:
    """Router bound to one runner's session service (the restore target).

    Pass either (``session_service`` + ``app_name``) or a ``session_resolver``.
    ``on_restored`` lets the host repoint its run loop at the restored session
    (the web UI uses this to continue the chat in place).
    """
    if session_resolver is None:
        if session_service is None or app_name is None:
            raise ValueError("pass either session_resolver or (session_service + app_name)")
        _svc, _app = session_service, app_name

        async def session_resolver() -> Tuple[object, str]:  # noqa: F811
            return _svc, _app

    store = store or get_default_store()
    router = APIRouter(
        prefix="/api/checkpoints", tags=["checkpoints"],
        dependencies=[Depends(require_checkpoint_admin)],
    )

    @router.get("")
    async def list_checkpoints(run_id: Optional[str] = None):
        return {"checkpoints": store.list(run_id)}

    @router.post("/runs")
    async def register_run(body: RunRegisterRequest):
        # Synapse v1: the platform registers its run_id + traceparent for a
        # contextId before message/send, so capture stamps the platform run_id.
        from CoScientist.checkpoints import synapse
        synapse.register_run(body.context_id, body.run_id, body.traceparent)
        return {"ok": True}

    @router.get("/{checkpoint_id}")
    async def get_manifest(checkpoint_id: str):
        try:
            manifest, _ = store.load(checkpoint_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"checkpoint {checkpoint_id} not found")
        return manifest.model_dump()

    @router.get("/{checkpoint_id}/bundle")
    async def download_bundle(checkpoint_id: str):
        path = store.bundle_path(checkpoint_id)
        if path is None:
            raise HTTPException(status_code=404, detail=f"checkpoint {checkpoint_id} not found")
        return FileResponse(path, filename=path.name, media_type="application/zip")

    @router.post("/{checkpoint_id}/restore")
    async def restore(checkpoint_id: str, body: RestoreRequest | None = None):
        body = body or RestoreRequest()
        if CheckpointPlugin.any_busy():
            # restore never mutates a live run; refuse while ANY runner in this
            # process has an active invocation (stores are process singletons)
            raise HTTPException(status_code=409, detail="an invocation is currently running; retry when idle")
        svc, target_app = await session_resolver()
        try:
            result = await restore_checkpoint(
                checkpoint_id,
                session_service=svc,
                app_name=target_app,
                compat=body.compat,
                store=store,
                import_stores=body.import_stores,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"checkpoint {checkpoint_id} not found")
        except CompatibilityError as exc:
            raise HTTPException(status_code=409, detail={"pin_mismatches": exc.mismatches})
        if on_restored is not None:
            await on_restored(result)
        return result

    return router
