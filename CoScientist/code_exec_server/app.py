"""FastAPI app for the code-exec server.

Exposes the contract the CoderToolset client expects:

  POST /submit  json={"command", "workspace_id", "timeout"} -> {"job_id", "status"}
  GET  /result?job_id=...  -> {"status", "stdout", "stderr", "exit_code", ...}
  GET  /health             -> {"status": "ok"}

Paths and workspace root are read from CoScientist settings (code_exec.*) so the
server and the agent stay configured from one place. Point the agent at this
server by setting CODE_EXEC__URL (e.g. http://localhost:8131) in the .env.

Run:  python -m CoScientist.code_exec_server.app
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from CoScientist.config import get_settings
from CoScientist.code_exec_server.runner import JobRunner

_settings = get_settings()
_CFG = _settings.code_exec

app = FastAPI(title="CoScientist Code-Exec Server")
runner = JobRunner(workspace_root=_CFG.workspace_root)


class SubmitRequest(BaseModel):
    command: str
    workspace_id: str = "default"
    timeout: int = _CFG.default_timeout


@app.post(_CFG.submit_path)
async def submit(req: SubmitRequest) -> dict:
    job = runner.submit(
        command=req.command,
        workspace_id=req.workspace_id,
        timeout=req.timeout,
    )
    return {"job_id": job.job_id, "status": job.status.value}


@app.get(_CFG.result_path)
async def result(job_id: str) -> dict:
    job = runner.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    return job.to_dict()


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    host = os.getenv("CODE_EXEC_HOST", "0.0.0.0")
    port = int(os.getenv("CODE_EXEC_PORT", "8131"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
