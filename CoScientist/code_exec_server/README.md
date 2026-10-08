# Code-Exec Server (CoderAgent sandbox)

A standalone HTTP service that executes the CoderAgent's shell/code/git commands
in an isolated, per-session workspace. Run it separately (ideally in its own
container) so arbitrary research code never runs inside the orchestrator process.

When the agent's `CODE_EXEC__URL` is **unset**, the CoderAgent falls back to a
local in-process sandbox and you don't need this server at all. Run this server
only when you want remote/isolated execution.

## HTTP contract

```
POST /submit   json={"command": str, "workspace_id": str, "timeout": int} -> {"job_id": str, "status": "running"}
GET  /result?job_id=<id>                                                  -> {"status", "stdout", "stderr", "exit_code", "workspace_id", "job_id"}
GET  /health                                                              -> {"status": "ok"}
```

`status` is one of `running | success | error | timeout | blocked`.

## Run it

From the repo root (the dir that contains the `CoScientist/` package):

```bash
# default host 0.0.0.0, port 8131
python -m CoScientist.code_exec_server.app

# custom host/port
CODE_EXEC_HOST=0.0.0.0 CODE_EXEC_PORT=8131 python -m CoScientist.code_exec_server.app
```

Health check:

```bash
curl -s http://localhost:8131/health      # {"status":"ok"}
```

## Point the agent at it

Set in `.env` (read by `CoScientist.config`):

```bash
CODE_EXEC__URL=http://localhost:8131
# optional tuning (defaults shown):
CODE_EXEC__CHECK_WAIT=15          # seconds check_job waits inline for a job
CODE_EXEC__DEFAULT_TIMEOUT=1800   # max seconds a single command may run
CODE_EXEC__WORKSPACE_ROOT=./workspace
```

With `CODE_EXEC__URL` set, the CoderToolset submits jobs over HTTP and polls for
results; without it, execution is local. Everything else (fire-and-forget,
`check_job`, concurrency, the safety blocklist, HITL approval) behaves the same.

## Local two-terminal smoke test ("remote" = localhost)

**Terminal 1** — start the server:

```bash
cd <repo-root>
CODE_EXEC_PORT=8131 python -m CoScientist.code_exec_server.app
```

**Terminal 2** — drive the toolset against it over HTTP:

```bash
cd <repo-root>
CODE_EXEC__URL=http://localhost:8131 python - <<'PY'
import asyncio
from CoScientist.tools.coder_tools import CoderToolset

async def main():
    ts = CoderToolset()
    r = await ts.execute_bash("echo hello && sleep 1 && echo done")
    print("submit:", r["status"], r["job_id"])
    print("result:", (await ts.check_job(r["job_id"]))["stdout"])

asyncio.run(main())
PY
```

You should see the server log `POST /submit` and `GET /result?...` 200s, and the
command output come back. The full automated check is:

```bash
python -m CoScientist.scripts.test_coder_sandbox   # runs both local and remote paths
```

## Notes

- Each `workspace_id` maps to `./<workspace_root>/<workspace_id>/`. The agent
  derives `workspace_id` from the session, so all of a session's commands share
  one workspace (clones/artifacts persist across calls).
- A hard safety blocklist (`rm -rf /`, `mkfs`, fork bombs, …) is enforced both
  client-side and here, as defense in depth.
- Jobs run in the background; `/result` returns `running` until they finish.
