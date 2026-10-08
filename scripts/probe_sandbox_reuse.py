"""Ask the sandbox service whether a specific container will take more work.

Each task gets its own container and its own /workspace, so continuing an
experiment means landing in the SAME one. The client supports that — the
submission carries `session_id` — but whether a container that has finished (the
console shows it as COOLDOWN) still accepts a follow-up is a property of the
service, not of this client. This finds out cheaply, with a trivial task.

Usage (needs the VPN that reaches SANDBOX_URL):

    python scripts/probe_sandbox_reuse.py 2f859f5b-02db-4561-80b7-8afc1b8d0540

Reports whether the service reused the container or silently made a new one, and
lists /workspace in whatever it got, so you can see if the files are there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, "/app")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sandbox_id")
    ap.add_argument("--url", default=os.getenv("SANDBOX_URL"))
    ap.add_argument("--dir", default="/workspace/golem_h1",
                    help="directory to look for after the probe")
    args = ap.parse_args()
    if not args.url:
        print("SANDBOX_URL is not set and --url was not given")
        return 2

    from CoScientist.tools.coder_tools import openhands_sandbox as sb

    print(f"1) status of {args.sandbox_id}")
    try:
        st = sb.get_sandbox_status(sandbox_id=args.sandbox_id, sandbox_url=args.url)
        print("   ", json.dumps(st, ensure_ascii=False)[:300])
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {type(exc).__name__}: {exc}")

    print(f"\n2) is {args.dir} still there?")
    try:
        ls = sb.list_sandbox_files(args.dir, sandbox_id=args.sandbox_id,
                                   sandbox_url=args.url)
        entries = (ls or {}).get("entries") or []
        print(f"    {len(entries)} entries")
        for e in entries[:8]:
            print(f"      {e.get('type','?'):<4} {e.get('name')}")
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {type(exc).__name__}: {exc}")

    print("\n3) submitting a trivial follow-up into that container")
    try:
        res = sb.run_sandbox_task(
            "Print the absolute path and `ls -la` of /workspace, then stop. "
            "Do not create, modify or delete anything.",
            sandbox_id=args.sandbox_id, sandbox_url=args.url, timeout=120, verbose=False,
        )
        reused = res.get("reused")
        got = res.get("sandbox_id")
        print(f"    reused={reused}  sandbox_id={got}")
        if reused and got == args.sandbox_id:
            print("    => the container takes follow-up work; keep the binding and continue in it")
        elif got and got != args.sandbox_id:
            print(f"    => the service made a NEW container ({got}); the old workspace is NOT there")
        print("    summary:", str(res.get("summary"))[:300])
    except Exception as exc:  # noqa: BLE001
        print(f"    ! {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
