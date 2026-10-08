"""Pull artifacts out of a sandbox container before it disappears.

Each OpenHands task gets its own container and its own /workspace, so anything a
task leaves behind is reachable only through that task's id, and only while the
container is still around (a finished task sits in COOLDOWN for a while, then
goes). The agent can LIST files in a sandbox but has no download tool, so work
products stay stranded there — which is how a finished dataset ended up
unreachable to the next task.

Usage (needs the VPN that reaches SANDBOX_URL):

    python scripts/rescue_sandbox_files.py 2f859f5b-02db-4561-80b7-8afc1b8d0540 \
        --dir /workspace/golem_h1 --out /app/rescued/golem_h1

Lists the remote directory, then downloads every file into --out, keeping the
layout. Prints what it could not fetch and exits non-zero, so a partial rescue
is never mistaken for a complete one.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, "/app")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sandbox_id", help="task id, e.g. 2f859f5b-02db-...")
    ap.add_argument("--dir", default="/workspace", help="remote directory to pull")
    ap.add_argument("--out", default="/app/rescued", help="local destination")
    ap.add_argument("--url", default=os.getenv("SANDBOX_URL"),
                    help="sandbox base URL (default: SANDBOX_URL)")
    args = ap.parse_args()

    if not args.url:
        print("SANDBOX_URL is not set and --url was not given")
        return 2

    from CoScientist.tools.coder_tools import openhands_sandbox as sb

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def walk(remote: str, depth: int = 0) -> list:
        """Remote files under `remote`, recursing into directories."""
        if depth > 4:
            return []
        try:
            listing = sb.list_sandbox_files(remote, sandbox_id=args.sandbox_id,
                                            sandbox_url=args.url)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! cannot list {remote}: {type(exc).__name__}: {exc}")
            return []
        entries = listing.get("entries") if isinstance(listing, dict) else None
        if not entries:
            return []
        found = []
        for e in entries:
            path, kind = e.get("path") or "", e.get("type")
            if not path or e.get("name", "").startswith((".git", ".venv", ".cache")):
                continue
            if kind == "dir":
                found += walk(path, depth + 1)
            else:
                found.append((path, int(e.get("size") or 0)))
        return found

    print(f"listing {args.dir} in sandbox {args.sandbox_id} …")
    files = walk(args.dir)
    if not files:
        print("nothing listed — the container is probably gone, or the path is wrong")
        return 1
    print(f"{len(files)} files, {sum(s for _, s in files) / 1e6:.1f} MB")

    failed = []
    for remote, size in sorted(files, key=lambda f: -f[1]):
        rel = remote.lstrip("/").replace(args.dir.lstrip("/") + "/", "", 1)
        local = out_root / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        try:
            sb.download_sandbox_file(remote, str(local), sandbox_id=args.sandbox_id,
                                             sandbox_url=args.url)
            print(f"  ok  {size/1e6:7.2f} MB  {rel}")
        except Exception as exc:  # noqa: BLE001
            failed.append((rel, f"{type(exc).__name__}: {exc}"))
            print(f"  !!  {rel}: {exc}")

    print(f"\nsaved to {out_root}")
    if failed:
        print(f"NOT RESCUED: {len(failed)}")
        for rel, why in failed[:10]:
            print(f"  {rel}: {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
