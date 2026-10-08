"""Rebuild a converted tool's image from its portable artefacts.

The venvs inside a built ``alembic-tool:<name>`` image are host-specific and
cannot be relocated, so what travels between machines is the low-weight
artefacts: the cloned source plus ``output/{tools,helpers,server.py,setup.sh}``.
The image is rebuilt from them on the target through
``docker/alembic/serve.Dockerfile``, which regenerates the venv by replaying
``setup.sh`` on a clean base.

That makes a bundle of these artefacts the thing worth keeping and shipping: it
is small enough to store, and it is enough to stand the tool up anywhere.

Both docker steps take an injectable ``runner``, so the logic is testable
without Docker. A failed rebuild is classified from its build log, because
"the build failed" is not actionable and the same handful of causes recur.

.. note::

   There is no command-line entry point for this yet. The functions here are
   the mechanism and are covered by tests, but nothing in the tree calls them,
   so shipping a tool to another machine currently means driving them from
   Python by hand. A CLI is the missing piece.

   For the same reason this path has never been run for real: the tests cover
   it with an injected runner, and no push to a registry, no rebuild on a
   target, no serve on the far side has been done end to end. Both are open:
   write the entry point, then verify it against a live registry and a second
   machine.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVE_DOCKERFILE = PROJECT_ROOT / "docker" / "alembic" / "serve.Dockerfile"
# The portable artefacts serve.Dockerfile copies (everything EXCEPT the venvs).
ARTEFACT_PATHS = (
    "repos",
    "output/tools",
    "output/helpers",
    "output/server.py",
    "output/setup.sh",
)
_BUILD_TIMEOUT = 3600  # seconds per rebuild


@dataclass
class RebuildResult:
    """Outcome of one portable rebuild."""

    name: str
    ok: bool
    failing_step: str = ""
    cause: str = ""
    log_tail: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ok": self.ok,
            "failing_step": self.failing_step,
            "cause": self.cause,
            "elapsed_sec": round(self.elapsed_sec, 1),
            "log_tail": self.log_tail,
        }


# ── failure classification (pure) ────────────────────────────────────────────

# Ordered (signature substrings, all lower-case) → (failing_step, cause). First
# match wins, so the specific transcript-portability signatures come before the
# generic ones. Signatures are kept narrow: broad tokens like "cairo"/"apt-get"
# also appear in the base image's own (successful) apt step and would false-match.
_FAILURE_SIGNATURES: tuple[tuple[tuple[str, ...], str, str], ...] = (
    # `uv venv` doesn't seed pip, but a transcript calls `.venv/bin/pip` directly.
    (("bin/pip: no such file", "no module named pip"), "setup.sh", "venv-missing-pip"),
    # intra-script cwd drift: a relative `cd .alembic/...` resolved against the
    # wrong dir after an earlier cd.
    ((" cd: ", "cd: .alembic"), "setup.sh", "cwd-drift"),
    (
        ("no matching distribution", "could not find a version"),
        "setup.sh",
        "dependency-unresolvable",
    ),
    (
        (
            "failed building wheel",
            "error: subprocess-exited-with-error",
            "metadata-generation-failed",
        ),
        "setup.sh",
        "wheel-build-failed",
    ),
    (
        (
            "pygraphviz",
            "no-build-isolation",
            "error: command 'gcc'",
            "error: command 'cc'",
        ),
        "setup.sh",
        "native-build-dep",
    ),
    (
        (
            "could not resolve host",
            "temporary failure in name resolution",
            "connection reset",
        ),
        "setup.sh",
        "network",
    ),
    (("e: unable to locate package",), "apt", "apt-package"),
    # generic fallback: a command exited 127 without a more specific signature.
    (("non-zero code: 127", "command not found"), "setup.sh", "command-not-found"),
)


def classify_failure(build_log: str) -> tuple[str, str]:
    """Best-effort (failing_step, cause) from a docker build log; ("", "") clean."""
    low = build_log.lower()
    for needles, step, cause in _FAILURE_SIGNATURES:
        if any(n in low for n in needles):
            return step, cause
    return ("build", "unknown")


def _tail(text: str, n: int = 40) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


# ── docker steps (injectable) ────────────────────────────────────────────────


def extract_artifacts(name: str, dest_root: Path, *, runner=subprocess.run) -> Path:
    """Copy the venv-free artefacts out of ``alembic-tool:<name>`` into a clean
    build context under ``dest_root``. Returns the context dir.

    The context contains only ``.alembic/<name>/{repos,output/{tools,helpers,
    server.py,setup.sh}}`` plus the ``docker/alembic`` serve files — **never a
    venv** — so the rebuild genuinely starts from low-weight artefacts.
    """
    ctx = dest_root / name
    (ctx / ".alembic" / name / "output").mkdir(parents=True, exist_ok=True)
    # serve.Dockerfile + its runtime helpers must be in the context.
    shutil.copytree(
        PROJECT_ROOT / "docker" / "alembic",
        ctx / "docker" / "alembic",
        dirs_exist_ok=True,
    )

    created = runner(
        ["docker", "create", f"alembic-tool:{name}"],
        capture_output=True,
        text=True,
        check=True,
    )
    cid = (created.stdout or "").strip()
    try:
        for rel in ARTEFACT_PATHS:
            src = f"{cid}:/work/.alembic/{name}/{rel}"
            dst = ctx / ".alembic" / name / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # A missing optional artefact (e.g. helpers) must not abort extraction.
            runner(
                ["docker", "cp", src, str(dst)],
                capture_output=True,
                text=True,
                check=False,
            )
    finally:
        runner(["docker", "rm", cid], capture_output=True, text=True, check=False)
    return ctx


def rebuild_tool(
    name: str,
    context_dir: Path,
    *,
    runner=subprocess.run,
    tag_prefix: str = "rebuild-test",
) -> RebuildResult:
    """Rebuild one tool from its context via serve.Dockerfile; classify the outcome."""
    started = time.monotonic()
    proc = runner(
        [
            "docker",
            "build",
            "-f",
            str(context_dir / "docker" / "alembic" / "serve.Dockerfile"),
            "--build-arg",
            f"REPO_NAME={name}",
            "-t",
            f"{tag_prefix}:{name}",
            str(context_dir),
        ],
        # The legacy builder prints "Step N/M" progress to stdout already, and
        # rejects BuildKit-only flags like --progress; keep the argv portable.
        capture_output=True,
        text=True,
        check=False,
        timeout=_BUILD_TIMEOUT,
    )
    elapsed = time.monotonic() - started
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    ok = proc.returncode == 0
    step, cause = ("", "") if ok else classify_failure(log)
    return RebuildResult(
        name=name,
        ok=ok,
        failing_step=step,
        cause=cause,
        log_tail="" if ok else _tail(log),
        elapsed_sec=elapsed,
    )
