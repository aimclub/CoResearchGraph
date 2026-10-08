"""Single source of truth for models, timeouts, and loop caps.

Everything tunable lives here so there is exactly one place to look. This
module imports nothing from the rest of the package, so any module may import
it at top level without risking a circular import.
"""
from __future__ import annotations

import os

# ── Model ───────────────────────────────────────────────────────────────────
# One knob for every agent. Default is the qwen dev model; the top-quality
# benchmark run sets MODEL=openrouter/z-ai/glm-5.2. Sampling params are passed
# to the provider ONLY when explicitly set (leaving them unset = provider
# default, which avoids the temperature-0 repetition loops observed on qwen).
# Determinism comes from the deterministic gates, not from pinning the sampler.
MODEL             = os.environ.get("MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
MODEL_TEMPERATURE = os.environ.get("MODEL_TEMPERATURE")  # None => unset
MODEL_TOP_P       = os.environ.get("MODEL_TOP_P")

# Optional target-task spec(s) (TM-Bench mode). JSON/YAML text, or a path (or
# comma/':'-separated paths) to task files, each {name, description, arguments,
# returns, example, ...}. A list runs several tasks against ONE repo (the STAMP
# dual-task case). Unset => native autonomous mode. ALEMBIC_TARGET_TASK is the
# old single-task spelling, still honoured.
TASKS = os.environ.get("ALEMBIC_TASKS") or os.environ.get("ALEMBIC_TARGET_TASK")

# Optional SOFT hint(s) for the explorer only (env: ALEMBIC_HINTS). Free-form
# text describing the kind of tool(s) the caller hopes to see mined among the
# others — NO forced name/signature and NO plan-gate enforcement, unlike TASKS.
# Composable with TASKS (required tools stay pinned; the hint just steers the
# rest). Unset => explorer proposes fully autonomously.
HINTS = os.environ.get("ALEMBIC_HINTS")

APP_NAME = "alembic_app"
USER_ID  = "user_1"

STAGES = ("explorer", "environment", "coder", "validator", "wrapper")

# ── Stage retry / repair budgets (R1: the target is the focus) ───────────────
# A stage whose exit gate fails is rolled back to its start-of-stage checkpoint
# and rerun with a short note about the previous failure.
STAGE_RESET      = int(os.environ.get("STAGE_RESET", "2"))       # extra loops per stage
DEBUGGING_ROUNDS = int(os.environ.get("DEBUGGING_ROUNDS", "10"))  # validator repair cap

# ── Per-stage wall-clock budgets (seconds) — OFF by default (R1) ─────────────
# Set ALEMBIC_TIMEOUT_<STAGE>=<seconds> to enable one. Loop breakers and
# per-command subprocess timeouts below still bound runaway behaviour.
STAGE_TIMEOUT: dict[str, int | None] = {
    s: int(v) if (v := os.environ.get(f"ALEMBIC_TIMEOUT_{s.upper()}")) else None
    for s in STAGES
}

# When a stage timeout IS set, reserve the last 15% of it to force a
# "write what you have" wrap-up before the hard timeout cancels everything.
REPORT_GRACE_FRACTION = 0.85

# ── Other wall-clock timeouts ─────────────────────────────────────────────────
BASH_TIMEOUT                = 15    # quick reads (ls/grep/head)
BASH_ENV_TIMEOUT            = 900   # slow installs / weight downloads
VENV_SETUP_TIMEOUT          = 600   # a single setup_venv subprocess
VENV_COMPAT_TIMEOUT         = 240   # compat-check script
IMPORT_CHECK_TIMEOUT        = 60    # per-file import check (heavy ML imports are slow)
TEST_TIMEOUT                = 300   # per-tool pytest run incl. test_invoc_ (R6); 5 min
INVOKE_TIMEOUT              = 120   # a single live function invocation (R6)
DEBUGGER_CALL_TIMEOUT       = 600   # one debugger round-trip
WRAPPER_CALL_TIMEOUT        = 600   # the fallback wrapper-agent round-trip

# ── Loop breakers ─────────────────────────────────────────────────────────────
MAX_STEPS         = 120   # hard ceiling on events per agent turn
MAX_TOOL_REPEATS  = 3     # abort on N identical consecutive tool calls
MAX_TOOL_CYCLE    = 3     # abort on N identical NON-consecutive calls (set-cycling)
MAX_GUARD_RETRIES = 3     # re-nudge an agent that missed write_report
MAX_TRANSIENT_FAULT_RETRIES = 2  # retry a silent provider fault, off-budget

# ── LLM request resilience (OpenRouter faults: 403 policy blocks, 5xx, timeouts) ─
# A transient provider error must not dump a stacktrace and kill the stage — the
# request is logged as a one-liner, backed off, and retried. We optimise for the
# best result, so by default there is NO retry cap (unbounded until it succeeds);
# each request still has its own timeout, so a hung call is retried, not stuck.
# Set ALEMBIC_LLM_RETRY_CAP=<n> to bound it (0 = fail fast, no retries).
LLM_REQUEST_TIMEOUT  = int(os.environ.get("ALEMBIC_LLM_REQUEST_TIMEOUT", "600"))  # per call
LLM_RETRY_BASE_DELAY = float(os.environ.get("ALEMBIC_LLM_RETRY_BASE_DELAY", "8"))  # 1st backoff
LLM_RETRY_MAX_DELAY  = float(os.environ.get("ALEMBIC_LLM_RETRY_MAX_DELAY", "90"))  # backoff ceiling
LLM_RETRY_CAP        = int(v) if (v := os.environ.get("ALEMBIC_LLM_RETRY_CAP")) else None

# Tools whose own instructions require repeated identical-arg calls — exempt
# from the non-consecutive cycle breaker but not the consecutive one.
TOOL_CYCLE_EXEMPT = frozenset({"check_venv_compat", "run_tool_tests"})

# ── Tool-selection caps (Plan gate) ───────────────────────────────────────────
MAX_TOOLS = 12   # hard cap on tools exposed per repo (matches ToolRosella)

# ── Output size caps ──────────────────────────────────────────────────────────
MAX_BYTES              = 40_000   # stdout/stderr text shown to the LLM
RESULT_MAX_LIST_ITEMS  = 20       # cap list length in a successful tool result
RESULT_MAX_STR_LEN     = 2_000    # cap string length in a successful tool result

# Sentinel that separates a function's real stdout from its JSON result.
RESULT_SENTINEL = "<<<ALEMBIC_RESULT>>>"
