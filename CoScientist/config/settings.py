"""
Application configuration using Pydantic Settings.
"""
import os as _os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Union

from dotenv import find_dotenv as _find_dotenv, load_dotenv as _load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_load_dotenv(_find_dotenv())

from rag_tools.config import Settings as ToolRAGSettings


ROOT_DIR = Path(__file__).parent.parent.absolute()


# =========================
# LLM CONFIG
# =========================
class LLMSettings(BaseModel):
    allowed_providers: List[str] = ["google-vertex", "azure"]

    service_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    main_url: Optional[str] = None
    scenario_url: Optional[str] = None
    main_model: Optional[str] = None
    scenario_model: Optional[str] = None

    # Dedicated model for the CoderAgent (a stronger model handles its multi-step
    # engineering / tool-use better). Falls back to main_model if unset. The
    # provider prefix in the model string (e.g. "openrouter/...") selects the
    # endpoint, so no separate URL is needed.
    coder_model: Optional[str] = None

    # Dedicated model for NirReportAgent. Writing a GOST 7.32-2017 report is a
    # long-form authoring job over a large evidence base, which is a different
    # workload from the short Markdown the aggregator assembles — so it gets its
    # own model instead of raising `reasoning` for every run. Falls back to
    # main_model when unset, which is exactly today's behaviour.
    nir_model: Optional[str] = None

    # A small, cheap model for the "view summary" button on an agent in the
    # execution log: it reads one agent's trace and writes a few lines on what
    # was done, with which tools, and what came out. Called only on request,
    # so a weak model is the right one. Falls back to the model half of
    # `summary_url` ("base;model"), then to main_model. LLM__AGENT_SUMMARY_MODEL.
    agent_summary_model: Optional[str] = None

    # Seconds to wait for a single completion before giving up. Without this a
    # provider that accepts the connection and then goes quiet never raises, so
    # the agent waits forever and the run looks frozen with nothing in the log.
    # A timeout turns that silence into a retryable error. Override with
    # LLM__REQUEST_TIMEOUT.
    request_timeout: int = 600

    openrouter_provider: Optional[str] = None
    openrouter_provider_sort: Optional[str] = None
    openrouter_provider_order: Optional[str] = None

    service_url: Optional[str] = None
    service_cc_url: Optional[str] = None

    vision_url: Optional[str] = None
    summary_url: Optional[str] = None
    marker_model: Optional[str] = None
    datasets_url: Optional[str] = None
    deepeval_url: Optional[str] = None


# =========================
# LLM CONFIG
# =========================
class ServicesSettings(BaseModel):
    tavily_api_key: Optional[str] = None
    openalex_api_key: Optional[str] = None
    openalex_email: Optional[str] = None
    proxy_url: Optional[str] = None


# =========================
# MED LLM CONFIG
# =========================
class MedLLMSettings(BaseModel):
    task_url: Optional[str] = None
    result_url: Optional[str] = None

    login: Optional[str] = None
    password: Optional[str] = None

    poll_interval: int = 10
    max_polls: int = 60

# =========================
# STORAGE
# =========================
class StorageSettings(BaseModel):
    root_dir: Path = ROOT_DIR

    parse_results: Optional[str] = None
    chroma_storage: Optional[str] = None
    papers_storage: Optional[str] = None
    ds_storage: Optional[str] = None
    img_storage: Optional[str] = None
    another_storage: Optional[str] = None
    memory_db: Optional[str] = None

    path_to_data: Optional[str] = None
    path_to_cvae_checkpoint: Optional[str] = None
    path_to_results: Optional[str] = None
    path_to_temp_files: Optional[str] = None
    my_papers: Optional[str] = None

    logging_path: Optional[str] = 'logs/'


# =========================
# HOSTS & PORTS
# =========================
class HostsPortsSettings(BaseModel):
    chroma_host: Optional[str] = None
    embedding_host: Optional[str] = None
    reranker_host: Optional[str] = None
    opencChemie_host: Optional[str] = None
    chem_services_host: Optional[str] = None
    retrosynthesis_services_host: Optional[str] = None

    chroma_port: Optional[str] = None
    embedding_port: Optional[str] = None
    reranker_port: Optional[str] = None
    opencChemie_port: Optional[str] = None
    chem_services_port: Optional[str] = None
    retrosynthesis_services_port: Optional[str] = None
  

# =========================
# COLLECTIONS
# =========================
class CollectionsSettings(BaseModel):
    summaries: Optional[str] = None
    texts: Optional[str] = None
    images: Optional[str] = None


# =========================
# S3
# =========================
class S3Settings(BaseModel):
    use_s3: bool = False
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket_name: Optional[str] = None
    # Lifetime of presigned artifact URLs, in seconds. 604800 (7 days) is the
    # SigV4 maximum. The default of the S3 client (360 s) expires before the
    # operator opens the report. Override via S3__PRESIGN_TTL.
    presign_ttl: int = 604800


# =========================
# ARTIFACTS (the session's own copy of what the run produced)
# =========================
class ArtifactsSettings(BaseModel):
    """Mirroring tool output into the session directory.

    A tool hands back a presigned link to its own storage, and one measured on a
    live run was valid for six minutes. Storing that string is how a figure
    becomes unreachable. Mirroring copies the bytes here while the link works.

    The caps exist because this writes to disk on every tool call. What exceeds
    one is recorded as a skipped artifact with a reason — never dropped quietly.
    """

    #: Master switch. Off means the run behaves exactly as it did before.
    enabled: bool = True
    #: Per file. Bigger than this is recorded as ``skipped/oversize``.
    max_file_mb: int = 100
    #: Per session, across every mirrored file.
    max_session_mb: int = 2048
    max_files_per_session: int = 500
    #: One fetch of one artifact. The link may already be dead; do not hang.
    download_timeout: int = 60
    #: Also push a copy to our S3, giving the artifact a second durable address
    #: that works from another host. Best effort — the local copy is the home.
    mirror_to_s3: bool = True


# =========================
# OPIK
# =========================
class OpikSettings(BaseModel):
    # Master switch for Opik tracing (env: OPIK__ENABLED). Off by default so the
    # app never ships spans to a (possibly rate-limited) Opik backend unless
    # explicitly opted in. When False, tracing is fully disabled process-wide.
    enabled: bool = False
    api_key: Optional[str] = None
    url_override: Optional[str] = None
    opik_project_name: Optional[str] = None


# =========================
# MCP
# =========================
class MCPSettings(BaseModel):
    paper_analysis_url: Optional[str] = None
    papers_search_url: Optional[str] = None
    moosechem_url: Optional[str] = None
    result_formatter_url: Optional[str] = None
    # The file vault (mcp-servers/vault-mcp-server). Two consumers read it:
    # worker agents get the upload/download pair as an ADK toolset, and
    # framework code calls it per request through tools/vault_client.py.
    # Unset means both drop out, and the run still completes.
    vault_url: Optional[str] = None

    # The "Автонормоконтроль" MCP (GOST 7.32-2017 NIR report rendering).
    # DELIBERATELY no default: the address is a moving target — the ITMO
    # instance and a locally hosted one swap places — and a literal in the code
    # would outlive whichever is current. Set MCP__NORMCONTROL_URL in .env;
    # unset means the NIR tools drop out and the run completes as before.
    normcontrol_url: Optional[str] = None


# =========================
# NIR report (GOST 7.32-2017, via the normcontrol MCP)
# =========================
class NIRSettings(BaseModel):
    """The GOST NIR report stage, off until an operator turns it on.

    An explicit flag rather than an inference from "is normcontrol_url set?":
    a reachable server is not a reason to spend a strong model on a 30-page
    document, so the choice stays the operator's. Both conditions together are
    :attr:`Settings.nir_ready`, which is what the agent is actually gated on.
    """

    enabled: bool = False
    # draft tolerates `<...>` placeholders and temporary pagination; production
    # additionally demands a real page_count and a complete page_map, which
    # nothing can supply before the DOCX has been laid out. Kept configurable so
    # a future two-pass flow can switch it without touching the call sites.
    mode: str = "draft"


# =========================
# HITL (Human-in-the-Loop)
# =========================
class HITLSettings(BaseModel):
    # Human-in-the-loop approval for outward-facing / hard-to-reverse actions.
    # OFF by default: a one-prompt autonomous run cannot pause for console
    # approval (ConsoleHITLHandler blocks on input(), which would hang a web/
    # headless run). The hard safety blocklist in coder_tools (_BLOCKED: rm -rf /,
    # mkfs, fork bombs, …) still refuses genuinely dangerous commands regardless.
    # Re-enable for interactive/supervised runs via HITL__ENABLED=true.
    enabled: bool = False

# =========================
# CONTEXT INITIALIZATION
# =========================
class ContextInitSettings(BaseModel):
    """The pre-stage that drafts the research frame, confirms it with the
    operator (structured web form, when HITL is on) and seeds it into the
    research graph before the orchestrator runs.

    ``enabled`` gates the whole pre-stage — referenced from system.yaml as
    ``${context_init.enabled}``. The gate is soft: the operator may submit the
    form with fields deferred (the agent fills working values), so a run never
    blocks indefinitely. Override via RESEARCH_FRAME (or CONTEXT_INIT__ENABLED).
    """
    enabled: bool = _os.getenv("RESEARCH_FRAME", _os.getenv("CONTEXT_INIT__ENABLED", "true")).lower() in ("true", "1", "yes")

    @model_validator(mode="before")
    @classmethod
    def _read_research_frame(cls, data):
        rf = _os.getenv("RESEARCH_FRAME")
        if rf is not None:
            val = rf.lower() in ("true", "1", "yes")
            if isinstance(data, dict):
                data["enabled"] = val
            elif data is None:
                data = {"enabled": val}
        return data


# =========================
# ORCHESTRATOR
# =========================
class OrchestratorSettings(BaseModel):
    # Whether the orchestrator uses the PlannerAgent (referenced from
    # system.yaml as ${orchestrator.use_planner}). When False, the planner is
    # not attached and the orchestrator prompt's planning step adapts — the
    # assembler keeps prompt and tools consistent automatically.
    #
    # Kept in sync with PlannerAgent.enabled in system.yaml: the planner agent
    # is disabled (it planned worse than the orchestrator coordinating inline),
    # so this is False too. With True while the agent is disabled, the
    # orchestrator prompt stays in TASK_MANAGEMENT mode expecting a plan nobody
    # creates -> it hammers update_task_status on phantom task ids and gives up.
    use_planner: bool = False

    # Shared provider-attempt quota, NOT an independent allowance for every
    # nested ADK runner. Only an explicit operator decision grants another quota.
    # Opaque remote agents have their own accounting and are outside this gate.
    max_llm_calls: int = Field(default=100, ge=1, le=100)

    @field_validator("max_llm_calls", mode="before")
    @classmethod
    def _clamp_legacy_llm_budget(cls, value):
        """Old environments used 3000; load them safely without breaking boot."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            return value
        if number > 100:
            import logging
            logging.getLogger(__name__).warning(
                "ORCHESTRATOR__MAX_LLM_CALLS=%s exceeds the shared quota; using 100", number,
            )
            return 100
        return value

# =========================
# CODE EXECUTION
# =========================
class CodeExecSettings(BaseModel):
    """Settings for the remote code-execution MCP server used by the CoderAgent.

    The server is expected to expose an HTTP API:
      - POST {submit_url}  with JSON {command, workspace_id, timeout} -> {job_id}
      - GET  {result_url}?job_id=... -> {status, stdout, stderr, exit_code}
    When `url` is empty the CoderToolset falls back to local subprocess execution.
    """
    url: Optional[str] = None             # base url of the code-exec MCP server
    submit_path: str = "/submit"
    result_path: str = "/result"
    poll_interval: int = 5                # seconds between status polls
    default_timeout: int = 7200           # per-command timeout (s) — big enough for
                                          # training/optimization (server may cap it;
                                          # checkpoint long work to disk regardless)
    exec_wait: int = 300                  # how long execute_bash waits inline for
                                          # the command to finish before handing
                                          # back a job_id — so the model gets the
                                          # result in ONE call instead of polling
    check_wait: int = 600                 # how long check_job blocks inline for a
                                          # running job before returning. This is an
                                          # async poll loop (no LLM round-trips), so a
                                          # single check_job waits up to 10 min for a
                                          # long job — the key to letting a long
                                          # autonomous run wait patiently instead of
                                          # burning dozens of polling turns.
    workspace_root: str = "./workspace"   # per-session sandbox root (local fallback)

# =========================
# WEB / RUNTIME SETTINGS
# =========================
import os as _os
from typing import Optional as _Optional

def _default_openrouter_provider_sort() -> str:
    explicit_sort = _os.getenv("OPENROUTER_PROVIDER_SORT", _os.getenv("LLM__OPENROUTER_PROVIDER_SORT", "")).strip().lower()
    if explicit_sort:
        return explicit_sort
    general = _os.getenv("OPENROUTER_PROVIDER", _os.getenv("LLM__OPENROUTER_PROVIDER", "")).strip().lower()
    if general in ("price", "throughput", "latency"):
        return general
    return "default"

def _default_openrouter_provider_order() -> str:
    explicit_order = _os.getenv("OPENROUTER_PROVIDER_ORDER", _os.getenv("LLM__OPENROUTER_PROVIDER_ORDER", "")).strip()
    if explicit_order:
        return explicit_order
    general = _os.getenv("OPENROUTER_PROVIDER", _os.getenv("LLM__OPENROUTER_PROVIDER", "")).strip()
    if general.lower() not in ("price", "throughput", "latency", "default", "auto", "none", ""):
        return general
    return ""

class WebSettings(BaseModel):
    """Runtime-tunable parameters configurable from the web UI.

    Unlike the rest of Settings (loaded once from .env), these can be
    mutated at runtime via ``/api/settings``.  The global ``settings``
    singleton is the single source of truth — all components read from it
    directly.
    """
    openrouter_provider_sort: str = _default_openrouter_provider_sort()   # "default" | "price" | "throughput" | "latency"
    openrouter_provider_order: str = _default_openrouter_provider_order() # e.g. "Together, DeepInfra" or empty
    start_mode: str = _os.getenv("START_MODE", "planner")             # "init" | "planner" | "orchestrator" | "orchestrator_planner"
    max_searches: int = int(_os.getenv("RESEARCH_AGENT_SEARCHES", "2"))           # WebSearchLimiter per-turn cap
    max_retries: int = int(_os.getenv("LLM_MAX_RETRIES", "3"))
    hitl_enabled: bool = _os.getenv("HITL__ENABLED", "false").lower() in ("true", "1", "yes")
    # One knob for every confirmation in the system: "auto" never asks, "basic"
    # waits ten minutes and treats silence as a REFUSAL, "debug" waits for the
    # human however long it takes. See CoScientist/hitl/mode.py for why silence
    # approves in none of them. Empty means "derive it from the two settings
    # below", so an existing stand keeps the behaviour it had.
    hitl_mode: str = _os.getenv("HITL__MODE", "").strip().lower()
    hitl_auto_approve_timeout: int = int(_os.getenv("HITL_AUTO_APPROVE_TIMEOUT", _os.getenv("HITL__AUTO_APPROVE_TIMEOUT", _os.getenv("HITL_TIMEOUT_SECONDS", "300"))))
    scope_hitl: bool = _os.getenv("ORCHESTRATOR__SCOPE_HITL", "false").lower() in ("true", "1", "yes")
    # Work Order: executor agents declare a contract (goal, assumptions, steps,
    # tools, side effects) before acting. Inert unless HITL is on.
    work_order_enabled: bool = _os.getenv("WORK_ORDER__ENABLED", "true").lower() in ("true", "1", "yes")
    # Veto window for compute-tier contracts: auto-approved after this many seconds.
    # -1 (default) disables auto-approval — the contract waits for the human.
    work_order_veto_seconds: int = int(_os.getenv("WORK_ORDER__VETO_SECONDS", "-1"))
    # Amendments per agent run before every further one needs a blocking review.
    work_order_max_amendments: int = int(_os.getenv("WORK_ORDER__MAX_AMENDMENTS", "3"))
    use_planner: bool = _os.getenv("ORCHESTRATOR__USE_PLANNER", "true").lower() in ("true", "1", "yes")
    planner_retrieval_enabled: bool = _os.getenv("PLANNER__RETRIEVAL_ENABLED", "true").lower() in ("true", "1", "yes")
    planner_graph_enabled: bool = _os.getenv("PLANNER__GRAPH_ENABLED", "true").lower() in ("true", "1", "yes")
    planner_critic_enabled: bool = _os.getenv("PLANNER__CRITIC_ENABLED", "false").lower() in ("true", "1", "yes")
    planner_critic_rounds: int = int(_os.getenv("PLANNER__CRITIC_ROUNDS", "1"))
    knowledge_graph_enabled: bool = _os.getenv("GRAPH__ENABLED", "true").lower() in ("true", "1", "yes")
    auto_clear_graph_enabled: bool = _os.getenv("GRAPH__AUTO_CLEAR", "false").lower() in ("true", "1", "yes")
    executor_tool_keep_score: float = float(_os.getenv("EXECUTOR_TOOL_KEEP_SCORE", "0.3"))
    executor_tool_abstain_score: float = float(_os.getenv("EXECUTOR_TOOL_ABSTAIN_SCORE", "0.2"))
    # The main profile's FEDOT.MAS reranker fallback (ExecutorSwitchAgent). The
    # Experiment Module's route decisions do not read it: their switch is
    # EXPERIMENTS__ROUTE_FEDOT.
    fedot_fallback_enabled: bool = _os.getenv("EXECUTOR__FEDOT_FALLBACK", "false").lower() in ("true", "1", "yes")
    # The clinical specialist: PubMed/PICO, study taxonomy and DICOM. A narrow
    # role, and a study that needs none of it pays for the agent in the
    # orchestrator's roster and in the router's choices — so it switches off.
    # Turning it off also withdraws `medical` as an execution route, because a
    # route whose agent is not in the tree is a route that cannot run: the
    # experiment planner is not offered it, the critique refuses it, and a task
    # already planned on it is blocked. Also a toggle in the web settings.
    medical_agent_enabled: bool = _os.getenv("MEDICAL__ENABLED", "true").lower() in ("true", "1", "yes")
    # The hypothesis generator's thinking budget. `high` by default because
    # ideation plus choosing what to test first is the most reasoning-bound job
    # in the system; turn it down per run with HYPOTHESES__REASONING=low when
    # the hypothesis is expected to be straightforward and the run is paying
    # for latency.
    hypotheses_reasoning: str = _os.getenv("HYPOTHESES__REASONING", "high")
    fedot_fallback_timeout_s: float = float(_os.getenv("EXECUTOR__FEDOT_FALLBACK_TIMEOUT", "900"))
    # MCP hub (Docker Hub, credentials in .env): searched before a build, optional auto-upload.
    alembic_hub_search_enabled: bool = _os.getenv("ALEMBIC_HUB__SEARCH_ENABLED", "true").lower() in ("true", "1", "yes")
    alembic_hub_auto_upload: bool = _os.getenv("ALEMBIC_HUB__AUTO_UPLOAD", "false").lower() in ("true", "1", "yes")
    # Whether an agent may start a conversion itself; the builds page always can.
    alembic_agent_build_enabled: bool = _os.getenv("ALEMBIC__AGENT_BUILD_ENABLED", "false").lower() in ("true", "1", "yes")
    sandbox_url: str = _os.getenv("SANDBOX_URL", "")
    coder_workspace_id: _Optional[str] = _os.getenv("CODER_WORKSPACE_ID")
    coder_mode: str = _os.getenv("CODER__MODE", "local")        # "local" | "openhands"
    merge_tasks_enabled: bool = _os.getenv("PLANNER__MERGE_TASKS", "true").lower() in ("true", "1", "yes")
    max_active_hypotheses: int = int(_os.getenv("HYPOTHESES__MAX_ACTIVE", "1"))
    # A node's write-up is offered by a button on every reportable card; this
    # decides whether a settled card also asks for one on its own. On by
    # default, and the browser may overrule it for one reader.
    node_report_auto: bool = _os.getenv(
        "NODE_REPORT__AUTO", "true").lower() in ("true", "1", "yes")
    use_proxy: bool = _os.getenv("USE_PROXY", "True").lower() in ("true", "1", "yes")
    opik_enabled: bool = _os.getenv("OPIK__ENABLED", "false").lower() in ("true", "1", "yes")
    auto_naming_enabled: bool = _os.getenv("AUTO_NAMING__ENABLED", "true").lower() in ("true", "1", "yes")
    # Default of the per-browser "Show internal agents and tools" switch. A
    # browser that flipped the switch keeps its own choice.
    show_internal_enabled: bool = _os.getenv("SHOW_INTERNAL__ENABLED", "false").lower() in ("true", "1", "yes")
    # Side-nav call graph: how long a group of parallel calls stays unfolded
    # after a click before it folds back.
    call_graph_collapse_seconds: int = int(_os.getenv("CALL_GRAPH__COLLAPSE_SECONDS", "10"))
    coscientist_username: _Optional[str] = _os.getenv("COSCIENTIST_USERNAME") or _os.getenv("DEFAULT_USERNAME")
    context_init_enabled: bool = _os.getenv("RESEARCH_FRAME", "true").lower() in ("true", "1", "yes")
    session_snapshots_dir: str = _os.getenv("SESSION_SNAPSHOTS_DIR", "session_snapshots")


# =========================
# RESEARCH CONTEXT GRAPH
# =========================
class ResearchGraphSettings(BaseModel):
    """The typed research blackboard agents write to (graph/research/).

    Distinct from the auto-recorded execution graph (graph/*). When enabled=False
    the research tools and prompt sections drop out entirely (the assembler makes
    the whole feature vanish, prompts stay consistent). Override via
    RESEARCH_GRAPH__ENABLED etc.
    """
    enabled: bool = True
    dir: str = "./graph_runs"              # snapshot directory (shared with graph_runs)
    active_file: str = "research_active.json"
    slice_depth_max: int = 2               # cap on get_context_slice depth
    slice_char_budget: int = 4000          # cap on a rendered context slice
    context_char_budget: int = 4000        # cap on the orchestrator trigger digest
    # A research spans many prompts, so browser refresh and Web Stop never wipe
    # it. ``reset_session_state(..., reset_research=None)`` consults this flag
    # when an explicit maintenance reset is requested. A new session id already
    # resolves to a separate empty graph.
    reset_on_session: bool = False


# =========================
# EXPERIMENT MODULE (v0)
# =========================
class ExperimentsSettings(BaseModel):
    """Settings for the isolated Experiment Module profile.

    Values are read through the main ``Settings`` object, so the canonical
    environment names use the nested ``EXPERIMENTS__*`` form.
    """

    # The one FEDOT.MAS switch of the Experiment Module (EXPERIMENTS__ROUTE_FEDOT).
    # experiments.yaml attaches FedotAgent on it, and every route decision -
    # the planner prompt, its context, the critique, start_task, fallback and
    # the Alembic post-build route - asks state_machine.fedot_route_available,
    # which also requires FedotAgent to be listed under ExperimentExecutorAgent.
    # Off by default: FEDOT.MAS is not reliable enough to be a default route,
    # and ReAct over the bound MCP tools (react_tools) covers the same tasks.
    # EXECUTOR__FEDOT_FALLBACK is a different switch: the main profile's
    # reranker fallback.
    route_fedot: bool = False
    route_coder_mcp: bool = False
    route_alembic: bool = False
    # Which side of the Coder/Alembic fork a task that reuses a repository
    # takes when nobody answers the review (HITL mode `auto`, or a timeout):
    # "coder" runs the code directly, "alembic_build" wraps it as an MCP tool
    # first. Coder stays the default because it skips the container build;
    # a study that exists to leave a reusable tool behind sets
    # EXPERIMENTS__ALEMBIC_ROUTE_DEFAULT=alembic_build.
    alembic_route_default: Literal["coder", "alembic_build"] = "coder"
    task_max_attempts: int = Field(default=2, ge=1, le=2)
    # Cumulative across routes and automatic replans of the same operation.
    task_max_total_attempts: int = Field(default=3, ge=1, le=10)
    max_recovery_discovery_rounds: int = Field(default=2, ge=0, le=2)
    control_same_failure_limit: int = Field(default=3, ge=2, le=5)
    control_no_progress_limit: int = Field(default=5, ge=2, le=10)
    max_plan_tasks: int = Field(default=8, ge=1, le=20)
    # How many times a rejected result review may send the module back to
    # planning. `task_max_attempts` bounds retries of ONE task; nothing used to
    # bound redoing the whole plan, and a stale phase turned that into an
    # unbounded loop (observed 2026-09-01: five plans in one run, the first of
    # which had already finished every task successfully). 0 disables replanning
    # entirely; the cap is counted across the whole experiment run.
    max_replan_rounds: int = Field(default=1, ge=0, le=5)
    # Outer hops after a result-review reject. Skip planner at replan_count >=
    # this. The dispatch budget in coalesce.py reads this one because it lives on
    # the orchestrator State, i.e. it survives the AgentTool boundary, unlike the
    # counter kept inside the runtime.
    max_replans: int = Field(default=2, ge=1, le=8)
    # Inner schema/critique regenerations of ExperimentPlan within one planner
    # hop, counted as CONSECUTIVE failures and reset on every plan that
    # validates. Was 8 hardcoded, which is up to six wasted rounds on a costly
    # planner; 2 proved too tight once a human HITL edit re-entered planning, so
    # this leaves room for one human round plus a couple of genuine planner
    # mistakes without letting a broken plan burn eight planner calls.
    max_plan_revisions: int = Field(default=4, ge=1, le=8)
    # Which FEDOT engine backs the fedot_mas route.
    #
    # "mas" (default) is the single-shot routing config. "maw" is a fixed
    # Sequential/Parallel/Loop pipeline whose config is designed in two calls -
    # an agent pool, then the pipeline tree. MAW was tried as a fix for config
    # generation failures and measured WORSE on the same ask (2026-09-02): 3 of
    # 11 pipelines completed against MAS's 25 of 46, and 4010s against 499s.
    # The reason is that the dominant failure is not config size but the model
    # not writing to output_key at all, so splitting the call into two just
    # doubles the places that can fail, each with its own 4-attempt retry.
    # Kept selectable because the pipeline shape is still the better model for
    # deterministic multi-step work once generation is reliable.
    fedot_engine: Literal["maw", "mas"] = "mas"
    require_task_design: bool = True
    # When True (default), schema invents baselines/metrics for weak planners so
    # completeness majors for unspecified/empty design cannot fire. Set False
    # (EXPERIMENTS__LENIENT_PLANNER=false) to preserve unspecified* sentinels.
    lenient_planner: bool = True
    # Route fallback chains after a failed attempt. Default: fedot → react → coder.
    # Override via EXPERIMENTS__FALLBACK_*. A route that is switched off is
    # skipped, so a chain never falls back into FEDOT.MAS while it is off.
    fallback_fedot_mas: list[str] = Field(
        default_factory=lambda: ["fedot_mas", "react_tools", "coder"]
    )
    fallback_react_tools: list[str] = Field(default_factory=lambda: ["react_tools", "coder"])
    fallback_coder: list[str] = Field(default_factory=lambda: ["coder"])
    fallback_alembic_build: list[str] = Field(
        default_factory=lambda: ["alembic_build", "coder"]
    )
    fallback_research: list[str] = Field(default_factory=lambda: ["research"])
    fallback_medical: list[str] = Field(default_factory=lambda: ["medical"])

    alembic_timeout_s: float = Field(default=1800.0, gt=0)
    alembic_poll_s: float = Field(default=5.0, gt=0)
    fedot_timeout_s: float = Field(default=600.0, gt=0)
    react_timeout_s: float = Field(default=600.0, gt=0)
    coder_timeout_s: float = Field(default=7200.0, gt=0)
    research_timeout_s: float = Field(default=600.0, gt=0)
    medical_timeout_s: float = Field(default=600.0, gt=0)
    # The two reviews that do NOT follow the global HITL switch: the module
    # asks for them even when "ask for my approval" is off, and a window that
    # runs out PAUSES the run instead of approving it (fail closed — an
    # experiment is never approved because nobody was watching). So these two
    # flags are the only way to let an experiment through unattended. Read at
    # call time, so the Approvals tab takes effect on the next review.
    plan_auto_approve: bool = False
    result_auto_approve: bool = False
    plan_review_timeout_s: float = Field(default=300.0, gt=0)
    result_review_timeout_s: float = Field(default=300.0, gt=0)
    complexity_warning_tasks: int = Field(default=6, ge=1, le=8)


# =========================
# CHECKPOINTS
# =========================
class CheckpointSettings(BaseModel):
    """Run-state snapshots at module boundaries (checkpoints/ package).

    OFF by default: enabling adds the CheckpointPlugin to every runner and a
    /api/checkpoints router to the A2A/web apps. Override via
    CHECKPOINTS__ENABLED / CHECKPOINTS__DIR. Management API additionally requires
    CHECKPOINTS__API_TOKEN; capture itself remains available without a token.
    """

    enabled: bool = False
    dir: str = "./checkpoints_data"
    api_token: Optional[SecretStr] = Field(default=None, min_length=32)


# =========================
# SYNAPSE v1 ADAPTER
# =========================
class SynapseSettings(BaseModel):
    """Synapse platform v1 contract bridge (checkpoints/synapse.py)."""

    enabled: bool = False
    callback_url: Optional[str] = None
    bundle_base_url: Optional[str] = None
    otlp_endpoint: Optional[str] = None


# =========================
# BLIND REVIEW GUARD
# =========================
class BlindSettings(BaseModel):
    """Blind-review guard for validation runs with a known answer.

    A refuted claim is re-checked by the system; the guard keeps the outcome
    of that check out of what search tools return (agents/blind.yaml wires it
    onto the search-capable agents). Env: BLIND__ENABLED, BLIND__CUTOFF_YEAR,
    BLIND__BLOCKLIST (JSON list), BLIND__CLAIM, BLIND__JUDGE, BLIND__JUDGE_MODEL.
    """

    enabled: bool = False
    # Last publication year the run may read. OpenAlex queries get the filter;
    # other results are dropped when they carry a later date.
    cutoff_year: Optional[int] = None
    # Case-insensitive terms that identify the refutation: authors, titles,
    # method names, DOIs. A query naming one is refused; a result block naming
    # one is dropped or redacted.
    blocklist: List[str] = Field(default_factory=list)
    # One sentence naming the claim under review, for the judge.
    claim: str = ""
    # An LLM judge reads the blocks that passed the blocklist and drops those
    # that reveal the outcome of the check without using a listed term.
    judge: bool = True
    judge_model: Optional[str] = None  # falls back to llm.main_model
    judge_timeout: float = 45.0
    judge_max_items: int = 25
    judge_max_chars: int = 1500


# =========================
# CRITIC
# =========================
class CriticSettings(BaseModel):
    """Critic LLM callback parameters (pre-action, post-action, plan critic)."""
    timeout: float = 90.0
    http_timeout_ratio: float = 0.75
    max_attempts: int = 2
    max_tokens: int = 7000
    model: Optional[str] = None  # Dedicated model for the Critic callbacks; falls back to llm.main_model if unset
    # Model "thinking" for the critic, in system.yaml's vocabulary: False/"off",
    # or "minimal"|"low"|"medium"|"high". A verdict is a short judgement against
    # an explicit checklist, and reasoning tokens are spent from `max_tokens` —
    # thinking too hard truncates the JSON it was supposed to return. None
    # leaves the provider's default alone.
    reasoning: Optional[Union[bool, str]] = "low"

    @property
    def http_timeout(self) -> float:
        return self.timeout * self.http_timeout_ratio


# =========================
# AGENTS (per-agent overrides of system.yaml)
# =========================
class AgentOverride(BaseModel):
    """What the operator changed about one agent, over its system.yaml entry.

    Every field is optional: None means "as declared in the YAML". Read when a
    session's agent tree is assembled (schema.AgentConfig and the assembler),
    so a change reaches the next session, not the running one.
    """

    enabled: Optional[bool] = None
    # Same vocabulary as `reasoning:` in system.yaml: "off", or
    # "minimal" | "low" | "medium" | "high".
    reasoning: Optional[str] = None
    # "main" | "coder" | "nir" | a literal litellm model string.
    model: Optional[str] = None
    # The budget of the agent's limiter callback (registry.ToolLimit): search
    # calls or calls per tool, whichever that limiter counts.
    limit: Optional[int] = None


class AgentsSettings(BaseModel):
    """Per-agent overrides set from the web UI's Agents section.

    Environment: ``AGENTS__OVERRIDES`` holds the whole map as JSON, e.g.
    ``{"MedicalAgent": {"enabled": false}, "ResearchAgent": {"reasoning": "low"}}``,
    and ``AGENTS__DEFAULT_REASONING`` replaces `defaults.reasoning` of the
    profile for every agent that declares none of its own.
    """

    default_reasoning: Optional[str] = None
    overrides: dict[str, AgentOverride] = Field(default_factory=dict)


# =========================
# MAIN SETTINGS
# =========================
class Settings(BaseSettings):
    """Main application settings."""

    llm: LLMSettings = LLMSettings()
    services: ServicesSettings = ServicesSettings()
    med_llm: MedLLMSettings = MedLLMSettings()
    storage: StorageSettings = StorageSettings()
    hosts_ports: HostsPortsSettings = HostsPortsSettings()
    collections: CollectionsSettings = CollectionsSettings()
    s3: S3Settings = S3Settings()
    artifacts: ArtifactsSettings = ArtifactsSettings()
    opik: OpikSettings = OpikSettings()
    hitl: HITLSettings = HITLSettings()
    nir: NIRSettings = NIRSettings()
    context_init: ContextInitSettings = ContextInitSettings()
    orchestrator: OrchestratorSettings = OrchestratorSettings()
    code_exec: CodeExecSettings = CodeExecSettings()
    tool_rag: ToolRAGSettings = ToolRAGSettings()
    mcp: MCPSettings = MCPSettings()
    web: WebSettings = WebSettings()
    research_graph: ResearchGraphSettings = ResearchGraphSettings()
    experiments: ExperimentsSettings = ExperimentsSettings()
    checkpoints: CheckpointSettings = CheckpointSettings()
    synapse: SynapseSettings = SynapseSettings()
    critic: CriticSettings = CriticSettings()
    agents: AgentsSettings = AgentsSettings()
    blind: BlindSettings = BlindSettings()

    model_config = SettingsConfigDict(
        env_file=".env",          
        env_nested_delimiter="__",     # IMPORTANT for nesting
        extra="ignore"
    )

    @property
    def nir_buildable(self) -> bool:
        """There is a normcontrol server to reach, so the agent can be built.

        system.yaml attaches NirReportAgent on ``enabled: ${nir_buildable}``,
        and attachment is decided ONCE, when the agent tree is assembled at
        import. ``nir.enabled`` deliberately does not appear here: it is an
        operator switch the web UI flips per session, and a gate read at build
        time could never see that. Without the server there is nothing to gate —
        the toolset is dropped as unconfigured, and an agent advertising a
        capability it does not have is worse than an absent one.

        Whether the operator is actually *asked* remains a runtime decision, in
        ``reporting/nir/callback.py``, which reads ``nir_ready`` on every call.
        """
        return bool(self.mcp.normcontrol_url)

    @property
    def nir_ready(self) -> bool:
        """The NIR stage is on AND there is a normcontrol server to reach.

        The runtime gate: what decides whether the operator sees the question
        at all. Read through a property rather than fixed at construction so a
        setting changed at runtime (the web UI writes some) still decides
        correctly.
        """
        return bool(self.nir.enabled and self.mcp.normcontrol_url)


# Global instance
settings = Settings()
_settings_context: ContextVar[Settings | None] = ContextVar(
    "coscientist_settings_context",
    default=None,
)


def get_settings() -> Settings:
    """Return this run's settings snapshot, or the process defaults.

    Context variables are inherited by asyncio tasks, which lets concurrent web
    sessions use different agent selections without mutating the global object.
    """
    return _settings_context.get() or settings


@contextmanager
def settings_scope(value: Settings | None) -> Iterator[Settings]:
    """Temporarily bind ``value`` as :func:`get_settings` in this context."""
    if value is None:
        yield settings
        return
    token = _settings_context.set(value)
    try:
        yield value
    finally:
        _settings_context.reset(token)
