"""
Application configuration using Pydantic Settings.
"""
import os as _os
from pathlib import Path
from typing import List, Optional, Union

from dotenv import find_dotenv as _find_dotenv, load_dotenv as _load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator
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

    # Seconds to wait for a single completion before giving up. Without this a
    # provider that accepts the connection and then goes quiet never raises, so
    # the agent waits forever and the run looks frozen with nothing in the log.
    # A timeout turns that silence into a retryable error. Override with
    # LLM__REQUEST_TIMEOUT.
    request_timeout: int = 180

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
    result_formatter_url: Optional[str] = None
    # The file vault (mcp-servers/vault-mcp-server). Two consumers read it:
    # worker agents get the upload/download pair as an ADK toolset, and
    # framework code calls it per request through tools/vault_client.py.
    # Unset means both drop out, and the run still completes.
    vault_url: Optional[str] = None


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

    # Upper bound on LLM calls for one top-level run, passed to ADK's RunConfig.
    # ADK defaults to 500, which a long autonomous research run (many CoderAgent
    # debug/poll iterations) hits and gets cut off mid-work. Raised so a single
    # prompt can drive a long job to completion; still finite as a runaway-cost
    # backstop. Override via ORCHESTRATOR__MAX_LLM_CALLS.
    max_llm_calls: int = 3000

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
    hitl_auto_approve_timeout: int = int(_os.getenv("HITL_AUTO_APPROVE_TIMEOUT", _os.getenv("HITL__AUTO_APPROVE_TIMEOUT", _os.getenv("HITL_TIMEOUT_SECONDS", "300"))))
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
    fedot_fallback_enabled: bool = _os.getenv("EXECUTOR__FEDOT_FALLBACK", "true").lower() in ("true", "1", "yes")
    fedot_fallback_timeout_s: float = float(_os.getenv("EXECUTOR__FEDOT_FALLBACK_TIMEOUT", "900"))
    sandbox_url: str = _os.getenv("SANDBOX_URL", "")
    coder_workspace_id: _Optional[str] = _os.getenv("CODER_WORKSPACE_ID")
    coder_mode: str = _os.getenv("CODER__MODE", "local")        # "local" | "openhands"
    merge_tasks_enabled: bool = _os.getenv("PLANNER__MERGE_TASKS", "true").lower() in ("true", "1", "yes")
    max_active_hypotheses: int = int(_os.getenv("HYPOTHESES__MAX_ACTIVE", "1"))
    use_proxy: bool = _os.getenv("USE_PROXY", "True").lower() in ("true", "1", "yes")
    opik_enabled: bool = _os.getenv("OPIK__ENABLED", "false").lower() in ("true", "1", "yes")
    auto_naming_enabled: bool = _os.getenv("AUTO_NAMING__ENABLED", "true").lower() in ("true", "1", "yes")
    # Default of the per-browser "Show internal agents and tools" switch. A
    # browser that flipped the switch keeps its own choice.
    show_internal_enabled: bool = _os.getenv("SHOW_INTERNAL__ENABLED", "false").lower() in ("true", "1", "yes")
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
# AUTH
# =========================
class AuthSettings(BaseModel):
    """Shared-password gate for the web UI.

    Deliberately NOT part of WebSettings. That group is runtime-mutable: the
    unauthenticated ``POST /api/settings`` writes into it field by field, and
    ``GET /api/settings`` reads it back out to the browser. A credential there
    is one careless key mapping away from being both writable and readable by
    a caller. Nothing in the web layer mutates this group.

    Override with AUTH__PASSWORD_HASH, AUTH__SECRET_KEY and so on.
    """

    enabled: bool = True

    # Salted PBKDF2 digest of the one password for the whole deployment, in
    # the form "pbkdf2_sha256:<rounds>:<salt>:<digest>". Generate it with
    # `python -m CoScientist auth-hash`. Missing or malformed means the gate
    # cannot open, and every request gets 503 — see CoScientist/web/auth.py.
    # The app never falls back to serving without a credential.
    password_hash: str = ""

    # The password in clear text. A fallback for local runs, kept because the
    # digest form needs a generation step. It loses to password_hash when both
    # are set, and the server reports it at startup. Do not use it on a
    # deployment: anyone who reads the .env file can sign in.
    password: Optional[str] = None

    # HMAC key for the session cookie. Unset mints a fresh random key at every
    # boot, so a restart logs everyone out. That matches the rest of the app:
    # the ADK session service is in-memory and a restart already wipes it.
    secret_key: Optional[str] = None

    # Cookie lifetime in seconds. Default one week.
    session_max_age: int = 7 * 24 * 60 * 60

    # `Secure` on the session cookie. Unset means decide per request from
    # X-Forwarded-Proto: Secure under HTTPS, not Secure over plain HTTP. That
    # is the only setting correct for both deployments, because a browser
    # never sends a Secure cookie back over plain HTTP — pinning it true
    # without TLS accepts the password and then loops back to /login. Set it
    # true to demand HTTPS, or false to force plain HTTP and accept that the
    # password crosses the network in the clear.
    cookie_secure: Optional[bool] = None

    # Comma-separated origins allowed to open the WebSocket, e.g.
    # "https://cosci.example.org". SameSite=Lax does not reliably cover a
    # WebSocket handshake, so the origin needs its own check. An empty list
    # falls back to same-origin, which depends on the proxy passing the Host
    # header through. Name the origin here to stop depending on that.
    allowed_origins: str = ""

    # Online-guessing brake: failed logins per client IP per window. One
    # password on the public internet makes this the main residual risk.
    max_login_attempts: int = 10
    login_window_seconds: int = 15 * 60

    @property
    def origin_list(self) -> List[str]:
        # Lowercased: browsers send the host lowercased whatever the user
        # typed, so a hand-written "https://CoSci.example.org" would otherwise
        # never match.
        return [
            o.strip().rstrip("/").lower()
            for o in self.allowed_origins.split(",")
            if o.strip()
        ]

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
    opik: OpikSettings = OpikSettings()
    hitl: HITLSettings = HITLSettings()
    context_init: ContextInitSettings = ContextInitSettings()
    orchestrator: OrchestratorSettings = OrchestratorSettings()
    code_exec: CodeExecSettings = CodeExecSettings()
    tool_rag: ToolRAGSettings = ToolRAGSettings()
    mcp: MCPSettings = MCPSettings()
    web: WebSettings = WebSettings()
    research_graph: ResearchGraphSettings = ResearchGraphSettings()
    checkpoints: CheckpointSettings = CheckpointSettings()
    synapse: SynapseSettings = SynapseSettings()
    critic: CriticSettings = CriticSettings()
    auth: AuthSettings = AuthSettings()

    model_config = SettingsConfigDict(
        env_file=".env",          
        env_nested_delimiter="__",     # IMPORTANT for nesting
        extra="ignore"
    )


# Global instance
settings = Settings()


def get_settings() -> Settings:
    return settings
