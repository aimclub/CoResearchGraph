"""
CoScientist - Main entry point

Runs the multi-agent scientific discovery pipeline:
- Hypothesis generation
- Research
- Experimentation (FEDOT)
- Orchestration
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import os
from typing import Optional
import logging
from uuid import uuid4

from google.adk.sessions import InMemorySessionService
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.runners import Runner
from google.adk.agents.run_config import RunConfig
from google.genai import types

from CoScientist.config import get_settings, ReportConfig
from CoScientist.agents import orchestrator_agent, root_agent, run_root, build_for_mode
from CoScientist.reporting import finalize_report, RunResult
from CoScientist.tools.coder_tools import coder_toolset
from CoScientist.agents.callbacks import cleanup_uploaded_papers
from CoScientist.hitl.tool import hitl_toolset
from CoScientist.hitl import (
    AbstractHITLHandler,
    HITLRequest,
    HITLResponse,
)
from CoScientist.utils.text import strip_thinking

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compaction_config():
    """Build the events-compaction config: when an agent's prompt grows past the
    token threshold, ADK summarizes older events (with the agent's own model),
    keeping the last N raw events. Disable with AGENT_CONTEXT_TOKEN_THRESHOLD=0.
    """
    try:
        from google.adk.apps.app import EventsCompactionConfig
        threshold = int(os.getenv("AGENT_CONTEXT_TOKEN_THRESHOLD", "150000"))
        if threshold <= 0:
            return None
        return EventsCompactionConfig(
            compaction_interval=int(os.getenv("AGENT_COMPACTION_INTERVAL", "15")),
            overlap_size=int(os.getenv("AGENT_COMPACTION_OVERLAP", "2")),
            token_threshold=threshold,
            event_retention_size=int(os.getenv("AGENT_CONTEXT_RETENTION", "12")),
        )
    except Exception:  # noqa: BLE001 — compaction is best-effort, never block startup
        return None


def reset_session_state(
    user_id: str,
    session_id: str,
    *,
    reset_research: Optional[bool] = None,
) -> None:
    """Explicitly reset graph state for one session only.

    TaskTracker state belongs to ADK session state and semantic memory is global
    across the installation, so neither is touched here.
    """
    try:
        from CoScientist.graph.memory import reset_knowledge_graph
        reset_knowledge_graph(user_id=user_id, session_id=session_id)
    except Exception:  # noqa: BLE001
        pass

    try:
        should_reset = (
            get_settings().research_graph.reset_on_session
            if reset_research is None
            else reset_research
        )
        if should_reset:
            from CoScientist.graph.research.store import get_research_graph
            get_research_graph(
                user_id=user_id,
                session_id=session_id,
            ).reset(archive=True)
    except Exception:  # noqa: BLE001
        pass


def _s3_csv_preview(url: str, max_rows: int = 10, max_bytes: int = 200_000) -> str:
    """Best-effort: download a presigned-S3 CSV and return a small text preview
    (header + first rows of Smiles + key property columns). Returns '' on any failure.

    Lets the final answer be formed from the ACTUAL S3 file contents rather than a bare
    link or unverified prose (F010.A6).
    """
    import urllib.request
    import csv
    import io
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read(max_bytes).decode("utf-8", "replace")
        rows = list(csv.reader(io.StringIO(raw)))
        if not rows:
            return ""
        hdr = rows[0]
        prefer = ("Smiles", "QED", "LogP", "Synthetic Accessibility", "Validity")
        keep = [i for i, h in enumerate(hdr) if h in prefer] or list(range(min(5, len(hdr))))
        out = [" | ".join(hdr[i] for i in keep)]
        for row in rows[1:1 + max_rows]:
            out.append(" | ".join(row[i] if i < len(row) else "" for i in keep))
        extra = len(rows) - 1 - max_rows
        if extra > 0:
            out.append(f"… (+{extra} more rows)")
        return "\n".join(out)
    except Exception:
        return ""


class CoScientistManager:
    """
    Main manager for CoScientist (ADK-based execution).
    """

    def __init__(
        self,
        app_name: str = "coscientist_app",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        hitl_handler: Optional[AbstractHITLHandler] = None,
        session_service: Optional[BaseSessionService] = None,
        initial_state: Optional[dict] = None,
    ):
        self.app_name = app_name
        self.user_id = user_id or f"user_{uuid4().hex}"
        self.session_id = session_id or f"session_{uuid4().hex}"

        # Extra session state to seed at session creation, for callers that
        # drive the system programmatically (scripts, reproductions, harnesses)
        # and need something in state before the first turn.
        self._initial_state = dict(initial_state or {})

        # Web mode injects one shared service so managers can reopen existing
        # sessions. CLI mode falls back to a private in-memory service.
        self.session_service: Optional[BaseSessionService] = session_service
        self.runner: Optional[Runner] = None
        self._run_error: Optional[Exception] = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()

        # HITL setup
        self._hitl_handler = hitl_handler


    async def initialize(self):
        """Initialize session + runner."""
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return

            if self.session_service is None:
                self.session_service = InMemorySessionService()
            session = await self.session_service.get_session(
                app_name=self.app_name,
                user_id=self.user_id,
                session_id=self.session_id,
            )
            if session is None:
                from CoScientist.graph.session_scope import (
                    GRAPH_SCOPE_SESSION_KEY,
                    GRAPH_SCOPE_USER_KEY,
                )
                await self.session_service.create_session(
                    app_name=self.app_name,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    state={
                        "active_tasks": [],
                        **self._initial_state,
                        # Written last: session scoping is the manager's own and
                        # a caller must not be able to redirect it.
                        GRAPH_SCOPE_USER_KEY: self.user_id,
                        GRAPH_SCOPE_SESSION_KEY: self.session_id,
                    },
                )
            from google.adk.apps.app import App
            from CoScientist.logging.event_logger import EventLoggerPlugin
            from CoScientist.logging.tool_activity import ToolActivityPlugin
            from CoScientist.logging.agent_output import AgentOutputPlugin
            from CoScientist.logging.metrics import UsageMetricsPlugin
            from CoScientist.graph.plugin import GraphMemoryPlugin
            from CoScientist.graph.research.validator import BackgroundValidatorPlugin
            from CoScientist.agents.truncation_plugin import ToolResultTruncationPlugin
            from CoScientist.tools.mcp_artifact_plugin import McpArtifactCapturePlugin
            from CoScientist.verify.gate_plugin import ArtifactGatePlugin
            from CoScientist.agents.loop_guard_plugin import RepeatCallGuardPlugin
            from CoScientist.tools.session_scope_plugin import SessionScopePlugin

            # Build the agent system (reads start_mode + tunable params from settings).
            system = build_for_mode()

            app = App(
                name=self.app_name,
                root_agent=system.run_root,
                plugins=[
                    # First: deterministically refuse training on a fabricated
                    # dataset (before_tool gate) — fabrication buys nothing.
                    ArtifactGatePlugin(),
                    # Refuse the Nth identical tool call: any agent can fall into a
                    # tight loop (a collector once repeated one web search 39 times).
                    RepeatCallGuardPlugin(),
                    EventLoggerPlugin(),
                    # Observer: reports tool use from nested AgentTool runners
                    # too, which the top-level event stream cannot see. Inert
                    # unless a consumer (the Web UI) registered a sink.
                    ToolActivityPlugin(),
                    # Observer: posts the final answer of the agents flagged
                    # `report_output` (hypotheses, research, execution report),
                    # which otherwise only exists inside the caller's
                    # function_response. Inert without a sink.
                    AgentOutputPlugin(),
                    # Prices every model call, in nested AgentTool runners too,
                    # so one session total covers the whole agent tree.
                    UsageMetricsPlugin(),
                    GraphMemoryPlugin(),
                    BackgroundValidatorPlugin(),
                    # Fills user_id / session_id into the tool calls that declare
                    # them, so an MCP server scopes its S3 keys correctly and the
                    # model never has to copy an id by hand.
                    SessionScopePlugin(),
                    # Capture artifact (figure/table) URLs from tool results BEFORE
                    # truncation can drop them, so the report collector downloads them.
                    McpArtifactCapturePlugin(),
                    # Keep truncation last so observers receive full results.
                    ToolResultTruncationPlugin(),
                ],
                events_compaction_config=_compaction_config(),
            )
            self.runner = Runner(app=app, session_service=self.session_service)

            if self._hitl_handler:
                hitl_toolset._handler = self._hitl_handler
                if getattr(coder_toolset, "_hitl_handler", None) is not None:
                    if hasattr(coder_toolset._hitl_handler, 'set_delegate'):
                        coder_toolset._hitl_handler.set_delegate(self._hitl_handler)
                    else:
                        coder_toolset._hitl_handler = self._hitl_handler

            self._initialized = True

    async def _set_state(self, key: str, value) -> None:
        """Best-effort write into the live session state (in-memory)."""
        try:
            session = await self.session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id,
            )
            if session is not None:
                session.state[key] = value
        except Exception as exc:
            logger.warning("could not set session state %r: %s", key, exc)

    @staticmethod
    def _final_text(event) -> Optional[str]:
        """Answer text of a final-response event, skipping thinking parts."""
        if not (event.is_final_response() and event.content and event.content.parts):
            if event.actions and event.actions.escalate:
                return f"Escalation: {getattr(event, 'error_message', None) or 'Unknown error'}"
            return None
        parts = event.content.parts
        # Thinking models emit a separate `thought` part before the answer; prefer
        # the non-thought text, falling back to any text so we never drop it.
        answer = "\n".join(
            p.text for p in parts
            if getattr(p, "text", None) and not getattr(p, "thought", False)
        )
        final = answer or "\n".join(p.text for p in parts if getattr(p, "text", None)) or None
        if final:
            final = strip_thinking(final)
        return final or None

    async def run(
        self,
        query: str,
        verbose: bool = True,
        report_config: Optional[ReportConfig] = None,
    ) -> RunResult:
        """Run the full pipeline and package the report.

        The whole lifecycle (orchestrator → Result Aggregator) is ONE ADK
        invocation over ``run_root``, so it is a single trace with the aggregator
        as the terminal stage. Returns a :class:`RunResult` (``markdown``,
        ``report_dir``, ``manifest``); ``str(result)`` is the markdown, so legacy
        callers that treated the return value as a string keep working.
        """
        report_config = report_config or ReportConfig()
        await self.initialize()

        # The aggregator's format_results reads report_config mid-invocation, so it
        # must be in state BEFORE the run starts.
        await self._set_state("report_config", report_config.to_state())
        self._run_error = None

        content = types.Content(role="user", parts=[types.Part(text=query)])

        # When an agent hallucinates a tool it doesn't have, the unknown-tool guard
        # replaces the model turn with a corrective message. On the ROOT agent that
        # text-only turn is treated by ADK as the final answer, so the run would end
        # on the correction ("… not in your tool list …"). Detect that and CONTINUE
        # the same session (graph + workspace persist) with a nudge instead of
        # stopping — bounded so a persistently-confused model can't loop forever.
        _GUARD_SIG = "not in your tool list"
        _nudge = types.Content(role="user", parts=[types.Part(text=(
            "Continue the task — do not stop. Call ONLY tools you actually have: "
            "delegate literature/research to ResearchAgent and coding/experiments "
            "to CoderAgent or TaskExecutorAgent instead of calling their tools "
            "yourself. Keep working until the concrete deliverable is produced."))])
        msg = content

        # The report is the LAST final-response text of the run — i.e. the terminal
        # aggregator stage. If a mid-run failure (e.g. a slow MCP tool's 300s
        # timeout) aborts the invocation, we swallow it and fall through to the
        # deterministic finalizer below so captured artifacts still reach the user.
        report_markdown = ""
        for _attempt in range(6):
            self._run_error = None
            try:
                async for event in self.runner.run_async(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=msg,
                    # Lift ADK's 500-LLM-call default so a long autonomous run driven
                    # by a single prompt isn't cut off mid-work (finite cost backstop).
                    run_config=RunConfig(
                        max_llm_calls=get_settings().orchestrator.max_llm_calls
                    ),
                ):
                    if verbose:
                        print(
                            f"[Event] {event.author} | {type(event).__name__} | "
                            f"Final={event.is_final_response()}"
                        )
                    text = self._final_text(event)
                    if text is not None:
                        report_markdown = text
            except Exception as exc:
                self._run_error = exc
                logger.error(
                    "run loop raised (%s: %s); attempting partial delivery from captured "
                    "S3 artifacts.", type(exc).__name__, str(exc)[:200],
                )
                break

            if (self._run_error is None and _GUARD_SIG in (report_markdown or "")
                    and _attempt < 5):
                logger.warning(
                    "guard-correction surfaced as the final answer (hallucinated "
                    "tool); continuing the same session with a nudge."
                )
                msg = _nudge
                continue
            break

        # Read session state once, for the S3 fallback and reference extraction.
        try:
            session = await self.session_service.get_session(
                app_name=self.app_name, user_id=self.user_id, session_id=self.session_id,
            )
            state = dict(getattr(session, "state", None) or {}) if session else {}
        except Exception:
            state = {}

        # Safety net: if the aggregator produced nothing (e.g. the run stopped
        # early), surface any captured S3 artifacts so results still reach the
        # user. In the normal path the aggregator already embeds these, so we
        # only fall back when there is no report text.
        if not report_markdown.strip():
            arts = state.get("fedot_artifacts")
            if arts:
                blocks = []
                for a in arts:
                    url = a.get("url")
                    if not url:
                        continue
                    cnt = a.get("generated_count")
                    tag = f" ({cnt} molecules)" if cnt else ""
                    preview = await asyncio.to_thread(_s3_csv_preview, url)
                    block = f"**Generated result{tag}** — [download full CSV]({url})"
                    if preview:
                        block += f"\n```\n{preview}\n```"
                    blocks.append(block)
                if blocks:
                    report_markdown = "## Captured results\n\n" + "\n\n".join(blocks)

        # What this run cost, per agent. Logged rather than returned: the caller
        # hands `final_response` to a user (or to another model), and a bill is
        # not part of the answer. Disable with LOG_USAGE_METRICS=0.
        if os.getenv("LOG_USAGE_METRICS", "1") != "0":
            try:
                from CoScientist.logging.metrics import log_report
                log_report((self.user_id, self.session_id))
            except Exception as exc:  # noqa: BLE001 - reporting never fails a run
                logger.debug("Usage report unavailable: %s", exc)

        if not report_markdown.strip():
            if self._run_error is not None:
                report_markdown = (
                    f"The run stopped early ({type(self._run_error).__name__}) before producing a "
                    "result, and no partial artifacts were captured. This is usually a slow MCP "
                    "tool hitting its timeout or a transient model/network error — please retry."
                )
            else:
                report_markdown = (
                    "I couldn't complete this request within the available steps — the orchestrator "
                    "did not reach a tool that produced a result. Please retry or narrow the request."
                )

        # Index this research so LATER runs can find and reuse it. The graph is
        # already persisted per session, but nothing ever read it back — without
        # an index every run rediscovers what the system already settled. Derived
        # and best-effort: indexing must never affect the answer.
        try:
            from CoScientist.graph.research.index import index_research
            from CoScientist.graph.research.store import get_research_graph

            rg = get_research_graph(user_id=self.user_id, session_id=self.session_id)
            index_research(rg.full(), path=str(getattr(rg, "_path", "")),
                           user_id=self.user_id, session_id=self.session_id)
        except Exception:  # noqa: BLE001
            pass

        # Package the deliverable: report.md + LaTeX (per config) + MANIFEST.json.
        return await asyncio.to_thread(
            finalize_report, self.session_id, report_markdown, report_config, state,
        )

    async def close(self):
        """Cleanup session-related resources and uploaded paper artifacts."""
        if self.runner is not None:
            try:
                await self.runner.close()
            except Exception as exc:  # noqa: BLE001 - continue local cleanup
                logger.error(
                    "Warning: failed to close runner for session %s: %s",
                    self.session_id,
                    exc,
                )
            finally:
                self.runner = None
                self._initialized = False
        try:
            await asyncio.to_thread(cleanup_uploaded_papers, self.user_id, self.session_id)
        except Exception as exc:
            logger.error(f"Warning: failed to cleanup uploaded papers for session {self.session_id}: {exc}")
        # The ledger is process memory that grows with the number of sessions a
        # long-lived host serves, so a closed session gives its entry back.
        try:
            from CoScientist.logging.metrics import reset_session
            reset_session(key=(self.user_id, self.session_id))
        except Exception:  # noqa: BLE001 - nothing here is worth a failed close
            pass

# Convenience functions
async def create_manager() -> CoScientistManager:
    """Create and initialize a CoScientistManager."""
    manager = CoScientistManager()
    await manager.initialize()
    return manager


# Export public API
__all__ = [
    # Main classes
    "CoScientistManager",
    "ReportConfig",
    "RunResult",
    # Functions
    "create_manager",
]

# CLI entrypoint — thin shim. Prefer: python -m CoScientist cli
if __name__ == "__main__":
    from CoScientist.cli import run_repl

    run_repl()
