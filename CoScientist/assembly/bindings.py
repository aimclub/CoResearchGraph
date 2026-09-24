"""Registers every concrete tool, callback, agent class, output schema and
planner in the assembly registry — the names ``system.yaml`` refers to.

This module is the ONE place that maps names to implementations and carries the
prompt documentation (ToolDoc) for every tool. The assembler renders each
agent's "available tools" prompt section from these docs, so the docs here and
the wiring can never diverge: a tool that is not attached is not documented,
and a documented tool is attached.

Tool factories import their modules lazily so that merely loading the registry
does not construct MCP sessions or read service settings.
"""
from __future__ import annotations

from importlib import import_module
from typing import Optional

from CoScientist.assembly.registry import (
    REGISTRY,
    CallbackEntry,
    ToolDoc,
    ToolLimit,
    ToolEntry,
)

# ── Tools ────────────────────────────────────────────────────────────────────

def _websearch():
    from CoScientist.tools import websearch_toolset_instance
    return websearch_toolset_instance

def _paper_analysis():
    from CoScientist.tools import paper_analysis_toolset_instance
    return paper_analysis_toolset_instance


def _papers_search():
    from CoScientist.tools import papers_search_toolset_instance
    return papers_search_toolset_instance


def _vault():
    from CoScientist.tools import vault_toolset_instance
    return vault_toolset_instance


def _retrieval():
    from CoScientist.tools import retrieval_toolset_instance
    return retrieval_toolset_instance


def _mcp_server_search():
    from CoScientist.tools import search_mcp_servers
    return [search_mcp_servers]


def _fedot():
    from CoScientist.tools import fedot_toolset_instance
    return fedot_toolset_instance


def _result_formatter():
    from CoScientist.tools import result_formatter_tool
    return result_formatter_tool


def _nir_report():
    """The GOST NIR report toolset, or None when the MCP is not configured.

    Returning None (rather than a toolset that fails on first call) lets the
    entry be `optional`, so NirReportAgent simply has nothing to offer and the
    run completes the way it does today.
    """
    from CoScientist.config import get_settings

    if not get_settings().mcp.normcontrol_url:
        return None
    from CoScientist.tools.nir_report_tool import nir_report_tools
    return nir_report_tools


def _dynamic_tools():
    from CoScientist.tools import dynamic_mcp_toolset_instance
    return dynamic_mcp_toolset_instance


def _medical():
    from CoScientist.tools import med_toolset_instance
    return med_toolset_instance


def _coder():
    """Local coder toolset — dropped when the web UI switches it off, leaving
    the coder family to work through the OpenHands `sandbox` tools only."""
    if not _is_local_coder():
        return None
    from CoScientist.tools import coder_toolset_instance
    return coder_toolset_instance


def _alembic():
    from CoScientist.tools.alembic_tools import ALEMBIC_TOOLS
    return ALEMBIC_TOOLS

def _verify():
    from CoScientist.verify.tools import verify_toolset
    return verify_toolset.get_tools(None)


def _sandbox():
    """OpenHands sandbox tools — absent when no sandbox URL is configured."""
    from CoScientist.tools.coder_tools.sandbox_tools import get_sandbox_tools
    return get_sandbox_tools() or None

def _task_tracker():
    from CoScientist.tools import task_tracker_instance
    return task_tracker_instance


def _experiment_control():
    from CoScientist.experiments.runtime import experiment_control_toolset
    return experiment_control_toolset


def _create_plan_tool():
    from CoScientist.tools.task_tracker import create_plan_tool
    return [create_plan_tool()]


def _sleep_tool():
    from google.adk.tools import FunctionTool
    from CoScientist.tools.sleep_tool import sleep_tool
    return [FunctionTool(sleep_tool)]

def _read_result_tool():
    from google.adk.tools import FunctionTool
    from CoScientist.tools.read_result_tool import read_result
    return [FunctionTool(read_result)]

def _web_flag(field: str) -> bool:
    """Read a per-tool switch off ``settings.web`` (set from the web UI)."""
    try:
        from CoScientist.config import get_settings
        return bool(getattr(get_settings().web, field))
    except Exception:  # noqa: BLE001
        return True


def _is_local_coder() -> bool:
    try:
        from CoScientist.config import get_settings
        return get_settings().web.coder_mode == "local"
    except Exception:  # noqa: BLE001
        return True


def _graph():
    """Knowledge-graph reader toolset — dropped when the graph is switched off,
    which also stops GraphMemoryPlugin from recording (graph/plugin.py)."""
    if not _web_flag("knowledge_graph_enabled"):
        return None
    from CoScientist.graph.agent_tools import graph_reader_instance
    return graph_reader_instance


def _planner_retrieval():
    if not _web_flag("planner_retrieval_enabled"):
        return None
    return _retrieval()


def _planner_graph():
    if not _web_flag("planner_graph_enabled"):
        return None
    return _graph()


def _research_graph_enabled() -> bool:
    try:
        from CoScientist.config import get_settings
        return get_settings().research_graph.enabled
    except Exception:  # noqa: BLE001
        return False


def _research_graph():
    if not _research_graph_enabled():
        return None
    from CoScientist.graph.research.agent_tools import research_worker_toolset
    return research_worker_toolset


def _research_graph_orchestrator():
    if not _research_graph_enabled():
        return None
    from CoScientist.graph.research.agent_tools import research_orchestrator_toolset
    return research_orchestrator_toolset


def _research_graph_readonly():
    if not _research_graph_enabled():
        return None
    from CoScientist.graph.research.agent_tools import research_reporter_toolset
    return research_reporter_toolset

REGISTRY.register_tool(ToolEntry(
    key="websearch",
    factory=_websearch,
    runtime_resolved=True,  # Tavily MCP — tool surface comes from the remote server
    docs=(
        ToolDoc(
            name="tavily_search",
            signature="tavily_search(query)",
            purpose="General web search.",
        ),
        ToolDoc(
            name="tavily_extract",
            signature="tavily_extract(urls)",
            purpose="Read the content of specific pages/URLs.",
        ),
        ToolDoc(
            name="tavily_crawl",
            signature="tavily_crawl(url)",
            purpose="Crawl a site starting from a URL when one page is not enough.",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="paper_analysis",
    factory=_paper_analysis,
    optional=True,  # built only when MCP__PAPER_ANALYSIS_URL is configured
    runtime_resolved=True,
    docs=(
        ToolDoc(
            name="explore_scientific_database",
            signature="explore_scientific_database(task)",
            purpose=(
                "RAG over the internal scientific-literature corpus "
                "(deployed paper-analysis MCP). Returns an answer with supporting "
                "excerpts, not a list of papers - use find_papers_in_db for that."
            ),
        ),
        ToolDoc(
            name="explore_scientific_database",
            signature="explore_scientific_database(task)",
            purpose="RAG search over an internal scientific literature database.",
        ),
        ToolDoc(
            name="find_papers_in_db",
            signature="find_papers_in_db(task)",
            purpose=(
                "Finds the papers in the internal scientific literature database "
                "that are relevant to a topic. Returns a list of papers (title, "
                "domain, field, relevance scores), not an answer."
            ),
        ),
        ToolDoc(
            name="get_papers_database_statistics",
            signature="get_papers_database_statistics()",
            purpose=(
                "Reports what the internal scientific literature database "
                "holds: the number of unique papers and their share per "
                "research domain and field. Reads metadata only - it cannot "
                "answer questions about what the papers say."
            ),
        ),
        ToolDoc(
            name="explore_my_papers",
            signature="explore_my_papers(question, s3_keys)",
            purpose="Answers questions using user-uploaded or previously downloaded papers.",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="vault",
    factory=_vault,
    optional=True,  # built only when MCP__VAULT_URL is configured
    runtime_resolved=True,
    docs=(
        ToolDoc(
            name="get_upload_link",
            signature="get_upload_link(filename, feature=None)",
            purpose=(
                "Returns a one-hour upload_url for a new file, plus the bucket "
                "and s3_key that identify it for good. Upload with a plain HTTP "
                "PUT and no extra headers. Report the bucket and the s3_key, "
                "never the URL: the URL expires and the object does not."
            ),
        ),
        ToolDoc(
            name="get_download_link",
            signature="get_download_link(s3_key)",
            purpose=(
                "Turns an s3_key from an earlier step back into a one-hour "
                "download URL. Use it when a link you were given no longer works."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="papers_search",
    factory=_papers_search,
    optional=True,  # built only when MCP__PAPERS_SEARCH_URL is configured
    runtime_resolved=True,
    docs=(
        ToolDoc(
            name="search_papers",
            signature="search_papers(keywords)",
            purpose=(
                "Searches scientific papers in OpenAlex using keywords. "
                "Does NOT download full paper files. Argument name is "
                "`keywords`, not `query`. Optional `email` / `api_key` overlay "
                "OpenAlex credentials from env/headers."
            ),
        ),
        ToolDoc(
            name="download_papers_from_search",
            signature="download_papers_from_search(keywords)",
            purpose=(
                "Searches and downloads papers for downstream analysis. "
                "Optional `email` / `api_key` overlay OpenAlex credentials."
            ),
        ),
    ),
))

_RETRIEVAL_DOCS = (
    ToolDoc(
        name="retrieve_tools",
        signature="retrieve_tools(query)",
        purpose=(
            "Searches the MCP registry by capability. Returns ranked tool "
            "records with tool name, server_id, full description, input_schema, "
            "and score; use the metadata to determine exact requirement coverage."
        ),
    ),
    ToolDoc(
        name="get_server_info",
        signature="get_server_info(server_id)",
        purpose="Returns server metadata.",
    ),
)

REGISTRY.register_tool(ToolEntry(
    key="retrieval",
    factory=_retrieval,
    docs=_RETRIEVAL_DOCS,
))

# Same toolset as "retrieval", gated on the web setting so the planner's MCP
# discovery can be switched off from the UI without touching any other agent.
REGISTRY.register_tool(ToolEntry(
    key="planner_retrieval",
    factory=_planner_retrieval,
    optional=True,  # dropped when WEB__PLANNER_RETRIEVAL_ENABLED is false
    docs=_RETRIEVAL_DOCS,
))

REGISTRY.register_tool(ToolEntry(
    key="task_tracker",
    factory=_task_tracker,
    runtime_resolved=True,  # BaseToolset — tool surface comes from get_tools()
    docs=(
        ToolDoc(
            name="get_active_tasks",
            signature="get_active_tasks()",
            purpose="Get tasks from the current ADK session",
        ),
        ToolDoc(
            name="update_task_status",
            signature="update_task_status(task_id, status, notes=None)",
            purpose="Set task status to DONE/FAILED/IN_PROGRESS",
        ),
    ),
))

_GRAPH_DOCS = (
    ToolDoc(
        name="read_research_graph",
        signature="read_research_graph()",
        purpose="Read the shared knowledge graph: roster + every step so far.",
    ),
    ToolDoc(
        name="get_graph_history",
        signature="get_graph_history(limit)",
        purpose="Chronological history of steps taken in this session.",
    ),
    ToolDoc(
        name="get_agents_info",
        signature="get_agents_info()",
        purpose="Structured info about all agents in the system.",
    ),
)

REGISTRY.register_tool(ToolEntry(
    key="experiment_control",
    factory=_experiment_control,
    runtime_resolved=True,
    docs=(
        ToolDoc(
            name="get_experiment_plan",
            signature="get_experiment_plan()",
            purpose="Read the approved experiment plan and task/attempt runtime.",
        ),
        ToolDoc(
            name="start_task",
            signature="start_task(task_id)",
            purpose="Create one fresh attempt and immutable scoped route envelope.",
        ),
        ToolDoc(
            name="record_result",
            signature="record_result(task_id, attempt_id, result)",
            purpose="Validate and persist the attempt's only terminal TaskResult.",
        ),
        ToolDoc(
            name="retry_task",
            signature="retry_task(task_id)",
            purpose="Authorize a retryable failure to use a new attempt.",
        ),
        ToolDoc(
            name="fallback_task",
            signature="fallback_task(task_id, reason)",
            purpose="Advance to the next route in the finite acyclic fallback chain.",
        ),
        ToolDoc(
            name="skip_task",
            signature="skip_task(task_id, reason)",
            purpose="Skip an optional task and persist a skipped TaskResult.",
        ),
        ToolDoc(
            name="amend_task",
            signature="amend_task(task_id, patch, reason)",
            purpose="Amend an unstarted runtime task; material changes return to review.",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="graph",
    factory=_graph,
    optional=True,  # dropped when WEB__KNOWLEDGE_GRAPH_ENABLED is false
    runtime_resolved=True,  # BaseToolset — tool surface comes from get_tools()
    docs=_GRAPH_DOCS,
))

# Same toolset as "graph", gated on the web setting (planner only).
REGISTRY.register_tool(ToolEntry(
    key="planner_graph",
    factory=_planner_graph,
    optional=True,  # dropped when WEB__PLANNER_GRAPH_ENABLED is false
    runtime_resolved=True,
    docs=_GRAPH_DOCS,
))

# ── Research Context Graph ────────────────────────────────────────────────────
# The typed blackboard (CoScientist/graph/research). Two surfaces: workers get
# read + research_commit; the orchestrator additionally gets init/triggers/focus.
# Both optional (drop out when RESEARCH_GRAPH__ENABLED is false) and
# runtime_resolved (BaseToolset — real tool names come from get_tools()).
_RESEARCH_COMMIT_DOC = ToolDoc(
    name="research_commit",
    signature="research_commit(nodes, edges, status_updates)",
    purpose=("Record your results in the shared research graph in ONE "
             "transaction (validated + applied all-or-nothing). You may only "
             "write types/edges/status changes your role allows."),
    usage=(
        'create a node: {"type": "Evidence", "attrs": {...}, "status"?: "...", "ref"?: "e_new"}',
        'enrich an existing node: {"id": "EB1", "attrs": {...}} (no "type")',
        'edge: {"type": "supports", "from": "E4", "to": "H2"} — use "#e_new" to point at a node created in this call',
        'status change: {"id": "H2", "status": "under_verification", "reason"?: "..."}',
        "on ok=false, read errors, fix the payload, and call it again (nothing was saved).",
    ),
)
_RESEARCH_SLICE_DOC = ToolDoc(
    name="research_context_slice",
    signature="research_context_slice(node_id, depth=1)",
    purpose="Get one node plus its 1–2 hop neighborhood (the focused view to work from).",
)
_RESEARCH_OVERVIEW_DOC = ToolDoc(
    name="research_overview",
    signature="research_overview()",
    purpose="Compact index of the whole research graph (ids, types, statuses, labels).",
)
_RESEARCH_PROVENANCE_DOC = ToolDoc(
    name="research_provenance",
    signature="research_provenance(node_id)",
    purpose="Trace a node back to the root question (chain of nodes/edges + sources).",
)
_RESEARCH_WORKER_DOCS = (_RESEARCH_COMMIT_DOC, _RESEARCH_SLICE_DOC,
                         _RESEARCH_OVERVIEW_DOC, _RESEARCH_PROVENANCE_DOC)
_RESEARCH_ORCH_DOCS = _RESEARCH_WORKER_DOCS + (
    ToolDoc(
        name="research_init",
        signature="research_init(question, attrs, constraints, tools, resources, empirical_bases)",
        purpose=("Start a NEW research: create the root ResearchQuestion + its "
                 "context star. Call once at the start; archives any active graph."),
    ),
    ToolDoc(
        name="research_prior",
        signature="research_prior(query, limit)",
        purpose=("Search PAST researches (previous runs) related to a question: "
                 "their hypotheses with verdicts, the methods/tools used and the "
                 "conclusions. Consult BEFORE planning so settled work is reused "
                 "instead of re-derived; each hit reports why it matched."),
    ),
    ToolDoc(
        name="research_triggers",
        signature="research_triggers()",
        purpose=("Evaluate the decision triggers (READY / BLOCKED / REFUTE / "
                 "CLOSABLE / PENDING / TOOLS / RESOURCES / QUESTIONS / PROGRESS)."),
    ),
    ToolDoc(
        name="research_set_focus",
        signature="research_set_focus(node_id)",
        purpose=("Set the node the NEXT delegated worker focuses on — it "
                 "receives that node's slice automatically. Call before delegating."),
    ),
)

REGISTRY.register_tool(ToolEntry(
    key="research_graph",
    factory=_research_graph,
    optional=True,
    runtime_resolved=True,
    docs=_RESEARCH_WORKER_DOCS,
))

REGISTRY.register_tool(ToolEntry(
    key="research_graph_orchestrator",
    factory=_research_graph_orchestrator,
    optional=True,
    runtime_resolved=True,
    docs=_RESEARCH_ORCH_DOCS,
))

# Read-only surface for the Result Aggregator: overview / slice / provenance,
# no research_commit (the reporter reads the finished graph, never mutates it).
REGISTRY.register_tool(ToolEntry(
    key="research_graph_readonly",
    factory=_research_graph_readonly,
    optional=True,
    runtime_resolved=True,
    docs=(_RESEARCH_OVERVIEW_DOC, _RESEARCH_SLICE_DOC, _RESEARCH_PROVENANCE_DOC),
))

REGISTRY.register_tool(ToolEntry(
    key="verify",
    factory=_verify,
    docs=(
        ToolDoc(
            name="validate_dataset",
            signature="validate_dataset(path)",
            purpose=(
                "Deterministically check that a dataset file holds REAL, diverse "
                "molecules (RDKit-valid SMILES + a fitness/SA column). Call it on "
                "your training dataset BEFORE training — training is BLOCKED until a "
                "real dataset validates; toy/placeholder data (integers, one "
                "repeated molecule, a synthetic fallback) is rejected."),
        ),
        ToolDoc(
            name="validate_training",
            signature="validate_training(checkpoint_path, loss_log)",
            purpose=(
                "Deterministically check a training result: a real saved checkpoint "
                "and a loss that actually decreased over >=1 epoch."),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="create_plan_tool",
    factory=_create_plan_tool,
    docs=(
        ToolDoc(
            name="create_plan",
            signature="create_plan(tasks)",
            purpose=(
                "Replace all tasks with a new plan. Each task needs title, "
                "description and assignee, plus `id` and `parent_id` to state "
                "which task must run first. Tasks are stored in execution order."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="sleep",
    factory=_sleep_tool,
    docs=(
        ToolDoc(
            name="sleep_tool",
            signature="sleep_tool(minutes)",
            purpose=(
                "Pause before your next tool call instead of checking again "
                "immediately — use this to space out status/log checks on a "
                "long-running job (e.g. one that takes hours) instead of "
                "polling it every turn. Capped at 10 minutes per call; call it "
                "again afterwards if you need to wait longer."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="read_result",
    factory=_read_result_tool,
    docs=(
        ToolDoc(
            name="read_result",
            signature="read_result(ref, find=None, keys=None)",
            purpose=(
                "Read values from a shortened tool result (result_truncated + "
                "result_s3) stored whole in S3: ref is the result_s3 link or s3_key, "
                "find the value names, keys the top-level fields."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="mcp_server_search",
    factory=_mcp_server_search,
    docs=(
        ToolDoc(
            name="search_mcp_servers",
            signature="search_mcp_servers(query)",
            purpose=(
                "Searches public MCP registries and returns up to 15 matching "
                "servers with descriptions, metadata, and links."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="fedot",
    factory=_fedot,
    docs=(
        ToolDoc(
            name="fedot_tool",
            signature="fedot_tool(task_description)",
            purpose="Builds and executes a multi-agent pipeline to solve the task.",
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="result_formatter",
    factory=_result_formatter,
    docs=(
        ToolDoc(
            name="format_results",
            signature="format_results()",
            purpose=(
                "Collect every figure and data table this run produced (from session "
                "artifacts and the sandbox workspace) into the per-run report folder and "
                "return ready-to-embed Markdown blocks (image embeds + tables). Call FIRST."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="nir_report",
    factory=_nir_report,
    optional=True,  # built only when MCP__NORMCONTROL_URL is configured
    docs=(
        ToolDoc(
            name="nir_report_outline",
            signature="nir_report_outline()",
            purpose=(
                "Show the planned GOST 7.32-2017 report: section ids, the evidence "
                "recorded for each, figures awaiting captions, and gaps. Call FIRST "
                "and write only from what it returns."
            ),
        ),
        ToolDoc(
            name="nir_report_draft",
            signature=(
                "nir_report_draft(research_title, report_title, abstract_text, keywords, "
                "introduction_paragraphs, conclusion_paragraphs, section_texts, "
                "figure_captions=None, terms=None, abbreviations=None)"
            ),
            purpose=(
                "Assemble your prose into the GOST document and check it locally — "
                "no network, so iterate freely. Returns unwritten sections, contract "
                "problems and style warnings."
            ),
        ),
        ToolDoc(
            name="nir_report_submit",
            signature="nir_report_submit()",
            purpose=(
                "Validate the draft on the normcontrol server, build the DOCX and "
                "return a permanent download link plus the server's warnings."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="dynamic_tools",
    factory=_dynamic_tools,
    runtime_resolved=True,  # tool surface is the task's MCP servers, resolved per turn from state
    docs=(
        ToolDoc(
            name="<dynamic MCP tools>",
            signature="(varies)",
            purpose=(
                "The MCP tools selected for THIS task by the tool-prep pipeline "
                "(filtered_tools/deployed_mcps). Call them directly to run the work."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="medical",
    factory=_medical,
    docs=(
        ToolDoc(
            name="search_pubmed",
            signature="search_pubmed(keyword, num_results)",
            purpose=(
                "Find peer-reviewed literature on a clinical topic, drug, "
                "condition, or intervention (10 results by default)."
            ),
        ),
        ToolDoc(
            name="get_pico",
            signature="get_pico(title, abstract)",
            purpose=(
                "Extract Population / Intervention / Comparison / Outcome "
                "structure from a paper abstract."
            ),
        ),
        ToolDoc(
            name="get_study_taxonomy",
            signature="get_study_taxonomy(title, abstract)",
            purpose=(
                "Classify a paper's study design (observational vs experimental "
                "vs literature review, with subtypes)."
            ),
        ),
        ToolDoc(
            name="analyze_medical_image",
            signature="analyze_medical_image(artifact_id, question)",
            purpose=(
                "Interpret an uploaded DICOM or image file; provides differential "
                "diagnosis and ICD-10 codes."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="coder",
    factory=_coder,
    optional=True,  # dropped when WEB__CODER_LOCAL_TOOLS_ENABLED is false
    docs=(
        ToolDoc(
            name="execute_bash",
            signature="execute_bash(command, timeout)",
            purpose=(
                "Run a shell command in the session sandbox and WAIT for it: "
                "stdout, stderr and exit_code come back in this single call for "
                "almost everything (git clone, pip install, scripts, data "
                "processing). Only a genuinely long job that outlives the inline "
                "wait returns status \"running\" with a `job_id` to check later."
            ),
            usage=(
                "Use it for scripts, building/testing code, git (clone, checkout, "
                "commit, push, pull, diff, log), and data processing.",
                "You can run several independent commands; each call returns when "
                "its command finishes (or hands back a job_id for a long job).",
            ),
        ),
        ToolDoc(
            name="check_job",
            signature="check_job(job_id)",
            purpose=(
                "Check a long job that execute_bash handed back as still "
                "\"running\". You normally do NOT need this — execute_bash "
                "already waits and returns the result directly."
            ),
            usage=(
                "If the job is still running, do other work and check once "
                "later — never poll in a tight loop.",
            ),
        ),
        ToolDoc(
            name="read_file",
            signature="read_file(file_path, start_line, end_line)",
            purpose="Read code, config, and data files (completes immediately).",
        ),
        ToolDoc(
            name="write_file",
            signature="write_file(file_path, content)",
            purpose="Author code, config, and data files (completes immediately).",
        ),
        ToolDoc(
            name="list_directory",
            signature="list_directory(path)",
            purpose="List files in a directory.",
        ),
        ToolDoc(
            name="install_package",
            signature="install_package(package_name, upgrade)",
            purpose=(
                "Pip-install Python dependencies; like execute_bash it waits "
                "inline and returns the result (a very slow install may hand "
                "back a `job_id` for check_job)."
            ),
        ),
    ),
))

REGISTRY.register_tool(ToolEntry(
    key="alembic",
    factory=_alembic,
    docs=(
        ToolDoc(
            name="build_mcp_server",
            signature="build_mcp_server(repo_url, force_rebuild)",
            purpose=(
                "Start an Alembic build: turn a scientific GitHub repository into "
                "a served MCP tool server (clone -> env -> generated+validated "
                "tools -> FastMCP server in Docker)."
            ),
            usage=(
                "Returns immediately with a job_id; the build itself runs in the "
                "background and takes tens of minutes — report the job_id and "
                "do NOT poll it in a tight loop, check back later instead.",
                "Reuses an already running/done build for the same repo_url "
                "unless force_rebuild=true is passed.",
            ),
        ),
        ToolDoc(
            name="check_mcp_build",
            signature="check_mcp_build(job_id)",
            purpose=(
                "Check the status of a build started by build_mcp_server: "
                "\"running\" with the current pipeline stage and a log tail, "
                "\"done\" with the served mcp_url/image/container, or \"failed\" "
                "with the error tail of the build log."
            ),
        ),
        ToolDoc(
            name="list_mcp_builds",
            signature="list_mcp_builds()",
            purpose=(
                "List every Alembic build known to this process (running and "
                "finished) — use it to find a build from an earlier "
                "delegation/session (e.g. a lost job_id)."
            ),
        ),
    ),
))

_SANDBOX_TAIL_DOCS = (
    ToolDoc(
        name="check_sandbox_task",
        signature="check_sandbox_task()",
        purpose=(
            "Pick up the result of a sandbox task that came back "
            "\"running\". You normally do NOT need it — run_sandbox_task "
            "already waits and returns the result."
        ),
        usage=(
            "It waits inline; if the answer is still \"running\", do other "
            "work and check once later — never poll in a tight loop.",
        ),
    ),
    ToolDoc(
        name="list_sandbox_files",
        signature="list_sandbox_files(path)",
        purpose=(
            "List files in the sandbox workspace — use it to VERIFY that the "
            "artifacts the sandbox agent reported really exist before you "
            "rely on them."
        ),
    ),
    ToolDoc(
        name="fetch_sandbox_artifact",
        signature="fetch_sandbox_artifact(path)",
        purpose=(
            "Bring a file OUT of the sandbox into durable storage and get back "
            "a link that keeps working. Listing only proves a file exists; this "
            "is how a plot, a checkpoint, a results table or a generated "
            "dataset reaches the report and the reader — including files the "
            "sandbox agent never uploaded. A directory comes back as one ZIP."
        ),
    ),
)


def _sandbox_docs():
    """Sandbox docs, phrased for whether the local coder toolset is also there.

    With both, the sandbox is the escalation path for heavy jobs and the two
    workspaces must not be confused. Alone, it IS the way the agent runs
    anything, so the guidance must not point back at execute_bash.
    """
    if _is_local_coder():
        run_usage = (
            "The sandbox is a SEPARATE machine from your execute_bash "
            "workspace — files do NOT cross between them. Data goes in via "
            "`dataset_url`; results come back as the summary.",
            "It is bound to your session: the first call creates it, later "
            "calls continue in the SAME sandbox with its files and memory "
            "intact — so build one experiment up over several calls.",
            "Pass `new_sandbox=True` ONLY for an independent experiment on a "
            "clean machine; everything the previous one produced is lost.",
            "Say exactly what the deliverable is and where to write it — you "
            "cannot watch it work, you only get its report back.",
            "For ordinary code, shell and git work keep using execute_bash.",
        )
    else:
        run_usage = (
            "This is your ONLY way to run anything: you have no local shell, "
            "so every command, script, clone and install happens here. Data "
            "goes in via `dataset_url`; results come back as the summary.",
            "`task` is the task you were given, forwarded as it is — the agent "
            "on the other side plans and writes the code itself.",
            "It is bound to your session: the first call creates it, later "
            "calls continue in the SAME sandbox with its files and memory "
            "intact — so successive tasks build on each other.",
            "Pass `new_sandbox=True` ONLY for an independent experiment on a "
            "clean machine; everything the previous one produced is lost.",
            "Send the WHOLE task in one call — each call spins up a full "
            "coding agent; you cannot watch it work, you only get its report.",
        )
    return (
        ToolDoc(
            name="run_sandbox_task",
            signature="run_sandbox_task(task, dataset_url, new_sandbox)",
            purpose=(
                "Delegate a HEAVY / long-running / GPU-bound job (training runs, "
                "large data processing, long experiments) to an autonomous agent "
                "in the OpenHands sandbox. Waits inline and returns that agent's "
                "report; hands back status \"running\" only if the job outlives "
                "the wait."
            ),
            usage=run_usage,
        ),
    ) + _SANDBOX_TAIL_DOCS


REGISTRY.register_tool(ToolEntry(
    key="sandbox",
    factory=_sandbox,
    # Dropped silently in deployments where SANDBOX_URL is unset — the prompt
    # then never advertises a sandbox the agent does not have.
    optional=True,
    docs=_sandbox_docs,
))

# HITL tools are not a YAML-listed tool entry: the assembler attaches them via
# the per-agent `hitl: true` flag (when HITL is globally enabled) and appends
# these docs so the prompt always matches.
HITL_TOOL_DOCS = (
    ToolDoc(
        name="request_approval",
        signature="request_approval(agent_name, message, context)",
        purpose=(
            "(HITL) Ask the human to approve or reject a proposed action before "
            "proceeding. Returns 'approved' (bool) and optional 'feedback'."
        ),
    ),
    ToolDoc(
        name="request_selection",
        signature="request_selection(agent_name, message, options, default_option=None)",
        purpose=(
            "(HITL) Ask the human to choose one of several options you generated "
            "(e.g. hypotheses or plans). An optional default_option controls auto mode; "
            "otherwise auto selects the first option. Returns 'selected' and 'approved'."
        ),
    ),
)

# Attached the same way, by the per-agent `work_order: true` flag.
WORK_ORDER_TOOL_DOCS = (
    ToolDoc(
        name="declare_work_order",
        signature=(
            "declare_work_order(goal, done_criteria, assumptions, steps, planned_tools, "
            "expected_outcome, fallback)"
        ),
        purpose=(
            "(Work Order) Declare your contract BEFORE your first external action: "
            "goal, assumptions (a list of strings), "
            "steps, tools, expected outcome. "
            "Returns status approved / revise / rejected."
        ),
    ),
    ToolDoc(
        name="update_work_order",
        signature="update_work_order(reason, add_tools, add_steps)",
        purpose=(
            "(Work Order) Amend the approved contract when you need a tool or step "
            "it does not cover. The human reviews the diff."
        ),
    ),
    ToolDoc(
        name="update_work_step",
        signature="update_work_step(step_id, status, note)",
        purpose=(
            "(Work Order) Mark a step in_progress / done / skipped as you go, so the "
            "human can follow the plan live."
        ),
    ),
    ToolDoc(
        name="submit_work_report",
        signature=(
            "submit_work_report(summary, findings, done_verdict, done_evidence, "
            "actual_outcome, artifacts)"
        ),
        purpose=(
            "(Work Order) Before your final answer, report what you found (with "
            "evidence) and produced. The human accepts it, sends it back for rework "
            "or rejects it. Returns status accepted / revise / rejected."
        ),
    ),
)


# ── Callbacks ────────────────────────────────────────────────────────────────

def _cb(key: str, kind: str, func=None, factory=None, limit: Optional[ToolLimit] = None) -> None:
    REGISTRY.register_callback(CallbackEntry(key=key, kind=kind, func=func, factory=factory, limit=limit))


def _save_uploaded_artifacts():
    from CoScientist.agents.callbacks import before_model_modifier
    return before_model_modifier


def _seed_coder_workspace():
    from CoScientist.tools.coder_tools import seed_coder_workspace
    return seed_coder_workspace


def _inject_medical_artifacts():
    from CoScientist.agents.callbacks import med_agent_before_model
    return med_agent_before_model


def _inject_uploaded_papers():
    from CoScientist.agents.callbacks import papers_agent_before_model
    return papers_agent_before_model


def _log_research_tool_calls():
    from CoScientist.agents.callbacks import print_research_agent_tool_call
    return print_research_agent_tool_call


def _capture_mcp_artifacts():
    from CoScientist.agents.callbacks import capture_mcp_artifacts
    return capture_mcp_artifacts


def _mirror_plan_after_create():
    from CoScientist.agents.callbacks import mirror_plan_after_create
    return mirror_plan_after_create


def _mirror_plan_before_agent():
    from CoScientist.agents.callbacks import mirror_plan_before_agent
    return mirror_plan_before_agent


def _skip_retriever_context():
    from CoScientist.agents.callbacks import before_tool_reranker_model
    return before_tool_reranker_model


def _shortlist_reranker_tools():
    from CoScientist.agents.callbacks import shortlist_reranker_tools
    return shortlist_reranker_tools


def _collect_reranked_tools():
    from CoScientist.agents.callbacks import after_tool_reranker_agent
    return after_tool_reranker_agent


def _collect_reranked_tools_from_model():
    from CoScientist.agents.callbacks import after_tool_reranker_model
    return after_tool_reranker_model


def _collect_reranked_mcps():
    from CoScientist.agents.callbacks import after_fullset_reranker_agent
    return after_fullset_reranker_agent


def _redirect_when_no_tools():
    from CoScientist.agents.callbacks import redirect_when_no_tools
    return redirect_when_no_tools


def _inject_fedot_candidates():
    from CoScientist.agents.callbacks import inject_fedot_candidates
    return inject_fedot_candidates


def _announce_attached_tools():
    from CoScientist.agents.callbacks import announce_attached_tools
    return announce_attached_tools


def _before_get_task():
    from CoScientist.agents.callbacks import before_get_task
    return before_get_task

def _inject_original_query():
    from CoScientist.agents.callbacks import inject_original_query
    return inject_original_query

def _inject_upstream_artifacts():
    # Kept for default system.yaml / non-EM profiles. EM uses
    # seed_upstream_from_resolved_inputs at start_task instead.
    from CoScientist.tools.fedot_artifact_handoff import inject_upstream_artifacts
    return inject_upstream_artifacts


def _inject_graph_root():
    from CoScientist.agents.callbacks import inject_graph_root
    return inject_graph_root


def _inject_dataset_context():
    from CoScientist.agents.callbacks import inject_dataset_context
    return inject_dataset_context


def _inject_report_language():
    from CoScientist.agents.callbacks import inject_report_language
    return inject_report_language


def _user_links():
    from CoScientist.agents.callbacks import user_links
    return user_links


def _resolve_link_refs():
    from CoScientist.agents.callbacks import resolve_link_refs
    return resolve_link_refs


def _register_tool_result_links():
    from CoScientist.agents.callbacks import register_tool_result_links
    return register_tool_result_links


def _expand_link_refs():
    from CoScientist.agents.callbacks import expand_link_refs
    return expand_link_refs


def _redact_link_urls():
    from CoScientist.agents.callbacks import redact_link_urls
    return redact_link_urls


def _inject_research_context(ctx):
    """before_agent callback seeding state['research_context']. The orchestrator
    (root) gets the overview + trigger digest; a worker gets its focus slice.
    Which branch is baked in at build time from the agent's role."""
    from CoScientist.graph.research.agent_tools import make_inject_research_context
    is_root = bool(getattr(ctx.config, "root", False))
    return make_inject_research_context(is_root=is_root)


def _global_max_searches() -> int:
    from CoScientist.config import get_settings
    return get_settings().web.max_searches


# The search budget of RESEARCH_AGENT_SEARCHES (Settings → Tools), unless the
# agent's row in Settings → Agents sets its own.
SEARCH_LIMIT = ToolLimit(kind="searches", default=_global_max_searches,
                         setting="researchAgent.maxSearches", minimum=0)


def _agent_max_searches(ctx) -> int:
    from CoScientist.assembly.schema import agent_limit
    return agent_limit(ctx.config.name, _global_max_searches())


def _web_search_limiter(ctx):
    from CoScientist.agents.callbacks.tool_callbacks import SearchLimiter
    return SearchLimiter(max_searches=_agent_max_searches(ctx)).limit_searches


def _count_research_searches():
    from CoScientist.agents.callbacks.tool_callbacks import SearchLimiter
    from CoScientist.config import get_settings
    return SearchLimiter(max_searches=get_settings().web.max_searches).record_search_result


def _reset_research_searches():
    from CoScientist.agents.callbacks.tool_callbacks import SearchLimiter
    from CoScientist.config import get_settings
    return SearchLimiter(max_searches=get_settings().web.max_searches).reset_search_budget


def _tavily_search_limiter(ctx):
    from CoScientist.agents.callbacks.tool_callbacks import TavilySearchLimiter
    return TavilySearchLimiter(max_searches=_agent_max_searches(ctx)).limit_searches


def _paper_search_guard():
    from CoScientist.agents.callbacks.tool_callbacks import PaperSearchGuard
    return PaperSearchGuard().guard_paper_search


def _forbid_explore_my_papers():
    from CoScientist.agents.callbacks.tool_callbacks import ForbidExploreMyPapersGuard
    return ForbidExploreMyPapersGuard().guard_tool


def _sanitize_json_output():
    from CoScientist.agents.callbacks import sanitize_json_output
    return sanitize_json_output


def _unwrap_model_response_args():
    from CoScientist.agents.callbacks import unwrap_model_response_args
    return unwrap_model_response_args


def _guard_unknown_tools(ctx):
    """after_model guard capturing the agent's REAL tool names from its context,
    so a hallucinated tool call is corrected instead of crashing the run.

    The valid set must include BOTH the agent's function tools AND its
    subordinate AgentTools: sub-agents (e.g. CoderAgent, DatasetCollectorAgent)
    are legitimate call targets but are attached outside `tool_entries`, so
    leaving them out makes the guard false-block real delegations.

    Agents whose tool surface is resolved at runtime (dynamic MCP toolsets,
    e.g. ExperimentAgent) can't be guarded — their real tools aren't known at
    build time — so skip the guard for them to avoid blocking valid calls."""
    from CoScientist.agents.callbacks import make_unknown_tool_guard
    docs = [d for e in ctx.tool_entries for d in e.resolved_docs()]
    # A placeholder doc (name in <angle brackets>, e.g. "<dynamic MCP tools>")
    # marks a toolset whose real tool names are resolved per turn from state
    # (ExperimentAgent's dynamic MCP tools) — we can't enumerate them at build
    # time, so skip the guard rather than false-block valid calls. Fixed
    # BaseToolsets (graph, task_tracker) are runtime_resolved too but DO declare
    # their real tool names in docs, so they stay guarded.
    if any(d.name.startswith("<") for d in docs):
        return None
    names = [d.name for d in docs]
    names += [s.name for s in ctx.subordinates]  # subordinate AgentTools
    return make_unknown_tool_guard(names)


def _finish_after_plan_registered():
    from CoScientist.agents.callbacks import make_plan_registration_guard
    return make_plan_registration_guard()


def _pre_action_critique(ctx):
    from CoScientist.agents.callbacks import make_pre_action_critique
    return make_pre_action_critique(REGISTRY.prompt("pre_action_critic")(ctx))


def _post_action_critique(ctx):
    from CoScientist.agents.callbacks import make_post_action_critique
    return make_post_action_critique(REGISTRY.prompt("post_action_critic")(ctx))


def make_plan_critic(ctx):
    """The planner's plan critic, for agents declaring ``critic:`` in the YAML.

    Not an ADK callback (no callback can make the planner redo its roadmap):
    the assembler passes it to the session agent, which owns the review loop.
    Built here anyway so the assembler keeps looking things up instead of
    importing agent internals — and so the critic's prompt is rendered from the
    same PromptContext that wires the agents its plans may assign work to.
    """
    from CoScientist.agents.callbacks import make_plan_critique
    return make_plan_critique(REGISTRY.prompt("plan_critic")(ctx))


def _hitl_before_model():
    from CoScientist.agents.common import hitl_handler
    from CoScientist.hitl.callbacks import make_hitl_before_callback
    return make_hitl_before_callback(hitl_handler)


def _ask_pipeline_scope():
    from CoScientist.agents.common import hitl_handler
    from CoScientist.hitl.pipeline_scope import make_ask_pipeline_scope_callback
    return make_ask_pipeline_scope_callback(hitl_handler)


def _ask_nir_report():
    from CoScientist.agents.common import hitl_handler
    from CoScientist.reporting.nir.callback import make_ask_nir_report_callback
    return make_ask_nir_report_callback(hitl_handler)


def _enforce_pipeline_scope_hops():
    from CoScientist.hitl.pipeline_scope import enforce_pipeline_scope_hops
    return enforce_pipeline_scope_hops


def _mark_pipeline_scope_lane():
    from CoScientist.hitl.pipeline_scope import mark_pipeline_scope_lane
    return mark_pipeline_scope_lane
def _hitl_before_tool():
    from CoScientist.agents.common import hitl_handler
    from CoScientist.hitl.callbacks import make_hitl_before_tool_callback
    return make_hitl_before_tool_callback(hitl_handler, target_tools=("run_sandbox_task",))


# Plain callbacks are registered through tiny lazy factories that ignore the
# context — so importing bindings never drags in S3/opik/etc. transitively.
_cb("save_uploaded_artifacts", "before_model", factory=lambda ctx: _save_uploaded_artifacts())
# Pin the coder sandbox to the ADK session (one workspace per session).
_cb("seed_coder_workspace", "before_model", factory=lambda ctx: _seed_coder_workspace())
_cb("inject_medical_artifacts", "before_model", factory=lambda ctx: _inject_medical_artifacts())
_cb("inject_uploaded_papers", "before_model", factory=lambda ctx: _inject_uploaded_papers())
_cb("log_research_tool_calls", "after_tool", factory=lambda ctx: _log_research_tool_calls())
_cb("capture_mcp_artifacts", "after_tool", factory=lambda ctx: _capture_mcp_artifacts())
# The registered plan becomes the research graph's method column, deterministically.
# Two hooks because either path can be the one that fires: `create_plan` belongs to
# an agent that ships disabled, and an operator can register a roadmap from the web.
_cb("mirror_plan_after_create", "after_tool", factory=lambda ctx: _mirror_plan_after_create())
_cb("mirror_plan_before_agent", "before_agent", factory=lambda ctx: _mirror_plan_before_agent())
_cb("skip_retriever_context", "before_model", factory=lambda ctx: _skip_retriever_context())
# Cross-encoder pre-pass: hand the LLM reranker a short list, not everything.
_cb("shortlist_reranker_tools", "before_agent", factory=lambda ctx: _shortlist_reranker_tools())
_cb("collect_reranked_tools", "after_agent", factory=lambda ctx: _collect_reranked_tools())
_cb(
    "collect_reranked_tools_from_model",
    "after_model",
    factory=lambda ctx: _collect_reranked_tools_from_model(),
)
_cb("collect_reranked_mcps", "after_agent", factory=lambda ctx: _collect_reranked_mcps())
# Coder↔Executor redirect: abstain to CoderAgent when no tool matched the task.
_cb("redirect_when_no_tools", "before_agent", factory=lambda ctx: _redirect_when_no_tools())
# Reranker fallback: show FedotAgent the candidate pool fedot_tool will receive.
_cb("inject_fedot_candidates", "before_agent", factory=lambda ctx: _inject_fedot_candidates())
# State the executor's real tool list, so a catalogue "nothing matched" verdict
# earlier in the conversation cannot override the tools it is actually holding.
_cb("announce_attached_tools", "before_model", factory=lambda ctx: _announce_attached_tools())
# Load active tasks into agent state before the agent runs.
_cb("before_get_task", "before_agent", factory=lambda ctx: _before_get_task())
# Project prior MCP CSV columns onto the current tools' input_schema arg names.
# EM profile omits this — start_task seeds via seed_upstream_from_resolved_inputs.
_cb(
    "inject_upstream_artifacts",
    "before_agent",
    factory=lambda ctx: _inject_upstream_artifacts(),
)
_cb("inject_original_query", "before_model", factory=lambda ctx: _inject_original_query())
# Give the orchestrator/planner the knowledge-graph root (agents + history) up front.
_cb("inject_graph_root", "before_agent", factory=lambda ctx: _inject_graph_root())
# Seed state['research_context'] from the research blackboard (role-dependent).
_cb("inject_research_context", "before_agent", factory=_inject_research_context)
# Tell the agent about the dataset archive the user attached in the web UI; it
# decides itself which calls need the link.
_cb("inject_dataset_context", "before_agent", factory=lambda ctx: _inject_dataset_context())
# Report language the user picked for this session: inject the whole block
# (headings, substitution rule, glossary), not a bare language name.
_cb("inject_report_language", "before_agent", factory=lambda ctx: _inject_report_language())
_cb("user_links", "before_agent", factory=lambda ctx: _user_links())
_cb("redact_link_urls", "before_model", factory=lambda ctx: _redact_link_urls())
_cb("resolve_link_refs", "before_tool", factory=lambda ctx: _resolve_link_refs())
_cb("register_tool_result_links", "after_tool", factory=lambda ctx: _register_tool_result_links())
_cb("expand_link_refs", "after_model", factory=lambda ctx: _expand_link_refs())
# Human-In-The-Loop approval callback before model/agent/tool execution.
_cb("hitl_before_model", "before_model", factory=lambda ctx: _hitl_before_model())
_cb("hitl_before_agent", "before_agent", factory=lambda ctx: _hitl_before_model())
_cb("ask_pipeline_scope", "before_agent", factory=lambda ctx: _ask_pipeline_scope())
# Asks, once per session, whether the run should also produce a GOST 7.32-2017
# NIR report, and collects the title-page requisites nothing else knows. Inert
# unless NIR__ENABLED, MCP__NORMCONTROL_URL and HITL are all on.
_cb("ask_nir_report", "before_agent", factory=lambda ctx: _ask_nir_report())
_cb(
    "enforce_pipeline_scope_hops",
    "after_model",
    factory=lambda ctx: _enforce_pipeline_scope_hops(),
)
_cb(
    "mark_pipeline_scope_lane",
    "after_tool",
    factory=lambda ctx: _mark_pipeline_scope_lane(),
)
_cb("hitl_before_tool", "before_tool", factory=lambda ctx: _hitl_before_tool())
# Limit web search calls per agent turn.
_cb("WebSearchLimiter", "before_tool", factory=_web_search_limiter, limit=SEARCH_LIMIT)
_cb("count_research_searches", "after_tool", factory=lambda ctx: _count_research_searches())
_cb("reset_research_searches", "before_agent", factory=lambda ctx: _reset_research_searches())
# A separate per-agent quota for EconomicsAgent / ReactorAgent Tavily fallback.
_cb("TavilySearchLimiter", "before_tool", factory=_tavily_search_limiter, limit=SEARCH_LIMIT)
# Clamp OpenAlex result sets before the request reaches the remote papers MCP.
_cb("PaperSearchGuard", "before_tool", factory=lambda ctx: _paper_search_guard())
# Forbid ResearchAgent from calling explore_my_papers (reserved for PaperRetriever).
_cb("ForbidExploreMyPapers", "before_tool", factory=lambda ctx: _forbid_explore_my_papers())
# Catch hallucinated tool calls (e.g. `find`) and correct instead of crashing.
_cb("guard_unknown_tools", "after_model", factory=_guard_unknown_tools)
from CoScientist.agents.callbacks.pilot_delegation import require_pilot_delegations
_cb("require_pilot_delegations", "after_model", func=require_pilot_delegations)
# End the planner's turn once its plan is registered, so it cannot loop
# re-registering to undo create_plan's own normalisation.
_cb("finish_after_plan_registered", "after_model",
    factory=lambda ctx: _finish_after_plan_registered())
# Trim prose/fences/trailing text around a JSON answer BEFORE strict
# output_schema validation (providers don't always honour response_format).
_cb("sanitize_json_output", "after_model", factory=lambda ctx: _sanitize_json_output())
def _stage_tz_draft():
    from CoScientist.context_init.tz_agent import stage_tz_draft
    return stage_tz_draft


# The assembled ТЗ, staged where `{tz_draft?}` can reach it.
_cb("stage_tz_draft", "before_agent", factory=lambda ctx: _stage_tz_draft())
# The same answer given through set_model_response (agents with tools AND an
# output_schema): lift it out of a wrapper key the model invented, so schema
# validation does not silently turn it into an empty object.
_cb("unwrap_model_response_args", "before_tool",
    factory=lambda ctx: _unwrap_model_response_args())


def _brief_hypotheses_regime():
    from CoScientist.agents.callbacks.hypothesis_brief import brief_hypotheses_regime
    return brief_hypotheses_regime


# What the run can already measure, staged where `{hypothesis_brief?}` reaches
# the hypothesis generator: with tools in hand it aims the claim at them, with
# none it spends the effort on judging the candidates instead.
_cb("brief_hypotheses_regime", "before_agent",
    factory=lambda ctx: _brief_hypotheses_regime())


def _guard_report_without_execution():
    from CoScientist.agents.callbacks.report_guard import (
        guard_report_without_execution,
    )
    return guard_report_without_execution


# The aggregator is a `pipeline.post` stage, so it runs whether or not anything
# was executed — and the experiment module has a branch that stops quietly. This
# is what stops a full scientific report being written over a plan that never
# ran, and what stages `{report_unexecuted_note?}` when only part of it did.
_cb("guard_report_without_execution", "before_agent",
    factory=lambda ctx: _guard_report_without_execution())
# ── Experiment Module callbacks ──────────────────────────────────────────────
# Every EM callback is a plain (context-independent) function, so they are
# registered table-driven: (registry key, hook, "package:attr"), one lazy
# import per resolve. Keys and hooks must stay in sync with experiments.yaml.
_EM = "CoScientist.experiments"
_EM_CALLBACKS: tuple[tuple[str, str, str], ...] = (
    ("prepare_experiment_user_turn", "before_agent", f"{_EM}.runtime:prepare_experiment_user_turn"),
    # Bounded planner context plus hard AgentTool route guard.
    ("build_experiment_context", "before_agent", f"{_EM}.context:build_experiment_context"),
    ("commit_experiment_hypotheses", "after_agent", f"{_EM}.hypotheses:commit_experiment_hypotheses"),
    ("persist_experiment_em_request", "before_agent", f"{_EM}.hypotheses:persist_experiment_em_request"),
    ("bootstrap_research_question_if_empty", "before_agent", f"{_EM}.hypotheses:bootstrap_research_question_if_empty"),
    ("seed_hypotheses_from_em_request", "before_model", f"{_EM}.hypotheses:seed_hypotheses_from_em_request"),
    ("enforce_hypothesis_research_commit", "after_model", f"{_EM}.hypotheses:enforce_hypothesis_research_commit"),
    ("normalize_em_hypothesis_commit", "after_model", f"{_EM}.hypotheses:normalize_em_hypothesis_commit"),
    ("capture_hypotheses_after_research_commit", "after_tool", f"{_EM}.hypotheses:capture_hypotheses_after_research_commit"),
    ("reset_experiment_retrieval_budget", "before_agent", f"{_EM}.context:reset_experiment_retrieval_budget"),
    ("enforce_experiment_retrieval_budget", "after_model", f"{_EM}.context:enforce_experiment_retrieval_budget"),
    ("snapshot_experiment_discovered_capabilities", "after_agent", f"{_EM}.context:snapshot_experiment_discovered_capabilities"),
    ("stash_experiment_retrieved_capabilities", "before_agent", f"{_EM}.context:stash_experiment_retrieved_capabilities"),
    # Same snapshot, after ToolRetriever finishes (reranker clears accumulated_tools).
    ("persist_experiment_retrieved_capabilities", "after_agent", f"{_EM}.context:stash_experiment_retrieved_capabilities"),
    ("skip_executor_without_runtime", "before_agent", f"{_EM}.context:skip_executor_without_runtime"),
    # After ToolPreparer: lit/knowledge asks with no compute signal → NO_MATCHING_TOOL
    # before Hypotheses/Plan/Coder burn budget on unrelated inventory.
    ("assess_experiment_inventory_feasibility", "after_agent", f"{_EM}.runtime:assess_experiment_inventory_feasibility"),
    ("skip_when_experiment_not_feasible", "before_agent", f"{_EM}.runtime:skip_when_experiment_not_feasible"),
    ("skip_when_experiment_stage_complete", "before_agent", f"{_EM}.runtime:skip_when_experiment_stage_complete"),
    ("guard_experiment_route", "before_tool", f"{_EM}.runtime:guard_route_agent_tool"),
    ("pin_alembic_build_args", "before_tool", f"{_EM}.runtime:pin_alembic_build_args"),
    ("pin_fedot_alembic_task", "before_tool", f"{_EM}.runtime:pin_fedot_alembic_task"),
    ("await_alembic_job_if_experiment", "after_tool", f"{_EM}.runtime:await_alembic_job_if_experiment"),
    ("force_schema_s3_upload", "before_tool", f"{_EM}.runtime:force_schema_s3_upload"),
    ("force_molecule_generator_s3_upload", "before_tool", f"{_EM}.runtime:force_molecule_generator_s3_upload"),
    ("mark_experiment_route_returned", "after_tool", f"{_EM}.runtime:on_route_agent_returned"),
    ("capture_experiment_tool_results", "after_tool", f"{_EM}.runtime:capture_experiment_tool_results"),
    ("enforce_pending_record_result", "after_model", f"{_EM}.runtime:enforce_pending_record_result"),
    ("enforce_continue_until_reporting", "after_model", f"{_EM}.runtime:enforce_continue_until_reporting"),
    ("rewrite_mismatched_control_action", "after_model", f"{_EM}.runtime:rewrite_mismatched_control_action"),
    # Collapse parallel ExperimentModuleAgent fan-out into one merged request.
    ("coalesce_experiment_module_calls", "after_model", f"{_EM}.runtime:coalesce_experiment_module_calls"),
    ("suppress_experiment_module_after_completed", "after_model", f"{_EM}.runtime:suppress_experiment_module_after_completed"),
)


def _em_lazy_factory(path: str):
    module_name, attr = path.split(":", 1)
    return lambda ctx: getattr(import_module(module_name), attr)


for _key, _hook, _path in _EM_CALLBACKS:
    _cb(_key, _hook, factory=_em_lazy_factory(_path))

# Critic callbacks: their LLM prompts embed the orchestrator's current roster.
_cb("pre_action_critique", "after_model", factory=_pre_action_critique)


def _spoiler_guard_query(ctx):
    from CoScientist.agents.callbacks.spoiler_guard import guard_from_settings
    return guard_from_settings().guard_query


def _spoiler_guard_result(ctx):
    from CoScientist.agents.callbacks.spoiler_guard import guard_from_settings
    return guard_from_settings().guard_result


# Blind-review guard (agents/blind.yaml): first in both chains, so a refused
# query never reaches the tool and a filtered result is what the counters and
# the link registry behind it see.
_cb("guard_spoiler_query", "before_tool", factory=_spoiler_guard_query)
_cb("guard_spoiler_result", "after_tool", factory=_spoiler_guard_result)
_cb("post_action_critique", "after_tool", factory=_post_action_critique)


# ── Agent classes / output schemas / planners ────────────────────────────────

def _register_classes() -> None:
    from CoScientist.agents.custom_agents import (
        ExecutorSwitchAgent,
        WebToolsDeployerAgent,
    )
    from CoScientist.hitl.session_agent import SessionAgent
    from CoScientist.context_init.agent import ContextInitSessionAgent
    from CoScientist.context_init.tz_agent import TZSpecSessionAgent
    from CoScientist.experiments.review import ExperimentReviewSessionAgent

    REGISTRY.register_agent_class("session", SessionAgent)
    REGISTRY.register_agent_class("web_tools_deployer", WebToolsDeployerAgent)
    # Runs ONE of its children: the normal executor, or the reranker fallback.
    REGISTRY.register_agent_class("executor_switch", ExecutorSwitchAgent)
    # Context-init pre-stage: the review shows a STRUCTURED FORM (research frame)
    # and seeds the confirmed frame into the research graph.
    REGISTRY.register_agent_class("context_init_session", ContextInitSessionAgent)
    # The ТЗ stage of the general profile: issues the document from the
    # frame the stage above confirmed. No review of its own — the frame was
    # the review, and this is that frame written out to GOST 19.201-78.
    REGISTRY.register_agent_class("tz_spec_session", TZSpecSessionAgent)
    REGISTRY.register_agent_class("experiment_review", ExperimentReviewSessionAgent)


def _register_schemas() -> None:
    from CoScientist.storage import MCPRanking, ToolRanking
    from CoScientist.context_init.models import ResearchFrame
    from CoScientist.context_init.tz_agent import TZProse
    from CoScientist.experiments.schemas import (
        ExperimentPlan,
        ExperimentTask,
        PlanCritique,
        TaskResult,
    )

    REGISTRY.register_output_schema("tool_ranking", ToolRanking)
    REGISTRY.register_output_schema("mcp_ranking", MCPRanking)
    # Framing entities of the meta-model, filled per run (context_init pre-stage).
    REGISTRY.register_output_schema("research_frame", ResearchFrame)
    REGISTRY.register_output_schema("tz_prose", TZProse)
    REGISTRY.register_output_schema("experiment_plan", ExperimentPlan)
    REGISTRY.register_output_schema("experiment_task", ExperimentTask)
    REGISTRY.register_output_schema("task_result", TaskResult)
    REGISTRY.register_output_schema("plan_critique", PlanCritique)


def _register_planners() -> None:
    from google.adk.planners import PlanReActPlanner

    REGISTRY.register_planner("plan_react", PlanReActPlanner)


_register_classes()
_register_schemas()
_register_planners()
