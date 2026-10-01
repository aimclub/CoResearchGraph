"""Callbacks for ResearchAgent and user-uploaded paper state."""

import logging
import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai.types import Part

from CoScientist.chemical_utils.smiles_extraction import extract_reactions, extract_smiles
from CoScientist.paper_parser.s3_connection import s3_service
from CoScientist.graph.session_scope import session_key

logger = logging.getLogger(__name__)

_PAPER_STATE_KEY = "uploaded_paper_s3_keys"

# RAG/paper tools ResearchAgent can call that might surface molecule/reaction
# data — everything else (websearch, task_tracker) is skipped without even trying.
# The live paper-analysis/papers-search MCP servers are versioned independently
# of this repo (`runtime_resolved=True` in assembly/bindings.py — their tool
# surface is discovered live, not pinned here), so this set carries both the
# current deployed names AND older ones we've seen, to survive a rename on
# either side without silently going dark. Confirmed live 2026-08-31:
# paper_analysis v3.1.1 exposes explore_scientific_database (renamed from
# explore_scientific_database) + explore_my_papers + find_papers_in_db +
# find_relevant_data_in_db; papers_search v3.1.0 exposes search_entity +
# search_papers + download_papers_from_search.
_SMILES_SOURCE_TOOLS = {
    "explore_scientific_database",  # pre-3.x name — kept for older deployments
    "explore_scientific_database",
    "explore_my_papers",
    "find_papers_in_db",
    "find_relevant_data_in_db",
    "search_entity",
    "search_papers",
    "download_papers_from_search",
}
_LITERATURE_SMILES_STATE_KEY = "literature_smiles"
_LITERATURE_SMILES_SUMMARY_STATE_KEY = "literature_smiles_summary"
_LITERATURE_REACTIONS_STATE_KEY = "literature_reactions"
_LITERATURE_REACTIONS_SUMMARY_STATE_KEY = "literature_reactions_summary"
_USER_ID_ENV = "USER_ID"
_SESSION_ID_ENV = "SESSION_ID"
_UPLOADED_PAPERS_PATH_ENV = "STORAGE__UPLOADED_PAPERS"
_DEFAULT_LOCAL_PAPERS_ROOT = Path(__file__).resolve().parents[2] / "local_papers"

_upload_locks: dict[str, asyncio.Lock] = {}


async def papers_agent_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """ResearchAgent-level callback: prepare uploaded papers and inject state into the user's prompt."""
    await ensure_local_papers_uploaded(callback_context)

    s3_keys: List[str] = callback_context.state.get(_PAPER_STATE_KEY, [])
    downloaded_keys: List[str] = callback_context.state.get("downloaded_paper_s3_keys", [])
    all_keys = s3_keys + downloaded_keys

    if not all_keys:
        reminder = Part(
            text=(
                "[Available uploaded papers] No pre-uploaded papers are available for this session. "
                "Do not invent S3 keys. "
                "You may still call explore_my_papers if you have S3 keys from a previous download_papers_from_search result."
            )
        )
        for content in reversed(llm_request.contents):
            if content.role == "user":
                content.parts = [reminder] + list(content.parts or [])
                break
        return None

    paper_list = ", ".join(all_keys)
    reminder = Part(
        text=(
            "[Available papers] The following S3 keys are available: "
            f"{paper_list}. "
            "Use these s3_keys when calling explore_my_papers."
        )
    )

    for content in reversed(llm_request.contents):
        if content.role == "user":
            content.parts = [reminder] + list(content.parts or [])
            break

    return None


async def ensure_local_papers_uploaded(callback_context: CallbackContext) -> None:
    """Upload local papers to S3 and register their keys in session state."""
    user_id, session_id = session_key(callback_context)
    scope_key = f"{user_id}:{session_id}"
    _upload_locks.setdefault(scope_key, asyncio.Lock())

    async with _upload_locks[scope_key]:
        if callback_context.state.get(_PAPER_STATE_KEY):
            print(
                "[S3 papers] already registered for "
                f"user={user_id} session={session_id}: "
                f"{callback_context.state[_PAPER_STATE_KEY]}",
                flush=True,
            )
            return

        papers_dir = _resolve_local_papers_dir()
        if papers_dir is None or not papers_dir.exists() or not papers_dir.is_dir():
            print(
                f"[S3 papers] local directory not found: {papers_dir}",
                flush=True,
            )
            logger.debug("No local papers directory found for uploaded papers.")
            return

        pdf_files = [
            path
            for path in sorted(papers_dir.iterdir())
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]

        if not pdf_files:
            print(f"[S3 papers] no PDF files found in {papers_dir}", flush=True)
            logger.debug("Local uploaded papers directory is empty: %s", papers_dir)
        else:
            print(
                f"[S3 papers] found {len(pdf_files)} PDF(s) in {papers_dir}; "
                f"uploading to bucket={s3_service.bucket_name!r} "
                f"prefix={user_id}/{session_id}/uploaded_papers",
                flush=True,
            )
            logger.info("Found %d local PDF(s) for upload in %s", len(pdf_files), papers_dir)

        prefix = f"{user_id}/{session_id}/uploaded_papers"
        uploaded_keys: List[str] = []

        if pdf_files:
            for pdf_path in pdf_files:
                try:
                    s3_service.upload_file_object(prefix, pdf_path.name, str(pdf_path))
                    s3_key = f"{prefix}/{pdf_path.name}"
                    uploaded_keys.append(s3_key)
                    print(f"[S3 papers] UPLOAD OK: {s3_key}", flush=True)
                    logger.info("Uploaded local paper to S3: %s", s3_key)
                except Exception as exc:
                    print(
                        f"[S3 papers] UPLOAD FAILED: {pdf_path}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    logger.warning(
                        "Failed to upload local paper %s to S3: %s",
                        pdf_path,
                        exc,
                    )

        if not uploaded_keys:
            try:
                existing_keys = s3_service.list_objects(prefix)
            except Exception as exc:
                print(
                    f"[S3 papers] LIST FAILED for {prefix}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                existing_keys = []
            if existing_keys:
                uploaded_keys = existing_keys
                print(
                    f"[S3 papers] found existing object(s): {existing_keys}",
                    flush=True,
                )
                logger.info(
                    "No new uploads; found existing S3 keys under prefix %s: %s",
                    prefix,
                    existing_keys,
                )
            else:
                print(
                    f"[S3 papers] no objects found under prefix {prefix}",
                    flush=True,
                )
                logger.debug("No S3 keys found under prefix %s", prefix)

        if uploaded_keys:
            callback_context.state[_PAPER_STATE_KEY] = uploaded_keys
            logger.info(
                "Registered uploaded paper S3 keys in session state: %s",
                uploaded_keys,
            )


def cleanup_uploaded_papers(user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
    """Delete uploaded paper objects from S3 for the given user/session."""
    user_id = user_id or _get_user_id()
    session_id = session_id or _get_session_id()
    prefix = f"{user_id}/{session_id}/uploaded_papers"

    existing_keys = s3_service.list_objects(prefix)
    if not existing_keys:
        logger.info("No uploaded paper objects to clean in S3 for prefix %s", prefix)
        return

    logger.info(
        "Cleaning up %d uploaded paper object(s) from S3 under prefix %s",
        len(existing_keys),
        prefix,
    )

    try:
        s3_service.clean_up_by_prefix(prefix)
        logger.info("Completed cleanup of uploaded papers under prefix %s", prefix)
    except Exception as exc:
        logger.warning(
            "Failed to clean up uploaded paper objects under prefix %s: %s",
            prefix,
            exc,
        )

    session_key = f"{user_id}:{session_id}"
    _upload_locks.pop(session_key, None)


def _resolve_local_papers_dir() -> Optional[Path]:
    custom_path = os.getenv(_UPLOADED_PAPERS_PATH_ENV)
    if custom_path:
        resolved = Path(custom_path)
        if resolved.exists() and resolved.is_dir():
            return resolved
        logger.warning(
            "Configured %s=%r does not exist or is not a directory; falling back to default.",
            _UPLOADED_PAPERS_PATH_ENV,
            custom_path,
        )

    if _DEFAULT_LOCAL_PAPERS_ROOT.exists() and _DEFAULT_LOCAL_PAPERS_ROOT.is_dir():
        return _DEFAULT_LOCAL_PAPERS_ROOT

    return None


def capture_literature_smiles(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """ResearchAgent after_tool callback: pull SMILES out of RAG/paper-search
    results and hand them to the design stage via session state.

    ADK only keeps the LLM's paraphrase of a tool result under output_key —
    the raw response is gone once the agent turn ends. Molecule-bearing
    literature tools (explore_scientific_database, explore_my_papers,
    search_papers, download_papers_from_search) are prompted to copy SMILES
    verbatim into their answer, but nothing marks which substring is one, and
    a paraphrase could still drop or mangle it. Scanning the raw tool_response
    here, at the tool-call boundary, catches it before that can happen.
    """
    if tool.name not in _SMILES_SOURCE_TOOLS:
        return

    try:
        found = extract_smiles(str(tool_response))
        if not found:
            return

        existing: List[str] = tool_context.state.get(_LITERATURE_SMILES_STATE_KEY, [])
        merged = existing + [s for s in found if s not in existing]
        tool_context.state[_LITERATURE_SMILES_STATE_KEY] = merged
        tool_context.state[_LITERATURE_SMILES_SUMMARY_STATE_KEY] = _render_smiles_summary(merged)
        logger.info(
            "[ResearchAgent] captured %d new SMILES from %s (%d total in session)",
            len(found),
            tool.name,
            len(merged),
        )
    except Exception as exc:
        logger.warning("Failed to extract SMILES from %s result: %s", tool.name, exc)


def _render_smiles_summary(smiles_list: List[str]) -> str:
    if not smiles_list:
        return ""
    lines = "\n".join(f"- {s}" for s in smiles_list)
    return f"SMILES молекул, найденных литературным RAG-поиском ({len(smiles_list)}):\n{lines}"


def capture_literature_reactions(
    tool: BaseTool,
    args: Dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> None:
    """ResearchAgent after_tool callback: pull REACTION SMILES (reactants>
    agents>products, at least one reactant and one product) out of RAG/paper-
    search results and hand them to the retrosynthesis stage via session state.

    Same tool-call-boundary rationale as capture_literature_smiles, and same
    RAG/paper tools — a route/synthesis literature query (LIT-02/LIT-03-style:
    "маршруты синтеза") comes back through the same tools as a
    molecule-analogue query, just with a different answer shape. Molecules and
    reactions are mutually exclusive by charset (see smiles_extraction.py), so
    running both extractors over the same response is safe and non-redundant.
    """
    if tool.name not in _SMILES_SOURCE_TOOLS:
        return

    try:
        found = extract_reactions(str(tool_response))
        if not found:
            return

        existing: List[str] = tool_context.state.get(_LITERATURE_REACTIONS_STATE_KEY, [])
        merged = existing + [r for r in found if r not in existing]
        tool_context.state[_LITERATURE_REACTIONS_STATE_KEY] = merged
        tool_context.state[_LITERATURE_REACTIONS_SUMMARY_STATE_KEY] = _render_reactions_summary(merged)
        logger.info(
            "[ResearchAgent] captured %d new reaction(s) from %s (%d total in session)",
            len(found),
            tool.name,
            len(merged),
        )
    except Exception as exc:
        logger.warning("Failed to extract reactions from %s result: %s", tool.name, exc)


def _render_reactions_summary(reactions: List[str]) -> str:
    if not reactions:
        return ""
    lines = "\n".join(f"- {r}" for r in reactions)
    return f"Реакции (reactants>agents>products), найденные литературным RAG-поиском ({len(reactions)}):\n{lines}"


def _get_user_id() -> str:
    return os.getenv(_USER_ID_ENV, "user_1")


def _get_session_id() -> str:
    return os.getenv(_SESSION_ID_ENV, "session_001")
