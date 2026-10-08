"""Callbacks for ResearchAgent and user-uploaded paper state."""

import logging
import os
import asyncio
from pathlib import Path
from typing import List, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai.types import Part

from CoScientist.paper_parser.s3_connection import s3_service
from CoScientist.graph.session_scope import session_key

logger = logging.getLogger(__name__)

_PAPER_STATE_KEY = "uploaded_paper_s3_keys"
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
            return

        papers_dir = _resolve_local_papers_dir()
        if papers_dir is None or not papers_dir.exists() or not papers_dir.is_dir():
            logger.debug("No local papers directory found for uploaded papers.")
            return

        pdf_files = [
            path
            for path in sorted(papers_dir.iterdir())
            if path.is_file() and path.suffix.lower() == ".pdf"
        ]

        if not pdf_files:
            logger.debug("Local uploaded papers directory is empty: %s", papers_dir)
        else:
            logger.info("Found %d local PDF(s) for upload in %s", len(pdf_files), papers_dir)

        prefix = f"{user_id}/{session_id}/uploaded_papers"
        uploaded_keys: List[str] = []

        if pdf_files:
            for pdf_path in pdf_files:
                try:
                    s3_service.upload_file_object(prefix, pdf_path.name, str(pdf_path))
                    s3_key = f"{prefix}/{pdf_path.name}"
                    uploaded_keys.append(s3_key)
                    logger.info("Uploaded local paper to S3: %s", s3_key)
                except Exception as exc:
                    logger.warning(
                        "Failed to upload local paper %s to S3: %s",
                        pdf_path,
                        exc,
                    )

        if not uploaded_keys:
            existing_keys = s3_service.list_objects(prefix)
            if existing_keys:
                uploaded_keys = existing_keys
                logger.info(
                    "No new uploads; found existing S3 keys under prefix %s: %s",
                    prefix,
                    existing_keys,
                )
            else:
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


def _get_user_id() -> str:
    return os.getenv(_USER_ID_ENV, "user_1")


def _get_session_id() -> str:
    return os.getenv(_SESSION_ID_ENV, "session_001")
