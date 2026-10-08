"""Callbacks for medical image handling across the agent pipeline."""

import hashlib
from typing import List

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse, LlmRequest
from google.genai.types import Part

_STATE_KEY = "uploaded_medical_artifacts"


async def before_model_modifier(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Orchestrator-level callback: save uploaded files as artifacts and register their IDs.

    For every Part that carries inline_data:
      1. Derive a deterministic artifact ID from the file content.
      2. Save the Part as an artifact (idempotent).
      3. Register the artifact ID in session state under _STATE_KEY.
      4. Prepend a text Part annotating the artifact ID so the LLM knows what to pass downstream.
    """
    for content in llm_request.contents:
        if not content.parts:
            continue

        modified_parts: List[Part] = []
        for part in content.parts:
            if part.inline_data:
                modified_parts.extend(
                    await _process_inline_data(part, callback_context)
                )
            else:
                modified_parts.append(part)

        content.parts = modified_parts


async def med_agent_before_model(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """MedicalAgent-level callback: inject available artifact IDs from session state.

    Reads the registry written by before_model_modifier and prepends a reminder
    to the current turn so the agent always knows which artifacts are available,
    regardless of what the orchestrator included in its call.
    """
    uploads: List[str] = callback_context.state.get(_STATE_KEY, [])
    if not uploads:
        return None

    artifact_list = ", ".join(uploads)
    reminder = Part(
        text=f"[Available medical image artifacts] artifact_ids: {artifact_list}"
    )

    for content in reversed(llm_request.contents):
        if content.role == "user":
            content.parts = [reminder] + list(content.parts or [])
            break

    return None


# ─── helpers ─────────────────────────────────────────────────────────────────

async def _process_inline_data(
    part: Part, callback_context: CallbackContext
) -> List[Part]:
    artifact_id = _make_artifact_id(part)

    existing = await callback_context.list_artifacts()
    if artifact_id not in existing:
        await callback_context.save_artifact(filename=artifact_id, artifact=part)

    # Register in session state so any agent can find it later
    uploads: List[str] = list(callback_context.state.get(_STATE_KEY, []))
    if artifact_id not in uploads:
        uploads.append(artifact_id)
        callback_context.state[_STATE_KEY] = uploads

    return [
        Part(text=f"[Uploaded file] artifact_id={artifact_id}"),
        # Raw bytes are stored in the artifact store; do not forward to the LLM.
        # LiteLLM cannot handle binary MIME types (e.g. application/dicom).
    ]


def _make_artifact_id(part: Part) -> str:
    display_name = (part.inline_data.display_name or "file").rsplit("/", 1)[-1]
    content_hash = hashlib.sha256(part.inline_data.data).hexdigest()[:16]

    if "." in display_name:
        ext = display_name.rsplit(".", 1)[-1]
    else:
        ext = part.inline_data.mime_type.split("/")[-1]

    return f"upload_{content_hash}.{ext}"
