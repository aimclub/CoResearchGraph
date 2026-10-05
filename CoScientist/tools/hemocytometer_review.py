"""Human-in-the-loop manual review for hemocytometer images.

A separate ToolEntry (not part of HITLToolset) so it attaches only where
"hemocytometer" itself is attached in system.yaml, instead of leaking into
every agent that has hitl: true. See hitl/tool.py's wait_for_external_response
for why this goes through the same handler as request_approval/request_selection.
"""
import uuid
from typing import Optional

from google.adk.tools.tool_context import ToolContext

from CoScientist.config import get_settings
from CoScientist.graph.session_scope import session_key
from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.hitl.tool import wait_for_external_response

settings = get_settings()

# A manual review needs real human time (open the link, adjust params, look
# at the image) — much longer than the approve/reject default timeout.
_REVIEW_TIMEOUT_SECONDS = 1800.0


async def request_hemocytometer_review(
    tool_context: ToolContext,
    message: Optional[str] = None,
) -> list[dict]:
    """Ask a human to manually review/adjust one or more hemocytometer images in
    the standalone web UI, and WAIT for them to submit before returning.

    Use this (instead of analyze_hemocytometer_image) when the user wants to
    visually inspect the photo(s) or try different chamber parameters
    themselves before you use the numbers — e.g. "let me check it myself
    first". The human can compute as many images as they want in the UI
    before clicking "Отправить результат агенту" ONCE — that single click
    submits everything they computed in that session as a batch. This call
    blocks until they do, which can take a while — there's no shortcut, a
    human needs real time to open the link and look at the image. It always
    returns what they actually submitted, never an empty or fabricated result.

    Args:
        message: Optional extra context to show the human (e.g. why you're
            asking them to review it). A link and instructions are always
            included automatically — you don't need to repeat the URL.

    Returns:
        A list of dicts, one per image the human computed and submitted —
        each with cell_count, concentration_cells_per_ml, camera_type,
        dilution, in the order they were computed.
    """
    user_id, session_id = session_key(tool_context)
    request_id = str(uuid.uuid4())
    ui_url = (settings.mcp.hemocytometer_ui_url or "").rstrip("/")
    review_url = f"{ui_url}?user_id={user_id}&request_id={request_id}"

    text = f"{message}\n\n" if message else ""
    text += (
        "Перейди по ссылке, выполни анализ вручную и нажми «Отправить "
        f"результат агенту», когда будешь готов: {review_url}"
    )

    request = HITLRequest(
        agent_name=tool_context.agent_name,
        action_type=HITLAction.PROVIDE_INPUT,
        message=text,
        context={
            "review_url": review_url,
            "_session": {"user_id": user_id, "session_id": session_id},
        },
        invoked_via="tool",
        timeout_seconds=_REVIEW_TIMEOUT_SECONDS,
        # No automatic mode can fabricate a real cell count — this must reach
        # a human even when HITL__MODE=auto would otherwise silently approve it.
        requires_human=True,
    )
    response = await wait_for_external_response(request, request_id)
    return (response.form_values or {}).get("results", [])
