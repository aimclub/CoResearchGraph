"""A per-call MCP client for the normcontrol server's NIR tools.

Modelled on ``tools/vault_client.py``, and for the same reason: every call opens
its own session and closes it. The ADK session manager caches a session and
binds its exit stack to the loop that created it, so a shared toolset driven
from a fresh loop fails on the second call. The raw MCP SDK carries no such
state.

There is a second reason here. The agent cannot hold the ``assets`` argument:
one 287 KB figure is ~383 000 base64 characters, and a model has no business
emitting that. So the toolset is wrapped rather than handed over, and the
wrapper fills ``assets`` from disk.

No function raises. A missing URL, an unreachable server or a refusal all come
back as a result dict the caller can report. A run that produced results must
still produce a report.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from CoScientist.config import get_settings
from CoScientist.reporting.nir import contract

logger = logging.getLogger(__name__)

#: A render lays out a whole DOCX, embeds fonts and uploads to S3. The vault's
#: 30 s is far too short; the reference client in the MCP repo uses 180 s.
_TIMEOUT = 300.0

#: Only one DOCX renders at a time server-wide — a concurrent call is refused
#: outright, not queued. These are the waits between retries of that refusal.
_BUSY_BACKOFF = (2.0, 5.0, 10.0, 20.0)

TOOL_VALIDATE = "nir_report_validate"
TOOL_RENDER = "nir_report_render"


def normcontrol_url() -> Optional[str]:
    """Read at call time, so a test or a redeploy can point this elsewhere.

    There is no default. The address moves between a self-hosted instance and
    the ITMO one, and a literal in the code would outlive whichever is current.
    """
    return get_settings().mcp.normcontrol_url


def _failure(error: str, message: str) -> Dict[str, Any]:
    return {"ok": False, "valid": False, "error": error, "errors": [], "warnings": [], "message": message}


def _payload(result: Any) -> Optional[Dict[str, Any]]:
    """Unwrap the tool reply.

    FastMCP returns a dict as ``structuredContent`` and *also* as JSON inside
    ``content[].text``. Prefer the structured form and fall back to parsing the
    text, so this keeps working if either side changes which it sends.
    """
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if not isinstance(text, str):
            continue
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


async def _call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """One tool call against the normcontrol MCP."""
    url = normcontrol_url()
    if not url:
        return _failure(
            "not_configured",
            "MCP__NORMCONTROL_URL не задан — сервер отчётов о НИР не настроен",
        )

    # Imported here so importing this module does not pull the MCP stack into a
    # process that never generates a NIR report.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    try:
        async with streamablehttp_client(url, timeout=_TIMEOUT) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
    except Exception as exc:  # noqa: BLE001 - the MCP must never sink a run
        logger.warning("normcontrol: %s failed (%s)", tool_name, exc)
        return _failure("unreachable", f"Сервер нормоконтроля недоступен: {exc}")

    payload = _payload(result)
    if payload is None:
        logger.warning("normcontrol: %s returned no readable payload", tool_name)
        return _failure("bad_response", "Сервер вернул ответ, который не удалось разобрать")
    return payload


async def _call_with_retry(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call, retrying only the one error a retry can fix.

    ``server_busy`` means another report was rendering at that moment. Every
    other failure is either ours (the document is wrong) or the deployment's
    (fonts missing, storage down), and repeating the call would only add load.
    """
    payload = await _call(tool_name, arguments)
    for delay in _BUSY_BACKOFF:
        if payload.get("error") != contract.ERROR_SERVER_BUSY:
            return payload
        logger.info("normcontrol: server busy, retrying %s in %.0fs", tool_name, delay)
        await asyncio.sleep(delay)
        payload = await _call(tool_name, arguments)
    return payload


def _arguments(
    values: Dict[str, Any],
    assets: Optional[Dict[str, str]],
    mode: str,
    page_count: Optional[int],
    page_map: Optional[Dict[str, int]],
) -> Dict[str, Any]:
    arguments: Dict[str, Any] = {"values": values, "mode": mode}
    if assets:
        arguments["assets"] = assets
    # Omitted rather than sent as null: the server types these as
    # ``PositiveInt | None`` with strict validation, and a stray 0 or "" is an
    # invalid_arguments refusal.
    if page_count:
        arguments["page_count"] = page_count
    if page_map:
        arguments["page_map"] = page_map
    return arguments


async def nir_validate(
    values: Dict[str, Any],
    *,
    assets: Optional[Dict[str, str]] = None,
    mode: str = contract.DEFAULT_MODE,
    page_count: Optional[int] = None,
    page_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Check the document without building anything.

    Returns the server's own shape: ``ok``/``valid``, field-level ``errors``,
    ``warnings`` and ``applied_defaults``. Cheap, and it never touches the
    render lock — so the author can iterate here before spending a render.
    """
    return await _call_with_retry(
        TOOL_VALIDATE, _arguments(values, assets, mode, page_count, page_map)
    )


async def nir_render(
    values: Dict[str, Any],
    *,
    assets: Optional[Dict[str, str]] = None,
    mode: str = contract.DEFAULT_MODE,
    page_count: Optional[int] = None,
    page_map: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Build the DOCX and return the server's download link.

    On success the payload carries ``output_docx`` (presigned, 24 hours),
    ``sha256``, ``task_id``, ``mime_type`` and ``expires_at``. That link points
    into *their* bucket and expires, so the caller is expected to fetch the file
    and keep it — see ``tools/nir_report_tool.py``.
    """
    return await _call_with_retry(
        TOOL_RENDER, _arguments(values, assets, mode, page_count, page_map)
    )


async def list_tools() -> Optional[list]:
    """Tool names the server advertises, or None when it cannot be reached.

    Used by the smoke check and by the callback's availability probe; keeping it
    here means the connection details live in exactly one module.
    """
    url = normcontrol_url()
    if not url:
        return None
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    try:
        async with streamablehttp_client(url, timeout=30.0) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listing = await session.list_tools()
        return [tool.name for tool in listing.tools]
    except Exception as exc:  # noqa: BLE001
        logger.warning("normcontrol: cannot list tools at %s (%s)", url, exc)
        return None


__all__ = [
    "normcontrol_url",
    "nir_validate",
    "nir_render",
    "list_tools",
    "TOOL_VALIDATE",
    "TOOL_RENDER",
]
