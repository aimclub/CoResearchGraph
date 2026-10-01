"""A confirmation that has to go through the web interface.

For actions that must never run unreviewed (the rig campaign): with HITL
switched off or no web interface attached the action is blocked instead of
running without a card.
"""
from __future__ import annotations

from typing import Optional

from CoScientist.config import get_settings
from CoScientist.hitl.handler import AbstractHITLHandler, DelegatingHITLHandler


def _web_attached(handler: AbstractHITLHandler) -> bool:
    while isinstance(handler, DelegatingHITLHandler):
        handler = handler.delegate
    try:
        from CoScientist.web.handler import WebHITLHandler
    except Exception:  # noqa: BLE001 — no web package, no web interface
        return False
    return isinstance(handler, WebHITLHandler)


def unavailable_reason(handler: AbstractHITLHandler) -> Optional[str]:
    """Why the web interface cannot be asked right now, or None."""
    if not get_settings().web.hitl_enabled:
        return "HITL is switched off"
    if not _web_attached(handler):
        return "the web interface is not attached"
    return None


__all__ = ["unavailable_reason"]
