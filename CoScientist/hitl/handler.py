"""HITL handlers — abstract interface and implementations."""

import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod

from CoScientist.hitl.models import HITLRequest, HITLResponse, HITLAction
from CoScientist.hitl.resolver import resolve_auto, resolve_timeout


def _auto_approves() -> bool:
    """Whether this run answers its own confirmations (`HITL__MODE=auto`).

    Imported inside the function rather than at module scope: `hitl.mode` reads
    the settings, and this module is imported while they are still being built.
    """
    try:
        from CoScientist.hitl.mode import auto_approves

        return auto_approves()
    except Exception:  # noqa: BLE001 — an unreadable mode still asks the human
        return False


class AbstractHITLHandler(ABC):
    """Abstract interface for handling HITL requests.

    Implement this for different UIs: console, web chat, Telegram, etc.
    """

    @abstractmethod
    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        """Process a HITL request and return the human's response."""
        ...

    async def notify(self, payload: dict) -> None:
        """Tell the human something without waiting for an answer.

        Used for Work Order notices (a read-tier contract, step progress, a
        deviation). ``payload`` may carry ``_session`` like a request context.
        Default: log it — a UI without a notice channel loses nothing but the
        heads-up.
        """
        logging.getLogger(__name__).info(
            "[HITL notice] %s %s", payload.get("agent_name"), payload.get("kind"),
        )


class DelegatingHITLHandler(AbstractHITLHandler):
    """A handler that delegates to another handler, allowing runtime swapping."""

    def __init__(self, delegate: AbstractHITLHandler):
        self.delegate = delegate

    def set_delegate(self, delegate: AbstractHITLHandler):
        self.delegate = delegate

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        return await self.delegate.handle_request(request)

    async def notify(self, payload: dict) -> None:
        await self.delegate.notify(payload)

    def __deepcopy__(self, memo):
        # Return self so that workflow deepcopies share the same handler instance
        return self



class ConsoleHITLHandler(AbstractHITLHandler):
    """Simple console-based HITL handler (for local development/testing)."""

    async def notify(self, payload: dict) -> None:
        text = payload.get("text")
        if text and sys.stdin.isatty():
            print(f"\n[HITL notice] {payload.get('agent_name')}: {text}")
        else:
            await super().notify(payload)

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        # `auto` means nobody is asked — including here. Without this the one
        # mode whose whole point is that it runs unattended blocked on `input()`
        # forever on an interactive CLI run.
        if _auto_approves() and not request.requires_human:
            logging.getLogger(__name__).info(
                "[HITL] answered by the mode, not a human (HITL__MODE=auto): %s",
                request.agent_name)
            return resolve_auto(request)
        # No console attached (A2A/uvicorn server, headless run, cron): reading
        # stdin would hang the process — or steal another server's stdin — which
        # is exactly how HITL "did not work" when served over A2A. Fall through
        # with an explicit, logged decision instead of blocking. Agents served
        # over A2A get the non-blocking long-running tools (hitl/a2a_tools.py);
        # this is the safety net for any other headless path.
        if not sys.stdin.isatty():
            # The default is REFUSE, and it follows from the modes: silence
            # approves in none of them, and a headless process is silence. It
            # used to default to approve, which meant `debug` — the mode that
            # exists so a run cannot leave without you — approved every request
            # instantly on any box with no terminal, side-effect Work Orders
            # included. `HITL_HEADLESS_POLICY=approve` still says otherwise
            # explicitly, for a lane that wants it.
            approve = os.getenv("HITL_HEADLESS_POLICY", "reject").lower() == "approve"
            logging.getLogger(__name__).warning(
                "[HITL] no console attached — auto-%s for %s: %s",
                "approving" if approve else "rejecting", request.agent_name, request.message[:160],
            )
            # `timed_out` on the refusal, and `instructions` left empty: nobody
            # DECIDED this. Callers that tell a human's no from an absence read
            # that flag — the experiment plan review is one, and without it a
            # cron run recorded "rejected by the operator" against an operator
            # who was never there — and every other caller reads `instructions`
            # as the operator's own words.
            if approve:
                return resolve_auto(request)
            return resolve_timeout(reason="no_console_attached")
        print(f"\n{'=' * 60}")
        print(f"[HITL] Agent '{request.agent_name}' requests: {request.action_type.value}. Invoked_via: {request.invoked_via}")
        print(f"Message: {request.message}")

        if request.context and "output" in request.context:
            print(f"\nPROPOSED PLAN/OUTPUT:")
            print(f"{'-' * 30}")
            print(f"{request.context['output']}")
            print(f"{'-' * 30}")

        if request.options:
            print("\nOptions:")
            for i, opt in enumerate(request.options, 1):
                print(f"  {i}. {opt}")

        if request.form:
            return await self._prompt_form(request)
        if request.action_type == HITLAction.SELECT and request.options:
            return await self._prompt_select(request)

        is_simple_toggle = (request.invoked_via == "callback" and request.action_type == HITLAction.APPROVE)
        
        print("\nAction Menu:")
        if is_simple_toggle:
            print("  1. Approve (Proceed with agent execution)")
            print("  2. Reject (Skip this agent's execution)")
        else:
            print("  1. Approve (Accept and proceed)")
            print("  2. Edit (Provide feedback / request changes to agent)")
            print("  3. Stop program (Exit completely)")
        
        while True:
            choice = await asyncio.to_thread(input, f"\nSelect action (1-{2 if is_simple_toggle else 3}): ")
            choice = choice.strip()

            if choice == "1":
                return HITLResponse(
                    action=HITLAction.APPROVE,
                    approved=True
                )
            elif choice == "2":
                if is_simple_toggle:
                    return HITLResponse(
                        action=HITLAction.REJECT,
                        approved=False,
                        instructions="Human rejected execution."
                    )
                else:
                    feedback = await asyncio.to_thread(input, "Enter your feedback/changes: ")
                    return HITLResponse(
                        action=HITLAction.EDIT,
                        approved=False,
                        instructions=feedback
                    )
            elif choice == "3" and not is_simple_toggle:
                print("\nStopping program execution based on user request...")
                sys.exit(0)
            else:
                print(f"Invalid choice. Please enter a valid option.")

    async def _prompt_select(self, request: HITLRequest) -> HITLResponse:
        n = len(request.options)
        while True:
            raw = await asyncio.to_thread(input, f"\nSelect option (1-{n}): ")
            raw = raw.strip()
            if raw.isdigit():
                idx = int(raw) - 1
                if 0 <= idx < n:
                    return HITLResponse(
                        action=HITLAction.SELECT,
                        approved=True,
                        selected_option=request.options[idx],
                    )
            print(f"Invalid choice. Please enter 1-{n}.")

    async def _prompt_form(self, request: HITLRequest) -> HITLResponse:
        form_values: dict = {}
        intro = (request.form or {}).get("intro")
        if intro:
            print(f"\n{intro}")
        for block in (request.form or {}).get("blocks") or []:
            title = block.get("title") or ""
            if title:
                print(f"\n{title}")
            answers: dict = {}
            for field in block.get("fields") or []:
                name = field.get("name") or ""
                if not name:
                    continue
                raw = await asyncio.to_thread(input, f"  {name}: ")
                value = raw.strip()
                if value:
                    answers[name] = value
            if answers:
                form_values[title] = answers
        return HITLResponse(
            action=HITLAction.APPROVE,
            approved=True,
            form_values=form_values or None,
        )
