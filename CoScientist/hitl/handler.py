"""HITL handlers — abstract interface and implementations."""

import asyncio
import logging
import os
import sys
from abc import ABC, abstractmethod

from CoScientist.hitl.models import HITLRequest, HITLResponse, HITLAction


class AbstractHITLHandler(ABC):
    """Abstract interface for handling HITL requests.

    Implement this for different UIs: console, web chat, Telegram, etc.
    """

    @abstractmethod
    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        """Process a HITL request and return the human's response."""
        ...


class DelegatingHITLHandler(AbstractHITLHandler):
    """A handler that delegates to another handler, allowing runtime swapping."""

    def __init__(self, delegate: AbstractHITLHandler):
        self.delegate = delegate

    def set_delegate(self, delegate: AbstractHITLHandler):
        self.delegate = delegate

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        return await self.delegate.handle_request(request)

    def __deepcopy__(self, memo):
        # Return self so that workflow deepcopies share the same handler instance
        return self



class ConsoleHITLHandler(AbstractHITLHandler):
    """Simple console-based HITL handler (for local development/testing)."""

    async def handle_request(self, request: HITLRequest) -> HITLResponse:
        # No console attached (A2A/uvicorn server, headless run, cron): reading
        # stdin would hang the process — or steal another server's stdin — which
        # is exactly how HITL "did not work" when served over A2A. Fall through
        # with an explicit, logged decision instead of blocking. Agents served
        # over A2A get the non-blocking long-running tools (hitl/a2a_tools.py);
        # this is the safety net for any other headless path.
        if not sys.stdin.isatty():
            approve = os.getenv("HITL_HEADLESS_POLICY", "approve").lower() != "reject"
            logging.getLogger(__name__).warning(
                "[HITL] no console attached — auto-%s for %s: %s",
                "approving" if approve else "rejecting", request.agent_name, request.message[:160],
            )
            return HITLResponse(
                action=HITLAction.APPROVE if approve else HITLAction.REJECT,
                approved=approve,
                instructions=("No human was reachable (headless process); answered "
                              f"automatically with '{'approve' if approve else 'reject'}'."),
            )
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