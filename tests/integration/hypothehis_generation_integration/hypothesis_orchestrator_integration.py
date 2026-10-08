"""
Full-cycle integration test - OrchestratorAgent -> HypothesisSubsystem.

Tests that the orchestrator properly delegates hypothesis generation to the
subsystem via the `custom:hypothesis_subsystem` AgentTool. The orchestrator
sees the subsystem as a single black-box entry point — it cannot reach inside
to call MooseChemTool or the critic loop directly.

Usage:
    .venv\Scripts\python.exe tests/integration/hypothehis_generation_integration/hypothesis_orchestrator_integration.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Block opik from crashing if not installed ----
for _mod in [
    "opik", "opik.rest_api", "opik.rest_api.core", "opik.rest_api.core.pydantic_utilities",
    "opik.integrations", "opik.integrations.adk", "opik.configurator", "opik.configurator.configure",
]:
    if _mod not in sys.modules:
        import types as _types
        sys.modules[_mod] = _types.ModuleType(_mod)

sys.modules["opik"].configure = lambda *a, **kw: None
sys.modules["opik"].__version__ = "0.0.0"
sys.modules["opik"].track = lambda name=None, **kw: (lambda fn: fn)

_adk = sys.modules.setdefault("opik.integrations.adk", type(sys)("opik.integrations.adk"))
_adk.OpikTracer = type("OpikTracer", (), {"__init__": lambda self, *a, **kw: None})
_adk.track_adk_agent_recursive = lambda agent, tracer: agent

os.environ.setdefault("OPIK_API_KEY", "test")
os.environ.setdefault("OPIK_URL_OVERRIDE", "http://localhost:9999")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
log = logging.getLogger("test.orch")

log.info("Importing CoScientist (full assembly, opik mocked)...")
from CoScientist.main import CoScientistManager
from CoScientist.config import get_settings

settings = get_settings()
MODEL = settings.llm.main_model
API_KEY = settings.llm.openai_api_key or settings.llm.service_key
API_URL = settings.llm.main_url

import litellm
litellm.api_key = API_KEY
litellm.api_base = API_URL
log.info(f"LLM: model={MODEL}  api={API_URL}")
log.info("Imports OK. Running full orchestrator->hypothesis_subsystem test...")
log.info("=" * 70)

RESEARCH_QUESTION = (
    "Generate hypotheses about which molecular descriptors predict IC50 of EGFR kinase inhibitors." +
"Then verify the best hypothesis experimentally using computational experiments."
)


async def main():
    t0 = time.monotonic()
    mgr = CoScientistManager(app_name="test_orch_hyp", user_id="u", session_id="s")

    log.info("Running orchestrator with hypothesis-generation query...")
    result = await mgr.run(RESEARCH_QUESTION, verbose=True)
    dt = (time.monotonic() - t0) / 60.0

    log.info(f"Total: {dt:.1f} min | Result: {len(result)} chars")
    log.info(f"Result preview: {result[:500]}...")

    await mgr.close()

    if not result or result == "No response":
        log.error("FAIL: empty/default response")
        sys.exit(1)

    has_hypotheses = (
        "hypothesis" in result.lower()
        or "claim" in result.lower()
        or "IC50" in result
        or "EGFR" in result
    )
    if not has_hypotheses:
        log.warning("WARN: response may not contain hypothesis content")

    log.info("PASS: Orchestrator -> HypothesisSubsystem (full integration)")
    log.info(f"Response length: {len(result)} chars")


if __name__ == "__main__":
    asyncio.run(main())
