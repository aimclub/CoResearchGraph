"""Agent definitions.

Five LLM agents — explorer, environment, coder, debugger, wrapper — all on one
MODEL. The validator is NOT an agent: validation is a deterministic code loop
in main.py that calls the debugger as a subroutine. The wrapper agent is a
fallback only: server.py is rendered deterministically (tools/codegen.py) and
the agent runs solely when the compile/import gate fails.
"""
import asyncio

import litellm

litellm.suppress_debug_info = True

from loguru import logger
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

from alembic import config
from alembic.tools import (
    bash, bash_env, check_venv_compat, clone_repo, invoke_tool_function,
    read_file, read_output_file, run_tool_tests, search, setup_venv,
    update_file, write_file, write_report,
)
from alembic.instructions import (
    coder_instruction, debugger_instruction, environment_instruction,
    explorer_instruction, wrapper_instruction,
)
from alembic.agent_runtime import _retryable_llm_error, _short_err


class ResilientLiteLlm(LiteLlm):
    """LiteLlm that survives OpenRouter/provider faults. A transient error is
    logged as ONE line, backed off (exponential, capped), and the request is
    retried — unbounded by default (config.LLM_RETRY_CAP is None), so a
    no-timeout run keeps trying until it gets a real answer instead of dumping a
    stacktrace and failing the stage. Only pre-yield failures retry, so a fault
    mid-stream is never double-emitted."""

    async def generate_content_async(self, llm_request, stream: bool = False):
        attempt = 0
        while True:
            produced = False
            try:
                async for resp in super().generate_content_async(llm_request, stream=stream):
                    produced = True
                    yield resp
                return
            except Exception as e:  # noqa: BLE001 — classify, don't crash the run
                if produced or not _retryable_llm_error(e):
                    raise
                attempt += 1
                if config.LLM_RETRY_CAP is not None and attempt > config.LLM_RETRY_CAP:
                    logger.error(f"[llm] {_short_err(e)} — retry cap "
                                 f"({config.LLM_RETRY_CAP}) exhausted, giving up.")
                    raise
                delay = min(config.LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                            config.LLM_RETRY_MAX_DELAY)
                logger.warning(f"[llm] provider fault ({_short_err(e)}); "
                               f"retry {attempt} in {delay:.0f}s.")
                await asyncio.sleep(delay)


def _model() -> ResilientLiteLlm:
    """One model for every agent. Sampling params only when explicitly set via
    env (leaving them unset avoids the temperature-0 loops seen on qwen). A
    per-request timeout bounds hung calls; ResilientLiteLlm retries faults."""
    sampling = {"timeout": config.LLM_REQUEST_TIMEOUT, "num_retries": 0}
    if config.MODEL_TEMPERATURE is not None:
        sampling["temperature"] = float(config.MODEL_TEMPERATURE)
    if config.MODEL_TOP_P is not None:
        sampling["top_p"] = float(config.MODEL_TOP_P)
    return ResilientLiteLlm(model=config.MODEL, **sampling)


def _const(text: str):
    """Wrap a static instruction as an InstructionProvider. ADK runs `{var}`
    session-state templating on *string* instructions and raises KeyError on
    any literal brace-identifier — our prompts are full of `{}` code examples,
    so pass a callable, which ADK does NOT template (bypass_state_injection)."""
    return lambda _ctx: text


explorer_agent = Agent(
    name="explorer",
    model=_model(),
    description="Clones a scientific GitHub repo and reports its functionality, environment needs, and proposed tools.",
    instruction=_const(explorer_instruction),
    tools=[clone_repo, read_file, bash, search, write_report],
)

environment_agent = Agent(
    name="environment",
    model=_model(),
    description="Builds the Python virtual environment(s) for the repository from a computed layout.",
    instruction=_const(environment_instruction),
    tools=[setup_venv, bash_env, bash, check_venv_compat],
)

coder_agent = Agent(
    name="coder",
    model=_model(),
    description="Implements each verified tool as a plain Python function with smoke and invocation tests.",
    instruction=_const(coder_instruction),
    tools=[bash, read_file, write_file, read_output_file, update_file],
)

debugger_agent = Agent(
    name="debugger",
    model=_model(),
    description="Fixes a batch of reported failures — installs missing deps or edits tool/test files — and re-runs them to confirm.",
    instruction=_const(debugger_instruction),
    tools=[read_output_file, update_file, bash, bash_env,
           invoke_tool_function, run_tool_tests],
)

wrapper_agent = Agent(
    name="wrapper",
    model=_model(),
    description="Fallback fixer for the generated FastMCP server when its compile/import gate fails.",
    instruction=_const(wrapper_instruction),
    tools=[read_output_file, update_file, bash],
)
