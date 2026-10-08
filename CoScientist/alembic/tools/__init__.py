"""Agent-facing tools + deterministic gate helpers for the alembic pipeline.

Flat import surface — ``from alembic.tools import clone_repo, ...`` — split by
concern across submodules:

    paths     workdir layout, current repo, per-repo path helpers
    shell     bash / bash_env (+ env-command recording for setup.sh)
    fs        clone/read/search repo, read/write output and reports
    venv      setup_venv, check_venv_compat, ensure_pytest
    invoke    artefact checks, per-tool pytest, live invocation, server check
    analysis  AST symbol table / target verification / layout (plan gate)
    codegen   deterministic server.py / setup.sh / code.py renderers
"""
from alembic.common import get_repo_name
from alembic.tools.paths import WORKDIR, set_current_repo
from alembic.tools.shell import bash, bash_env, start_env_recording, stop_env_recording
from alembic.tools.fs import (
    clone_repo, read_file, search, read_report,
    write_file, read_output_file, update_file, write_report,
)
from alembic.tools.venv import setup_venv, check_venv_compat, ensure_pytest
from alembic.tools.invoke import (
    check_server, check_tool_artefacts, invoke_tool_function, run_tool_tests,
)

__all__ = [
    "WORKDIR", "get_repo_name", "set_current_repo",
    "bash", "bash_env", "start_env_recording", "stop_env_recording",
    "clone_repo", "read_file", "search", "read_report",
    "write_file", "read_output_file", "update_file", "write_report",
    "setup_venv", "check_venv_compat", "ensure_pytest",
    "check_server", "check_tool_artefacts", "invoke_tool_function", "run_tool_tests",
]
