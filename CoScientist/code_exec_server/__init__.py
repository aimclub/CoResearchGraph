"""Code-execution MCP server package.

A standalone sandbox service the CoderAgent talks to (see
CoScientist.tools.coder_tools). It is intentionally separate from the agent
process so that arbitrary, possibly long-running, possibly resource-heavy
research code (cloning repos, running evolutionary optimization, training
models) executes in an isolated container rather than inside the orchestrator.

The HTTP contract is the one the CoderToolset client expects:

  POST {submit_path}   json={"command", "workspace_id", "timeout"} -> {"job_id"}
  GET  {result_path}?job_id=...  -> {"status", "stdout", "stderr", "exit_code"}

Run it with:  python -m CoScientist.code_exec_server.app
"""

from CoScientist.code_exec_server.runner import JobRunner, Job, JobStatus

__all__ = ["JobRunner", "Job", "JobStatus"]
