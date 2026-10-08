wrapper_instruction = '''
You fix the generated FastMCP server so it compiles and imports. server.py was
rendered by code from the tool functions in tools/ — it failed its compile/
import gate and the error is in your opening message. Make the MINIMAL fix.

Rules:
- `read_output_file("server.py")`, fix the reported error only, then
  `update_file("server.py", <full corrected file>)`.
- server.py must keep: a top-level `mcp = FastMCP(...)` instance, one
  `@mcp.tool()` function per tool that forwards to
  `_call("<name>", {...kwargs...})`, and imports limited to
  fastmcp/json/subprocess/shutil/pathlib/importlib/uuid (heavy repo imports
  belong in the tool functions, never here).
- server.py must also keep its S3 pass-through: the `importlib.util`-based
  load of `helpers/s3_transfer.py` into `_s3` (guarded by `try/except` —
  a missing/broken helper file falls back to a no-op `_S3Unavailable` stub
  whose `s3_enabled()` returns `False`, never a hard import failure), and in
  `_call()` the `_s3.s3_enabled()` check that builds a scratch dir, calls
  `_s3.prepare_kwargs(...)` before the subprocess, and calls
  `_s3.publish_result(...)` after it inside a `try/finally` that always
  removes the scratch dir. Do not add a module-level `import boto3` —
  s3_transfer.py imports it lazily inside its own functions so a server with
  no S3 configured never needs it installed. If `helpers/s3_transfer.py`
  itself is unavailable or unfixable, it is fine to fall back to the
  `_S3Unavailable` stub (or drop the S3 layer entirely, keeping only the
  plain `_call()` subprocess path) rather than block the compile/import gate
  on it — S3 support is an enhancement, not a required part of serving.
- Every generated `@mcp.tool()` function also keeps its two extra trailing
  parameters, `user_id: str = ""` and `session_id: str = ""` — optional S3
  vault scope hints that `SessionScopePlugin` fills in automatically at the
  ADK tool-call boundary, present on EVERY tool regardless of whether S3 is
  configured (the decision is baked into the signature at codegen time, not
  made at runtime). They are forwarded to `_call(name, {...kwargs...},
  user_id, session_id)` as separate positional args, kept OUT of the
  `{...kwargs...}` payload dict — never pass them to the helper function or
  to `run_function.py` — UNLESS the tool's own signature already declares a
  same-named parameter, in which case that parameter's value stays in the
  kwargs payload as usual, but `""` (not the variable) is passed to `_call()`
  for that scope slot: the tool's own `user_id`/`session_id` is domain data
  (e.g. a database row id) with nothing to do with the S3 namespace, and must
  never leak into it. That is only true for a NON-EMPTY value, though —
  `SessionScopePlugin` fills in ANY declared parameter left blank at the ADK
  boundary, scope-named or not, so a caller that leaves the tool's own
  `user_id`/`session_id` empty still gets it silently populated from the
  graph's scope, same as the dedicated scope params. `_call()` in turn
  resolves the S3 key prefix via `_s3_scope(user_id, session_id)`, which
  prefers the pair when BOTH are non-empty and otherwise falls back to the
  `X-Coscientist-*` request headers (a dormant fallback nothing currently
  sends), then to `local`/`default`.
- Verify with `bash("<output>/.venv/bin/python -m py_compile <output>/server.py && echo OK")`.
- Finish with one line: what was wrong, what you changed.
'''
