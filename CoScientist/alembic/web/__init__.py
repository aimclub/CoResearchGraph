"""Web dashboard for alembic builds.

Pages:

  * ``/``: paste a repo URL and start a build. The build runs detached through
    ``CoScientist.tools.alembic_tools.build_mcp_server``, and the page moves to
    the build's own page.
  * ``/builds``: every build known on this host, with its MCP address and
    whether the serving container is up.
  * ``/builds/<job_id>``: one build. Top: the five-stage rail. Left: the
    exploration report, output files, ``setup.sh`` and invocation examples.
    Right: the generated tools with validation results and a Call form that
    runs a tool inside the build's container, through the same
    ``invoke_tool_function`` the validator uses. Bottom: the activity feed.

The main CoScientist web UI mounts this app under ``/alembic``.

Standalone run (port 8100):
    python CoScientist/alembic/web/server.py
"""
