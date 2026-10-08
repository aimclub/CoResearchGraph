"""Entry point for `adk web` / `adk api_server`.

Set A2A_MODE=1 to use the A2A orchestrator (sub-agents must be running).
Default (A2A_MODE unset) uses the in-process ADK orchestrator.

Exported as an ADK ``App`` so the event-logger plugin rides along: every
agent's thoughts, tool calls and tool results are printed to the console,
same as the A2A servers do. Disable with LOG_AGENT_EVENTS=0.
"""
import os

from google.adk.apps import App

from CoScientist.logging.event_logger import EventLoggerPlugin
from CoScientist.logging.metrics import UsageMetricsPlugin
from CoScientist.graph.plugin import GraphMemoryPlugin
from CoScientist.graph.research.validator import BackgroundValidatorPlugin
from CoScientist.agents.truncation_plugin import ToolResultTruncationPlugin
from CoScientist.tools.mcp_artifact_plugin import McpArtifactCapturePlugin
from CoScientist.tools.session_scope_plugin import SessionScopePlugin
from CoScientist.main import _compaction_config

if os.getenv("A2A_MODE"):
    from CoScientist.a2a.orchestrator import orchestrator_a2a_agent as root_agent
else:
    from CoScientist.agents import orchestrator_agent as root_agent

# adk web keys sessions by the agents-dir entry name, so App.name must match
# the directory name ("CoScientist"). Truncation is last so the logger sees the
# full tool result before the model gets a context-bounded copy; compaction
# summarizes the context once it crosses the token threshold.
app = App(
    name="CoScientist",
    root_agent=root_agent,
    plugins=[
        EventLoggerPlugin(),
        UsageMetricsPlugin(),
        GraphMemoryPlugin(),
        BackgroundValidatorPlugin(),
        # An MCP server builds its S3 key from user_id and session_id. Without
        # this plugin every adk web user writes to the same unknown_user prefix.
        SessionScopePlugin(),
        # Capture runs before truncation, so it still sees the full URL.
        McpArtifactCapturePlugin(),
        ToolResultTruncationPlugin(),
    ],
    events_compaction_config=_compaction_config(),
)
