"""Configuration for the Dynamic Execution Graph service/client."""
import os

GRAPH_HOST = os.getenv("GRAPH_HOST", "localhost")
GRAPH_PORT = int(os.getenv("GRAPH_PORT", "8010"))
GRAPH_URL = f"http://{GRAPH_HOST}:{GRAPH_PORT}"

# Where the service snapshots each run's graph as JSON (persistence/replay).
GRAPH_SNAPSHOT_DIR = os.getenv("GRAPH_SNAPSHOT_DIR", "./graph_runs")

# Master switch: when off, the client is a no-op and nothing touches the service.
GRAPH_ENABLED = os.getenv("GRAPH_ENABLED", "1") not in ("0", "false", "False")
