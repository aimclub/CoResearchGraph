"""Logging module."""
from CoScientist.logging.logger import get_logger
from CoScientist.logging.opik_tracer import multi_agent_tracer, get_multi_agent_tracer

__all__ = ["get_logger", 
            "multi_agent_tracer",
            "get_multi_agent_tracer"]
