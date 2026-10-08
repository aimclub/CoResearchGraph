"""Configuration module."""
from CoScientist.config.settings import settings, Settings, get_settings
from CoScientist.config.report import ReportConfig, LATEX_MODES

__all__ = ["settings", "Settings", "get_settings", "ReportConfig", "LATEX_MODES"]
