"""Configuration module."""
from CoScientist.config.settings import (
    ExperimentsSettings,
    Settings,
    get_settings,
    settings_scope,
    settings,
)
from CoScientist.config.report import ReportConfig, LATEX_MODES

__all__ = [
    "ExperimentsSettings",
    "settings",
    "Settings",
    "get_settings",
    "settings_scope",
    "ReportConfig",
    "LATEX_MODES",
]
