"""Checkpoint subsystem: save/restore full run state at module boundaries.

Design: CHECKPOINT_DESIGN.md (implementation) + SynapseNmas.md (platform
contract this stays forward-compatible with).
"""
from CoScientist.checkpoints.api import make_checkpoint_router
from CoScientist.checkpoints.capture import capture_checkpoint
from CoScientist.checkpoints.model import CheckpointManifest, HitlPending
from CoScientist.checkpoints.plugin import CheckpointPlugin
from CoScientist.checkpoints.restore import CompatibilityError, restore_checkpoint
from CoScientist.checkpoints.store import LocalZipStore, get_default_store

__all__ = [
    "CheckpointManifest",
    "CheckpointPlugin",
    "CompatibilityError",
    "HitlPending",
    "LocalZipStore",
    "capture_checkpoint",
    "get_default_store",
    "make_checkpoint_router",
    "restore_checkpoint",
]
