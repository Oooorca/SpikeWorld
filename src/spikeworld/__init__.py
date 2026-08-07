"""SpikeWorld reference implementation."""

from .adaptation import ACTION_CAP, FAST_BYTES, MechanismRouter, PersistentFastState
from .model import ModelConfig, SpikeWorld

__all__ = [
    "ACTION_CAP",
    "FAST_BYTES",
    "MechanismRouter",
    "ModelConfig",
    "PersistentFastState",
    "SpikeWorld",
]

__version__ = "0.1.0"
