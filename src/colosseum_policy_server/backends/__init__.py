"""Public contracts for robot-specific local inference plugins."""
from .base import LocalModel, PolicyBackend

__all__ = ["LocalModel", "PolicyBackend"]
