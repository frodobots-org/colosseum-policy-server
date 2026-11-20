"""Public contracts for robot-specific local inference plugins."""
from .base import LocalModel, PlanResult, PolicyBackend

__all__ = ["LocalModel", "PlanResult", "PolicyBackend"]
