"""Legacy import/entry point for Franka VLA model adapters; no robot control."""
from ..model_adapters.franka_vla import FrankaVLAAdapter
from ..model_adapters import franka_vla as _implementation

DroidBackend = FrankaVLAAdapter


def create_backend(options):
    return DroidBackend(options)


def __getattr__(name):
    return getattr(_implementation, name)
