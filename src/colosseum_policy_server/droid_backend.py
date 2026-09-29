"""Compatibility imports; DROID implementation lives in backends.droid."""
from .backends.droid import *  # noqa: F401,F403
from .backends import droid as _implementation


def __getattr__(name):
    return getattr(_implementation, name)
