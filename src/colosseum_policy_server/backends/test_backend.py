"""Test-only joint-position echo; no model loading or hardware commands."""
from __future__ import annotations

import numpy as np

from .base import LocalModel
from .. import colosseum_pb2 as pb
from ..tensors import tensor_to_numpy


def create_backend(options):
    return TestBackend(options)


class TestBackend:
    __test__ = False  # This is a runtime plugin, not a pytest test class.
    verification_only = True

    def __init__(self, options):
        self.fields = options.get('state_fields', ['joint_position', 'gripper_position'])
        if not isinstance(self.fields, list) or not self.fields or any(not isinstance(f, str) or not f for f in self.fields) or len(set(self.fields)) != len(self.fields):
            raise ValueError('state_fields must be distinct nonempty state names in action order')

    async def infer(self, model: LocalModel, observation: pb.Observation) -> np.ndarray:
        if model.action_space != 'joint_position':
            raise ValueError('Loopback requires absolute joint_position actions')
        values = []
        for field in self.fields:
            if field not in observation.state:
                raise ValueError(f'Missing loopback state: {field}')
            value = tensor_to_numpy(observation.state[field])
            if value.ndim != 1 or value.size == 0:
                raise ValueError(f'Loopback state must be a nonempty vector: {field}')
            values.append(value)
        action = np.concatenate(values).astype(np.float32)
        if action.size != model.action_dim or not np.isfinite(action).all():
            raise ValueError('Loopback state does not match the action dimension or contains invalid values')
        return action[None, :]
