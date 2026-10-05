"""Inference-side robot contracts. Hardware drivers live in the Client."""
from dataclasses import dataclass


def canonical_robot(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('robot_type must be a nonempty string')
    return 'franka' if value == 'droid' else value


def session_robot(model, request, options):
    configured = model.backend_options.get('robot_type', options.get('robot_type'))
    # Older DROID clients did not send robot_type. Preserve that legacy path.
    robot = canonical_robot(request.get('robot_type') or configured or 'franka')
    if configured is not None and canonical_robot(configured) != robot:
        raise ValueError('Client robot_type does not match the configured model')
    return robot


@dataclass(frozen=True)
class JointRobotContract:
    name: str
    joint_count: int
    state_key: str = 'joint_position'

    @property
    def action_dim(self):
        return self.joint_count + 1

    def validate(self, model):
        if model.action_space != 'joint_position' or model.action_dim != self.action_dim:
            raise ValueError(f'{self.name} LLM requires {self.action_dim}-D joint_position')


def llm_robot_contract(robot):
    # Add a verified contract here when its observation/action conventions are
    # implemented. Never infer embodiment from action dimension alone.
    if robot != 'franka':
        raise ValueError(f'LLM robot adapter {robot!r} is not implemented')
    return JointRobotContract('franka', 7)
