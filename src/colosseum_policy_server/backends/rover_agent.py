"""Cloud LLMs driving the Earth Rover Mini+ behind the Local Protocol.

The arm agents (``inspect_agent``) move joints through Inspect Robots' tools.
A rover has no joints, so this backend runs its own small loop on the same
provider clients (Responses, Messages, Chat Completions or the Claude Code
login) and the same Client-owned API keys.

Stop and look: each turn the model sees the latest camera frame and the GPS
distance and direction to the goal, then makes exactly one tool call.
``drive`` returns a short ``base_velocity`` chunk followed by zeros, so the
rover is standing still while the model thinks about the next turn. ``done``
and ``give_up`` end the trial with a zero ``done`` plan; the operator scores it.
"""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from .base import LocalModel, PlanResult
from ..model_adapters.observation import _rgb
from ..model_adapters.robots import ROVER_ROBOT, session_robot
from ..tensors import tensor_to_numpy

OFFICIAL_LLM_URLS = {
    'openai': 'https://api.openai.com/v1',
    'x-ai': 'https://api.x.ai/v1',
    'anthropic': 'https://api.anthropic.com/v1',
}
PROVIDERS = {'gpt-': ('openai', 'responses'), 'grok-': ('x-ai', 'chat'), 'claude-': ('anthropic', 'messages')}
ROVER_SENSORS = ('front_image', 'rear_image')
EARTH_RADIUS_M = 6_371_000.0
MAX_TURN_ATTEMPTS = 4
# The Claude Code CLI must use the operator's subscription login, never API billing.
CLAUDE_CODE_BLOCKED_ENV = frozenset({
    'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL', 'ANTHROPIC_PROFILE',
    'ANTHROPIC_FEDERATION_RULE_ID', 'ANTHROPIC_ORGANIZATION_ID', 'CLAUDE_CODE_USE_BEDROCK',
    'CLAUDE_CODE_USE_VERTEX', 'CLAUDE_CODE_USE_FOUNDRY', 'CLAUDE_CODE_USE_ANTHROPIC_AWS'})

DEFAULT_ROVER_NOTES = (
    "The Earth Rover Mini+ is a small four-wheeled outdoor robot, about 30 cm long. Its front camera "
    "sits about 14 cm above the ground, so kerbs, steps and low plants look large and nearby objects "
    "fill the view. It drives on pavements and paths in public places where people walk. It cannot "
    "climb kerbs or steps: driving onto a raised edge can flip it.")

SYSTEM_PROMPT = """You are driving a robot to a destination, one short move at a time.

{notes}

How each turn works:
- You get the latest camera image{rear}, the GPS distance and direction to the goal, what your last move did, and the time left.
- Reply with exactly one tool call. `drive` moves the rover for up to {max_seconds:g} s and then it stops. It stays still while you decide the next move, and you see the result in the next turn.
- `forward` is the speed, from -{max_reverse:g} (reverse) to {max_forward:g}. `turn` is the turning rate, from -{max_turn:g} to {max_turn:g}: positive turns left, negative turns right. Use forward 0 with a turn to rotate in place. The rover responds differently on different ground, so check what each move actually did before the next one.
- Call `done` when you have reached the destination: the GPS distance is below about {arrive_m:g} m, or you can see you are at the place the instruction describes. GPS can be off by about 5 m, so trust the camera for the last metres.
- Call `give_up` only if the rover is stuck or the goal cannot be reached safely. Both `done` and `give_up` end the trial.

Safety comes first:
- Stay on paved paths. Never drive towards kerbs, steps, ledges, water, plants or road traffic.
- Keep well clear of people, animals and bicycles. If one is close or coming towards you, wait (drive with forward 0 and turn 0) or move away.
- If you cannot see clearly ahead, rotate in place to look before driving.
- Prefer several short moves over one long move when anything is near."""


def _wrap_deg(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def bearing_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> tuple[float, float]:
    """Bearing in degrees clockwise from north, and distance in metres, from point 1 to point 2."""
    north = math.radians(lat2 - lat1) * EARTH_RADIUS_M
    east = math.radians(lon2 - lon1) * EARTH_RADIUS_M * math.cos(math.radians((lat1 + lat2) / 2.0))
    return math.degrees(math.atan2(east, north)) % 360.0, math.hypot(north, east)


def _state64(observation, name: str, size: int) -> np.ndarray | None:
    """Navigation values stay float64: float32 GPS is only good to about a metre."""
    if name not in observation.state:
        return None
    value = np.asarray(tensor_to_numpy(observation.state[name]), dtype=np.float64).reshape(-1)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f'invalid observation state {name}')
    return value


def _number(settings, name, default, low, high):
    value = settings.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= float(value) <= high:
        raise ValueError(f'{name} must be a number from {low:g} to {high:g}')
    return float(value)


def tool_schemas(limits) -> list[dict[str, Any]]:
    note = {'type': 'string', 'description': 'What you see and why you chose this command.'}
    return [
        {'type': 'function', 'function': {
            'name': 'drive',
            'description': ('Drive for a few seconds, then stop. forward is the speed (negative reverses); '
                            'turn is the turning rate, positive = left. forward 0 with a turn rotates in place.'),
            'parameters': {'type': 'object', 'additionalProperties': False,
                'required': ['forward', 'turn', 'seconds', 'note'],
                'properties': {
                    'forward': {'type': 'number', 'minimum': -limits['max_reverse'], 'maximum': limits['max_forward']},
                    'turn': {'type': 'number', 'minimum': -limits['max_turn'], 'maximum': limits['max_turn']},
                    'seconds': {'type': 'number', 'minimum': limits['min_seconds'], 'maximum': limits['max_seconds']},
                    'note': note}}}},
        {'type': 'function', 'function': {
            'name': 'done',
            'description': 'You have reached the destination. Ends the trial.',
            'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['note'],
                           'properties': {'note': note}}}},
        {'type': 'function', 'function': {
            'name': 'give_up',
            'description': 'The rover is stuck or the goal cannot be reached safely. Ends the trial.',
            'parameters': {'type': 'object', 'additionalProperties': False, 'required': ['note'],
                           'properties': {'note': note}}}},
    ]


def _evicted(messages: list[dict[str, Any]], horizon: int, *, mark_anchor: bool,
             keep: frozenset[int] = frozenset()) -> list[dict[str, Any]]:
    """Keep camera frames only in the newest ``horizon`` messages that carry them.

    Messages in ``keep`` (the goal photo) always keep their images.
    """
    with_images = [i for i, message in enumerate(messages) if i not in keep
                   and isinstance(message.get('content'), list)
                   and any(part.get('type') == 'image_url' for part in message['content'])]
    stale = with_images[:-horizon] if horizon else with_images
    if not stale:
        return list(messages)
    view = list(messages)
    for index in stale:
        message = messages[index]
        parts = [part for part in message['content'] if part.get('type') != 'image_url']
        parts.append({'type': 'text', 'text': '[camera frame elided]'})
        view[index] = {**message, 'content': parts}
    if mark_anchor:
        # A stable cache breakpoint on the newest elided turn (Messages API only).
        view[stale[-1]] = {**view[stale[-1]], 'cache_anchor': True}
    return view


class HeadingEstimate:
    """Compass heading plus an offset learned from straight drives measured by GPS.

    The rover's magnetometer can be tens of degrees off near buildings; the
    course over ground of a few metres of straight driving is more reliable.
    One capped move covers only a metre or two, so consecutive straight moves
    are joined into one segment (GeNIE's estimator does the same).
    """

    def __init__(self, *, min_move_m: float = 4.0, max_turn_deg: float = 12.0, gain: float = 0.5):
        self.offset_deg, self.calibrated = 0.0, False
        self.min_move_m, self.max_turn_deg, self.gain = min_move_m, max_turn_deg, gain
        self.track: list[tuple[float, float, float]] = []

    def heading(self, compass_deg: float) -> float:
        return (compass_deg + self.offset_deg) % 360.0

    def observe(self, before, after, command) -> None:
        """Record the fixes around one command; (latitude, longitude, compass) each."""
        if command is None or command['forward'] <= 0 or abs(command['turn']) > 0.05:
            self.track = []
            return
        if not self.track:
            self.track = [before]
        self.track.append(after)
        reference = self.track[0][2]
        if any(abs(_wrap_deg(fix[2] - reference)) > self.max_turn_deg for fix in self.track):
            self.track = [after]  # It turned (or the compass jumped): start a new segment.
            return
        course, moved = bearing_distance(*self.track[0][:2], *self.track[-1][:2])
        if moved < self.min_move_m:
            return
        compass = reference + sum(_wrap_deg(fix[2] - reference) for fix in self.track) / len(self.track)
        error = _wrap_deg(course - (compass + self.offset_deg))
        self.offset_deg = _wrap_deg(self.offset_deg + self.gain * error)
        self.calibrated = True
        self.track = [after]


def create_backend(options: Mapping[str, Any]) -> 'RoverAgentBackend':
    return RoverAgentBackend(options)


class RoverAgentBackend:
    """One isolated LLM conversation per trial; bounded base_velocity commands."""

    control_mode = 'base_velocity'
    observation_extras = ()

    def __init__(self, options: Mapping[str, Any], *, transport=None, clock=time.monotonic):
        self.options = dict(options)
        self.transport = transport  # Tests inject httpx.MockTransport.
        self.clock = clock
        self.client = None
        self.model = None
        self._inflight = None

    # Session -----------------------------------------------------------------

    async def start_session(self, model: LocalModel, request: Mapping[str, Any]) -> None:
        if self.client is not None or self._inflight is not None:
            raise RuntimeError('previous rover agent session has not closed')
        if session_robot(model, request, self.options) != ROVER_ROBOT:
            raise ValueError('rover agent requires robot_type earth_rover_mini_plus')
        if model.model_type != 'llm' or model.action_space != 'base_velocity' or model.action_dim != 2:
            raise ValueError('rover LLM requires a 2-D base_velocity contract')
        settings = {**self.options, **model.backend_options}
        hz = model.control_hz
        self.limits = {
            'max_forward': _number(settings, 'max_forward', 0.3, 0.05, 1.0),
            'max_reverse': _number(settings, 'max_reverse', 0.2, 0.0, 1.0),
            'max_turn': _number(settings, 'max_turn', 0.6, 0.05, 1.0),
            'max_seconds': _number(settings, 'max_seconds', 2.0, 0.2, 10.0),
            'min_seconds': 1 / hz,
        }
        self.settle_steps = round(_number(settings, 'settle_seconds', 0.5, 0.0, 3.0) * hz)
        self.arrive_m = _number(settings, 'arrive_m', 8.0, 1.0, 50.0)
        longest = round(self.limits['max_seconds'] * hz) + self.settle_steps
        if longest > model.max_horizon:
            raise ValueError(f'max_horizon {model.max_horizon} is shorter than the longest drive ({longest} steps)')
        self.sensors = tuple(settings.get('sensors', ('front_image',)))
        if (not self.sensors or self.sensors[0] != 'front_image' or len(set(self.sensors)) != len(self.sensors)
                or not set(self.sensors) <= set(ROVER_SENSORS)):
            raise ValueError('sensors must be front_image, optionally followed by rear_image')
        horizon = settings.get('image_horizon', 2)
        if type(horizon) is not int or horizon < 1:
            raise ValueError('image_horizon must be a positive integer')
        self.image_horizon = horizon
        self.effort = settings.get('effort', 'low')
        notes = str(settings.get('robot_notes') or DEFAULT_ROVER_NOTES).strip()
        self.system = SYSTEM_PROMPT.format(
            notes=notes, rear=' (front, then rear)' if 'rear_image' in self.sensors else '',
            arrive_m=self.arrive_m, **self.limits)
        self.client, self.wire, self.provider = self._client(model, settings, request)
        self.model = model
        self.run_id = str(request.get('run_id', ''))
        # The Client sends the rover task's limits in client_options: the Local
        # Protocol task has no navigation field.
        navigation = request.get('navigation') if isinstance(request.get('navigation'), Mapping) else {}
        timeout = navigation.get('timeout_s')
        self.timeout_s = float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else None
        self.log_dir = Path(settings.get('log_dir', 'agent-logs'))
        self.log_id = uuid4().hex  # Never use a peer's run_id as a path.
        self.messages: list[dict[str, Any]] = [{'role': 'system', 'content': self.system}]
        self.started_at = None
        self.last_step = -1
        self.previous = None  # (latitude, longitude, compass) before the last command
        self.last_command = None
        self.heading = HeadingEstimate()
        self.calls = 0
        self.usage: dict[str, int] = {}
        self.turns: list[dict[str, Any]] = []
        self.stop_reason = self.stop_note = None

    def _client(self, model, settings, request):
        from inspect_robots_agent import AnthropicClient, ChatClient, Provider, ResponsesClient

        provider, wire = next(((p, w) for prefix, (p, w) in PROVIDERS.items() if model.name.startswith(prefix)),
                              (None, None))
        if provider is None:
            raise ValueError('Unsupported LLM model name')
        # Router names may carry a suffix; the provider gets its own model ID.
        api_model = settings.get('api_model', model.name)
        prefix = next(prefix for prefix in PROVIDERS if model.name.startswith(prefix))
        if (not isinstance(api_model, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._:-]*', api_model)
                or not api_model.startswith(prefix)):
            raise ValueError('api_model must be a model ID for the assigned LLM provider')
        auth_method = request.get('auth_method', 'api-key')
        if auth_method not in {'api-key', 'claude-code'}:
            raise ValueError('Unsupported LLM authentication method')
        if auth_method == 'claude-code':
            if provider != 'anthropic' or request.get('api_key'):
                raise ValueError('Claude Code login is only valid for Claude without an API key')
            from inspect_robots_agent._claude_code import ClaudeCodeClient
            env = {key: value for key, value in os.environ.items() if key not in CLAUDE_CODE_BLOCKED_ENV}
            return (ClaudeCodeClient(api_model, command=settings.get('claude_command'),
                                     timeout_s=settings.get('claude_timeout_s', 120), env=env),
                    'claude-code', provider)
        api_key = request.get('api_key')
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError('Client must supply its API key for this LLM')
        official = OFFICIAL_LLM_URLS[provider]
        endpoint = settings.get('llm_endpoint', 'router')
        if endpoint not in {'router', 'official'}:
            raise ValueError('llm_endpoint must be router or official')
        base_url = str(settings.get('api_base_url') or (official if endpoint == 'official' else model.url)).rstrip('/')
        if settings.get('api_base_url') and base_url != official:
            raise ValueError("api_base_url override must be the selected provider's official API URL")
        if provider == 'anthropic' and not urlsplit(base_url).path.strip('/'):
            base_url += '/v1'
        wire = settings.get('api_format', wire)
        target = Provider(base_url=base_url, api_key=api_key, model=api_model, wire=wire)
        transport = self.transport
        if provider == 'anthropic' and settings.get('api_auth', 'api_key') == 'bearer':
            from .llm_transport import BearerTransport
            transport = BearerTransport(api_key, transport)
        if wire == 'messages':
            client = AnthropicClient(target, max_output_tokens=int(settings.get('max_output_tokens', 16000)),
                                     transport=transport)
        elif wire == 'responses':
            client = ResponsesClient(target, transport=transport)
        elif wire == 'chat':
            client = ChatClient(target, transport=transport)
        else:
            raise ValueError(f'unsupported api_format {wire!r}')
        return client, wire, provider

    # Inference ---------------------------------------------------------------

    async def infer(self, model, observation):
        if self.client is None or model != self.model:
            raise RuntimeError('rover agent session was not prepared for this model')
        if self._inflight is not None:
            raise RuntimeError('previous inference is still running')
        # Provider clients are synchronous. Shield the worker so a timeout
        # cannot detach a conversation that is still being written.
        task = asyncio.create_task(asyncio.to_thread(self._infer, observation))
        self._inflight = task
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._inflight = None

    def _stopped(self):
        return PlanResult(np.zeros((1, 2), dtype=np.float32), done=True)

    def _infer(self, observation):
        if observation.control_step <= self.last_step:
            raise ValueError('control_step must advance')
        self.last_step = observation.control_step
        if self.stop_reason is not None:
            return self._stopped()
        if self.started_at is None:
            self.started_at = self.clock()
        images = {item.sensor_id: item for item in observation.sensors}
        frames = [(name, _rgb(images.get(name), name)) for name in self.sensors]
        navigation = _state64(observation, 'navigation_state', 9)
        if navigation is None:
            raise ValueError('observation is missing navigation_state')
        goal = _state64(observation, 'goal_gps', 2)
        if len(self.messages) == 1:
            self.messages.append(self._goal_message(observation, images.get('goal_image'), goal))
        text, summary = self._status(observation.control_step, navigation, goal)
        content = [{'type': 'text', 'text': text}]
        from inspect_robots_agent import png_data_url
        for name, pixels in frames:
            if len(frames) > 1:
                content.append({'type': 'text', 'text': f'{name}:'})
            content.append({'type': 'image_url', 'image_url': {'url': png_data_url(pixels)}})
        self.messages.append({'role': 'user', 'content': content})
        turn = {'step': observation.control_step, **summary}
        self.turns.append(turn)
        self.previous = (float(navigation[0]), float(navigation[1]), float(navigation[2]))
        for _ in range(MAX_TURN_ATTEMPTS):
            message = self._complete()
            if not message.tool_calls:
                self.messages.append({'role': 'user', 'content': 'Reply with exactly one tool call: drive, done or give_up.'})
                continue
            first, *extra = message.tool_calls
            result, outcome = self._execute(first)
            self.messages.append({'role': 'tool', 'tool_call_id': first.id, 'content': result})
            for call in extra:
                self.messages.append({'role': 'tool', 'tool_call_id': call.id,
                                      'content': 'Ignored: send one command per turn.'})
            if outcome is None:
                continue
            turn['command'] = {'tool': first.name, **outcome}
            if first.name in {'done', 'give_up'}:
                self.stop_reason, self.stop_note = first.name, outcome.get('note', '')
                return self._stopped()
            self.last_command = outcome
            hz = self.model.control_hz
            steps = max(1, round(outcome['seconds'] * hz))
            actions = np.zeros((steps + self.settle_steps, 2), dtype=np.float32)
            actions[:steps] = (outcome['forward'], outcome['turn'])
            return actions
        raise RuntimeError(f'LLM produced no valid command in {MAX_TURN_ATTEMPTS} attempts')

    def _complete(self):
        # messages[1] is the goal: instruction, GPS and the destination photo, if any.
        outgoing = _evicted(self.messages, self.image_horizon, mark_anchor=self.wire == 'messages',
                            keep=frozenset({1}))
        message = self.client.complete(outgoing, tool_schemas(self.limits), reasoning_effort=self.effort)
        self.calls += 1
        for key, value in (message.usage or {}).items():
            if isinstance(value, int):
                self.usage[key] = self.usage.get(key, 0) + value
        self.messages.append(message.raw())
        return message

    def _execute(self, call):
        """Validate one tool call. Returns (tool result text, accepted command or None)."""
        try:
            arguments = json.loads(call.arguments or '{}')
        except json.JSONDecodeError:
            return 'Rejected: arguments must be a JSON object. Try again.', None
        if not isinstance(arguments, dict):
            return 'Rejected: arguments must be a JSON object. Try again.', None
        note = arguments.get('note')
        if not isinstance(note, str) or not note.strip():
            return 'Rejected: note is required. Describe what you see and why. Try again.', None
        if call.name in {'done', 'give_up'}:
            return f'Accepted: {call.name}. The trial ends.', {'note': note.strip()[:500]}
        if call.name != 'drive':
            return f'Rejected: unknown tool {call.name!r}. Use drive, done or give_up.', None
        limits = self.limits
        bounds = {'forward': (-limits['max_reverse'], limits['max_forward']),
                  'turn': (-limits['max_turn'], limits['max_turn']),
                  'seconds': (limits['min_seconds'], limits['max_seconds'])}
        values = {}
        for name, (low, high) in bounds.items():
            raw = arguments.get(name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
                return f'Rejected: {name} must be a number. Try again.', None
            if not low - 1e-9 <= raw <= high + 1e-9:
                return f'Rejected: {name} must be between {low:g} and {high:g}. Try again.', None
            values[name] = float(min(max(raw, low), high))
        return (f"Accepted: forward {values['forward']:+.2f}, turn {values['turn']:+.2f} for "
                f"{values['seconds']:.1f} s, then stop. The next message shows the result.",
                {**values, 'note': note.strip()[:500]})

    # Observation text --------------------------------------------------------

    def _goal_message(self, observation, goal_image, goal):
        parts = [{'type': 'text', 'text': f'Instruction from the evaluator: {observation.instruction.strip() or "(none)"}'}]
        if goal is not None:
            parts[0]['text'] += f'\nDestination GPS: {goal[0]:.6f}, {goal[1]:.6f}.'
        if goal_image is not None:
            from inspect_robots_agent import png_data_url
            parts.append({'type': 'text', 'text': 'Photo of the destination, taken from where the rover should stop:'})
            parts.append({'type': 'image_url', 'image_url': {'url': png_data_url(_rgb(goal_image, 'goal_image'))}})
        if self.timeout_s is not None:
            parts[0]['text'] += f'\nTime limit: {self.timeout_s:.0f} s, including your thinking time.'
        return {'role': 'user', 'content': parts}

    def _status(self, step, navigation, goal):
        # navigation_state: latitude, longitude, heading_deg, speed, battery, rpm_1..rpm_4.
        # The speed field's unit is not documented, so the model does not get it.
        latitude, longitude, compass, battery = (float(navigation[i]) for i in (0, 1, 2, 4))
        lines = []
        elapsed = self.clock() - self.started_at
        timing = f'Turn {len(self.turns) + 1}, {elapsed:.0f} s elapsed'
        if self.timeout_s is not None:
            timing += f', about {max(0.0, self.timeout_s - elapsed):.0f} s left'
        lines.append(timing + '.')
        summary: dict[str, Any] = {'elapsed_s': round(elapsed, 1)}
        if self.previous is not None and self.last_command is not None:
            _, moved = bearing_distance(self.previous[0], self.previous[1], latitude, longitude)
            turned = _wrap_deg(compass - self.previous[2])
            self.heading.observe(self.previous, (latitude, longitude, compass), self.last_command)
            # Compass degrees grow clockwise, so a left turn lowers them.
            direction = '' if abs(turned) < 1 else (' (left)' if turned < 0 else ' (right)')
            lines.append(f'Your last move: about {moved:.1f} m by GPS; the compass changed by '
                         f'{turned:+.0f}°{direction}.')
            summary.update(moved_m=round(moved, 2), turned_deg=round(turned, 1))
        heading = self.heading.heading(compass)
        if goal is not None:
            bearing, distance = bearing_distance(latitude, longitude, float(goal[0]), float(goal[1]))
            relative = _wrap_deg(bearing - heading)
            side = 'straight ahead' if abs(relative) < 10 else (
                f'{abs(relative):.0f}° to your {"left" if relative < 0 else "right"}')
            lines.append(f'Goal: {distance:.1f} m away, {side}. (Goal bearing {bearing:.0f}°, your heading '
                         f'{heading:.0f}° from the compass{" corrected by GPS" if self.heading.calibrated else ""}; '
                         'the compass can be off by tens of degrees.)')
            summary.update(goal_m=round(distance, 2), goal_relative_deg=round(relative, 1))
        else:
            lines.append('Goal: follow the instruction; there is no GPS destination.')
        lines.append(f'Battery {battery:.0f}%.')
        return '\n'.join(lines), summary

    # Cleanup -----------------------------------------------------------------

    async def end_session(self):
        task = self._inflight
        if task is not None:
            # A timed-out HTTP call must finish before the client closes or a new trial starts.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            self._inflight = None
        client, self.client = self.client, None
        if client is None:
            return
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            record = {
                'run_id': self.run_id, 'model': self.model.public_spec(), 'robot_type': ROVER_ROBOT,
                'wire': self.wire, 'stop_reason': self.stop_reason, 'stop_note': self.stop_note,
                'llm_calls': self.calls, 'usage': self.usage, 'limits': self.limits,
                'heading_offset_deg': round(self.heading.offset_deg, 1), 'turns': self.turns,
                'transcript': _sanitized(self.messages),
            }
            (self.log_dir / f'{self.log_id}.json').write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        finally:
            close = getattr(client, 'close', None)
            if close is not None:
                close()
            self.model = None


def _sanitized(messages):
    """The conversation without camera frames, for the local session record."""
    result = copy.deepcopy(messages)
    for message in result:
        message.pop('cache_anchor', None)
        content = message.get('content')
        if isinstance(content, list):
            message['content'] = [part if part.get('type') != 'image_url'
                                  else {'type': 'text', 'text': '[camera frame]'} for part in content]
    return result
