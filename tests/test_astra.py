import asyncio
import base64
from dataclasses import replace
import io
import json
import time

import numpy as np
import pytest
from PIL import Image

from colosseum_policy_server.sdk import Observation, ImageFrame
from colosseum_policy_server.policies.astra.planner import Astra
from colosseum_policy_server.policies.astra.policy import AstraPolicy, permitted_observation
from colosseum_policy_server.policies.astra.controller import Controller, ControllerConfig


def recorded_observation():
    data = io.BytesIO()
    Image.new('RGB', (80, 48), 'green').save(data, format='JPEG')
    images = tuple(ImageFrame(name, 'JPEG', time.time_ns(), 80, 48, data.getvalue())
                   for name in ('front_image', 'goal_image'))
    return Observation('session-1', 1, 1000, time.time_ns(), 0, 'Reach the view.',
        {'orientation': np.array([20.0]), 'latitude': np.array([51.123456]),
         'longitude': np.array([-1.23456]), 'private_judge': np.array([99999])}, images, time.monotonic_ns())


async def decision(policy, obs):
    await policy.action(obs)
    await asyncio.wait_for(policy.pending, 2)
    return await policy.action(obs)


@pytest.mark.asyncio
async def test_photo_inputs_and_no_position_leakage():
    requests = []
    def transport(payload):
        requests.append(payload)
        return '{"p":[{"a":"S","v":0}],"d":true,"s":"Arrived"}'
    planner = Astra(model='fixture-photo-model', transport=transport)
    policy = AstraPolicy(planner)
    obs = recorded_observation()
    # Use the parser's full form so this verifies actual accepted plan syntax.
    planner._transport = lambda payload: requests.append(payload) or json.dumps({'plan': [{'action': 'stop'}], 'done': True, 'thought': 'Arrived'})
    actions, done, metadata = await decision(policy, obs)
    assert done and not np.any(actions)
    sent = json.dumps(requests)
    assert '51.123456' not in sent and 'longitude' not in sent and 'private_judge' not in sent
    images = [item for message in requests[0]['input'] for item in message['content'] if item['type'] == 'input_image']
    assert len(images) == 2
    for image in images:
        Image.open(io.BytesIO(base64.b64decode(image['image_url'].split(',')[1]))).verify()
    assert metadata['policy']['model'] == 'fixture-photo-model'
    assert len(metadata['policy']['source_sha256']) == 64
    assert metadata['policy']['controller']['heading_sign'] == -1
    await policy.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('answer', ['malformed', '{"plan":[{"action":"forward","amount":1}],"done":true}'])
async def test_failures_stop_and_remain_failures(answer):
    policy = AstraPolicy(Astra(model='fixture', transport=lambda _: answer))
    actions, done, metadata = await decision(policy, recorded_observation())
    assert not done and not np.any(actions) and metadata['error']
    await policy.close()


@pytest.mark.asyncio
async def test_slow_inference_does_not_block_control_responses():
    planner = Astra(model='fixture', transport=lambda _: (time.sleep(.15) or '{"plan":[{"action":"forward","amount":0.1}],"done":false}'))
    policy = AstraPolicy(planner)
    obs = recorded_observation()
    started = time.monotonic()
    actions, done, metadata = await policy.action(obs)
    assert time.monotonic() - started < .1 and not np.any(actions)
    assert metadata['phase'] == 'thinking'
    await policy.pending
    actions, done, metadata = await policy.action(obs)
    assert actions.shape == (3, 2) and actions[0, 0] == pytest.approx(.15)
    await policy.close()


def test_freshness_and_rear_guard():
    obs = recorded_observation()
    with pytest.raises(ValueError, match='stale'):
        permitted_observation(replace(obs, images=(replace(obs.images[0], capture_time_ns=1), obs.images[1])))
    controller = Controller()
    controller.load(Astra._parse('{"plan":[{"action":"backward","amount":0.2}],"done":false}'))
    assert controller.step(0, {}) == (0, 0)
    assert 'rear' in controller.error


def test_feedback_turn_and_configurable_heading_sign():
    controller = Controller(ControllerConfig(angular=.2))
    controller.load(Astra._parse('{"plan":[{"action":"left","amount":30}],"done":false}'))
    assert controller.step(0, {'orientation': 10}) == (0, .2)
    assert controller.step(.1, {'orientation': 345}) == (0, 0)
    assert not controller.moves
    with pytest.raises(ValueError):
        ControllerConfig(linear=2)
