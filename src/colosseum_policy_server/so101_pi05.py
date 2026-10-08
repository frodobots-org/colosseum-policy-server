"""LeRobot pi0.5 SO100/101 worker; absolute joint targets in dataset units.

Loads local assets only. Uses the checkpoint's saved pre/post processors so
state/action normalization and tokenization match training. Camera inputs are
slots, not fixed views: the external image fills camera_0 and the wrist image
camera_1; remaining slots are zero images flagged as padding.
"""
import json
from pathlib import Path

import numpy as np


class Pi05SO101Runtime:
    action_dim = 6
    image_keys = ('external_cam', 'wrist_cam')

    def __init__(self, checkpoint, *, device='cuda', num_steps=10):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        import lerobot.policies.pi05.processor_pi05  # noqa: F401  registers saved processor steps

        self.torch = torch
        checkpoint = Path(checkpoint)
        raw = json.loads((checkpoint / 'config.json').read_text())
        inputs, outputs = raw['input_features'], raw['output_features']
        self.slots = sorted(key for key in inputs if key.startswith('observation.images.camera_'))
        if (raw.get('type') != 'pi05' or raw.get('use_relative_actions', False) or len(self.slots) < 2
                or set(inputs) != {*self.slots, 'observation.state'}
                or inputs['observation.state']['shape'] != [6] or outputs['action']['shape'] != [6]):
            raise ValueError('Expected an absolute 6-D LeRobot pi0.5 SO100/101 checkpoint with camera slots')
        tokenizer = checkpoint / 'tokenizer/tokenizer.model'
        if not tokenizer.is_file():
            raise ValueError('pi0.5 SO100/101 checkpoint is missing tokenizer/tokenizer.model')
        config = PI05Config.from_pretrained(str(checkpoint), local_files_only=True)
        config.device = device
        config.compile_model = False
        config.num_inference_steps = num_steps
        config.text_tokenizer_name = str(tokenizer)
        self.shape = tuple(inputs[self.slots[0]]['shape'])
        self.policy = PI05Policy.from_pretrained(str(checkpoint), config=config,
                                                local_files_only=True).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            config, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})

    def infer(self, payload):
        torch = self.torch
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('pi0.5 SO101 requires 6 finite state values')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        batch = {'observation.state': torch.from_numpy(state), 'task': instruction}
        for index, slot in enumerate(self.slots):
            if index < len(self.image_keys):
                image = np.asarray(payload[self.image_keys[index]])
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
                    raise ValueError('pi0.5 SO101 image must be nonempty uint8 HWC RGB')
                batch[slot] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
            else:
                batch[slot] = torch.zeros(self.shape, dtype=torch.float32)
            batch[f'{slot}_is_pad'] = torch.tensor([index >= len(self.image_keys)])  # already batched
        with torch.inference_mode():
            self.policy.reset()
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        actions = np.asarray(actions.detach().float().cpu().numpy(), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return {'actions': actions[:, :6]}
