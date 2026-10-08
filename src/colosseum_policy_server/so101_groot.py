"""LeRobot GR00T N1.7 SO100/101 worker; absolute joint targets in dataset units.

Loads local assets only. Uses the checkpoint's saved pre/post processors and its
bundled vlm_processor so normalization and tokenization match training. Camera
inputs are slots, not fixed views: the external image fills camera_0 and the
wrist image camera_1. The remaining slots are left out instead of zero-padded,
so the VLM sees the two real views only.
"""
import json
from pathlib import Path

import numpy as np


class GrootSO101Runtime:
    action_dim = 6
    image_keys = ('external_cam', 'wrist_cam')

    def __init__(self, checkpoint, *, base_model, device='cuda'):
        import torch
        from lerobot.policies.groot.configuration_groot import GrootConfig
        from lerobot.policies.groot.modeling_groot import GrootPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.torch = torch
        checkpoint = Path(checkpoint)
        raw = json.loads((checkpoint / 'config.json').read_text())
        inputs, outputs = raw['input_features'], raw['output_features']
        self.slots = sorted(key for key in inputs if key.startswith('observation.images.camera_'))
        if (raw.get('type') != 'groot' or raw.get('embodiment_tag') != 'new_embodiment'
                or raw.get('use_relative_actions', False) or raw.get('n_obs_steps') != 1
                or len(self.slots) < 2 or set(inputs) != {*self.slots, 'observation.state'}
                or inputs['observation.state']['shape'] != [6] or outputs['action']['shape'] != [6]):
            raise ValueError('Expected an absolute 6-D LeRobot GR00T SO100/101 checkpoint with camera slots')
        processor = checkpoint / 'vlm_processor'
        if not processor.is_dir():
            raise ValueError('GR00T SO100/101 checkpoint is missing vlm_processor/')
        self.chunk = raw['chunk_size']
        config = GrootConfig.from_pretrained(str(checkpoint), local_files_only=True)
        config.device = device
        config.base_model_path = str(Path(base_model).resolve())
        self.policy = GrootPolicy.from_pretrained(str(checkpoint), config=config,
                                                 local_files_only=True).to(device).eval()
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={
                'groot_n1_7_vlm_encode_v1': {'model_name': str(processor.resolve()), 'device': device},
                'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        # The first call is about 2 s slower than the rest; spend it here, not in the trial.
        blank = np.zeros((480, 640, 3), np.uint8)
        self.infer({'external_cam': blank, 'wrist_cam': blank, 'instruction': 'warm up',
                    'state': np.zeros(6, np.float32)})

    def infer(self, payload):
        state = np.asarray(payload['state'], dtype=np.float32)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError('GR00T SO101 requires 6 finite state values')
        instruction = payload['instruction']
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Instruction must be nonempty text')
        batch = {'observation.state': self.torch.from_numpy(state), 'task': instruction}
        for source, slot in zip(self.image_keys, self.slots):
            frame = np.asarray(payload[source])
            if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3 or not frame.size:
                raise ValueError('GR00T SO101 image must be nonempty uint8 HWC RGB')
            batch[slot] = self.torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1), dtype=np.float32) / 255)
        # No select_action queue: each /act call predicts and postprocesses a full chunk.
        with self.torch.inference_mode():
            actions = self.post(self.policy.predict_action_chunk(self.pre(batch)))
        actions = actions.detach().float().cpu().numpy()
        if actions.shape != (1, self.chunk, 6) or not np.isfinite(actions).all():
            raise ValueError(f'GR00T SO101 must return a finite 1 x {self.chunk} x 6 action tensor')
        return {'actions': actions[0].astype(np.float32)}
