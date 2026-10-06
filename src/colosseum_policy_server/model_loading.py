"""Weight loading and inference entry points; heavyweight imports are lazy.

Install the relevant upstream SDK in the model worker's own environment.
All checkpoint/tokenizer/statistics paths are provided by the operator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .pi05_contract import load_quantile_stats, prepare_droid_request, droid_actions_from_lerobot


def patch_molmo_bf16(checkpoint):
    """Apply the deployed dtype fixes to the explicitly selected local source.

    Unlike the old deployment helper, never scan or modify unrelated HF caches.
    Transformers copies this local source to its dynamic-module cache on load.
    A backup is created once; weights are never modified.
    """
    source = Path(checkpoint) / "modeling_molmoact2.py"
    original = source.read_text(encoding="utf-8")
    updated = original
    for before, after, marker in (
        ("device=device,\n            dtype=torch.float32,\n            generator=generator,",
         "device=device,\n            dtype=source_tensor.dtype,  # patched_bf16_dtype\n            generator=generator,",
         "patched_bf16_dtype"),
        ("return value.detach().cpu().numpy().astype(np.float32, copy=False)",
         "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
         "patched_bf16_to_array"),
    ):
        if marker in updated:
            continue
        if updated.count(before) != 1:
            raise ValueError("Molmo dtype fix does not match this checkpoint's model source")
        updated = updated.replace(before, after, 1)
    if updated != original:
        backup = source.with_suffix(".py.before-colosseum-bf16")
        if not backup.exists():
            with backup.open("x", encoding="utf-8") as stream:
                stream.write(original)
        source.write_text(updated, encoding="utf-8")


class MolmoRuntime:
    def __init__(self, checkpoint, *, device="cuda:0", num_steps=10,
                 dtype="bfloat16", patch_bf16=False, robot_type="franka"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from PIL import Image

        if robot_type not in {'franka', 'yam'}:
            raise ValueError('Molmo robot_type must be franka or yam')
        self.action_dim = 14 if robot_type == 'yam' else 8
        self.image_keys = ('top_cam', 'left_cam', 'right_cam') if robot_type == 'yam' else ('external_cam', 'wrist_cam')
        self.norm_tag = 'yam_dual_molmoact2' if robot_type == 'yam' else 'franka_droid'
        self.torch, self.Image, self.num_steps = torch, Image, num_steps
        if patch_bf16:
            patch_molmo_bf16(checkpoint)
        if dtype != "float32":
            source = (Path(checkpoint) / "modeling_molmoact2.py").read_text(encoding="utf-8")
            if not all(marker in source for marker in ("patched_bf16_dtype", "patched_bf16_to_array")):
                raise ValueError("Molmo bfloat16 requires the deployed dtype fixes; use --patch-molmo-bf16")
        self.processor = AutoProcessor.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=True,
            extra_special_tokens={},
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(checkpoint), local_files_only=True, trust_remote_code=True,
            torch_dtype=getattr(torch, dtype),
        ).to(device).eval()

        target_dtype = next(self.model.parameters()).dtype

        def move_and_cast(inputs, device):
            out = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(device)
                    if value.is_floating_point() and value.dtype != target_dtype:
                        value = value.to(target_dtype)
                out[key] = value
            return out

        self.model._move_inputs_to_device = move_and_cast

    def infer(self, request):
        state = np.asarray(request["state"], dtype=np.float32).reshape(-1)
        if state.shape != (self.action_dim,) or not np.isfinite(state).all():
            raise ValueError(f"Molmo requires {self.action_dim} finite state values")
        images = []
        for key in self.image_keys:
            array = np.asarray(request[key])
            if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
                raise ValueError("Molmo image must be nonempty HWC RGB")
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            images.append(self.Image.fromarray(array))
        with self.torch.inference_mode():
            output = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=request["instruction"], state=state,
                norm_tag=self.norm_tag, inference_action_mode="continuous",
                enable_depth_reasoning=False, num_steps=int(request.get("num_steps", self.num_steps)),
                normalize_language=True, enable_cuda_graph=False,
            )
        actions = output.actions
        if self.torch.is_tensor(actions):
            actions = actions.detach().float().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions}


class Pi05Runtime:
    def __init__(self, checkpoint, tokenizer, stats, *, device="cuda", num_steps=10):
        import torch
        from transformers import AutoTokenizer
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies.pi05 import PI05Policy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

        self.torch, self.device, self.num_steps = torch, device, num_steps
        self.keys = OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        self.stats = load_quantile_stats(stats)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer), local_files_only=True)
        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.device, config.dtype, config.compile_model = device, "float32", False
        self.model = PI05Policy.from_pretrained(str(checkpoint), config=config, strict=True).to(device).eval()

    def infer(self, request):
        prepared = prepare_droid_request(request, self.stats)
        tokens = self.tokenizer([prepared.prompt], padding="max_length", padding_side="right", truncation=True,
                                max_length=200, return_tensors="pt")
        torch = self.torch
        state_key, tokens_key, mask_key = self.keys
        batch = {state_key: torch.from_numpy(prepared.normalized_state)[None].to(self.device),
                 tokens_key: tokens["input_ids"].to(self.device),
                 mask_key: tokens["attention_mask"].bool().to(self.device)}
        for target, source in (("base_0_rgb", "exterior_image_1_left"), ("left_wrist_0_rgb", "wrist_image_left")):
            image = np.asarray(request[f"observation/{source}"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("pi05 image must be uint8 HWC RGB")
            batch[f"observation.images.{target}"] = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None].to(self.device).float() / 255
        with torch.no_grad():
            raw = self.model.predict_action_chunk(batch, num_steps=self.num_steps)[0].float().cpu().numpy()
        return {"actions": droid_actions_from_lerobot(raw, self.stats),
                "raw_actions_15x32": raw.astype(np.float32),
                "conversion": "OpenPI DROID quantile unnormalize 32 dimensions then [:, :8]"}


def load_lap(checkpoint, tokenizer):
    import dataclasses
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    import lap.models.tokenizer as tokenization
    tokenization.PALIGEMMA_TOKENIZER_MODEL_PATH = str(tokenizer)
    from lap.training import config
    from lap.policies.policy_config_adapter import create_trained_policy
    cfg = config.get_config("lap")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, stop_action_to_vlm_grad=False))
    return create_trained_policy(cfg, str(checkpoint))


def load_groot(checkpoint, *, device="cuda", processor=None):
    if processor is not None:
        from transformers import Qwen3VLProcessor
        from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
        def local_processor(_name, transformers_loading_kwargs):
            options = dict(transformers_loading_kwargs)
            options["local_files_only"] = True
            return Qwen3VLProcessor.from_pretrained(str(processor), **options)
        processing.build_processor = local_processor
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    return Gr00tPolicy(embodiment_tag=EmbodimentTag.resolve("oxe_droid_relative_eef_relative_joint"),
                      model_path=str(checkpoint), device=device, strict=True)
