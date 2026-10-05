"""Shared wire observation decoding; no robot or model assumptions."""
import numpy as np
from .. import colosseum_pb2 as pb
from ..tensors import tensor_to_numpy

def _rgb(image: pb.Image | None, sensor_id: str) -> np.ndarray:
    if image is None or image.encoding != pb.RAW_RGB or image.width < 1 or image.height < 1 or len(image.data) != image.width * image.height * 3:
        raise ValueError(f"invalid RGB sensor {sensor_id}")
    return np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, 3).copy()


def _state(observation: pb.Observation, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in observation.state:
        raise ValueError(f"observation is missing {name}")
    value = np.asarray(tensor_to_numpy(observation.state[name]), dtype=np.float32).reshape(-1)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"invalid observation state {name}")
    return value
