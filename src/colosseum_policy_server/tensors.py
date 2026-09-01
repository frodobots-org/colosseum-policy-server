from __future__ import annotations

import numpy as np

from . import colosseum_pb2 as pb

_NUMPY_TO_PROTO = {
    np.dtype("float32"): pb.FLOAT32,
    np.dtype("float64"): pb.FLOAT64,
    np.dtype("uint8"): pb.UINT8,
    np.dtype("int32"): pb.INT32,
    np.dtype("int64"): pb.INT64,
}
_PROTO_TO_NUMPY = {value: key for key, value in _NUMPY_TO_PROTO.items()}


def tensor_from_numpy(array: np.ndarray) -> pb.Tensor:
    contiguous = np.ascontiguousarray(array)
    dtype = _NUMPY_TO_PROTO.get(contiguous.dtype)
    if dtype is None:
        raise TypeError(f"unsupported tensor dtype: {contiguous.dtype}")
    return pb.Tensor(shape=contiguous.shape, dtype=dtype, data=contiguous.tobytes())


def tensor_to_numpy(tensor: pb.Tensor) -> np.ndarray:
    dtype = _PROTO_TO_NUMPY.get(tensor.dtype)
    if dtype is None:
        raise TypeError(f"unsupported protobuf tensor dtype: {tensor.dtype}")
    expected = int(np.prod(tensor.shape, dtype=np.int64)) * dtype.itemsize
    if len(tensor.data) != expected:
        raise ValueError(f"tensor byte length mismatch: expected {expected}, got {len(tensor.data)}")
    return np.frombuffer(tensor.data, dtype=dtype).reshape(tuple(tensor.shape))
