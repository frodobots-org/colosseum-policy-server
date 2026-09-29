"""NumPy wire codecs shared by protocol clients and isolated model workers."""
from __future__ import annotations

import base64
from typing import Any, Mapping

import numpy as np


def _json_array(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {"__numpy__": base64.b64encode(value.tobytes()).decode("ascii"), "dtype": value.dtype.str, "shape": list(value.shape)}


def _json_decode(value: Any, *, max_values: int = 1_000_000) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value)
    if not isinstance(value, Mapping) or set(value) != {"__numpy__", "dtype", "shape"}:
        raise ValueError("model actions have an invalid array encoding")
    dtype, shape = np.dtype(value["dtype"]), value["shape"]
    if dtype.kind not in "biuf":
        raise ValueError("model actions require numeric array data")
    if not isinstance(shape, list) or not shape or any(type(item) is not int or item < 1 for item in shape) or int(np.prod(shape)) > max_values:
        raise ValueError("model actions have an invalid shape")
    try:
        data = base64.b64decode(value["__numpy__"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("model actions have invalid base64") from exc
    if len(data) != int(np.prod(shape)) * dtype.itemsize:
        raise ValueError("model actions have invalid byte length")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def _msgpack_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in "OVc":
        raise ValueError("unsupported array dtype")
    if isinstance(value, np.ndarray):
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot MessagePack {type(value).__name__}")


def _msgpack_object(value: dict[Any, Any]) -> Any:
    value = {key.decode("ascii") if isinstance(key, bytes) else key: item for key, item in value.items()}
    if "nd" in value:
        dtype = np.dtype(value["type"])
        if dtype.kind in "OVc":
            raise ValueError("unsupported array dtype")
        array = np.frombuffer(value["data"], dtype=dtype)
        return array.reshape(tuple(value["shape"])).copy() if value["nd"] else array[0]
    if "__ndarray__" in value:
        if np.dtype(value["dtype"]).kind in "OVc":
            raise ValueError("unsupported array dtype")
        return np.frombuffer(value["data"], dtype=np.dtype(value["dtype"])).reshape(tuple(value["shape"])).copy()
    if "__npgeneric__" in value:
        return np.dtype(value["dtype"]).type(value["data"])
    return value


def _groot_default(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.kind in "OVc":
            raise ValueError("unsupported array dtype")
        return {b"nd": isinstance(value, np.ndarray), b"type": array.dtype.str,
                b"shape": array.shape, b"data": array.tobytes()}
    raise TypeError("unsupported MessagePack value")
