"""Wire format of the model API: msgpack messages, with numpy arrays packed as raw bytes.

A request is a map ``{"op": <name>, ...arguments}``; the reply is ``{"ok": true, "result": ...}`` or
``{"ok": false, "error": "<message>"}``. Arrays travel as ``{"__nd__": true, "dtype", "shape",
"data"}`` (no copy beyond the bytes themselves), so images cost what their pixels cost.
"""

from __future__ import annotations

import msgpack
import numpy as np


def _default(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object:  # e.g. instructions: strings
        return {"__obj__": True, "shape": list(obj.shape), "items": obj.ravel().tolist()}
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__nd__": True, "dtype": arr.dtype.str, "shape": list(arr.shape), "data": arr.tobytes()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def _hook(obj):
    if obj.get("__obj__"):
        out = np.empty(len(obj["items"]), dtype=object)
        out[:] = obj["items"]
        return out.reshape(obj["shape"])
    if obj.get("__nd__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj


def pack(obj) -> bytes:
    return msgpack.packb(obj, default=_default, use_bin_type=True)


def unpack(data: bytes):
    return msgpack.unpackb(data, object_hook=_hook, raw=False)
