from __future__ import annotations

import mujoco

from franka_sim.backends.base import Backend
from franka_sim.backends.cpu import CpuBackend

BACKENDS = ("cpu", "warp")


def make_backend(name: str, model: mujoco.MjModel, num_envs: int = 1, **kwargs) -> Backend:
    """Create a physics backend: ``"cpu"`` (MuJoCo C) or ``"warp"`` (MuJoCo Warp, GPU or CPU)."""
    if name == "cpu":
        return CpuBackend(model, num_envs, **kwargs)
    if name == "warp":
        from franka_sim.backends.warp import WarpBackend  # optional dependency

        return WarpBackend(model, num_envs, **kwargs)
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")


__all__ = ["BACKENDS", "Backend", "CpuBackend", "make_backend"]
