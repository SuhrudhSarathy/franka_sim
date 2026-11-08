"""Franka Panda simulator on MuJoCo (CPU) and MuJoCo Warp, built for behavior cloning."""

import os as _os
import sys as _sys


def _default_gl_backend(environ=_os.environ, platform: str = _sys.platform) -> str | None:
    """OpenGL backend for MuJoCo rendering when the user did not choose one (``MUJOCO_GL``).

    MuJoCo defaults to GLFW, which needs a display. On a headless Linux server (no ``DISPLAY`` /
    ``WAYLAND_DISPLAY``) use EGL instead: GPU rendering with the NVIDIA driver, or Mesa's software
    EGL. Set ``MUJOCO_GL=osmesa`` yourself on machines without any EGL library.
    """
    if environ.get("MUJOCO_GL") or not platform.startswith("linux"):
        return None
    if environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"):
        return None
    return "egl"


# MuJoCo reads MUJOCO_GL once, when it is first imported, so this must run before importing it.
if (_gl := _default_gl_backend()) is not None:
    if "mujoco" in _sys.modules:
        import warnings as _warnings

        _warnings.warn("mujoco was imported before franka_sim on a headless machine; rendering will "
                       "fail. Import franka_sim first or set MUJOCO_GL=egl.", stacklevel=2)
    _os.environ["MUJOCO_GL"] = _gl

from franka_sim.env import ACTION_DIMS, FrankaEnv
from franka_sim.tasks import TASKS, Task, make_task, register_task

__all__ = ["ACTION_DIMS", "TASKS", "FrankaEnv", "Task", "make", "make_task", "register_task"]


def make(task: str = "reach", **kwargs) -> FrankaEnv:
    """Create a batched ``FrankaEnv``. See ``FrankaEnv`` for arguments."""
    return FrankaEnv(task, **kwargs)
