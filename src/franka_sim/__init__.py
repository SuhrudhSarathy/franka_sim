"""Franka Panda simulator on MuJoCo (CPU) and MuJoCo Warp, built for behavior cloning."""

from franka_sim.env import ACTION_DIMS, FrankaEnv
from franka_sim.tasks import TASKS, Task, make_task, register_task

__all__ = ["ACTION_DIMS", "TASKS", "FrankaEnv", "Task", "make", "make_task", "register_task"]


def make(task: str = "reach", **kwargs) -> FrankaEnv:
    """Create a batched ``FrankaEnv``. See ``FrankaEnv`` for arguments."""
    return FrankaEnv(task, **kwargs)
