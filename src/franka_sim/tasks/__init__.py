from __future__ import annotations

from collections.abc import Callable

from franka_sim.tasks.base import Evaluation, Task
from franka_sim.tasks.buttons import ButtonPressTask
from franka_sim.tasks.gears import GearAssemblyTask
from franka_sim.tasks.hammer import HammerNailsTask
from franka_sim.tasks.insertion import InsertionTask
from franka_sim.tasks.kitting import KittingTask
from franka_sim.tasks.pcb import PCBInsertionTask
from franka_sim.tasks.pegs import PegInsertionTask
from franka_sim.tasks.reach import ReachTask

TASKS: dict[str, Callable[..., Task]] = {
    "reach": ReachTask,
    "hammer_nails": HammerNailsTask,
    "peg_insertion": PegInsertionTask,
    "buttons": ButtonPressTask,
    "gear_assembly": GearAssemblyTask,
    "kitting": KittingTask,
    "pcb_insertion": PCBInsertionTask,
}


def make_task(task: str | Task, **kwargs) -> Task:
    if isinstance(task, Task):
        return task
    try:
        return TASKS[task](**kwargs)
    except KeyError:
        raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}") from None


def register_task(name: str, factory: Callable[..., Task]) -> None:
    TASKS[name] = factory


__all__ = [
    "TASKS", "ButtonPressTask", "Evaluation", "GearAssemblyTask", "HammerNailsTask", "InsertionTask", "KittingTask",
    "PCBInsertionTask", "PegInsertionTask", "ReachTask", "Task", "make_task",
    "register_task",
]
