from __future__ import annotations

from collections.abc import Callable
from functools import partial

from franka_sim.tasks.base import Evaluation, Task
from franka_sim.tasks.buttons import ButtonPressTask
from franka_sim.tasks.gears import GearAssemblyTask
from franka_sim.tasks.hammer import HammerNailsTask
from franka_sim.tasks.insertion import InsertionTask
from franka_sim.tasks.kitting import KittingTask
from franka_sim.tasks.pcb import PCBInsertionTask
from franka_sim.tasks.pegs import PegInsertionTask
from franka_sim.tasks.reach import ReachTask

# Registered tasks. ``<family>`` asks for one target chosen by the condition (language adherence);
# ``<family>_long`` asks for every target of the scene, in any order (long horizon, completion).
TASKS: dict[str, Callable[..., Task]] = {
    "reach": ReachTask,
    "hammer_nails": HammerNailsTask,
    "hammer_nails_long": partial(HammerNailsTask, long=True),
    "buttons": ButtonPressTask,
    "buttons_long": partial(ButtonPressTask, sequence_length=4, ordered=False),
    "peg_insertion": PegInsertionTask,
    "peg_insertion_long": partial(PegInsertionTask, mode="all"),
    "gear_assembly": GearAssemblyTask,
    "gear_assembly_long": partial(GearAssemblyTask, mode="all"),
    "kitting": KittingTask,
    "kitting_long": partial(KittingTask, mode="all"),
    "pcb_insertion": PCBInsertionTask,
}

# Episode horizons of the registered tasks: 2x the longest of 64 scripted demos (seed 1), rounded up
# to 50 steps (20 Hz control). Re-measure with scripts/evaluate.py --policy scripted after changing
# a task or an expert.
HORIZONS: dict[str, int] = {
    "reach": 100,
    "hammer_nails": 200, "hammer_nails_long": 350,
    "buttons": 100, "buttons_long": 300,
    "peg_insertion": 250, "peg_insertion_long": 700,
    "gear_assembly": 250, "gear_assembly_long": 1000,
    "kitting": 400, "kitting_long": 1500,
    "pcb_insertion": 250,
}


def make_task(task: str | Task, **kwargs) -> Task:
    """Instantiate a registered task; ``kwargs`` override the variant's defaults."""
    if isinstance(task, Task):
        return task
    try:
        factory = TASKS[task]
    except KeyError:
        raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}") from None
    inst = factory(**kwargs)
    inst.name = task  # recordings store this name, so a variant is rebuilt as the same variant
    if task in HORIZONS and not kwargs:  # custom kwargs keep the task's own default horizon
        inst.max_episode_steps = HORIZONS[task]
    return inst


def register_task(name: str, factory: Callable[..., Task]) -> None:
    TASKS[name] = factory


__all__ = [
    "HORIZONS", "TASKS", "ButtonPressTask", "Evaluation", "GearAssemblyTask", "HammerNailsTask", "InsertionTask", "KittingTask",
    "PCBInsertionTask", "PegInsertionTask", "ReachTask", "Task", "make_task",
    "register_task",
]
