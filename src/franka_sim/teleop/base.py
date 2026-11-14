"""Teleop device interface.

A device is polled once per control step. It reports a *twist*, the operator's desired end-effector
motion as 6 values in ``[-1, 1]`` (``vx vy vz wx wy wz``, world frame: robot facing +x, its left
+y, up +z), plus discrete *events* since the last poll. ``TeleopSession`` scales the twist by the
current speed into ``delta_ee_pose`` commands. Optionally a device also shows the operator what the
robot sees (``show``).

Events (strings):

========== =====================================================================================
``toggle``  take control from the learner / hand control back to it
``grip``    toggle the gripper (open <-> closed)
``pause``   pause / resume the clock
``save``    end the episode now and save it
``discard`` drop the episode and start a new one
``quit``    stop the session (the current episode is dropped)
``faster``  / ``slower``: scale the teleop speed
========== =====================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVENTS = ("toggle", "grip", "pause", "save", "discard", "quit", "faster", "slower")


@dataclass
class TeleopInput:
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6))  # each in [-1, 1]
    events: list[str] = field(default_factory=list)
    fine: bool = False  # precision mode: the session slows the twist down

    @property
    def moving(self) -> bool:
        return bool(np.any(np.abs(self.twist) > 1e-3))


class TeleopDevice:
    """Base class: override ``poll``; ``show`` and ``close`` are optional."""

    def poll(self) -> TeleopInput:
        raise NotImplementedError

    def show(self, frames: list[np.ndarray], hud: dict) -> None:  # noqa: B027 - optional
        pass

    def close(self) -> None:  # noqa: B027 - optional
        pass


class ScriptedDevice(TeleopDevice):
    """Replays a list of ``TeleopInput`` (one per poll, then idle): for tests and demos."""

    def __init__(self, inputs: list[TeleopInput]):
        self.inputs = list(inputs)
        self.shown = 0

    def poll(self) -> TeleopInput:
        return self.inputs.pop(0) if self.inputs else TeleopInput()

    def show(self, frames, hud) -> None:
        self.shown += 1
