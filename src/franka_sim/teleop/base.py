"""Teleop device interface.

A device is polled once per control step. It reports a *twist*, the operator's desired end-effector
motion as 6 values in ``[-1, 1]`` in an *operator frame* (forward, left, up, then three rotations),
plus discrete *events* since the last poll. ``TeleopSession`` maps the twist to the world with
``twist_to_world`` and scales it by the current speed into ``delta_ee_pose`` commands.

Operator frames (``TELEOP_FRAMES``):

``wrist`` (default)  follows the gripper, as seen in the ``wrist`` camera. Forward is the gripper's
                     approach axis (into the camera image), left/up are image left/up. Rotations:
                     ``wz`` spins the gripper about its approach axis, ``wy`` tilts the view up,
                     ``wx`` turns it right. The frame moves with the *commanded* gripper pose.
``world``            fixed: forward +x (away from the base), left +y, up +z; ``wx wy wz`` are roll,
                     pitch, yaw about those axes. Optionally a device also shows the operator what the
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
TELEOP_FRAMES = ("wrist", "world")

# Wrist frame in the end-effector (``gripper`` site) frame, whose +z is the approach axis. The wrist
# camera sits behind the hand looking along the approach axis: image right is exactly the site's -y,
# image up is mostly the site's -x (the camera is tilted ~33 deg; we snap to the gripper's axes so
# "forward" is a clean approach motion). Columns: forward, left, up.
_WRIST_TRANSLATION = np.column_stack([(0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0)])
# Rotation axes for (wx, wy, wz): about -up (turn right), -left (tilt up), -forward (spin; with the
# gripper pointing down this is world yaw, as in the world frame).
_WRIST_ROTATION = np.column_stack([(1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)])


def twist_to_world(twist: np.ndarray, frame: str, ee_quat: np.ndarray | None = None) -> np.ndarray:
    """Map an operator-frame twist (6,) to the world frame; ``ee_quat`` (wxyz) is needed for ``wrist``."""
    twist = np.asarray(twist, dtype=float)
    if frame == "world":
        return twist.copy()
    if frame != "wrist":
        raise ValueError(f"unknown teleop frame {frame!r}; expected one of {TELEOP_FRAMES}")
    import mujoco

    rot = np.empty(9)
    mujoco.mju_quat2Mat(rot, np.asarray(ee_quat, dtype=np.float64))
    rot = rot.reshape(3, 3)
    return np.concatenate([rot @ _WRIST_TRANSLATION @ twist[:3], rot @ _WRIST_ROTATION @ twist[3:]])


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
