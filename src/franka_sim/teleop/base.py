"""Teleop device interface.

A device is polled once per control step. It reports a *twist*, the operator's desired end-effector
motion as 6 values in ``[-1, 1]`` in an *operator frame* (forward, left, up, then three rotations),
plus discrete *events* since the last poll. ``TeleopSession`` maps the twist to the world with
``twist_to_world`` and scales it by the current speed into ``delta_ee_pose`` commands.

Operator frames (``TELEOP_FRAMES``). Every motion pivots at the gripper tip (``gripper`` site):

``world`` (default)  ``vx vy vz``: world +x (away from the base), +y (robot's left), +z (up).
``heading``          ``vz`` is world up, but forward/left turn with the gripper's yaw (forward is
                     the gripper's x axis projected onto the floor).

In both, ``wz`` spins the gripper about its own approach axis (the axis the wrist camera looks
along; with the gripper pointing down this is a yaw), and ``wx`` / ``wy`` roll / pitch about the
frame's forward / left axes. Frames use the *commanded* gripper orientation.

Events (strings):

========== =====================================================================================
``toggle``  take control from the learner / hand control back to it
``grip``    toggle the gripper (open <-> closed)
``pause``   pause / resume the clock
``save``    end the episode now and save it
``discard`` drop the episode and start a new one
``quit``    stop the session (the current episode is dropped)
``frame``   switch to the next operator frame
``faster``  / ``slower``: scale the teleop speed
========== =====================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EVENTS = ("toggle", "grip", "pause", "save", "discard", "quit", "faster", "slower", "frame")
TELEOP_FRAMES = ("world", "heading")


def _quat_to_mat(q) -> np.ndarray:
    import mujoco

    out = np.empty(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=np.float64))
    return out.reshape(3, 3)


def frame_axes(frame: str, ee_quat) -> np.ndarray:
    """Operator frame axes (columns: forward, left, up) in the world, for the gripper at ``ee_quat``."""
    if frame == "world":
        return np.eye(3)
    if frame != "heading":
        raise ValueError(f"unknown teleop frame {frame!r}; expected one of {TELEOP_FRAMES}")
    x = _quat_to_mat(ee_quat)[:, 0]  # the gripper's x axis: horizontal while it points down
    yaw = np.arctan2(x[1], x[0]) if np.hypot(x[0], x[1]) > 1e-3 else 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def twist_to_world(twist, frame: str, ee_quat) -> np.ndarray:
    """Map an operator-frame twist (6,) to a world-frame one (linear, then rotation vector)."""
    twist = np.asarray(twist, dtype=float)
    axes = frame_axes(frame, ee_quat)
    spin = -_quat_to_mat(ee_quat)[:, 2]  # about the approach axis, signed like world yaw when pointing down
    rot = axes[:, 0] * twist[3] + axes[:, 1] * twist[4] + spin * twist[5]
    return np.concatenate([axes @ twist[:3], rot])


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
