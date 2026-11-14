"""Franka Panda constants and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

ASSET_DIR = Path(__file__).parent / "assets"
PANDA_XML = ASSET_DIR / "panda" / "mjx_panda.xml"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("finger_joint1", "finger_joint2")
ARM_ACTUATORS = tuple(f"actuator{i}" for i in range(1, 8))
GRIPPER_ACTUATOR = "actuator8"
EE_SITE = "gripper"
ROBOT_BODIES = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
)

# The "ready" pose: gripper pointing straight down above the workspace.
HOME_QPOS = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])

# Joint velocity limits of the Panda (rad/s, from the Franka datasheet). Every arm controller keeps
# its joint targets within these rates.
JOINT_VEL_LIMITS = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
# Joint acceleration limits (rad/s^2, Franka datasheet): the command brakes in time to stop at its
# target, so large jumps do not overshoot.
JOINT_ACC_LIMITS = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])

# Arm joint position servos: force = KP * (ctrl - q) - KV * qdot, plus the joint's own damping.
# KP is Menagerie's. Menagerie's KV leaves the wrist (joints 5-7) at a damping ratio of ~0.3 (30%
# overshoot, visible ringing) and joints 2-4 at ~0.75; these KV values make every joint about
# critically damped (zeta ~1.0-1.2) at the home pose, using the joint-space inertia there.
ARM_KP = np.array([1000.0, 1000.0, 750.0, 750.0, 300.0, 300.0, 300.0])
ARM_KV = np.array([20.0, 45.0, 22.0, 22.0, 12.0, 13.0, 10.0])

# The ``gripper`` site sits between the finger pads; the fingertips reach this far below it.
FINGERTIP_OFFSET = 0.012

# Finger joint range is [0, 0.04] per finger; the gripper actuator commands one finger,
# an equality constraint mirrors the other.
FINGER_MAX = 0.04
GRIPPER_MAX_WIDTH = 2 * FINGER_MAX

# Gripper servo (see scene.build_spec): force = KP * (ctrl - q) - KV * qdot, clamped to MAX_FORCE.
GRIPPER_KP = 3000.0
GRIPPER_KV = 60.0
GRIPPER_MAX_FORCE = 70.0


def gripper_ctrl(closure: np.ndarray) -> np.ndarray:
    """Map gripper closure in [0, 1] (0 = open, 1 = closed) to actuator control."""
    return FINGER_MAX * (1.0 - np.clip(closure, 0.0, 1.0))
