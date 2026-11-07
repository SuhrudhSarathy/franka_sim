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

# The ``gripper`` site sits between the finger pads; the fingertips reach this far below it.
FINGERTIP_OFFSET = 0.012

# Finger joint range is [0, 0.04] per finger; the gripper actuator commands one finger,
# an equality constraint mirrors the other.
FINGER_MAX = 0.04
GRIPPER_MAX_WIDTH = 2 * FINGER_MAX


def gripper_ctrl(closure: np.ndarray) -> np.ndarray:
    """Map gripper closure in [0, 1] (0 = open, 1 = closed) to actuator control."""
    return FINGER_MAX * (1.0 - np.clip(closure, 0.0, 1.0))
