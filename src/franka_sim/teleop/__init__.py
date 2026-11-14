"""Teleoperation: devices that turn an operator's input into end-effector motion (see ``base``)."""

from franka_sim.teleop.base import EVENTS, ScriptedDevice, TeleopDevice, TeleopInput

__all__ = ["EVENTS", "KeyboardTeleop", "ScriptedDevice", "TeleopDevice", "TeleopInput"]


def __getattr__(name):  # pygame is imported only when the keyboard console is used
    if name == "KeyboardTeleop":
        from franka_sim.teleop.keyboard import KeyboardTeleop

        return KeyboardTeleop
    raise AttributeError(name)
