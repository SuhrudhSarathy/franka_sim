"""Run franka-sim as a proxy robot: a real-time simulated Franka behind a web UI and a model API.

``SimRobot`` is the robot (one world, commands held like a robot driver holds them), ``RobotServer``
serves it, and ``franka_sim.client.RobotClient`` talks to it. Needs the ``server`` extra.
"""

from franka_sim.server.robot import SimRobot

__all__ = ["RobotServer", "SimRobot", "serve"]


def __getattr__(name):  # the web stack is imported only when a server is actually built
    if name in ("RobotServer", "serve"):
        from franka_sim.server import app

        return getattr(app, name)
    raise AttributeError(name)
