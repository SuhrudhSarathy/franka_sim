"""``RobotClient``: drive a franka-sim robot server (``scripts/serve.py``) from a model process.

The server is a proxy robot: the simulation runs on its own clock and holds your last command, as a
real robot driver does. Two ways to use it::

    from franka_sim.client import RobotClient

    with RobotClient("ws://localhost:8000/ws", cameras=("front", "wrist"), image_size=(224, 224)) as robot:
        # 1. Real time (the default): read observations and stream commands at your own rate.
        obs = robot.observe()                          # {"joint_pos", ..., "rgb_front", "instruction"}
        robot.act([0.0, 0.005, 0.0, 0, 0, 0, 0.0], mode="delta_ee_pose")

        # 2. Lockstep (evaluation): pause the clock; each step advances exactly one control period.
        robot.pause()
        obs, info = robot.reset(seed=100_000, options={"target": 2})
        obs, reward, terminated, truncated, info = robot.step(action, mode="joint_pos")
        robot.run()                                     # back to real time

Observations follow the benchmark's ``vision`` protocol (robot state, images, ``instruction``)
unless ``privileged=True``. Needs the ``server`` extra (``websockets``, ``msgpack``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from franka_sim.server.protocol import pack, unpack


class RobotError(RuntimeError):
    """The server rejected a request."""


class RobotClient:
    """Synchronous WebSocket client of ``RobotServer``.

    Args:
        url: the server's model endpoint, ``ws://host:port/ws``.
        cameras: cameras rendered into every observation (``rgb_<camera>``).
        image_size: ``(height, width)`` of those images (default: the server's).
        privileged: include the simulator's task state in observations (oracle use only).
        timeout: seconds to wait for a reply.
    """

    def __init__(self, url: str = "ws://127.0.0.1:8000/ws", *, cameras: Sequence[str] = (),
                 image_size: tuple[int, int] | None = None, privileged: bool = False, timeout: float = 60.0):
        from contextlib import ExitStack

        from websockets.sync.client import connect

        self.url = url
        self.cameras = list(cameras)
        self.image_size = list(image_size) if image_size else None
        self.privileged = privileged
        self.timeout = timeout
        self._stack = ExitStack()
        self._ws = self._stack.enter_context(connect(url, max_size=64 << 20, open_timeout=timeout))
        self.spec = self._call("info")

    # Plumbing -----------------------------------------------------------------------------------
    def _call(self, op: str, **kwargs):
        self._ws.send(pack({"op": op, **kwargs}))
        reply = unpack(self._ws.recv(timeout=self.timeout))
        if not reply["ok"]:
            raise RobotError(reply["error"])
        return reply["result"]

    def _obs_kwargs(self, cameras, image_size, privileged) -> dict:
        return {
            "cameras": list(self.cameras if cameras is None else cameras),
            "image_size": list(image_size) if image_size else self.image_size,
            "privileged": self.privileged if privileged is None else privileged,
        }

    # Robot interface ------------------------------------------------------------------------------
    def info(self) -> dict:
        """Tasks, cameras, action modes (name -> dims), control rate, whether the clock runs."""
        return self._call("info")

    def state(self) -> dict:
        """Robot and episode status: joints, end effector, gripper, success, progress, ..."""
        return self._call("state")

    def observe(self, cameras=None, image_size=None, privileged=None) -> dict:
        """The current observation (a dict of numpy arrays plus ``instruction``)."""
        return self._call("observe", **self._obs_kwargs(cameras, image_size, privileged))

    def act(self, action, mode: str = "ee_pose") -> None:
        """Real-time command (see ``SimRobot``): absolute targets are held, ``joint_vel`` stops after
        the server's watchdog, deltas are applied once. The last value is the gripper closure."""
        self._call("act", action=np.asarray(action, dtype=np.float64), mode=mode)

    def stop(self) -> None:
        """Cancel the command in force; the arm holds its commanded configuration."""
        self._call("stop")

    def step(self, action, mode: str = "ee_pose", cameras=None, image_size=None, privileged=None):
        """Lockstep (paused server): apply ``action`` for one control period.

        Returns ``obs, reward, terminated, truncated, info`` like a Gymnasium env.
        """
        out = self._call("step", action=np.asarray(action, dtype=np.float64), mode=mode,
                         **self._obs_kwargs(cameras, image_size, privileged))
        return out["obs"], out["reward"], out["terminated"], out["truncated"], out["info"]

    def reset(self, seed: int | None = None, options: dict | None = None, task: str | None = None,
              cameras=None, image_size=None, privileged=None):
        """New episode (optionally of another task). Returns ``obs, info``."""
        out = self._call("reset", seed=seed, options=options, task=task,
                         **self._obs_kwargs(cameras, image_size, privileged))
        return out["obs"], out["info"]

    def pause(self) -> dict:
        """Stop the real-time clock: the robot then advances only through ``step``."""
        return self._call("pause")

    def run(self) -> dict:
        """Restart the real-time clock."""
        return self._call("run")

    def close(self) -> None:
        self._stack.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
