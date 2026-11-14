"""``SimRobot``: one simulated Franka that behaves like a real robot behind a driver.

The simulation advances on its own clock (``tick``, called by the server at the control rate), and
commands are *held* the way a robot driver holds them:

* absolute targets (``joint_pos``, ``ee_pose``) stay in force until the next command;
* ``joint_vel`` is applied until ``watchdog`` seconds pass without a new command, then the arm stops;
* deltas (``delta_ee_pose``, ``joint_delta``) are accumulated between ticks and applied once.

With no command at all, the arm holds its commanded configuration. ``step`` is the alternative,
*lockstep* interface for evaluation: apply one action, advance exactly one control period and return
the transition, like ``FrankaEnv.step``.

``SimRobot`` is not thread-safe and must be driven from one thread: MuJoCo's OpenGL rendering is
bound to the thread that created the context (on macOS, the main thread). ``RobotServer`` takes care
of that.
"""

from __future__ import annotations

import time

import numpy as np

from franka_sim import robot as panda
from franka_sim.benchmark import ROBOT_KEYS
from franka_sim.controller import ACTION_DIMS, canonical_mode
from franka_sim.env import FrankaEnv
from franka_sim.ik import axis_angle_to_quat, quat_mul, quat_to_axis_angle
from franka_sim.tasks import TASKS


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


class SimRobot:
    """A single simulated Franka with a robot-driver interface.

    Args:
        task: task name (``franka_sim.TASKS``).
        control_freq: control rate (Hz) of ``tick`` and ``step``.
        image_size: default ``(height, width)`` of camera observations.
        watchdog: seconds a ``joint_vel`` command stays in force without being renewed.
        seed: seed of the first reset.
        env_kwargs: extra ``FrankaEnv`` arguments (``backend``, ``task_kwargs``, ...).
    """

    def __init__(self, task: str = "hammer_nails", *, control_freq: float = 20.0,
                 image_size: tuple[int, int] = (224, 224), watchdog: float = 0.25, seed: int | None = 0,
                 env_kwargs: dict | None = None):
        self.control_freq = control_freq
        self.image_size = tuple(image_size)
        self.watchdog = watchdog
        self.env_kwargs = dict(env_kwargs or {})
        self.env: FrankaEnv | None = None
        self._build(task)
        self.reset(seed=seed)

    # ------------------------------------------------------------------------------------------
    def _build(self, task: str) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; available: {sorted(TASKS)}")
        if self.env is not None:
            self.env.close()
        kwargs = {"control_freq": self.control_freq, "image_size": self.image_size, **self.env_kwargs}
        self.env = FrankaEnv(task, num_envs=1, action_mode="joint_pos", **kwargs)
        self.task_name = task

    @property
    def control_dt(self) -> float:
        return self.env.control_dt

    @property
    def cameras(self) -> list[str]:
        m = self.env.model
        return [m.cam(i).name for i in range(m.ncam)]

    def reset(self, seed: int | None = None, options: dict | None = None, task: str | None = None) -> dict:
        """Start a new episode (optionally of another task). Returns the ``info`` dict."""
        if task is not None and task != self.task_name:
            self._build(task)
        self.obs, info = self.env.reset(seed=seed, options=options or None)
        self.info = {k: np.asarray(v)[0] for k, v in info.items()}
        self.reward, self.terminated, self.truncated = 0.0, False, False
        self.episode_seed = seed
        self._grip = 0.0
        self._clear_command()
        self.last_command = {"mode": None, "source": None, "time": None}
        self.ticks = 0
        return self.info

    def _clear_command(self) -> None:
        self._held_mode: str | None = None  # absolute or joint_vel command in force
        self._held: np.ndarray | None = None
        self._held_time = 0.0
        self._delta_pos = np.zeros(3)  # accumulated delta_ee_pose
        self._delta_rot = np.array([1.0, 0.0, 0.0, 0.0])
        self._delta_joint = np.zeros(7)  # accumulated joint_delta
        self._has_delta = None  # which delta mode is pending

    # Commands -----------------------------------------------------------------------------------
    def command(self, action, mode: str, source: str = "api") -> None:
        """Set the command the robot follows from the next tick (see the module docstring)."""
        mode = canonical_mode(mode)
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape != (ACTION_DIMS[mode],):
            raise ValueError(f"{mode} actions have {ACTION_DIMS[mode]} values, got {a.shape[0]}")
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains NaN or inf")
        self._grip = float(np.clip(a[-1], 0.0, 1.0))
        arm = a[:-1]
        if mode in ("delta_ee_pose", "joint_delta"):
            if self._has_delta not in (None, mode):
                self._delta_pos[:] = 0.0
                self._delta_rot[:] = (1, 0, 0, 0)
                self._delta_joint[:] = 0.0
            self._has_delta = mode
            if mode == "delta_ee_pose":
                self._delta_pos += arm[:3]
                self._delta_rot = quat_mul(axis_angle_to_quat(arm[3:6]), self._delta_rot)
            else:
                self._delta_joint += arm
            self._held_mode = self._held = None  # a delta stream replaces any held target
        else:
            self._held_mode, self._held, self._held_time = mode, arm.copy(), time.monotonic()
            self._has_delta = None
        self.last_command = {"mode": mode, "source": source, "time": time.time()}

    def stop(self) -> None:
        """Drop every pending command: the arm holds its commanded configuration."""
        self._clear_command()

    def _next_action(self) -> tuple[np.ndarray, str]:
        if self._has_delta == "delta_ee_pose":
            arm = np.concatenate([self._delta_pos, quat_to_axis_angle(self._delta_rot)])
            mode = "delta_ee_pose"
        elif self._has_delta == "joint_delta":
            arm, mode = self._delta_joint.copy(), "joint_delta"
        elif self._held_mode == "joint_vel" and time.monotonic() - self._held_time > self.watchdog:
            self._held_mode = self._held = None
            arm, mode = np.zeros(7), "joint_vel"
        elif self._held_mode is not None:
            arm, mode = self._held, self._held_mode
        else:
            arm, mode = np.zeros(7), "joint_vel"  # hold the commanded configuration
        if self._has_delta is not None:
            self._has_delta = None
            self._delta_pos[:] = 0.0
            self._delta_rot[:] = (1, 0, 0, 0)
            self._delta_joint[:] = 0.0
        return np.append(arm, self._grip), mode

    # Advancing time ------------------------------------------------------------------------------
    def tick(self) -> None:
        """Advance one control period with the command in force (real-time mode)."""
        action, mode = self._next_action()
        self._advance(action, mode)

    def step(self, action, mode: str = "ee_pose", source: str = "api") -> dict:
        """Lockstep: apply ``action`` for exactly one control period and return the transition."""
        self.command(action, mode, source)
        self.tick()
        return self.transition()

    def _advance(self, action: np.ndarray, mode: str) -> None:
        obs, reward, term, trunc, info = self.env.step(action[None], action_mode=mode)
        self.obs = obs
        self.reward = float(reward[0])
        self.terminated, self.truncated = bool(term[0]), bool(trunc[0])
        self.info = {k: np.asarray(v)[0] for k, v in info.items()}
        self.ticks += 1

    # Reading the robot --------------------------------------------------------------------------
    def observation(self, cameras=(), image_size=None, privileged: bool = False) -> dict:
        """The current observation: robot state, ``instruction``, ``rgb_<camera>`` images, and with
        ``privileged`` the task's simulator state (object poses, targets: oracle use only)."""
        keys = self.obs.keys() if privileged else ROBOT_KEYS
        out = {k: np.asarray(self.obs[k][0]) for k in keys}
        h, w = image_size or self.image_size
        for cam in cameras:
            if cam not in self.cameras:
                raise ValueError(f"unknown camera {cam!r}; available: {self.cameras}")
            out[f"rgb_{cam}"] = self.env.render(cam, height=h, width=w)
        out["instruction"] = self.env.instructions()[0]
        return out

    def transition(self, cameras=(), image_size=None, privileged: bool = False) -> dict:
        return {
            "obs": self.observation(cameras, image_size, privileged),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": {k: _jsonable(v) for k, v in self.info.items()},
        }

    def state(self) -> dict:
        """JSON-able summary for dashboards."""
        env, o = self.env, self.obs
        lo, hi = env.ctrl_lo[env.arm_act], env.ctrl_hi[env.arm_act]
        cmd_pos, cmd_quat = env.controller.commanded_pose(0)
        return {
            "task": self.task_name,
            "instruction": env.instructions()[0],
            "seed": self.episode_seed,
            "conditioning": {k[5:]: _jsonable(v) for k, v in self.info.items() if k.startswith("cond_")},
            "joint_pos": o["joint_pos"][0].tolist(),
            "joint_vel": o["joint_vel"][0].tolist(),
            "joint_lo": lo.tolist(),
            "joint_hi": hi.tolist(),
            "joint_cmd": env.controller.q_cmd[0].tolist(),
            "ee_pos": o["ee_pos"][0].tolist(),
            "ee_quat": o["ee_quat"][0].tolist(),
            "ee_cmd_pos": cmd_pos.tolist(),
            "ee_cmd_quat": cmd_quat.tolist(),
            "home_ee_quat": env.home_ee_quat.tolist(),
            "gripper_width": float(o["gripper_width"][0, 0]),
            "gripper_cmd": self._grip,
            "gripper_max_width": panda.GRIPPER_MAX_WIDTH,
            "success": bool(self.info.get("success", False)),
            "progress": int(self.info.get("progress", 0)),
            "num_steps": int(env.task.num_steps),
            "wrong_target": bool(self.info.get("wrong_target", False)),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "elapsed_steps": int(env.elapsed_steps[0]),
            "max_episode_steps": int(env.max_episode_steps),
            "sim_time": float(env.sim.datas[0].time) if hasattr(env.sim, "datas") else None,
            "control_freq": self.control_freq,
            "last_command": self.last_command,
        }

    def describe(self) -> dict:
        """Static facts a client needs: tasks, cameras, action modes."""
        from franka_sim.benchmark import SUITE

        return {
            "task": self.task_name,
            "tasks": sorted(TASKS),
            "suite": {k: list(v) for k, v in SUITE.items()},
            "cameras": self.cameras,
            "action_modes": dict(ACTION_DIMS),
            "control_freq": self.control_freq,
            "image_size": list(self.image_size),
            "robot_keys": list(ROBOT_KEYS),
            "joint_names": list(panda.ARM_JOINTS),
            "home_qpos": panda.HOME_QPOS.tolist(),
        }

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
