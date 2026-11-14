"""Scripted (waypoint state-machine) policies that emit ``ee_pose`` actions.

A policy keeps per-env state (stage index, stage timer, smoothed command) so it works on a batch of
envs whose episodes start and end at different times: call ``reset(env_ids, obs)`` whenever envs
are reset and ``act(obs)`` every step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from franka_sim.ik import quat_mul
from franka_sim.tasks.base import yaw_quat


def quat_yaw(q: np.ndarray) -> np.ndarray:
    """Yaw (rotation about world z) of wxyz quaternion(s)."""
    w, x, y, z = np.moveaxis(np.asarray(q), -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(a, period: float = 2 * np.pi):
    """Wrap to ``[-period/2, period/2)``."""
    return (np.asarray(a) + period / 2) % period - period / 2


@dataclass
class Target:
    pos: np.ndarray
    yaw: float
    grip: float
    speed: float | None = None  # m per step; None = policy default
    noise: float = 1.0  # scale of the policy's action noise (0 for precise phases such as insertion)


class ScriptedPolicy:
    """Base class. Subclasses implement ``plan(i, obs, stage)``.

    ``plan`` returns ``(Target, done)`` for env ``i`` in its current ``stage``: the target pose to move
    toward, and whether the stage is complete (the policy then advances to ``stage + 1``). The
    commanded pose moves toward the target at most ``speed`` per step, which keeps demonstrations
    smooth.

    Args:
        env: the ``FrankaEnv`` (``action_mode="ee_pose"``, unless ``any_env_mode``).
        speed: default translation per step (m).
        noise: std-dev of Gaussian noise added to the commanded position (m), to diversify demos.
        seed: RNG seed for noise and per-episode variations.
        any_env_mode: allow an env in another action mode. The actions are still ``ee_pose``: step
            them with ``env.step(a, action_mode="ee_pose")`` or convert them with
            ``env.convert_action`` (this is how DAgger uses the expert as a labeler).
    """

    action_mode = "ee_pose"

    def __init__(self, env, speed: float = 0.02, yaw_speed: float = 0.15, noise: float = 0.0,
                 seed: int | None = None, any_env_mode: bool = False):
        if env.action_mode != "ee_pose" and not any_env_mode:
            raise ValueError("scripted policies emit ee_pose actions; create the env with action_mode='ee_pose'")
        self.env = env
        self.n = env.num_envs
        self.speed = speed
        self.yaw_speed = yaw_speed
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.base_quat = env.home_ee_quat  # gripper pointing down, fingers along world y
        self.stage = np.zeros(self.n, dtype=np.int64)
        self.timer = np.zeros(self.n, dtype=np.int64)
        self.cmd_pos = np.zeros((self.n, 3))
        self.cmd_yaw = np.zeros(self.n)
        self.memory: list[dict] = [{} for _ in range(self.n)]

    # ------------------------------------------------------------------------------------------
    def reset(self, env_ids, obs) -> None:
        for i in np.atleast_1d(env_ids):
            self.stage[i] = 0
            self.timer[i] = 0
            self.cmd_pos[i] = obs["ee_pos"][i]
            self.cmd_yaw[i] = wrap_angle(quat_yaw(quat_mul(obs["ee_quat"][i], _conj(self.base_quat))))
            self.memory[i] = {}

    def track(self, ee_pose: np.ndarray, env_ids=None) -> None:
        """Re-anchor the smoothed command to the robot's actual commanded pose ``(N, 7)``.

        The expert moves its command toward each target at a bounded speed, starting from its own
        previous command. When something else drove the robot (a learner, a human), call this
        before ``act`` so the expert's next action is a correction from where the robot really
        is, not from where the expert had left it."""
        ids = range(self.n) if env_ids is None else np.atleast_1d(env_ids)
        for i in ids:
            self.cmd_pos[i] = ee_pose[i, :3]
            self.cmd_yaw[i] = wrap_angle(quat_yaw(quat_mul(ee_pose[i, 3:7], _conj(self.base_quat))))

    def plan(self, i: int, obs: dict, stage: int) -> tuple[Target, bool]:
        raise NotImplementedError

    def finished(self, i: int) -> bool:
        """True once env ``i`` has run out of stages."""
        return bool(self.memory[i].get("finished", False))

    def act(self, obs: dict) -> np.ndarray:
        actions = np.zeros((self.n, 8))
        for i in range(self.n):
            for _ in range(4):  # allow a few instantaneous stage transitions per step
                target, done = self.plan(i, obs, int(self.stage[i]))
                if not done:
                    break
                self.stage[i] += 1
                self.timer[i] = 0
            self.timer[i] += 1
            speed = target.speed if target.speed is not None else self.speed
            delta = target.pos - self.cmd_pos[i]
            dist = np.linalg.norm(delta)
            if dist > speed:
                delta *= speed / dist
            self.cmd_pos[i] += delta
            dyaw = np.clip(wrap_angle(target.yaw - self.cmd_yaw[i]), -self.yaw_speed, self.yaw_speed)
            self.cmd_yaw[i] += dyaw
            sigma = self.noise * target.noise
            pos = self.cmd_pos[i] + (self.rng.normal(0, sigma, 3) if sigma > 0 else 0.0)
            quat = quat_mul(yaw_quat(self.cmd_yaw[i]), self.base_quat)
            actions[i] = np.concatenate([pos, quat, [target.grip]])
        return actions

    # Helpers -----------------------------------------------------------------------------------
    def reached(self, i: int, target: Target, tol: float = 0.01, yaw_tol: float = 0.05) -> bool:
        """The smoothed command has arrived at the target (position and yaw)."""
        return (np.linalg.norm(self.cmd_pos[i] - target.pos) < tol
                and abs(wrap_angle(self.cmd_yaw[i] - target.yaw)) < yaw_tol)

    def settled(self, i: int, obs: dict, target: Target, tol: float = 0.01) -> bool:
        """The measured end effector is within ``tol`` of the target (and the command arrived)."""
        return self.reached(i, target) and np.linalg.norm(obs["ee_pos"][i] - target.pos) < tol


def _conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])
