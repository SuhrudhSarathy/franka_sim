"""Task interface: a task adds objects to the scene, randomizes them on reset and scores episodes."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from franka_sim.backends.base import Backend


def yaw_quat(yaw) -> np.ndarray:
    """Quaternion(s) (wxyz) for rotation(s) about world z."""
    yaw = np.asarray(yaw, dtype=float)
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=-1)


def sample_separated(rng: np.random.Generator, n: int, low, high, min_dist: float, tries: int = 1000) -> np.ndarray:
    """Sample ``n`` 2D points uniformly in a box, at least ``min_dist`` apart (rejection sampling)."""
    low, high = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    pts: list[np.ndarray] = []
    for _ in range(tries):
        p = rng.uniform(low, high)
        if all(np.linalg.norm(p - q) >= min_dist for q in pts):
            pts.append(p)
            if len(pts) == n:
                return np.array(pts)
    raise RuntimeError(f"could not place {n} points {min_dist} apart in {low}..{high}")


def rotate2d(xy: np.ndarray, yaw) -> np.ndarray:
    """Rotate (..., 2) points by yaw (radians) about the origin."""
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.asarray(xy, dtype=float)
    return np.stack([c * xy[..., 0] - s * xy[..., 1], s * xy[..., 0] + c * xy[..., 1]], axis=-1)


@dataclass
class Evaluation:
    reward: np.ndarray  # (N,) float
    success: np.ndarray  # (N,) bool
    info: dict[str, np.ndarray] = field(default_factory=dict)


class Task:
    """Base class for tasks.

    Lifecycle: ``build(spec)`` once while the scene is assembled, ``bind(model, num_envs)`` once
    after compilation, then ``reset`` / ``observe`` / ``evaluate`` during rollouts. Per-env task
    variables (e.g. which nail to hit) live in numpy arrays sized ``num_envs`` on the task.
    """

    name: str = "task"
    max_episode_steps: int = 200

    def build(self, spec: mujoco.MjSpec) -> None:
        """Add bodies, geoms, sites, mocap bodies, ... to the scene."""

    def bind(self, model: mujoco.MjModel, num_envs: int) -> None:
        """Resolve names to ids and allocate per-env state."""
        self.model = model
        self.num_envs = num_envs

    def reset(self, env_ids: np.ndarray, qpos: np.ndarray, rng: np.random.Generator,
              options: dict | None = None) -> dict[str, np.ndarray]:
        """Randomize the task for ``env_ids``.

        ``qpos`` is a ``(len(env_ids), nq)`` array (robot already at its start pose) to edit in place,
        e.g. to place free-floating objects. Return extra ``MjData`` fields to write for these envs,
        e.g. ``{"mocap_pos": ...}``.
        """
        return {}

    def observe(self, sim: Backend) -> dict[str, np.ndarray]:
        """Task-specific observations, each ``(N, ...)``."""
        return {}

    def evaluate(self, sim: Backend) -> Evaluation:
        n = sim.num_envs
        return Evaluation(np.zeros(n), np.zeros(n, dtype=bool))

    def instruction(self, env_id: int) -> str:
        """Natural-language description of what env ``env_id`` should do (for conditioning)."""
        return self.name.replace("_", " ")

    def conditioning(self) -> dict[str, np.ndarray]:
        """Per-env conditioning variables (e.g. target index), each ``(N, ...)``."""
        return {}

    def mask_bodies(self, cond: dict) -> list[str] | None:
        """Bodies forming the target entity of one episode with conditioning ``cond`` (one env's
        values of ``conditioning()``), for target masks. ``None``: the task has no single target."""
        return None

    # Helpers -----------------------------------------------------------------------------------
    def free_joint_qadr(self, body: str) -> int:
        """qpos address of the free joint of ``body``."""
        jnt = self.model.body(body).jntadr[0]
        return int(self.model.jnt_qposadr[jnt])

    def default_mocap_pos(self, n: int) -> np.ndarray:
        """``(n, nmocap, 3)`` array of the mocap bodies' model positions, to edit in ``reset``."""
        pos = np.zeros((self.model.nmocap, 3))
        for b in range(self.model.nbody):
            if self.model.body_mocapid[b] >= 0:
                pos[self.model.body_mocapid[b]] = self.model.body_pos[b]
        return np.tile(pos, (n, 1, 1))

    def default_mocap_quat(self, n: int) -> np.ndarray:
        """``(n, nmocap, 4)`` array of the mocap bodies' model orientations, to edit in ``reset``."""
        quat = np.tile([1.0, 0.0, 0.0, 0.0], (self.model.nmocap, 1))
        for b in range(self.model.nbody):
            if self.model.body_mocapid[b] >= 0:
                quat[self.model.body_mocapid[b]] = self.model.body_quat[b]
        return np.tile(quat, (n, 1, 1))

    def joint_qadr(self, joint: str) -> int:
        return int(self.model.jnt_qposadr[self.model.joint(joint).id])
