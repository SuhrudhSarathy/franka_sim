"""Task interface: a task adds objects to the scene, randomizes them on reset and scores episodes."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from franka_sim.backends.base import Backend


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

    def joint_qadr(self, joint: str) -> int:
        return int(self.model.jnt_qposadr[self.model.joint(joint).id])
