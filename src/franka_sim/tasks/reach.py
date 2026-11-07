"""Reach a randomly placed target with the gripper."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim import robot
from franka_sim.backends.base import Backend
from franka_sim.tasks.base import Evaluation, Task


class ReachTask(Task):
    name = "reach"
    max_episode_steps = 100

    def __init__(self, low=(0.35, -0.25, 0.1), high=(0.65, 0.25, 0.45), threshold: float = 0.02):
        self.low, self.high = np.array(low), np.array(high)
        self.threshold = threshold

    def build(self, spec: mujoco.MjSpec) -> None:
        target = spec.worldbody.add_body(name="target", mocap=True, pos=[0.5, 0.0, 0.3])
        target.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0], rgba=[0.9, 0.2, 0.2, 0.6],
                        contype=0, conaffinity=0, group=1)

    def bind(self, model, num_envs):
        super().bind(model, num_envs)
        self.mocap_id = model.body("target").mocapid[0]
        self.site_id = model.site(robot.EE_SITE).id
        self.goal = np.tile((self.low + self.high) / 2, (num_envs, 1))

    def reset(self, env_ids, qpos, rng, options=None):
        goals = rng.uniform(self.low, self.high, size=(len(env_ids), 3))
        if options and "goal" in options:
            goals[:] = options["goal"]
        self.goal[env_ids] = goals
        mocap = self.default_mocap_pos(len(env_ids))
        mocap[:, self.mocap_id] = goals
        return {"mocap_pos": mocap}

    def observe(self, sim: Backend):
        return {"goal_pos": self.goal.copy()}

    def conditioning(self):
        return {"goal_pos": self.goal.copy()}

    def evaluate(self, sim: Backend) -> Evaluation:
        ee = sim.read("site_xpos")[:, self.site_id]
        dist = np.linalg.norm(ee - self.goal, axis=-1)
        return Evaluation(reward=-dist, success=dist < self.threshold, info={"distance": dist})

    def instruction(self, env_id):
        x, y, z = self.goal[env_id]
        return f"move the gripper to ({x:.2f}, {y:.2f}, {z:.2f})"
