"""Pick up a hammer and drive one selected nail (out of several) into a board."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim.backends.base import Backend
from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import Evaluation, Task, yaw_quat


class HammerNailsTask(Task):
    """Multi-object hammering with a conditioning target.

    The scene has a hammer resting on a stand and a board with ``num_nails`` colored nails. Each
    episode picks a target nail (randomly, or ``options={"target": i}`` on reset). The episode
    succeeds when the target nail is driven in (depth >= ``success_depth``) and no other nail is
    driven past ``wrong_depth``.

    Observations: ``hammer_pos``, ``hammer_quat``, ``hammer_grip_pos``, ``hammer_face_pos``,
    ``nail_pos`` (num_nails*3, top of each head), ``nail_depth`` (num_nails, 0 = up, 1 = driven),
    ``target_onehot``, ``target_nail_pos``.
    """

    name = "hammer_nails"
    max_episode_steps = 300

    def __init__(self, num_nails: int = 4, success_depth: float = 0.75, wrong_depth: float = 0.25,
                 nail_frictionloss: float = 6.0, randomize: bool = True):
        self.num_nails = num_nails
        self.success_depth = success_depth
        self.wrong_depth = wrong_depth
        self.nail_frictionloss = nail_frictionloss
        self.randomize = randomize

    def build(self, spec: mujoco.MjSpec) -> None:
        obj.add_hammer(spec)
        obj.add_hammer_rest(spec)
        obj.add_nail_board(spec, self.num_nails, frictionloss=self.nail_frictionloss)

    def bind(self, model, num_envs):
        super().bind(model, num_envs)
        self.hammer_body = model.body("hammer").id
        self.hammer_qadr = self.free_joint_qadr("hammer")
        self.rest_mocap = model.body("hammer_rest").mocapid[0]
        self.board_mocap = model.body("board").mocapid[0]
        self.grip_site = model.site("hammer_grip").id
        self.face_site = model.site("hammer_face").id
        self.nail_qadr = np.array([self.joint_qadr(f"nail{i}_joint") for i in range(self.num_nails)])
        self.nail_sites = np.array([model.site(f"nail{i}_top").id for i in range(self.num_nails)])
        self.target = np.zeros(num_envs, dtype=np.int64)

    def reset(self, env_ids, qpos, rng, options=None):
        options = options or {}
        n = len(env_ids)
        if self.randomize:
            board_xy = rng.uniform([0.55, 0.02], [0.62, 0.12], size=(n, 2))
            hammer_xy = rng.uniform([0.38, -0.33], [0.46, -0.25], size=(n, 2))
            hammer_yaw = rng.uniform(-0.2, 0.2, size=n)
        else:
            board_xy = np.tile([0.58, 0.07], (n, 1))
            hammer_xy = np.tile([0.42, -0.29], (n, 1))
            hammer_yaw = np.zeros(n)
        target = rng.integers(0, self.num_nails, size=n)
        if "target" in options:
            target[:] = options["target"]
        self.target[env_ids] = target

        a = self.hammer_qadr
        qpos[:, a:a + 2] = hammer_xy
        qpos[:, a + 2] = obj.HAMMER_HANDLE_Z
        qpos[:, a + 3:a + 7] = yaw_quat(hammer_yaw)
        qpos[:, self.nail_qadr] = 0.0

        mocap = self.default_mocap_pos(n)
        mocap[:, self.board_mocap, :2] = board_xy
        # The stand goes under the back of the handle, following the hammer's yaw.
        back = -0.07
        mocap[:, self.rest_mocap, 0] = hammer_xy[:, 0] + back * np.cos(hammer_yaw)
        mocap[:, self.rest_mocap, 1] = hammer_xy[:, 1] + back * np.sin(hammer_yaw)
        rest_quat = np.tile([1.0, 0, 0, 0], (n, self.model.nmocap, 1))
        rest_quat[:, self.rest_mocap] = yaw_quat(hammer_yaw)
        return {"mocap_pos": mocap, "mocap_quat": rest_quat}

    def nail_depth(self, sim: Backend) -> np.ndarray:
        return -sim.read("qpos")[:, self.nail_qadr] / obj.NAIL_TRAVEL

    def observe(self, sim: Backend):
        xpos, xquat, site = sim.read("xpos"), sim.read("xquat"), sim.read("site_xpos")
        nails = site[:, self.nail_sites]
        idx = np.arange(sim.num_envs)
        return {
            "hammer_pos": xpos[:, self.hammer_body],
            "hammer_quat": xquat[:, self.hammer_body],
            "hammer_grip_pos": site[:, self.grip_site],
            "hammer_face_pos": site[:, self.face_site],
            "nail_pos": nails.reshape(sim.num_envs, -1),
            "nail_depth": self.nail_depth(sim),
            "target_onehot": np.eye(self.num_nails)[self.target],
            "target_nail_pos": nails[idx, self.target],
        }

    def evaluate(self, sim: Backend) -> Evaluation:
        depth = np.clip(self.nail_depth(sim), 0.0, 1.0)
        idx = np.arange(sim.num_envs)
        target_depth = depth[idx, self.target]
        others = depth.copy()
        others[idx, self.target] = 0.0
        wrong = others.max(axis=1) > self.wrong_depth
        success = (target_depth >= self.success_depth) & ~wrong
        reward = target_depth - others.sum(axis=1)
        return Evaluation(reward, success, {"nail_depth": depth, "wrong_nail": wrong})

    def conditioning(self):
        return {"target": self.target.copy()}

    def instruction(self, env_id):
        color = obj.COLOR_NAMES[self.target[env_id] % len(obj.COLOR_NAMES)]
        return f"pick up the hammer and hit the {color} nail"
