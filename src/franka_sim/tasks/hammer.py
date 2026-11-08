"""Pick up a hammer and drive one selected nail (out of several) into a board, or all of them."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim.backends.base import Backend
from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import Evaluation, Task, rotate2d, sample_separated, yaw_quat


class HammerNailsTask(Task):
    """Multi-object hammering with a conditioning target.

    The scene has a hammer resting on a stand and a board with ``num_nails`` identical nails. Every
    reset samples the board pose and the nail positions on it (at least ``min_nail_dist`` apart).
    Nails are indexed by their position: nail 0 is the one furthest to the robot's left (+y), nail
    ``num_nails - 1`` the furthest to its right, so ``target = k`` always means "the k-th nail from
    the left". Each episode picks a target (randomly, or ``options={"target": k}``). The episode
    succeeds when the target nail is driven in (depth >= ``success_depth``) and no other nail is
    driven past ``wrong_depth``.

    With ``long=True`` (registered as ``hammer_nails_long``) every nail must be driven, in any order.
    Each episode still samples an order (``order``, a permutation; ``options={"order": [...]}``),
    which the scripted expert follows and the recording keeps, but success ignores it. The
    observations ``target_onehot`` / ``target_nail_pos`` then point at the next nail of that order
    that is not driven yet.

    Observations: ``hammer_pos``, ``hammer_quat``, ``hammer_grip_pos``, ``hammer_face_pos``,
    ``nail_pos`` (num_nails*3, top of each head), ``nail_depth`` (num_nails, 0 = up, 1 = driven),
    ``target_onehot``, ``target_nail_pos``.
    """

    name = "hammer_nails"
    max_episode_steps = 300

    def __init__(self, num_nails: int = 4, success_depth: float = 0.75, wrong_depth: float = 0.25,
                 nail_frictionloss: float = 6.0, min_nail_dist: float = 0.075, randomize: bool = True,
                 long: bool = False):
        self.num_nails = num_nails
        self.long = long
        self.success_depth = success_depth
        self.wrong_depth = wrong_depth
        self.nail_frictionloss = nail_frictionloss
        self.min_nail_dist = min_nail_dist
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
        self.nail_mocap = np.array([model.body(f"nail{i}_base").mocapid[0] for i in range(self.num_nails)])
        self.grip_site = model.site("hammer_grip").id
        self.face_site = model.site("hammer_face").id
        self.nail_qadr = np.array([self.joint_qadr(f"nail{i}_joint") for i in range(self.num_nails)])
        self.nail_sites = np.array([model.site(f"nail{i}_top").id for i in range(self.num_nails)])
        self.target = np.zeros(num_envs, dtype=np.int64)
        self.order = np.tile(np.arange(self.num_nails), (num_envs, 1))

    def reset(self, env_ids, qpos, rng, options=None):
        options = options or {}
        n = len(env_ids)
        if self.randomize:
            # Board anywhere in front of the robot, any modest yaw; the hammer lies on a random side
            # of the board (far enough to never overlap it) with a random heading.
            board_xy = rng.uniform([0.5, -0.08], [0.62, 0.08], size=(n, 2))
            board_yaw = rng.uniform(-0.4, 0.4, size=n)
            side = rng.choice([-1.0, 1.0], size=n)
            hammer_xy = np.stack([rng.uniform(0.36, 0.52, size=n),
                                  board_xy[:, 1] + side * rng.uniform(0.33, 0.38, size=n)], axis=1)
            hammer_yaw = rng.uniform(-1.0, 1.0, size=n)
        else:
            board_xy = np.tile([0.58, 0.07], (n, 1))
            board_yaw = np.zeros(n)
            hammer_xy = np.tile([0.42, -0.29], (n, 1))
            hammer_yaw = np.zeros(n)
        target = rng.integers(0, self.num_nails, size=n)
        if "target" in options:
            target[:] = options["target"]
        self.target[env_ids] = target
        for e in np.atleast_1d(env_ids):
            self.order[e] = options["order"] if "order" in options else rng.permutation(self.num_nails)

        a = self.hammer_qadr
        qpos[:, a:a + 2] = hammer_xy
        qpos[:, a + 2] = obj.HAMMER_HANDLE_Z
        qpos[:, a + 3:a + 7] = yaw_quat(hammer_yaw)
        qpos[:, self.nail_qadr] = 0.0

        mocap = self.default_mocap_pos(n)
        quat = self.default_mocap_quat(n)
        mocap[:, self.board_mocap, :2] = board_xy
        quat[:, self.board_mocap] = yaw_quat(board_yaw)
        lo, hi = -np.array(obj.NAIL_REGION_HALF), np.array(obj.NAIL_REGION_HALF)
        for k in range(n):
            if self.randomize:
                local = sample_separated(rng, self.num_nails, lo, hi, self.min_nail_dist)
            else:
                local = np.stack([np.zeros(self.num_nails), np.linspace(hi[1], lo[1], self.num_nails)], 1)
            world = board_xy[k] + rotate2d(local, board_yaw[k])
            world = world[np.argsort(-world[:, 1])]  # index 0 = furthest to the robot's left (+y)
            mocap[k, self.nail_mocap, :2] = world
            quat[k, self.nail_mocap] = yaw_quat(board_yaw[k])
        # The stand goes under the back of the handle, following the hammer's yaw.
        back = -0.07
        mocap[:, self.rest_mocap, 0] = hammer_xy[:, 0] + back * np.cos(hammer_yaw)
        mocap[:, self.rest_mocap, 1] = hammer_xy[:, 1] + back * np.sin(hammer_yaw)
        quat[:, self.rest_mocap] = yaw_quat(hammer_yaw)
        return {"mocap_pos": mocap, "mocap_quat": quat}

    def nail_depth(self, sim: Backend) -> np.ndarray:
        return -sim.read("qpos")[:, self.nail_qadr] / obj.NAIL_TRAVEL

    def current_target(self, depth: np.ndarray) -> np.ndarray:
        """Nail to work on now: the target, or (long) the next nail of ``order`` not driven yet."""
        if not self.long:
            return self.target
        idx = np.arange(len(depth))
        done = depth[idx[:, None], self.order] >= self.success_depth
        first = np.where(done.all(axis=1), self.num_nails - 1, np.argmax(~done, axis=1))
        return self.order[idx, first]

    def observe(self, sim: Backend):
        xpos, xquat, site = sim.read("xpos"), sim.read("xquat"), sim.read("site_xpos")
        nails = site[:, self.nail_sites]
        idx = np.arange(sim.num_envs)
        target = self.current_target(self.nail_depth(sim))
        return {
            "hammer_pos": xpos[:, self.hammer_body],
            "hammer_quat": xquat[:, self.hammer_body],
            "hammer_grip_pos": site[:, self.grip_site],
            "hammer_face_pos": site[:, self.face_site],
            "nail_pos": nails.reshape(sim.num_envs, -1),
            "nail_depth": self.nail_depth(sim),
            "target_onehot": np.eye(self.num_nails)[target],
            "target_nail_pos": nails[idx, target],
        }

    def evaluate(self, sim: Backend) -> Evaluation:
        depth = np.clip(self.nail_depth(sim), 0.0, 1.0)
        idx = np.arange(sim.num_envs)
        if self.long:
            driven = depth >= self.success_depth
            return Evaluation(depth.mean(axis=1), driven.all(axis=1),
                              {"nail_depth": depth, "progress": driven.sum(axis=1),
                               "wrong_target": np.zeros(sim.num_envs, dtype=bool)})
        target_depth = depth[idx, self.target]
        others = depth.copy()
        others[idx, self.target] = 0.0
        wrong = others.max(axis=1) > self.wrong_depth
        success = (target_depth >= self.success_depth) & ~wrong
        reward = target_depth - others.sum(axis=1)
        return Evaluation(reward, success, {"nail_depth": depth, "wrong_target": wrong,
                                            "progress": (target_depth >= self.success_depth).astype(np.int64)})

    def conditioning(self):
        if self.long:
            return {"order": self.order.copy()}
        return {"target": self.target.copy()}

    @property
    def num_steps(self) -> int:
        """Sub-goals per episode (nails to drive)."""
        return self.num_nails if self.long else 1

    def mask_bodies(self, cond):
        if self.long:
            return None  # multi-step: no mask
        return [f"nail{int(np.ravel(cond['target'])[0])}"]

    def instruction(self, env_id):
        if self.long:
            return "pick up the hammer and hit all the nails"
        k = int(self.target[env_id])
        if k == 0:
            which = "leftmost nail"
        elif k == self.num_nails - 1:
            which = "rightmost nail"
        else:
            which = f"{_ordinal(k + 1)} nail from the left"
        return f"pick up the hammer and hit the {which}"


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth"}.get(n, f"{n}th")
