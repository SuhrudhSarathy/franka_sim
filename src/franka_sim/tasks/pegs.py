"""Multi-peg insertion: several colored square pegs and a receptacle with a matching hole for each."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim.backends.base import Backend
from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import Evaluation, Task, sample_separated, yaw_quat


class PegInsertionTask(Task):
    """Insert colored pegs into the hole of the same color.

    ``mode="single"``: each episode picks one target peg (``options={"target": i}`` to choose); the
    episode succeeds when that peg is inside its own hole. ``mode="all"``: succeed when every peg
    is inserted (any order). Pegs start upright at random positions and yaw angles.

    Observations: ``peg_pos`` / ``peg_quat`` (flattened per peg), ``hole_pos`` (bottom centre of each
    hole), ``peg_inserted``, ``target_onehot``, ``target_peg_pos``, ``target_peg_quat``, ``target_hole_pos``.
    """

    name = "peg_insertion"
    max_episode_steps = 300

    def __init__(self, num_pegs: int = 3, mode: str = "single", max_yaw: float = 0.6, randomize: bool = True):
        if mode not in ("single", "all"):
            raise ValueError("mode must be 'single' or 'all'")
        self.num_pegs = num_pegs
        self.mode = mode
        self.max_yaw = max_yaw
        self.randomize = randomize
        if mode == "all":
            self.max_episode_steps = 250 * num_pegs

    def build(self, spec: mujoco.MjSpec) -> None:
        colors = [obj.COLORS[obj.COLOR_NAMES[i]] for i in range(self.num_pegs)]
        for i, c in enumerate(colors):
            obj.add_peg(spec, f"peg{i}", c, pos=[0.4, -0.2 + 0.1 * i, obj.PEG_HALF[2]])
        obj.add_receptacle(spec, colors)

    def bind(self, model, num_envs):
        super().bind(model, num_envs)
        self.peg_bodies = np.array([model.body(f"peg{i}").id for i in range(self.num_pegs)])
        self.peg_qadr = np.array([self.free_joint_qadr(f"peg{i}") for i in range(self.num_pegs)])
        self.peg_bottom = np.array([model.site(f"peg{i}_bottom").id for i in range(self.num_pegs)])
        self.hole_sites = np.array([model.site(f"hole{i}").id for i in range(self.num_pegs)])
        self.recept_mocap = model.body("receptacle").mocapid[0]
        self.target = np.zeros(num_envs, dtype=np.int64)

    def reset(self, env_ids, qpos, rng, options=None):
        options = options or {}
        n = len(env_ids)
        mocap = self.default_mocap_pos(n)
        for k in range(n):
            if self.randomize:
                rec_xy = rng.uniform([0.6, -0.06], [0.66, 0.06])
                pegs_xy = sample_separated(rng, self.num_pegs, [0.33, -0.3], [0.47, 0.3], min_dist=0.09)
                yaw = rng.uniform(-self.max_yaw, self.max_yaw, size=self.num_pegs)
            else:
                rec_xy = np.array([0.63, 0.0])
                pegs_xy = np.stack([np.full(self.num_pegs, 0.42),
                                    np.linspace(-0.2, 0.2, self.num_pegs)], axis=1)
                yaw = np.zeros(self.num_pegs)
            mocap[k, self.recept_mocap, :2] = rec_xy
            for i, a in enumerate(self.peg_qadr):
                qpos[k, a:a + 2] = pegs_xy[i]
                qpos[k, a + 2] = obj.PEG_HALF[2]
                qpos[k, a + 3:a + 7] = yaw_quat(yaw[i])
        target = rng.integers(0, self.num_pegs, size=n)
        if "target" in options:
            target[:] = options["target"]
        self.target[env_ids] = target
        return {"mocap_pos": mocap}

    def _state(self, sim: Backend):
        site = sim.read("site_xpos")
        bottoms, holes = site[:, self.peg_bottom], site[:, self.hole_sites]
        xy_err = np.linalg.norm(bottoms[..., :2] - holes[..., :2], axis=-1)
        depth = bottoms[..., 2] - holes[..., 2]
        inserted = (xy_err < 0.01) & (depth < 0.02)
        return bottoms, holes, inserted

    def observe(self, sim: Backend):
        xpos, xquat = sim.read("xpos")[:, self.peg_bodies], sim.read("xquat")[:, self.peg_bodies]
        _, holes, inserted = self._state(sim)
        idx = np.arange(sim.num_envs)
        return {
            "peg_pos": xpos.reshape(sim.num_envs, -1),
            "peg_quat": xquat.reshape(sim.num_envs, -1),
            "hole_pos": holes.reshape(sim.num_envs, -1),
            "peg_inserted": inserted.astype(np.float64),
            "target_onehot": np.eye(self.num_pegs)[self.target],
            "target_peg_pos": xpos[idx, self.target],
            "target_peg_quat": xquat[idx, self.target],
            "target_hole_pos": holes[idx, self.target],
        }

    def evaluate(self, sim: Backend) -> Evaluation:
        bottoms, holes, inserted = self._state(sim)
        dist = np.linalg.norm(bottoms - holes, axis=-1)
        idx = np.arange(sim.num_envs)
        if self.mode == "single":
            success = inserted[idx, self.target]
            reward = -dist[idx, self.target]
        else:
            success = inserted.all(axis=1)
            reward = -dist.sum(axis=1)
        return Evaluation(reward, success, {"peg_inserted": inserted})

    def conditioning(self):
        return {"target": self.target.copy()} if self.mode == "single" else {}

    def instruction(self, env_id):
        if self.mode == "all":
            return "insert every peg into the hole of the same color"
        color = obj.COLOR_NAMES[self.target[env_id] % len(obj.COLOR_NAMES)]
        return f"insert the {color} peg into the {color} hole"
