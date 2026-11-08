"""Shared machinery for pick-and-insert assembly tasks (pegs, gears, kitting, PCB cards).

An insertion task has ``num_objects`` free objects (bodies ``<prefix>{i}``) and one fixture (a mocap
body) carrying a goal site ``goal{i}`` per object: where object ``i``'s origin is when it is seated.
Each episode requests an ordered list of objects to insert (``targets``):

* ``mode="single"``: one random object (``options={"target": i}`` to choose),
* ``mode="all"``: every object, in random order,
* ``mode="subset"``: ``num_targets`` random objects (``options={"targets": [2, 0]}`` to choose).

Observations (the same keys for every insertion task, so one policy and one model fit all):

``obj_pos`` / ``obj_quat`` (flattened per object), ``goal_pos`` (flattened), ``fixture_quat``,
``obj_inserted`` (0/1 per object), ``targets_mask`` (1 for requested objects), ``target_onehot`` (the
next requested object that is not inserted yet), ``target_obj_pos``, ``target_obj_quat``,
``target_goal_pos``.

The scripted ``InsertionPolicy`` reads per-object hints from the task: ``grasp_offset`` (z of the
grasp point above the object origin), ``symmetry`` (yaw period of the object for alignment, ``None`` if
it is round), ``grasp_yaw`` (gripper yaw relative to the object) and ``tip_offset`` (z, in the object
frame, of the point that enters first, e.g. the bottom edge of a card) and ``grasp_width`` (object
width between the fingers, so the gripper opens only as far as needed in clutter).
"""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim.backends.base import Backend
from franka_sim.tasks.base import Evaluation, Task, rotate2d, yaw_quat


class InsertionTask(Task):
    obj_prefix = "obj"
    fixture = "fixture"
    object_names: tuple[str, ...] = ()
    # Tolerances on the object origin vs. its goal: horizontal distance, and height above the goal.
    xy_tol = 0.006
    z_tol = 0.012
    steps_per_object = 150
    # Objects start at least this far (m) from every goal on the fixture, so the open gripper never
    # disturbs a waiting object while it works at the fixture.
    fixture_clearance = 0.12

    def __init__(self, mode: str = "single", num_targets: int = 2, randomize: bool = True):
        if mode not in ("single", "all", "subset"):
            raise ValueError("mode must be 'single', 'all' or 'subset'")
        self.mode = mode
        self.randomize = randomize
        self._num_targets = num_targets

    # -- subclass hooks ------------------------------------------------------------------------
    @property
    def num_objects(self) -> int:
        return len(self.object_names)

    @property
    def num_targets(self) -> int:
        return {"single": 1, "all": self.num_objects, "subset": self._num_targets}[self.mode]

    @property
    def max_episode_steps(self) -> int:  # type: ignore[override]
        return 60 + self.steps_per_object * self.num_targets

    def sample_layout(self, rng: np.random.Generator):
        """Return ``(fixture_xy (2,), fixture_yaw, obj_xy (n, 2), obj_yaw (n,))`` for one env."""
        raise NotImplementedError

    def obj_rest_z(self, i: int) -> float:
        """Height of object ``i``'s origin when it rests on the table at reset."""
        raise NotImplementedError

    def extra_mocap(self, k: int, rng, mocap: np.ndarray, quat: np.ndarray, layout) -> None:
        """Place additional mocap bodies (e.g. a stand) for env-row ``k``."""

    grasp_offset: tuple[float, ...] = ()
    symmetry: tuple[float | None, ...] = ()
    grasp_yaw: tuple[float, ...] = ()
    tip_offset: tuple[float, ...] = ()
    grasp_width: tuple[float, ...] = ()

    # -- Task API ------------------------------------------------------------------------------
    def bind(self, model, num_envs):
        super().bind(model, num_envs)
        n = self.num_objects
        self.obj_bodies = np.array([model.body(f"{self.obj_prefix}{i}").id for i in range(n)])
        self.obj_qadr = np.array([self.free_joint_qadr(f"{self.obj_prefix}{i}") for i in range(n)])
        self.goal_sites = np.array([model.site(f"goal{i}").id for i in range(n)])
        self.fixture_mocap = model.body(self.fixture).mocapid[0]
        fixture_body = model.body(self.fixture).id
        all_goals = [model.site(j) for j in range(model.nsite) if model.site(j).name.startswith("goal")]
        self.goal_local = np.array([st.pos[:2] for st in all_goals if model.site_bodyid[st.id] == fixture_body])
        self.targets = np.zeros((num_envs, self.num_targets), dtype=np.int64)
        self.inserted = np.zeros((num_envs, n), dtype=bool)

    def reset(self, env_ids, qpos, rng, options=None):
        options = options or {}
        n = len(env_ids)
        mocap = self.default_mocap_pos(n)
        quat = self.default_mocap_quat(n)
        for k, e in enumerate(env_ids):
            for _attempt in range(100):
                layout = self.sample_layout(rng)
                fixture_xy, fixture_yaw, obj_xy, obj_yaw = layout
                goals = fixture_xy + rotate2d(self.goal_local, fixture_yaw)
                d = np.linalg.norm(obj_xy[:, None] - goals[None], axis=-1)
                if not self.randomize or d.min() >= self.fixture_clearance:
                    break
            mocap[k, self.fixture_mocap, :2] = fixture_xy
            quat[k, self.fixture_mocap] = yaw_quat(fixture_yaw)
            for i, a in enumerate(self.obj_qadr):
                qpos[k, a:a + 2] = obj_xy[i]
                qpos[k, a + 2] = self.obj_rest_z(i)
                qpos[k, a + 3:a + 7] = yaw_quat(obj_yaw[i])
            self.extra_mocap(k, rng, mocap, quat, layout)
            if "targets" in options:
                self.targets[e] = options["targets"]
            elif "target" in options:
                t = int(options["target"])
                rest = [j for j in rng.permutation(self.num_objects) if j != t]
                self.targets[e] = [t, *rest[: self.num_targets - 1]]
            else:
                self.targets[e] = rng.permutation(self.num_objects)[: self.num_targets]
        self.inserted[env_ids] = False
        return {"mocap_pos": mocap, "mocap_quat": quat}

    def _state(self, sim: Backend):
        pos = sim.read("xpos")[:, self.obj_bodies]
        goals = sim.read("site_xpos")[:, self.goal_sites]
        xy_err = np.linalg.norm(pos[..., :2] - goals[..., :2], axis=-1)
        dz = pos[..., 2] - goals[..., 2]
        inserted = (xy_err < self.xy_tol) & (dz < self.z_tol)
        return pos, goals, inserted

    def _next_target(self, inserted: np.ndarray) -> np.ndarray:
        """Index of the first requested object that is not inserted yet (last one if all are)."""
        idx = np.arange(self.num_envs)
        done = inserted[idx[:, None], self.targets]
        first = np.where(done.all(axis=1), self.num_targets - 1, np.argmax(~done, axis=1))
        return self.targets[idx, first]

    def observe(self, sim: Backend):
        pos, goals, inserted = self._state(sim)
        quat = sim.read("xquat")[:, self.obj_bodies]
        n = sim.num_envs
        idx = np.arange(n)
        nxt = self._next_target(inserted)
        mask = np.zeros((n, self.num_objects))
        mask[idx[:, None], self.targets] = 1.0
        return {
            "obj_pos": pos.reshape(n, -1),
            "obj_quat": quat.reshape(n, -1),
            "goal_pos": goals.reshape(n, -1),
            "fixture_quat": sim.read("mocap_quat")[:, self.fixture_mocap],
            "obj_inserted": inserted.astype(np.float64),
            "targets_mask": mask,
            "target_onehot": np.eye(self.num_objects)[nxt],
            "target_obj_pos": pos[idx, nxt],
            "target_obj_quat": quat[idx, nxt],
            "target_goal_pos": goals[idx, nxt],
        }

    def evaluate(self, sim: Backend) -> Evaluation:
        pos, goals, inserted = self._state(sim)
        idx = np.arange(sim.num_envs)[:, None]
        req = inserted[idx, self.targets]
        dist = np.linalg.norm(pos - goals, axis=-1)[idx, self.targets]
        success = req.all(axis=1)
        reward = req.mean(axis=1) - dist.sum(axis=1)
        return Evaluation(reward, success, {"obj_inserted": inserted, "num_inserted": req.sum(axis=1)})

    def conditioning(self):
        cond = {"targets": self.targets.copy()}
        if self.mode == "single":
            cond["target"] = self.targets[:, 0].copy()
        return cond

    def mask_bodies(self, cond):
        if self.mode != "single":
            return None  # multi-step: no mask
        return [f"{self.obj_prefix}{int(np.ravel(cond['target'])[0])}"]

    def describe(self, i: int) -> str:
        return self.object_names[i]

    def instruction(self, env_id):
        names = [self.describe(int(i)) for i in self.targets[env_id]]
        return self.instruction_template(names)

    def instruction_template(self, names: list[str]) -> str:
        if len(names) == 1:
            return f"insert the {names[0]}"
        return "insert the " + ", then the ".join(names)


def build_objects(spec: mujoco.MjSpec, prefix: str, builders) -> None:
    """Call ``builder(spec, f"{prefix}{i}")`` for each builder (helper for subclasses)."""
    for i, b in enumerate(builders):
        b(spec, f"{prefix}{i}")
