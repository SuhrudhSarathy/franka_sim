"""PCB insertion: take an expansion card out of its stand and seat it in a card-edge connector."""

from __future__ import annotations

import numpy as np

from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import yaw_quat
from franka_sim.tasks.insertion import InsertionTask

ORDINALS = ("leftmost", "middle", "rightmost")


class PCBInsertionTask(InsertionTask):
    """A card stands in a rack; the motherboard has ``num_slots`` connectors (1.4 mm clearance per
    side). Each episode picks the target connector (``options={"target": slot}``; slot 0 is the one
    furthest to the robot's left). Success: the card is fully seated in the target connector.

    Observations follow ``InsertionTask`` with the card as the single object; ``goal_pos`` is the seated
    card centre in the target connector, ``slot_pos`` lists it for every connector, ``slot_onehot``
    is the target connector.
    """

    name = "pcb_insertion"
    obj_prefix = "pcb"
    fixture = "motherboard"
    # Only a card that is inside the connector can get down to the seated height (z_tol); the
    # horizontal tolerance just allows for the connector's end clearance.
    xy_tol = 0.006
    z_tol = 0.004
    steps_per_object = 220
    grasp_offset = (0.012,)
    symmetry = (np.pi,)
    grasp_yaw = (0.0,)
    tip_offset = (-obj.PCB_HALF[2],)
    grasp_width = (2 * obj.PCB_HALF[1],)

    def __init__(self, num_slots: int = 3, randomize: bool = True):
        self.num_slots = num_slots
        self.object_names = ("card",)
        super().__init__("single", 1, randomize)

    # One card, several goals: the "objects" of InsertionTask are indexed by slot here.
    def build(self, spec):
        obj.add_pcb(spec, "pcb0")
        obj.add_card_rack(spec)
        obj.add_motherboard(spec, self.num_slots)

    def bind(self, model, num_envs):
        super().bind(model, num_envs)
        self.slot_goals = np.array([model.site(f"goal{i}").id for i in range(self.num_slots)])
        self.rack_mocap = model.body("card_rack").mocapid[0]
        self.slot = np.zeros(num_envs, dtype=np.int64)

    def obj_rest_z(self, i):
        return 0.002 + obj.PCB_HALF[2]

    def sample_layout(self, rng):
        if not self.randomize:
            return np.array([0.62, 0.0]), 0.0, np.array([[0.4, -0.28]]), np.zeros(1)
        board_xy = rng.uniform([0.56, -0.1], [0.64, 0.1])
        side = rng.choice([-1.0, 1.0])
        rack_xy = np.array([rng.uniform(0.35, 0.5), board_xy[1] + side * rng.uniform(0.27, 0.33)])
        return board_xy, rng.uniform(-0.5, 0.5), rack_xy[None], rng.uniform(-1.2, 1.2, size=1)

    def extra_mocap(self, k, rng, mocap, quat, layout):
        _, _, card_xy, card_yaw = layout
        mocap[k, self.rack_mocap, :2] = card_xy[0]
        quat[k, self.rack_mocap] = yaw_quat(card_yaw[0])

    def reset(self, env_ids, qpos, rng, options=None):
        options = dict(options or {})
        slot = options.pop("target", None)
        out = super().reset(env_ids, qpos, rng, options)
        self.slot[env_ids] = rng.integers(0, self.num_slots, size=len(env_ids)) if slot is None else slot
        return out

    # Goal of the card = seated in the target slot.
    def _state(self, sim):
        pos = sim.read("xpos")[:, self.obj_bodies]
        goals_all = sim.read("site_xpos")[:, self.slot_goals]
        goals = goals_all[np.arange(sim.num_envs), self.slot][:, None]
        xy_err = np.linalg.norm(pos[..., :2] - goals[..., :2], axis=-1)
        inserted = (xy_err < self.xy_tol) & (pos[..., 2] - goals[..., 2] < self.z_tol)
        return pos, goals, inserted

    def observe(self, sim):
        out = super().observe(sim)
        out["slot_pos"] = sim.read("site_xpos")[:, self.slot_goals].reshape(sim.num_envs, -1)
        out["slot_onehot"] = np.eye(self.num_slots)[self.slot]
        return out

    def conditioning(self):
        return {"target": self.slot.copy()}

    def instruction(self, env_id):
        s = int(self.slot[env_id])
        which = ORDINALS[s] if self.num_slots == 3 else f"slot {s}"
        return f"insert the card into the {which} connector"
