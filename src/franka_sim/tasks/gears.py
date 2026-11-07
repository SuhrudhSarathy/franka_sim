"""Gear assembly: slide three spur gears onto the shafts of a base plate (NIST-board style)."""

from __future__ import annotations

import numpy as np

from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import sample_separated
from franka_sim.tasks.insertion import InsertionTask

GEAR_RADII = (0.022, 0.03, 0.026)
GEAR_NAMES = ("small gear", "large gear", "medium gear")


class GearAssemblyTask(InsertionTask):
    """Pick gears lying flat on the table and slide each onto its own shaft (bore clearance 3 mm).

    Gears differ in size and finish (brass, steel, dark steel); shaft ``i`` fits gear ``i`` and the
    shafts are spaced so the assembled gears mesh. Default ``mode="all"``: assemble every gear in a
    random order (``targets`` condition). See ``InsertionTask`` for modes and observations.
    """

    name = "gear_assembly"
    obj_prefix = "gear"
    fixture = "gear_plate"
    object_names = GEAR_NAMES
    xy_tol = 0.004
    z_tol = 0.006
    steps_per_object = 170
    grasp_offset = (obj.grasp_offset_for_height(obj.GEAR_THICKNESS / 2),) * 3
    symmetry = (None, None, None)
    grasp_yaw = (0.0, 0.0, 0.0)
    tip_offset = (0.0, 0.0, 0.0)
    grasp_width = tuple(2 * (r - 0.002) for r in GEAR_RADII)

    def __init__(self, mode: str = "all", num_targets: int = 2, randomize: bool = True):
        super().__init__(mode, num_targets, randomize)

    def build(self, spec):
        for i, r in enumerate(GEAR_RADII):
            obj.add_gear(spec, f"gear{i}", r, obj.GEAR_COLORS[i], pos=[0.4, -0.2 + 0.15 * i, obj.GEAR_THICKNESS / 2])
        obj.add_gear_plate(spec, GEAR_RADII)

    def obj_rest_z(self, i):
        return obj.GEAR_THICKNESS / 2

    def sample_layout(self, rng):
        n = self.num_objects
        if not self.randomize:
            return np.array([0.62, 0.0]), 0.0, np.stack([np.full(n, 0.4), np.linspace(-0.22, 0.22, n)], 1), np.zeros(n)
        fixture_xy = rng.uniform([0.58, -0.1], [0.66, 0.1])
        gears = sample_separated(rng, n, [0.32, -0.36], [0.46, 0.36], min_dist=0.11)
        return fixture_xy, rng.uniform(-0.6, 0.6), gears, rng.uniform(-np.pi, np.pi, size=n)

    def instruction_template(self, names):
        if len(names) == 1:
            return f"put the {names[0]} on its shaft"
        return "assemble the gears: the " + ", then the ".join(names)
