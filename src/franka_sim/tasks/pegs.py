"""Multi-peg insertion: colored square pegs and a receptacle with a matching hole for each."""

from __future__ import annotations

import numpy as np

from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import sample_separated
from franka_sim.tasks.insertion import InsertionTask


class PegInsertionTask(InsertionTask):
    """Insert colored square pegs into the hole of the same color (see ``InsertionTask`` for modes,
    observations and conditioning). Pegs start upright anywhere on the table at any yaw; the
    receptacle is placed and rotated at random."""

    name = "peg_insertion"
    obj_prefix = "peg"
    fixture = "receptacle"
    xy_tol = 0.01
    z_tol = 0.02
    steps_per_object = 200

    long_instruction = "insert all the pegs into their holes"

    def __init__(self, num_pegs: int = 3, mode: str = "single", num_targets: int = 2, randomize: bool = True):
        self.num_pegs = num_pegs
        self.object_names = tuple(f"{obj.COLOR_NAMES[i]} peg" for i in range(num_pegs))
        self.grasp_offset = (0.015,) * num_pegs
        self.symmetry = (np.pi / 2,) * num_pegs
        self.tip_offset = (-obj.PEG_HALF[2],) * num_pegs
        self.grasp_yaw = (0.0,) * num_pegs
        self.grasp_width = (2 * obj.PEG_HALF[1],) * num_pegs
        super().__init__(mode, num_targets, randomize)

    def build(self, spec):
        colors = [obj.COLORS[obj.COLOR_NAMES[i]] for i in range(self.num_pegs)]
        for i, c in enumerate(colors):
            obj.add_peg(spec, f"peg{i}", c, pos=[0.4, -0.2 + 0.1 * i, obj.PEG_HALF[2]])
        obj.add_receptacle(spec, colors)

    def obj_rest_z(self, i):
        return obj.PEG_HALF[2]

    def sample_layout(self, rng):
        n = self.num_pegs
        if not self.randomize:
            return np.array([0.63, 0.0]), 0.0, np.stack([np.full(n, 0.42), np.linspace(-0.2, 0.2, n)], 1), np.zeros(n)
        fixture_xy = rng.uniform([0.56, -0.12], [0.66, 0.12])
        pegs = sample_separated(rng, n, [0.3, -0.35], [0.48, 0.35], min_dist=0.09)
        return fixture_xy, rng.uniform(-0.5, 0.5), pegs, rng.uniform(-np.pi, np.pi, size=n)

    def instruction_template(self, names):
        holes = [f"{nm} into the {nm.split()[0]} hole" for nm in names]
        return "insert the " + ", then the ".join(holes)
