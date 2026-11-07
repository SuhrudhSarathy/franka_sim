"""Kitting: place machined parts into their fitted compartments of a kit tray."""

from __future__ import annotations

import numpy as np

from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import sample_separated
from franka_sim.tasks.insertion import InsertionTask

KINDS = ("block", "cylinder", "bar", "cube")


class KittingTask(InsertionTask):
    """Parts lie scattered on the table; each goes into the compartment of the tray that fits it.

    Default ``mode="subset"`` with ``num_targets=2``: each episode asks for an ordered pick list of two
    parts (``options={"targets": [2, 0]}`` to choose), like filling a kit order. Parts must be placed
    aligned with their compartment. See ``InsertionTask`` for modes and observations.
    """

    name = "kitting"
    obj_prefix = "part"
    fixture = "kit_tray"
    object_names = KINDS
    xy_tol = 0.012
    z_tol = 0.012
    steps_per_object = 160

    def __init__(self, mode: str = "subset", num_targets: int = 2, randomize: bool = True):
        self.grasp_offset = tuple(obj.grasp_offset_for_height(obj.kit_part_footprint(k)[2]) for k in KINDS)
        self.symmetry = tuple(None if obj.KIT_PARTS[k][0] == obj.CYL else
                              (np.pi / 2 if k == "cube" else np.pi) for k in KINDS)
        # Close the fingers across the narrow side of elongated parts (part x = long axis).
        self.grasp_yaw = (0.0,) * len(KINDS)
        self.tip_offset = tuple(-obj.kit_part_footprint(k)[2] for k in KINDS)
        self.grasp_width = tuple(2 * obj.kit_part_footprint(k)[1] for k in KINDS)
        super().__init__(mode, num_targets, randomize)

    def build(self, spec):
        for i, k in enumerate(KINDS):
            obj.add_kit_part(spec, f"part{i}", k, pos=[0.4, -0.2 + 0.12 * i, obj.kit_part_footprint(k)[2]])
        obj.add_kit_tray(spec, KINDS)

    def obj_rest_z(self, i):
        return obj.kit_part_footprint(KINDS[i])[2]

    def sample_layout(self, rng):
        n = self.num_objects
        if not self.randomize:
            return np.array([0.62, 0.0]), 0.0, np.stack([np.full(n, 0.38), np.linspace(-0.27, 0.27, n)], 1), np.zeros(n)
        # The tray is long along its local y; keep the parts out of the gripper's reach of its walls.
        fixture_xy = rng.uniform([0.6, -0.08], [0.66, 0.08])
        parts = sample_separated(rng, n, [0.29, -0.36], [0.41, 0.36], min_dist=0.1)
        return fixture_xy, rng.uniform(-0.35, 0.35), parts, rng.uniform(-np.pi, np.pi, size=n)

    def instruction_template(self, names):
        return "kit the " + ", then the ".join(names)
