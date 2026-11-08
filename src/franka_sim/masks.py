"""Binary image masks of a task's target entity (the nail to hit, the connector to use, ...).

Masks come from MuJoCo's segmentation rendering, so they are exact. They are *amodal*: only the
target's geoms are drawn, so the arm, the hammer or a card in the slot never hides the target.
A task says which bodies form its target through ``Task.mask_bodies(cond)``; multi-step tasks
return ``None`` (no mask).
"""

from __future__ import annotations

from collections.abc import Sequence

import mujoco
import numpy as np

_MASK_GROUP = 5  # a geom group nothing else uses; target geoms are moved into it while rendering


def target_geoms(model: mujoco.MjModel, bodies: Sequence[str]) -> np.ndarray:
    """Ids of every geom attached to ``bodies``."""
    ids = [model.body(b).id for b in bodies]
    return np.flatnonzero(np.isin(model.geom_bodyid, ids))


def render_mask(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str | int,
                geoms: np.ndarray) -> np.ndarray:
    """``(H, W)`` bool mask of ``geoms`` seen from ``camera``, ignoring occlusion by anything else."""
    model = renderer.model
    saved = model.geom_group[geoms].copy()
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[_MASK_GROUP] = 1
    opt.sitegroup[:] = 0
    try:
        model.geom_group[geoms] = _MASK_GROUP
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=camera, scene_option=opt)
        seg = renderer.render()
    finally:
        renderer.disable_segmentation_rendering()
        model.geom_group[geoms] = saved
    # Compare with a plain int: comparing an array with the pybind enum is an elementwise Python
    # call (~30 ms per 256x256 image).
    is_target = np.zeros(model.ngeom + 1, dtype=bool)
    is_target[geoms] = True
    ids = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], model.ngeom)
    return is_target[ids]
