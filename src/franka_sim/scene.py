"""Build the MuJoCo scene (robot + table + cameras + task objects) with ``mujoco.MjSpec``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from franka_sim import robot

if TYPE_CHECKING:
    from franka_sim.tasks.base import Task

TABLE_HEIGHT = 0.0  # objects rest on the floor plane at z = 0; the robot base sits at the origin.


def _look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Quaternion (wxyz) for a camera at ``pos`` looking at ``target`` (MuJoCo cameras look along -z)."""
    pos, target, up = (np.asarray(v, dtype=float) for v in (pos, target, up))
    z = pos - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    mat = np.stack([x, y, z], axis=1).reshape(-1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat)
    return quat


def build_spec(task: Task | None = None, *, image_size: tuple[int, int] = (480, 640)) -> mujoco.MjSpec:
    """Return an ``MjSpec`` containing the Panda, a floor, lights, cameras and the task's objects."""
    spec = mujoco.MjSpec.from_file(str(robot.PANDA_XML))
    spec.modelname = f"franka_{task.name}" if task is not None else "franka"

    opt = spec.option
    opt.timestep = 0.002
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0

    # Gravity compensation keeps the position-controlled arm from sagging, so IK targets are tracked.
    for name in robot.ROBOT_BODIES:
        spec.body(name).gravcomp = 1.0

    # The menagerie gripper is a weak position servo (kp=350): it squeezes a 2 cm handle with only
    # ~4 N, so tools pivot in the grasp. A stiffer servo grips with ~30-40 N per finger (the real
    # Panda hand: 70 N continuous), clamped by the actuator's force range.
    grip = spec.actuator(robot.GRIPPER_ACTUATOR)
    grip.gainprm[0] = robot.GRIPPER_KP
    grip.biasprm[1] = -robot.GRIPPER_KP
    grip.biasprm[2] = -robot.GRIPPER_KV
    grip.forcerange = [-robot.GRIPPER_MAX_FORCE, robot.GRIPPER_MAX_FORCE]

    vis = spec.visual
    vis.global_.offheight = max(image_size[0], 480)
    vis.global_.offwidth = max(image_size[1], 640)
    vis.global_.azimuth = 150
    vis.global_.elevation = -25
    vis.headlight.ambient = [0.3, 0.3, 0.3]
    vis.headlight.diffuse = [0.5, 0.5, 0.5]

    spec.add_texture(
        name="skybox", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.6, 0.7, 0.8], rgb2=[0.1, 0.1, 0.15], width=512, height=512,
    )
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.55, 0.5, 0.45], rgb2=[0.5, 0.45, 0.4], width=300, height=300, mark=mujoco.mjtMark.mjMARK_EDGE,
        markrgb=[0.6, 0.55, 0.5],
    )
    floor_mat = spec.add_material(name="floor", texrepeat=[8, 8], reflectance=0.0)
    floor_mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    world = spec.worldbody
    world.add_light(name="key", pos=[0.5, -0.5, 1.8], dir=[-0.2, 0.3, -1], diffuse=[0.7, 0.7, 0.7],
                    castshadow=True)
    world.add_light(name="fill", pos=[1.5, 0.8, 1.5], dir=[-1, -0.5, -1], diffuse=[0.3, 0.3, 0.3],
                    castshadow=False)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="floor",
                   contype=1, conaffinity=1, condim=3)

    workspace = np.array([0.35, 0.0, 0.25])
    for name, pos in {
        "front": [1.4, 0.0, 1.05],
        "side": [0.45, -1.15, 0.65],
        "top": [0.5, 0.0, 1.4],
    }.items():
        up = (1.0, 0.0, 0.0) if name == "top" else (0.0, 0.0, 1.0)
        world.add_camera(name=name, pos=pos, quat=_look_at_quat(pos, workspace, up), fovy=50)
    # Wrist camera: mounted on the hand, looking along the gripper axis (+z of the hand frame).
    wrist_pos = [-0.075, 0.0, 0.045]
    spec.body("hand").add_camera(name="wrist", pos=wrist_pos,
                                 quat=_look_at_quat(wrist_pos, [0.0, 0.0, 0.16], up=(-1.0, 0.0, 0.0)), fovy=80)

    if task is not None:
        task.build(spec)
    return spec


def build_model(task: Task | None = None, **kwargs) -> mujoco.MjModel:
    return build_spec(task, **kwargs).compile()
