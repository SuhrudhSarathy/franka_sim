"""Batched Franka environment with a small, gym-like API."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim import robot
from franka_sim.backends import Backend, make_backend
from franka_sim.ik import DiffIK, axis_angle_to_quat, mat_to_quat, quat_mul
from franka_sim.masks import render_mask, target_geoms
from franka_sim.scene import build_spec
from franka_sim.tasks import Task, make_task

ACTION_DIMS = {
    "ee_delta": 7,  # dx dy dz (m), rx ry rz (rad, world-frame rotation vector), gripper
    "ee_pose": 8,  # x y z, qw qx qy qz (world frame), gripper
    "joint_pos": 8,  # 7 absolute joint targets (rad), gripper
    "joint_delta": 8,  # 7 joint target offsets from the current joint positions (rad), gripper
}


class FrankaEnv:
    """``num_envs`` Franka Panda worlds stepped together on a CPU or Warp backend.

    All observations, rewards and flags are batched: arrays with a leading ``num_envs`` axis, even
    when ``num_envs == 1``. The gripper command is a closure in ``[0, 1]`` (0 open, 1 closed).

    Args:
        task: task name (see ``franka_sim.tasks.TASKS``) or a ``Task`` instance.
        num_envs: number of parallel worlds.
        backend: ``"cpu"`` (MuJoCo) or ``"warp"`` (MuJoCo Warp).
        action_mode: one of ``ACTION_DIMS``.
        control_freq: policy frequency in Hz; physics runs at 500 Hz.
        cameras: camera names to render into observations (``rgb_<name>``): any of
            ``front``, ``side``, ``top``, ``wrist``.
        image_size: ``(height, width)`` of rendered observations.
        max_episode_steps: overrides the task's default horizon.
        init_noise: std-dev of Gaussian noise added to the start joint configuration on reset.
        randomize_start: sample the start end-effector pose on reset: a position offset from the
            home pose uniformly within ``start_pos_range`` (m, per axis) and a yaw within
            ``start_yaw_range`` (rad), solved with IK. ``False`` starts at the home pose.
        backend_kwargs: forwarded to the backend (e.g. ``device`` / ``nconmax`` for Warp).
    """

    def __init__(self, task: str | Task = "reach", *, num_envs: int = 1, backend: str = "cpu",
                 action_mode: str = "ee_delta", control_freq: float = 20.0, cameras=(),
                 image_size: tuple[int, int] = (128, 128), max_episode_steps: int | None = None,
                 init_noise: float = 0.02, randomize_start: bool = True,
                 start_pos_range=(0.08, 0.15, 0.08), start_yaw_range: float = 0.5, seed: int | None = None, task_kwargs: dict | None = None,
                 backend_kwargs: dict | None = None):
        if action_mode not in ACTION_DIMS:
            raise ValueError(f"unknown action_mode {action_mode!r}; expected one of {list(ACTION_DIMS)}")
        self.task = make_task(task, **(task_kwargs or {}))
        self.num_envs = num_envs
        self.action_mode = action_mode
        self.action_dim = ACTION_DIMS[action_mode]
        self.cameras = tuple(cameras)
        self.image_size = tuple(image_size)
        self.init_noise = init_noise
        self.randomize_start = randomize_start
        self.start_pos_range = np.asarray(start_pos_range, dtype=float)
        self.start_yaw_range = start_yaw_range
        self.max_episode_steps = max_episode_steps or self.task.max_episode_steps

        self.spec = build_spec(self.task, image_size=self.image_size)
        self.model: mujoco.MjModel = self.spec.compile()
        self.task.bind(self.model, num_envs)
        self.sim: Backend = make_backend(backend, self.model, num_envs, **(backend_kwargs or {}))
        self.ik = DiffIK(self.model)

        m = self.model
        self.control_dt = 1.0 / control_freq
        self.nsubsteps = max(1, round(self.control_dt / m.opt.timestep))
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in robot.ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in robot.ARM_JOINTS])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in robot.FINGER_JOINTS])
        self.arm_act = np.array([m.actuator(a).id for a in robot.ARM_ACTUATORS])
        self.grip_act = m.actuator(robot.GRIPPER_ACTUATOR).id
        self.ee_site = m.site(robot.EE_SITE).id
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()

        self._rng = np.random.default_rng(seed)
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.elapsed_steps = np.zeros(num_envs, dtype=np.int64)
        self.ctrl = np.zeros((num_envs, m.nu))
        # The last command resolved into every action representation (handy for recording).
        self.last_command = {
            "joint_pos": np.zeros((num_envs, 7)),
            "ee_pose": np.zeros((num_envs, 7)),
            "gripper": np.zeros(num_envs),
        }
        _, self.home_ee_quat = self.ik.site_pose(self._home_qpos())

    # ------------------------------------------------------------------------------------------
    def _home_qpos(self) -> np.ndarray:
        q = self.model.qpos0.copy()
        q[self.arm_qadr] = robot.HOME_QPOS
        q[self.finger_qadr] = robot.FINGER_MAX
        return q

    def reset(self, seed: int | None = None, env_ids=None, options: dict | None = None):
        """Reset ``env_ids`` (all by default). Returns ``(obs, info)`` for all envs."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        ids = np.arange(self.num_envs) if env_ids is None else np.atleast_1d(np.asarray(env_ids))
        n = len(ids)
        qpos = np.tile(self._home_qpos(), (n, 1))
        if self.randomize_start:
            home_pos, home_quat = self.ik.site_pose(qpos[0])
            margin = 0.2  # keep start configurations well inside the joint limits
            for k in range(n):
                for _attempt in range(20):
                    pos = home_pos + self._rng.uniform(-self.start_pos_range, self.start_pos_range)
                    yaw = self._rng.uniform(-self.start_yaw_range, self.start_yaw_range)
                    quat = quat_mul(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), home_quat)
                    q = qpos[k].copy()
                    for _ in range(5):  # IK takes bounded steps; iterate to converge from home
                        q[self.arm_qadr] = self.ik.solve(q, pos, quat)
                    arm = q[self.arm_qadr]
                    reached = np.linalg.norm(self.ik.site_pose(q)[0] - pos) < 0.005
                    if reached and np.all(arm > self.ik.lo + margin) and np.all(arm < self.ik.hi - margin):
                        qpos[k, self.arm_qadr] = arm
                        break
        if self.init_noise > 0:
            qpos[:, self.arm_qadr] += self._rng.normal(0, self.init_noise, size=(n, 7))
        extra = self.task.reset(ids, qpos, self._rng, options) or {}

        ctrl = np.zeros((n, self.model.nu))
        ctrl[:, self.arm_act] = qpos[:, self.arm_qadr]
        ctrl[:, self.grip_act] = robot.gripper_ctrl(np.zeros(n))
        self.sim.reset(ids)
        self.sim.write("qpos", qpos, ids)
        self.sim.write("ctrl", ctrl, ids)
        for field, values in extra.items():
            self.sim.write(field, values, ids)
        self.sim.forward()
        self.ctrl[ids] = ctrl
        self.elapsed_steps[ids] = 0

        obs = self._observe()
        self.last_command["joint_pos"][ids] = qpos[:, self.arm_qadr]
        self.last_command["ee_pose"][ids] = np.concatenate([obs["ee_pos"], obs["ee_quat"]], -1)[ids]
        self.last_command["gripper"][ids] = 0.0
        return obs, self._info()

    def step(self, action: np.ndarray):
        """Apply a ``(num_envs, action_dim)`` action. Returns ``obs, reward, terminated, truncated, info``."""
        action = np.asarray(action, dtype=np.float64).reshape(self.num_envs, self.action_dim)
        self.ctrl = self._action_to_ctrl(action)
        self.sim.step(self.ctrl, self.nsubsteps)
        self.elapsed_steps += 1
        ev = self.task.evaluate(self.sim)
        obs = self._observe()
        info = self._info()
        info.update(ev.info)
        info["success"] = ev.success
        truncated = self.elapsed_steps >= self.max_episode_steps
        terminated = ev.success.copy()
        return obs, ev.reward, terminated, truncated, info

    # ------------------------------------------------------------------------------------------
    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        qpos = self.sim.read("qpos")
        mode = self.action_mode
        grip = action[:, -1]
        joint_targets = np.empty((self.num_envs, 7))
        ee_targets = np.empty((self.num_envs, 7))
        if mode in ("joint_pos", "joint_delta"):
            joint_targets[:] = action[:, :7]
            if mode == "joint_delta":
                joint_targets += qpos[:, self.arm_qadr]
            for i in range(self.num_envs):
                q = qpos[i].copy()
                q[self.arm_qadr] = joint_targets[i]
                p, quat = self.ik.site_pose(q)
                ee_targets[i] = np.concatenate([p, quat])
        else:
            if mode == "ee_pose":
                pos, quat = action[:, :3], action[:, 3:7]
                quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
            else:  # ee_delta, relative to the measured end-effector pose
                site_pos = self.sim.read("site_xpos")[:, self.ee_site]
                site_mat = self.sim.read("site_xmat")[:, self.ee_site]
                pos = site_pos + action[:, :3]
                quat = np.stack([
                    quat_mul(axis_angle_to_quat(action[i, 3:6]), mat_to_quat(site_mat[i]))
                    for i in range(self.num_envs)
                ])
            for i in range(self.num_envs):
                joint_targets[i] = self.ik.solve(qpos[i], pos[i], quat[i])
            ee_targets[:] = np.concatenate([pos, quat], axis=-1)

        ctrl = self.ctrl.copy()
        ctrl[:, self.arm_act] = joint_targets
        ctrl[:, self.grip_act] = robot.gripper_ctrl(grip)
        ctrl = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        self.last_command["joint_pos"][:] = ctrl[:, self.arm_act]
        self.last_command["ee_pose"][:] = ee_targets
        self.last_command["gripper"][:] = np.clip(grip, 0.0, 1.0)
        return ctrl

    def _observe(self) -> dict[str, np.ndarray]:
        qpos = self.sim.read("qpos")
        qvel = self.sim.read("qvel")
        site_xpos = self.sim.read("site_xpos")
        site_xmat = self.sim.read("site_xmat")
        obs = {
            "joint_pos": qpos[:, self.arm_qadr],
            "joint_vel": qvel[:, self.arm_dadr],
            "gripper_width": qpos[:, self.finger_qadr].sum(-1, keepdims=True),
            "ee_pos": site_xpos[:, self.ee_site],
            "ee_quat": np.stack([mat_to_quat(m) for m in site_xmat[:, self.ee_site]]),
        }
        obs.update(self.task.observe(self.sim))
        for cam in self.cameras:
            obs[f"rgb_{cam}"] = np.stack([self.render(cam, env_id=i) for i in range(self.num_envs)])
        return obs

    def _info(self) -> dict:
        info = {"elapsed_steps": self.elapsed_steps.copy()}
        info.update({f"cond_{k}": v for k, v in self.task.conditioning().items()})
        return info

    def instructions(self) -> list[str]:
        return [self.task.instruction(i) for i in range(self.num_envs)]

    # ------------------------------------------------------------------------------------------
    def render(self, camera: str | int = "front", env_id: int = 0, height: int | None = None,
               width: int | None = None) -> np.ndarray:
        """RGB image ``(H, W, 3)`` uint8 of one env from a named camera (or -1 for the free camera)."""
        h, w = height or self.image_size[0], width or self.image_size[1]
        key = (h, w)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=h, width=w)
        renderer = self._renderers[key]
        renderer.update_scene(self.sim.mjdata(env_id), camera=camera)
        return renderer.render()

    def render_target_mask(self, camera: str | int = "top", env_id: int = 0, height: int | None = None,
                           width: int | None = None) -> np.ndarray | None:
        """Binary ``(H, W)`` uint8 mask (0/255) of env ``env_id``'s target entity (the nail to hit,
        the connector, the peg, the button), or ``None`` for tasks without a single target."""
        cond = {k: v[env_id] for k, v in self.task.conditioning().items()}
        bodies = self.task.mask_bodies(cond)
        if bodies is None:
            return None
        h, w = height or self.image_size[0], width or self.image_size[1]
        key = (h, w)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=h, width=w)
        mask = render_mask(self._renderers[key], self.sim.mjdata(env_id), camera,
                           target_geoms(self.model, bodies))
        return mask.astype(np.uint8) * 255

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
