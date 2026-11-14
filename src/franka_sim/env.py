"""Batched Franka environment with a small, gym-like API."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim import robot
from franka_sim.backends import Backend, make_backend
from franka_sim.controller import ACTION_ALIASES, ACTION_DIMS, ArmController, canonical_mode
from franka_sim.ik import DiffIK, mat_to_quat, quat_mul
from franka_sim.masks import render_mask, target_geoms
from franka_sim.scene import CAMERAS, build_spec
from franka_sim.tasks import Task, make_task

__all__ = ["ACTION_ALIASES", "ACTION_DIMS", "CAMERAS", "FrankaEnv"]


class FrankaEnv:
    """``num_envs`` Franka Panda worlds stepped together on a CPU or Warp backend.

    All observations, rewards and flags are batched: arrays with a leading ``num_envs`` axis, even
    when ``num_envs == 1``. The gripper command is a closure in ``[0, 1]`` (0 open, 1 closed).

    Args:
        task: task name (see ``franka_sim.tasks.TASKS``) or a ``Task`` instance.
        num_envs: number of parallel worlds.
        backend: ``"cpu"`` (MuJoCo) or ``"warp"`` (MuJoCo Warp).
        action_mode: one of ``ACTION_DIMS`` (see ``franka_sim.controller``): ``joint_pos``,
            ``joint_vel``, ``joint_delta``, ``ee_pose`` or ``delta_ee_pose`` (``ee_delta`` is an
            alias of ``delta_ee_pose``).
        control_freq: policy frequency in Hz; physics runs at 500 Hz.
        interp_substeps: the servo set-points are interpolated from the previous command to the new
            one across the control period, updated every ``interp_substeps`` physics steps (5 =
            100 Hz). ``None`` holds each command for the whole period (a staircase).
        velocity_feedforward: gain (0..1, or a bool) of the velocity feedforward: the servo
            set-points lead the command by ``gain * (kv + damping) / kp * velocity``. At 1 this
            cancels the position servos' tracking lag, so every joint follows the commanded path in
            step (without it each lags by its own time constant and the hand drifts several mm off
            straight-line paths). The cost: a command that jumps far in one step is overshot (~7 mm
            for a 10 cm jump at 1, ~1 mm at 0.5); stream commands smoothly, or lower the gain.
        cameras: camera names to render into observations (``rgb_<name>``), any of
            ``franka_sim.scene.CAMERAS``.
        image_size: ``(height, width)`` of rendered observations.
        max_episode_steps: overrides the task's default horizon.
        init_noise: std-dev of Gaussian noise added to the start joint configuration on reset.
        randomize_start: sample the start end-effector pose on reset: a position offset from the
            home pose uniformly within ``start_pos_range`` (m, per axis) and a yaw within
            ``start_yaw_range`` (rad), solved with IK. ``False`` starts at the home pose.
        backend_kwargs: forwarded to the backend (e.g. ``device`` / ``nconmax`` for Warp).
    """

    def __init__(self, task: str | Task = "reach", *, num_envs: int = 1, backend: str = "cpu",
                 action_mode: str = "delta_ee_pose", control_freq: float = 20.0,
                 interp_substeps: int | None = 5, velocity_feedforward: float = 1.0, cameras=(),
                 image_size: tuple[int, int] = (128, 128), max_episode_steps: int | None = None,
                 init_noise: float = 0.02, randomize_start: bool = True,
                 start_pos_range=(0.08, 0.15, 0.08), start_yaw_range: float = 0.5, seed: int | None = None, task_kwargs: dict | None = None,
                 backend_kwargs: dict | None = None, controller_kwargs: dict | None = None):
        action_mode = canonical_mode(action_mode)
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
        # Interpolation chunks: (fraction of the way to the new command, physics steps).
        nchunks = 1 if not interp_substeps else max(1, self.nsubsteps // interp_substeps)
        bounds = np.linspace(0, self.nsubsteps, nchunks + 1).round().astype(int)
        self._chunks = [((k + 1) / nchunks, int(b - a)) for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]))]
        unknown = set(self.cameras) - {m.cam(i).name for i in range(m.ncam)}
        if unknown:
            raise ValueError(f"unknown cameras {sorted(unknown)}; available: {[m.cam(i).name for i in range(m.ncam)]}")
        self.arm_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in robot.ARM_JOINTS])
        self.arm_dadr = np.array([m.jnt_dofadr[m.joint(j).id] for j in robot.ARM_JOINTS])
        self.finger_qadr = np.array([m.jnt_qposadr[m.joint(j).id] for j in robot.FINGER_JOINTS])
        self.arm_act = np.array([m.actuator(a).id for a in robot.ARM_ACTUATORS])
        self.grip_act = m.actuator(robot.GRIPPER_ACTUATOR).id
        self.ee_site = m.site(robot.EE_SITE).id
        self.ctrl_lo, self.ctrl_hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        # Servo lag time constants: a position servo tracking a ramp of velocity v trails it by tau * v.
        kp, kv = m.actuator_gainprm[self.arm_act, 0], -m.actuator_biasprm[self.arm_act, 2]
        self._servo_tau = float(velocity_feedforward) * (kv + m.dof_damping[self.arm_dadr]) / kp
        self.controller = ArmController(num_envs, self.control_dt, self.ctrl_lo[self.arm_act],
                                        self.ctrl_hi[self.arm_act], **(controller_kwargs or {}))

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
        self.controller.reset(ids, qpos[:, self.arm_qadr])
        self.elapsed_steps[ids] = 0

        obs = self._observe()
        self.last_command["joint_pos"][ids] = qpos[:, self.arm_qadr]
        self.last_command["ee_pose"][ids] = np.concatenate([obs["ee_pos"], obs["ee_quat"]], -1)[ids]
        self.last_command["gripper"][ids] = 0.0
        return obs, self._info()

    def step(self, action: np.ndarray, action_mode: str | None = None):
        """Apply a ``(num_envs, action_dim)`` action. Returns ``obs, reward, terminated, truncated, info``.

        ``action_mode`` overrides the env's mode for this step only (the action then has that mode's
        dimension); modes can be mixed freely from step to step.
        """
        mode = self.action_mode if action_mode is None else canonical_mode(action_mode)
        action = np.asarray(action, dtype=np.float64).reshape(self.num_envs, ACTION_DIMS[mode])
        prev = self.ctrl
        self.ctrl = self._action_to_ctrl(action, mode)
        self._advance(prev, self.ctrl)
        self.elapsed_steps += 1
        ev = self.task.evaluate(self.sim)
        obs = self._observe()
        info = self._info()
        info.update(ev.info)
        info["success"] = ev.success
        # Benchmark metrics every task reports: sub-goals done (0..task.num_steps) and whether a
        # target other than the requested one(s) was acted on (language adherence).
        info.setdefault("progress", ev.success.astype(np.int64))
        info.setdefault("wrong_target", np.zeros(self.num_envs, dtype=bool))
        truncated = self.elapsed_steps >= self.max_episode_steps
        terminated = ev.success.copy()
        return obs, ev.reward, terminated, truncated, info

    # ------------------------------------------------------------------------------------------
    def _action_to_ctrl(self, action: np.ndarray, mode: str) -> np.ndarray:
        q_meas = self.sim.read("qpos")[:, self.arm_qadr]
        grip = action[:, -1]
        cmd = self.controller.compute(mode, action[:, :-1], q_meas)
        ctrl = self.ctrl.copy()
        ctrl[:, self.arm_act] = cmd["joint_pos"]
        ctrl[:, self.grip_act] = robot.gripper_ctrl(grip)
        ctrl = np.clip(ctrl, self.ctrl_lo, self.ctrl_hi)
        self.last_command["joint_pos"][:] = ctrl[:, self.arm_act]
        self.last_command["ee_pose"][:] = cmd["ee_pose"]
        self.last_command["gripper"][:] = np.clip(grip, 0.0, 1.0)
        return ctrl

    def executed_action(self, mode: str | None = None) -> np.ndarray:
        """The last step's command restated as a ``mode`` action (default: the env's mode),
        gripper included: what to record as the label whoever (or whatever mode) produced it."""
        mode = self.action_mode if mode is None else mode
        return np.concatenate([self.controller.express(mode), self.last_command["gripper"][:, None]], axis=1)

    def convert_action(self, action: np.ndarray, from_mode: str, to_mode: str | None = None) -> np.ndarray:
        """Restate ``action`` (in ``from_mode``, at the current state) as a ``to_mode`` action
        (default: the env's mode) without stepping, e.g. to label states with an expert that speaks
        another action space. Executing the result in ``to_mode`` reproduces the same command."""
        from_mode = canonical_mode(from_mode)
        to_mode = self.action_mode if to_mode is None else to_mode
        action = np.asarray(action, dtype=np.float64).reshape(self.num_envs, ACTION_DIMS[from_mode])
        q_meas = self.sim.read("qpos")[:, self.arm_qadr]
        out = self.controller.preview(from_mode, action[:, :-1], q_meas)
        arm = self.controller.express(to_mode, out["last"])
        return np.concatenate([arm, np.clip(action[:, -1:], 0.0, 1.0)], axis=1)

    def _advance(self, prev_ctrl: np.ndarray, ctrl: np.ndarray) -> None:
        """Run one control period, ramping the arm set-points from ``prev_ctrl`` to ``ctrl``."""
        arm = self.arm_act
        lead = self._servo_tau * self.controller.v_cmd  # velocity feedforward (zero when disabled)
        for alpha, n in self._chunks:
            c = ctrl.copy()
            c[:, arm] = np.clip(prev_ctrl[:, arm] + alpha * (ctrl[:, arm] - prev_ctrl[:, arm]) + lead,
                                self.ctrl_lo[arm], self.ctrl_hi[arm])
            self.sim.step(c, n)

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
