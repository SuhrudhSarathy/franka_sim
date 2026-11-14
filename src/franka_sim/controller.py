"""Arm controllers: turn an action in any mode into joint position targets for the Panda's servos.

Every mode produces a *commanded joint configuration* ``q_cmd`` per control step, and every mode
keeps it physically plausible: the command moves within the Panda's joint velocity limits (and,
optionally, its acceleration limits) and stays inside the joint limits. Within the control
period the servo set-point is interpolated from the previous ``q_cmd`` to the new one, with a
velocity feedforward that cancels the servos' tracking lag (see ``FrankaEnv``), so the arm follows
the commanded path closely instead of a lagging staircase of set-points.

Action modes (the last entry is always the gripper closure, 0 open .. 1 closed):

=================  ===  ==========================================================================
mode               dim  arm part
=================  ===  ==========================================================================
``joint_pos``       8   7 absolute joint targets (rad)
``joint_vel``       8   7 joint velocities (rad/s), integrated onto the commanded configuration
``joint_delta``     8   7 joint offsets (rad) from the *measured* joint positions
``ee_pose``         8   absolute end-effector pose: x y z (m), qw qx qy qz (world frame)
``delta_ee_pose``   7   end-effector displacement from the *commanded* pose: dx dy dz (m, world
                        frame) and a rotation vector rx ry rz (rad, world frame)
=================  ===  ==========================================================================

The Cartesian modes are solved by ``MinkIK``: differential IK posed as quadratic programs with
mink (end-effector task + posture regularization, joint limits as hard constraints); the command then
moves toward the IK solution within the velocity and acceleration limits. Incremental modes (``joint_vel``, ``delta_ee_pose``) build on the previous command,
not on the measured state, so the arm does not drift under tracking lag; if contact holds the arm
back by more than ``max_lag`` the reference is pulled back toward the measured state, so it never
winds up.
"""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim import robot
from franka_sim.ik import axis_angle_to_quat, mat_to_quat, quat_conj, quat_mul, quat_to_axis_angle

ACTION_DIMS = {
    "joint_pos": 8,
    "joint_vel": 8,
    "joint_delta": 8,
    "ee_pose": 8,
    "delta_ee_pose": 7,
}
# Older names, still accepted.
ACTION_ALIASES = {"ee_delta": "delta_ee_pose"}


def canonical_mode(mode: str) -> str:
    mode = ACTION_ALIASES.get(mode, mode)
    if mode not in ACTION_DIMS:
        raise ValueError(f"unknown action_mode {mode!r}; expected one of {list(ACTION_DIMS)}")
    return mode


def arm_model() -> mujoco.MjModel:
    """The Panda arm alone (7 hinge joints, fingers welded open, no actuators): the IK model."""
    spec = mujoco.MjSpec.from_file(str(robot.PANDA_XML))
    for eq in list(spec.equalities):
        spec.delete(eq)
    for act in list(spec.actuators):
        spec.delete(act)
    for name in robot.FINGER_JOINTS:
        spec.delete(spec.joint(name))
    model = spec.compile()
    assert model.nq == 7, model.nq
    return model


class MinkIK:
    """Differential IK for the end-effector site, solved as a sequence of QPs with mink.

    ``solve(q, pos, quat)`` iterates from ``q`` toward a joint configuration that puts the site at
    the target pose: each iteration is a QP with the end-effector task, a small posture term pulling
    the redundant degree of freedom toward ``robot.HOME_QPOS``, and the joint limits as hard
    constraints. Velocity and acceleration limits are *not* applied here; ``ArmController`` moves the
    command toward the solution within them, so it can brake ahead of a far target.

    Args:
        iterations: maximum QP iterations per solve (stops early below ``tol``).
        position_cost / orientation_cost: weights of the end-effector task (per m / per rad).
        posture_cost: weight of the posture term.
        damping: Levenberg-Marquardt damping of the QP (regularizes singular configurations).
        limit_margin: keep this far (rad) inside the joint limits.
    """

    def __init__(self, *, iterations: int = 10, position_cost: float = 1.0, orientation_cost: float = 1.0,
                 posture_cost: float = 1e-2, damping: float = 1e-4, limit_margin: float = 0.01,
                 tol: float = 1e-5):
        import mink  # imported lazily: only the Cartesian modes need it

        self._mink = mink
        self.iterations = iterations
        self.damping = damping
        self.tol = tol
        self.model = arm_model()
        self.lo = self.model.jnt_range[:, 0] + limit_margin
        self.hi = self.model.jnt_range[:, 1] - limit_margin
        self.config = mink.Configuration(self.model, robot.HOME_QPOS.copy())
        self.ee_task = mink.FrameTask(robot.EE_SITE, "site", position_cost=position_cost,
                                      orientation_cost=orientation_cost)
        self.posture_task = mink.PostureTask(self.model, cost=posture_cost)
        self.posture_task.set_target(robot.HOME_QPOS)
        self.limits = [mink.ConfigurationLimit(self.model, min_distance_from_limits=limit_margin)]
        self.site_id = self.model.site(robot.EE_SITE).id

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """End-effector (pos, quat wxyz) at arm configuration ``q`` (7,)."""
        self.config.update(np.asarray(q, dtype=float))
        d = self.config.data
        return d.site_xpos[self.site_id].copy(), mat_to_quat(d.site_xmat[self.site_id])

    def solve(self, q: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """Arm configuration (7,) reaching the target pose, found by iterating from ``q``."""
        mink = self._mink
        self.config.update(np.clip(q, self.lo, self.hi))
        self.ee_task.set_target(mink.SE3(np.concatenate([quat, pos])))
        for _ in range(self.iterations):
            # With dt = 1 the QP's velocity is the configuration step itself.
            step = mink.solve_ik(self.config, [self.ee_task, self.posture_task], 1.0, "daqp",
                                 damping=self.damping, limits=self.limits)
            self.config.integrate_inplace(step, 1.0)
            if np.linalg.norm(self.ee_task.compute_error(self.config)) < self.tol:
                break
        return np.clip(self.config.q, self.lo, self.hi)


class ArmController:
    """Turns batched actions into commanded joint configurations, keeping per-env command state.

    Args:
        num_envs: batch size.
        dt: control period (s).
        joint_lo / joint_hi: arm joint limits (7,).
        max_lag: how far (rad, per joint) the commanded configuration may run ahead of the measured
            one in the incremental modes before it is pulled back (anti-windup).
        velocity_scale: fraction of the Panda's joint velocity limits the command may use.
        accel_scale: if set, also limit the command's acceleration to this fraction of the Panda's
            acceleration limits, braking in time to stop at the target. This softens far jumps
            (0.5: ~4 mm past a 10 cm jump instead of ~7 mm) but adds lag that destabilizes fast
            closed-loop policies (the insertion experts drop to 0.65-0.9 success), so it is off by
            default.
        ik_kwargs: forwarded to ``MinkIK``.
    """

    def __init__(self, num_envs: int, dt: float, joint_lo: np.ndarray, joint_hi: np.ndarray, *,
                 max_lag: float = 0.2, velocity_scale: float = 1.0, accel_scale: float | None = None,
                 ik_kwargs: dict | None = None):
        self.num_envs = num_envs
        self.dt = dt
        self.lo, self.hi = np.asarray(joint_lo, float), np.asarray(joint_hi, float)
        self.max_lag = max_lag
        self.vmax = velocity_scale * robot.JOINT_VEL_LIMITS
        self.amax = None if accel_scale is None else accel_scale * robot.JOINT_ACC_LIMITS
        self.ik = MinkIK(**(ik_kwargs or {}))
        self.q_cmd = np.tile(robot.HOME_QPOS, (num_envs, 1))
        self.v_cmd = np.zeros((num_envs, 7))  # velocity of the command over the last period

    def reset(self, env_ids: np.ndarray, q: np.ndarray) -> None:
        self.q_cmd[env_ids] = q
        self.v_cmd[env_ids] = 0.0

    def _limit_motion(self, prev: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Step from ``prev`` toward ``target`` within the velocity and acceleration limits.

        Per joint, the velocity is the one that reaches the target this period, capped by the
        velocity limit, by the largest speed from which the joint can still stop at the target
        (discrete-time braking with the acceleration limit), and by the change in velocity the
        acceleration limit allows from the previous period.
        """
        dt, a = self.dt, self.amax
        err = target - prev
        if a is None:  # velocity limits only
            return prev + np.clip(err, -self.vmax * dt, self.vmax * dt)
        brake = a * dt * (np.sqrt(0.25 + 2.0 * np.abs(err) / (a * dt * dt)) - 0.5)
        vel = np.sign(err) * np.minimum.reduce([np.abs(err) / dt, brake, self.vmax * np.ones_like(err)])
        vel = np.clip(vel, self.v_cmd - a * dt, self.v_cmd + a * dt)
        return prev + vel * dt

    def commanded_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return self.ik.fk(self.q_cmd[i])

    def _anchor(self, q_meas: np.ndarray) -> np.ndarray:
        """The previous command, pulled to within ``max_lag`` of the measured joints."""
        return q_meas + np.clip(self.q_cmd - q_meas, -self.max_lag, self.max_lag)

    def compute(self, mode: str, arm_action: np.ndarray, q_meas: np.ndarray, ee_meas_quat=None) -> dict:
        """New commanded joints for ``arm_action`` ``(N, dim - 1)``; updates ``q_cmd``.

        Returns ``{"joint_pos": (N, 7), "ee_pose": (N, 7)}``: the command as joint targets and as
        the end-effector pose it was solving for (the requested pose in the Cartesian modes, the
        forward kinematics of the joint command otherwise).
        """
        n = self.num_envs
        prev = self.q_cmd.copy()
        ee = np.empty((n, 7))
        if mode == "joint_pos":
            target = arm_action
        elif mode == "joint_delta":
            target = q_meas + arm_action
        elif mode == "joint_vel":
            vel = np.clip(arm_action, -self.vmax, self.vmax)
            prev = self._anchor(q_meas)
            target = prev + vel * self.dt
        elif mode in ("ee_pose", "delta_ee_pose"):
            if mode == "ee_pose":
                pos = arm_action[:, :3]
                quat = arm_action[:, 3:7] / np.linalg.norm(arm_action[:, 3:7], axis=-1, keepdims=True)
            else:
                prev = self._anchor(q_meas)
                pos, quat = np.empty((n, 3)), np.empty((n, 4))
                for i in range(n):
                    p0, q0 = self.ik.fk(prev[i])
                    pos[i] = p0 + arm_action[i, :3]
                    quat[i] = quat_mul(axis_angle_to_quat(arm_action[i, 3:6]), q0)
            target = np.stack([self.ik.solve(prev[i], pos[i], quat[i]) for i in range(n)])
            ee[:] = np.concatenate([pos, quat], axis=-1)
        else:
            raise ValueError(f"unknown action_mode {mode!r}")

        # Same limits for every mode: joint velocity and acceleration limits, then the position limits.
        q_new = np.clip(self._limit_motion(prev, target), self.lo, self.hi)
        self.v_cmd = (q_new - self.q_cmd) / self.dt
        if mode not in ("ee_pose", "delta_ee_pose"):
            for i in range(n):
                ee[i] = np.concatenate(self.ik.fk(q_new[i]))
        # What ``express`` needs to restate this command in another mode.
        self.last = {"q_before": self.q_cmd, "q_meas": np.asarray(q_meas, float).copy(), "q_new": q_new.copy(),
                     "ee": ee.copy()}
        self.q_cmd = q_new
        return {"joint_pos": q_new.copy(), "ee_pose": ee}

    def preview(self, mode: str, arm_action: np.ndarray, q_meas: np.ndarray) -> dict:
        """``compute`` without committing: the command state is left as it was. Returns the result of
        ``compute`` plus the bookkeeping ``express`` needs (as ``compute`` stores in ``last``)."""
        saved = (self.q_cmd.copy(), self.v_cmd.copy(), getattr(self, "last", None))
        try:
            out = self.compute(mode, arm_action, q_meas)
            out["last"] = self.last
        finally:
            self.q_cmd, self.v_cmd, self.last = saved
        return out

    def express(self, mode: str, last: dict | None = None) -> np.ndarray:
        """Restate a command (by default the last committed one) as an arm action in ``mode``.

        The result, executed in ``mode`` from the same state, reproduces the same joint command: e.g.
        a human's ``delta_ee_pose`` correction becomes the ``joint_pos`` (or ``ee_pose``, ...) label a
        policy with that action space should learn. ``(N, ACTION_DIMS[mode] - 1)``.
        """
        mode = canonical_mode(mode)
        last = self.last if last is None else last
        q_before, q_meas, q_new, ee = last["q_before"], last["q_meas"], last["q_new"], last["ee"]
        if mode == "joint_pos":
            return q_new.copy()
        if mode == "joint_delta":
            return q_new - q_meas
        if mode == "ee_pose":
            return ee.copy()
        anchor = q_meas + np.clip(q_before - q_meas, -self.max_lag, self.max_lag)
        if mode == "joint_vel":
            return (q_new - anchor) / self.dt
        out = np.empty((len(q_new), 6))  # delta_ee_pose, relative to the anchored commanded pose
        for i in range(len(q_new)):
            p0, r0 = self.ik.fk(anchor[i])
            out[i, :3] = ee[i, :3] - p0
            out[i, 3:] = quat_to_axis_angle(quat_mul(ee[i, 3:7], quat_conj(r0)))
        return out
