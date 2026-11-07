"""Differential inverse kinematics for the Panda end effector (damped least squares)."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim import robot


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_axis_angle(q: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle) of a unit quaternion, taking the shortest path."""
    out = np.empty(3)
    mujoco.mju_quat2Vel(out, q if q[0] >= 0 else -q, 1.0)
    return out


def axis_angle_to_quat(v: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(v))
    out = np.array([1.0, 0.0, 0.0, 0.0])
    if angle > 1e-12:
        mujoco.mju_axisAngle2Quat(out, v / angle, angle)
    return out


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(mat, dtype=np.float64).reshape(9))
    return out


class DiffIK:
    """Iterative damped-least-squares IK on a scratch ``MjData`` (host side, any backend).

    Solves for the 7 arm joints so the ``gripper`` site reaches a target position and orientation,
    with a weak null-space pull toward the home posture.
    """

    def __init__(self, model: mujoco.MjModel, *, damping: float = 1e-3, iterations: int = 10,
                 max_step: float = 0.2, posture_gain: float = 0.05, rot_weight: float = 0.5):
        self.model = model
        self.data = mujoco.MjData(model)
        self.damping = damping
        self.iterations = iterations
        self.max_step = max_step
        self.posture_gain = posture_gain
        self.rot_weight = rot_weight
        self.site_id = model.site(robot.EE_SITE).id
        self.qadr = np.array([model.jnt_qposadr[model.joint(j).id] for j in robot.ARM_JOINTS])
        self.dadr = np.array([model.jnt_dofadr[model.joint(j).id] for j in robot.ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(j).id for j in robot.ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in robot.ARM_JOINTS], 1]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def site_pose(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: (pos, quat wxyz) of the EE site for a full ``qpos``."""
        d = self.data
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site_id].copy(), mat_to_quat(d.site_xmat[self.site_id])

    def solve(self, qpos: np.ndarray, target_pos: np.ndarray, target_quat: np.ndarray | None) -> np.ndarray:
        """Return arm joint targets (7,) starting from the full ``qpos`` of one env."""
        m, d = self.model, self.data
        d.qpos[:] = qpos
        q = qpos[self.qadr].copy()
        for _ in range(self.iterations):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = target_pos - d.site_xpos[self.site_id]
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self.site_id)
            jp = self._jacp[:, self.dadr]
            if target_quat is not None:
                cur = mat_to_quat(d.site_xmat[self.site_id])
                err_r = quat_to_axis_angle(quat_mul(target_quat, quat_conj(cur)))
                jac = np.vstack([jp, self.rot_weight * self._jacr[:, self.dadr]])
                err = np.concatenate([err_p, self.rot_weight * err_r])
            else:
                jac, err = jp, err_p
            jjt = jac @ jac.T + self.damping * np.eye(jac.shape[0])
            dq = jac.T @ np.linalg.solve(jjt, err)
            # Null-space posture regularization toward the home configuration.
            pinv = jac.T @ np.linalg.inv(jjt)
            null = np.eye(7) - pinv @ jac
            dq += null @ (self.posture_gain * (robot.HOME_QPOS - q))
            norm = np.linalg.norm(dq)
            if norm > self.max_step:
                dq *= self.max_step / norm
            q = np.clip(q + dq, self.lo, self.hi)
            if np.linalg.norm(err) < 1e-4:
                break
        return q
