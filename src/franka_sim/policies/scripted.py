"""Scripted expert policies for every built-in task."""

from __future__ import annotations

import numpy as np

from franka_sim import robot
from franka_sim.policies.base import ScriptedPolicy, Target, quat_yaw, wrap_angle
from franka_sim.tasks import objects as obj

OPEN, CLOSED = 0.0, 1.0
UP = np.array([0.0, 0.0, 1.0])


class ReachPolicy(ScriptedPolicy):
    def plan(self, i, obs, stage):
        t = Target(obs["goal_pos"][i], 0.0, OPEN)
        if stage == 0:
            return t, self.settled(i, obs, t, tol=0.005)
        self.memory[i]["finished"] = True
        return t, False


class HammerPolicy(ScriptedPolicy):
    """Grasp the hammer handle, lift, hover the head over the target nail, strike until it is driven."""

    def __init__(self, env, max_strikes: int = 4, strike_speed: float = 0.035, **kwargs):
        super().__init__(env, **kwargs)
        self.max_strikes = max_strikes
        self.strike_speed = strike_speed
        self.success_depth = getattr(env.task, "success_depth", 0.75)

    def plan(self, i, obs, stage):
        mem = self.memory[i]
        ee = obs["ee_pos"][i]
        target_idx = int(np.argmax(obs["target_onehot"][i]))
        nail = obs["nail_pos"][i].reshape(-1, 3)[target_idx]
        face = obs["hammer_face_pos"][i]
        if "yaw" not in mem:
            mem["yaw"] = float(wrap_angle(quat_yaw(obs["hammer_quat"][i]), np.pi))
            mem["hover"] = 0.07 + self.rng.uniform(-0.01, 0.02)
            mem["strikes"] = 0
        yaw = mem["yaw"]

        if stage == 0:  # above the handle
            t = Target(obs["hammer_grip_pos"][i] + 0.12 * UP, yaw, OPEN)
            return t, self.settled(i, obs, t, tol=0.02)
        if stage == 1:  # descend around the handle
            t = Target(obs["hammer_grip_pos"][i] - 0.003 * UP, yaw, OPEN, speed=0.01)
            return t, self.settled(i, obs, t, tol=0.008)
        if stage == 2:  # close
            t = Target(self.cmd_pos[i].copy(), yaw, CLOSED)
            return t, self.timer[i] >= 10
        if stage == 3:  # lift
            t = Target(np.array([*self.cmd_pos[i][:2], 0.25]), yaw, CLOSED)
            return t, self.reached(i, t)
        if stage == 4:  # hover the hammer face above the target nail (closed loop on the face position)
            desired = nail + mem["hover"] * UP
            t = Target(ee + (desired - face), yaw, CLOSED)
            done = np.linalg.norm(desired - face) < 0.008 and self.timer[i] > 5
            if done:
                mem["strike_from"] = self.cmd_pos[i].copy()
                mem["strike_dz"] = (nail[2] - obj.NAIL_TRAVEL - 0.01) - face[2]
            return t, done
        if stage == 5:  # strike: drive the face down past the nail's travel
            t = Target(mem["strike_from"] + mem["strike_dz"] * UP, yaw, CLOSED, speed=self.strike_speed)
            depth = obs["nail_depth"][i][target_idx]
            return t, self.timer[i] >= 14 or depth >= 0.97
        if stage == 6:  # raise, then strike again if needed
            t = Target(mem["strike_from"].copy(), yaw, CLOSED)
            if not self.reached(i, t):
                return t, False
            mem["strikes"] += 1
            if obs["nail_depth"][i][target_idx] < self.success_depth and mem["strikes"] < self.max_strikes:
                self.stage[i] = 3  # -> advances to stage 4 (re-aim) on return
            return t, True
        mem["finished"] = True
        return Target(self.cmd_pos[i].copy(), yaw, CLOSED), False


class PegPolicy(ScriptedPolicy):
    """Grasp a peg from the top, align its yaw with the hole, lower it in, release. Repeats for
    every peg when the task is in ``mode="all"``."""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.mode = getattr(env.task, "mode", "single")

    def _peg(self, i, obs):
        mem = self.memory[i]
        if self.mode == "single":
            return int(np.argmax(obs["target_onehot"][i]))
        if "order" not in mem:
            mem["order"] = list(self.rng.permutation(len(obs["peg_inserted"][i])))
        return int(mem["order"][0])

    def plan(self, i, obs, stage):
        mem = self.memory[i]
        if self.mode == "all" and "order" in mem and not mem["order"]:
            mem["finished"] = True
            return Target(self.cmd_pos[i].copy(), self.cmd_yaw[i], OPEN), False
        k = self._peg(i, obs)
        peg = obs["peg_pos"][i].reshape(-1, 3)[k]
        peg_yaw = float(wrap_angle(quat_yaw(obs["peg_quat"][i].reshape(-1, 4)[k]), np.pi / 2))
        hole = obs["hole_pos"][i].reshape(-1, 3)[k]
        half_h = obj.PEG_HALF[2]
        ee = obs["ee_pos"][i]

        if stage == 0:
            mem["grasp_yaw"] = peg_yaw
            t = Target(peg + 0.12 * UP, peg_yaw, OPEN)
            return t, self.settled(i, obs, t, tol=0.02)
        if stage == 1:
            t = Target(peg + 0.015 * UP, mem["grasp_yaw"], OPEN, speed=0.01)
            return t, self.settled(i, obs, t, tol=0.006)
        if stage == 2:
            t = Target(self.cmd_pos[i].copy(), mem["grasp_yaw"], CLOSED)
            return t, self.timer[i] >= 10
        if stage == 3:
            t = Target(np.array([*self.cmd_pos[i][:2], 0.22]), mem["grasp_yaw"], CLOSED)
            return t, self.reached(i, t)
        if stage in (4, 5):
            # Rotate so the peg's yaw (mod 90 deg) matches the hole's, and steer the peg (not the
            # gripper) over the hole.
            yaw = float(self.cmd_yaw[i] - peg_yaw)
            height = 0.07 if stage == 4 else 0.004
            desired = hole + (half_h + height) * UP
            err = desired - peg
            if stage == 4:
                t = Target(ee + err, yaw, CLOSED)
                done = np.linalg.norm(err[:2]) < 0.003 and abs(err[2]) < 0.01 and abs(peg_yaw) < 0.03 \
                    and self.timer[i] > 5
                return t, done
            t = Target(ee + err, yaw, CLOSED, speed=0.006)
            return t, (peg[2] - hole[2] < half_h + 0.012) or self.timer[i] > 60
        if stage == 6:
            t = Target(self.cmd_pos[i].copy(), self.cmd_yaw[i], OPEN)
            return t, self.timer[i] >= 6
        if stage == 7:
            t = Target(np.array([*self.cmd_pos[i][:2], 0.22]), self.cmd_yaw[i], OPEN)
            if not self.reached(i, t):
                return t, False
            if self.mode == "all":
                mem["order"].pop(0)
                if mem["order"]:
                    self.stage[i] = -1  # -> 0 on return: next peg
                    return t, True
            return t, True
        mem["finished"] = True
        return Target(self.cmd_pos[i].copy(), self.cmd_yaw[i], OPEN), False


class ButtonPolicy(ScriptedPolicy):
    """Press the buttons of the sequence in order with closed fingers."""

    def plan(self, i, obs, stage):
        mem = self.memory[i]
        nb = len(obs["button_depth"][i])
        seq = np.argmax(obs["sequence_onehot"][i].reshape(-1, nb), axis=1)
        progress = int(obs["progress"][i][0])
        if progress >= len(seq) and stage == 0:
            mem["finished"] = True
            return Target(self.cmd_pos[i].copy(), 0.0, CLOSED), False
        b = seq[min(progress, len(seq) - 1)] if stage == 0 or "button" not in mem else mem["button"]
        top = obs["button_pos"][i].reshape(-1, 3)[b]
        if stage == 0:
            mem["button"] = b
            mem["hover"] = 0.06 + self.rng.uniform(0.0, 0.03)
            t = Target(top + mem["hover"] * UP, 0.0, CLOSED)
            return t, self.settled(i, obs, t, tol=0.01)
        if stage == 1:
            t = Target(top + (robot.FINGERTIP_OFFSET - 0.015) * UP, 0.0, CLOSED, speed=0.008)
            return t, obs["button_depth"][i][b] >= 0.9 or self.timer[i] > 40
        if stage == 2:
            t = Target(obs["button_pos"][i].reshape(-1, 3)[b] + mem["hover"] * UP, 0.0, CLOSED)
            if not self.reached(i, t):
                return t, False
            self.stage[i] = -1  # -> 0: next button in the sequence (or finish)
            return t, True
        return Target(self.cmd_pos[i].copy(), 0.0, CLOSED), False


POLICIES = {
    "reach": ReachPolicy,
    "hammer_nails": HammerPolicy,
    "peg_insertion": PegPolicy,
    "buttons": ButtonPolicy,
}


def make_policy(env, **kwargs) -> ScriptedPolicy:
    """The scripted expert for ``env.task``."""
    try:
        cls = POLICIES[env.task.name]
    except KeyError:
        raise ValueError(f"no scripted policy for task {env.task.name!r}") from None
    return cls(env, **kwargs)
