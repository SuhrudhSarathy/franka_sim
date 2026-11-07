"""Scripted expert policies for every built-in task."""

from __future__ import annotations

import mujoco
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


class InsertionPolicy(ScriptedPolicy):
    """Generic pick-align-insert expert for every ``InsertionTask`` (pegs, gears, kitting, PCB).

    For each requested object in turn: grasp it from the top at the task's grasp height and yaw, lift,
    steer the *object* (closed loop on its pose) above its goal while rotating it to the fixture's yaw
    (modulo the object's symmetry), lower it in, release and retreat. An object that was dropped or
    missed is simply picked again, since it is still the next unfinished target.
    """

    approach = 0.06  # height of the object above its goal before the final descent
    lift = 0.2

    def __init__(self, env, insert_speed: float = 0.006, **kwargs):
        super().__init__(env, **kwargs)
        self.task = env.task
        self.insert_speed = insert_speed

    def plan(self, i, obs, stage):
        mem = self.memory[i]
        task = self.task
        hold = Target(self.cmd_pos[i].copy(), self.cmd_yaw[i], mem.get("open", OPEN))
        remaining = (obs["targets_mask"][i] > 0) & (obs["obj_inserted"][i] < 0.5)
        if stage == 0:
            if not remaining.any():
                mem["finished"] = True
                return hold, False
            mem["k"] = int(np.argmax(obs["target_onehot"][i]))
            # Open only as wide as the object needs (+ margin): fewer collisions in clutter.
            width = task.grasp_width[mem["k"]] if task.grasp_width else robot.GRIPPER_MAX_WIDTH
            mem["open"] = float(np.clip(1.0 - (width + 0.03) / robot.GRIPPER_MAX_WIDTH, 0.0, 1.0))
        k = mem["k"]
        OPEN_K = mem["open"]
        pos = obs["obj_pos"][i].reshape(-1, 3)[k]
        yaw_obj = float(quat_yaw(obs["obj_quat"][i].reshape(-1, 4)[k]))
        goal = obs["goal_pos"][i].reshape(-1, 3)[k]
        yaw_goal = float(quat_yaw(obs["fixture_quat"][i]))
        sym = task.symmetry[k]
        up = task.grasp_offset[k] * UP
        ee = obs["ee_pos"][i]
        # Steer the point that enters first (e.g. a card's bottom edge), which stays correct even if
        # the object tilts in the grasp.
        tip_off = task.tip_offset[k] if task.tip_offset else 0.0
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, obs["obj_quat"][i].reshape(-1, 4)[k])
        tip = pos + rot.reshape(3, 3)[:, 2] * tip_off
        goal_tip = goal + tip_off * UP

        if stage in (3, 4, 5) and np.linalg.norm(pos + up - ee) > 0.03:
            # Lost (or never got) the object: release, back off and grasp it again.
            mem["regrasps"] = mem.get("regrasps", 0) + 1
            self.stage[i] = 5
            return hold, True

        if stage == 0:  # above the grasp point
            if self.timer[i] == 0 or "gyaw" not in mem:  # choose the grasp yaw once per pick
                if sym is None:  # round object: any yaw works; return toward the neutral wrist
                    mem["gyaw"] = float(np.clip(self.cmd_yaw[i], -1.0, 1.0))
                else:
                    mem["gyaw"] = float(wrap_angle(yaw_obj + task.grasp_yaw[k], min(sym, np.pi)))
            t = Target(pos + up + 0.12 * UP, mem["gyaw"], OPEN_K)
            return t, self.settled(i, obs, t, tol=0.02)
        if stage == 1:  # descend around the object (close anyway if blocked a few mm short)
            t = Target(pos + up, mem["gyaw"], OPEN_K, speed=0.01)
            return t, self.settled(i, obs, t, tol=0.006) or (self.reached(i, t) and self.timer[i] > 40)
        if stage == 2:  # close
            return Target(self.cmd_pos[i].copy(), mem["gyaw"], CLOSED), self.timer[i] >= 10
        if stage == 3:  # lift clear of everything (slowly at first, to pull the object out cleanly)
            z = max(self.lift, goal[2] + self.approach + 0.08)
            slow = self.cmd_pos[i][2] < pos[2] + task.grasp_offset[k] + 0.03 and self.timer[i] < 30
            t = Target(np.array([*self.cmd_pos[i][:2], z]), mem["gyaw"], CLOSED, speed=0.006 if slow else None)
            return t, self.reached(i, t)
        if stage in (4, 5):  # align the object over its goal, then lower it in
            dyaw = 0.0 if sym is None else float(wrap_angle(yaw_goal - yaw_obj, sym))
            yaw = float(self.cmd_yaw[i] + dyaw)
            if sym is not None and abs(yaw) > 2.2:  # an equivalent orientation that spares the wrist
                yaw -= np.sign(yaw) * sym
                dyaw = yaw - float(self.cmd_yaw[i])
            if stage == 4:
                err = goal_tip + self.approach * UP - tip
                if np.linalg.norm(err[:2]) > 0.01:  # travel at lift height, descend only once above
                    err[2] = 0.0
                done = (np.linalg.norm(err[:2]) < 0.0015 and abs(err[2]) < 0.01 and abs(dyaw) < 0.02
                        and self.timer[i] > 5)
                # Precise phases run without demo noise: it can exceed the fixture clearance.
                return Target(ee + err, yaw, CLOSED, noise=0.0 if np.linalg.norm(err[:2]) < 0.01 else 1.0), done
            err = goal_tip - 0.003 * UP - tip
            t = Target(ee + err, yaw, CLOSED, speed=self.insert_speed, noise=0.0)
            return t, (pos[2] - goal[2] < 0.5 * task.z_tol) or self.timer[i] > 60
        if stage == 6:  # release
            return hold, self.timer[i] >= 6
        if stage == 7:  # retreat, then the next object (or finish)
            t = Target(np.array([*self.cmd_pos[i][:2], max(self.lift, goal[2] + 0.12)]), self.cmd_yaw[i], OPEN_K)
            if not self.reached(i, t):
                return t, False
            self.stage[i] = -1
            return t, True
        return hold, False


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
    "peg_insertion": InsertionPolicy,
    "gear_assembly": InsertionPolicy,
    "kitting": InsertionPolicy,
    "pcb_insertion": InsertionPolicy,
    "buttons": ButtonPolicy,
}


def make_policy(env, **kwargs) -> ScriptedPolicy:
    """The scripted expert for ``env.task``."""
    from franka_sim.tasks.insertion import InsertionTask

    cls = POLICIES.get(env.task.name)
    if cls is None and isinstance(env.task, InsertionTask):
        cls = InsertionPolicy
    if cls is None:
        raise ValueError(f"no scripted policy for task {env.task.name!r}")
    return cls(env, **kwargs)
