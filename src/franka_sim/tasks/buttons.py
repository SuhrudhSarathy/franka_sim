"""Press colored buttons on a panel, either one target or a target sequence."""

from __future__ import annotations

import mujoco
import numpy as np

from franka_sim.backends.base import Backend
from franka_sim.tasks import objects as obj
from franka_sim.tasks.base import Evaluation, Task, yaw_quat


class ButtonPressTask(Task):
    """Press ``sequence_length`` distinct buttons in a given order.

    The sequence is sampled per episode, or set with ``options={"sequence": [2, 0]}`` (or
    ``{"target": i}`` when ``sequence_length == 1``). Pressing a button that is not next in the
    sequence fails the episode (``info["failed"]``).

    Observations: ``button_pos`` (top of each cap, flattened), ``button_depth`` (0 = up, 1 = fully
    pressed), ``sequence_onehot`` (sequence_length * num_buttons), ``progress`` (buttons done so far),
    ``next_button_pos``.
    """

    name = "buttons"

    def __init__(self, num_buttons: int = 4, sequence_length: int = 1, press_depth: float = 0.7,
                 randomize: bool = True):
        if not 1 <= sequence_length <= num_buttons:
            raise ValueError("sequence_length must be between 1 and num_buttons")
        self.num_buttons = num_buttons
        self.sequence_length = sequence_length
        self.press_depth = press_depth
        self.randomize = randomize
        self.max_episode_steps = 80 + 70 * sequence_length

    def build(self, spec: mujoco.MjSpec) -> None:
        obj.add_button_panel(spec, self.num_buttons)

    def bind(self, model, num_envs):
        super().bind(model, num_envs)
        self.panel_mocap = model.body("panel").mocapid[0]
        self.button_qadr = np.array([self.joint_qadr(f"button{i}_joint") for i in range(self.num_buttons)])
        self.button_sites = np.array([model.site(f"button{i}_top").id for i in range(self.num_buttons)])
        self.sequence = np.zeros((num_envs, self.sequence_length), dtype=np.int64)
        self.progress = np.zeros(num_envs, dtype=np.int64)
        self.failed = np.zeros(num_envs, dtype=bool)
        self.was_pressed = np.zeros((num_envs, self.num_buttons), dtype=bool)

    def reset(self, env_ids, qpos, rng, options=None):
        options = options or {}
        n = len(env_ids)
        for k, e in enumerate(env_ids):
            if "sequence" in options:
                self.sequence[e] = options["sequence"]
            elif "target" in options:
                self.sequence[e] = [options["target"]]
            else:
                self.sequence[e] = rng.permutation(self.num_buttons)[: self.sequence_length]
        self.progress[env_ids] = 0
        self.failed[env_ids] = False
        self.was_pressed[env_ids] = False
        mocap = self.default_mocap_pos(n)
        quat = self.default_mocap_quat(n)
        if self.randomize:
            mocap[:, self.panel_mocap, :2] = rng.uniform([0.45, -0.2], [0.65, 0.2], size=(n, 2))
            quat[:, self.panel_mocap] = yaw_quat(rng.uniform(-0.6, 0.6, size=n))
        else:
            mocap[:, self.panel_mocap, :2] = [0.57, 0.0]
        return {"mocap_pos": mocap, "mocap_quat": quat}

    def depth(self, sim: Backend) -> np.ndarray:
        return -sim.read("qpos")[:, self.button_qadr] / obj.BUTTON_TRAVEL

    def _next(self) -> np.ndarray:
        idx = np.minimum(self.progress, self.sequence_length - 1)
        return self.sequence[np.arange(self.num_envs), idx]

    def observe(self, sim: Backend):
        site = sim.read("site_xpos")[:, self.button_sites]
        onehot = np.eye(self.num_buttons)[self.sequence].reshape(self.num_envs, -1)
        return {
            "button_pos": site.reshape(self.num_envs, -1),
            "button_depth": self.depth(sim),
            "sequence_onehot": onehot,
            "progress": self.progress[:, None].astype(np.float64),
            "next_button_pos": site[np.arange(self.num_envs), self._next()],
        }

    def evaluate(self, sim: Backend) -> Evaluation:
        pressed = self.depth(sim) >= self.press_depth
        new = pressed & ~self.was_pressed
        self.was_pressed = pressed
        for e in np.flatnonzero(new.any(axis=1)):
            for b in np.flatnonzero(new[e]):
                if self.progress[e] < self.sequence_length and b == self.sequence[e, self.progress[e]]:
                    self.progress[e] += 1
                elif self.progress[e] < self.sequence_length:
                    self.failed[e] = True
        success = (self.progress >= self.sequence_length) & ~self.failed
        reward = self.progress / self.sequence_length - self.failed.astype(np.float64)
        return Evaluation(reward, success, {"button_depth": np.clip(self.depth(sim), 0, 1),
                                            "progress": self.progress.copy(), "failed": self.failed.copy()})

    def conditioning(self):
        return {"sequence": self.sequence.copy()}

    def mask_bodies(self, cond):
        if self.sequence_length != 1:
            return None  # multi-step: no mask
        return [f"button{int(np.ravel(cond['sequence'])[0])}"]

    def instruction(self, env_id):
        colors = [obj.COLOR_NAMES[b % len(obj.COLOR_NAMES)] for b in self.sequence[env_id]]
        if len(colors) == 1:
            return f"press the {colors[0]} button"
        return "press the " + ", then the ".join(colors) + " buttons in order"
