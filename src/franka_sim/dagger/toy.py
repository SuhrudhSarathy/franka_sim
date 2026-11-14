"""A deliberately imperfect learner, to try the DAgger loop before a real model exists.

``biased_expert`` is the scripted expert with a constant position offset (and optional noise) on
every command, converted to the env's action mode: it reaches for the wrong spot, so insertions
and strikes miss unless someone corrects it. It reads privileged observations, so use it with
``obs_mode="privileged"``::

    uv run --extra teleop mjpython scripts/dagger.py teleop --task peg_insertion \\
        --policy franka_sim.dagger.toy:biased_expert --policy-kwargs '{"bias": [0, 0.015, 0]}' \\
        --obs-mode privileged --viewer
"""

from __future__ import annotations

import numpy as np

from franka_sim.policies import make_policy


class BiasedExpert:
    def __init__(self, env, bias=(0.0, 0.015, 0.0), noise: float = 0.0, seed: int = 0):
        self.env = env
        self.action_mode = env.action_mode
        self.expert = make_policy(env, noise=noise, seed=seed, any_env_mode=True)
        self.bias = np.asarray(bias, dtype=float)

    def reset(self, env_ids, obs) -> None:
        self.expert.reset(env_ids, obs)

    def interrupt(self, env_ids, obs) -> None:
        """Someone else drove meanwhile: continue from the robot's commanded pose."""
        self.expert.track(self.env.last_command["ee_pose"], env_ids)

    def act(self, obs) -> np.ndarray:
        a = self.expert.act(obs)
        a[:, :3] += self.bias
        return self.env.convert_action(a, "ee_pose")


def biased_expert(env, **kwargs) -> BiasedExpert:
    return BiasedExpert(env, **kwargs)
