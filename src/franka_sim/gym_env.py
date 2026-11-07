"""Single-env ``gymnasium.Env`` wrapper around ``FrankaEnv``."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from franka_sim.env import FrankaEnv


class FrankaGymEnv(gym.Env):
    """A standard (unbatched) Gymnasium environment. Keyword arguments go to ``FrankaEnv``."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(self, task="reach", render_mode: str | None = None, render_camera: str = "front", **kwargs):
        kwargs.pop("num_envs", None)
        self.env = FrankaEnv(task, num_envs=1, **kwargs)
        self.render_mode = render_mode
        self.render_camera = render_camera
        self.metadata = {**self.metadata, "render_fps": int(round(1 / self.env.control_dt))}

        lo = np.full(self.env.action_dim, -np.inf, dtype=np.float32)
        hi = np.full(self.env.action_dim, np.inf, dtype=np.float32)
        lo[-1], hi[-1] = 0.0, 1.0
        self.action_space = spaces.Box(lo, hi, dtype=np.float32)

        obs, _ = self.env.reset()
        self.observation_space = spaces.Dict({
            k: (spaces.Box(0, 255, v.shape[1:], dtype=np.uint8) if k.startswith("rgb_")
                else spaces.Box(-np.inf, np.inf, v.shape[1:], dtype=np.float32))
            for k, v in obs.items()
        })

    @staticmethod
    def _unbatch(obs: dict) -> dict:
        return {k: (v[0] if k.startswith("rgb_") else v[0].astype(np.float32)) for k, v in obs.items()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset(seed=seed, options=options)
        info = {k: np.asarray(v)[0] for k, v in info.items()}
        info["instruction"] = self.env.instructions()[0]
        return self._unbatch(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(np.asarray(action)[None])
        info = {k: np.asarray(v)[0] for k, v in info.items()}
        return self._unbatch(obs), float(reward[0]), bool(terminated[0]), bool(truncated[0]), info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.env.render(self.render_camera)
        return None

    def close(self):
        self.env.close()
