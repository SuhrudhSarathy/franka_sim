"""A framework-agnostic behavior-cloning dataset over recorded HDF5 demos.

``BCDataset`` implements ``__len__`` / ``__getitem__`` returning dicts of numpy arrays, so it plugs
straight into ``torch.utils.data.DataLoader`` (or JAX / plain numpy loops). Each sample is an
observation history of ``obs_horizon`` steps and an action chunk of ``action_horizon`` steps (as
used by Diffusion Policy / ACT), padded by repeating the first/last step at episode boundaries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from franka_sim.recording import iter_episodes


class BCDataset:
    """Sliding windows over every (successful) episode, held in memory.

    Args:
        path: HDF5 file written by ``DemoWriter``.
        obs_keys: observation keys to include (default: all). Images stay uint8 ``(T, H, W, 3)``.
        action_key: ``"actions"`` (the env action), or ``"commands/joint_pos"`` / ``"commands/ee_pose"`` /
            ``"commands/delta_ee_pose"``
            to train on another action representation (the gripper is appended as the last dim).
        obs_horizon / action_horizon: window lengths.
        cond_keys: conditioning variables to include under ``cond/<key>`` (default: all).
        only_success: skip failed episodes.
    """

    def __init__(self, path: str | Path, obs_keys=None, action_key: str = "actions", obs_horizon: int = 1,
                 action_horizon: int = 1, cond_keys=None, only_success: bool = True):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.episodes = []
        self.instructions = []
        index = []
        for ep in iter_episodes(path, only_success=only_success):
            obs = {k: v for k, v in ep["obs"].items() if obs_keys is None or k in obs_keys}
            if action_key == "actions":
                actions = ep["actions"]
            else:
                group, name = action_key.split("/")
                actions = np.concatenate([ep[group][name], ep["commands"]["gripper"][:, None]], axis=1)
            cond = {k: v for k, v in ep["cond"].items() if cond_keys is None or k in cond_keys}
            e = len(self.episodes)
            self.episodes.append({"obs": obs, "actions": actions.astype(np.float32), "cond": cond})
            self.instructions.append(ep["attrs"]["instruction"])
            index.extend((e, t) for t in range(len(actions)))
        if not self.episodes:
            raise ValueError(f"no episodes in {path}")
        self.index = np.array(index, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        e, t = self.index[i]
        ep = self.episodes[e]
        T = len(ep["actions"])
        obs_t = np.clip(np.arange(t - self.obs_horizon + 1, t + 1), 0, T - 1)
        act_t = np.clip(np.arange(t, t + self.action_horizon), 0, T - 1)
        sample = {f"obs/{k}": v[obs_t] for k, v in ep["obs"].items()}
        sample["actions"] = ep["actions"][act_t]
        sample["action_pad_mask"] = (np.arange(t, t + self.action_horizon) < T)
        sample.update({f"cond/{k}": np.asarray(v) for k, v in ep["cond"].items()})
        sample["episode"] = e
        return sample

    def instruction(self, i: int) -> str:
        return self.instructions[self.index[i][0]]

    def stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Per-dimension min / max / mean / std of actions and non-image observations (for normalization)."""
        out = {}
        keys = {"actions": [ep["actions"] for ep in self.episodes]}
        for k in self.episodes[0]["obs"]:
            if self.episodes[0]["obs"][k].dtype != np.uint8:
                keys[f"obs/{k}"] = [ep["obs"][k] for ep in self.episodes]
        for k, arrs in keys.items():
            a = np.concatenate(arrs, axis=0)
            out[k] = {"min": a.min(0), "max": a.max(0), "mean": a.mean(0), "std": a.std(0)}
        return out
