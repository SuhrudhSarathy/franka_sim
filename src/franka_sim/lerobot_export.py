"""Convert recorded HDF5 demos (see ``recording.py``) into a LeRobot dataset (v3 format).

Requires the ``lerobot`` extra: ``uv sync --extra lerobot``.

Per frame the dataset holds:

* ``observation.state`` (8): 7 arm joint positions + gripper width, the robot's proprioception.
* ``observation.ee_pose`` (7): end-effector position + quaternion (wxyz).
* ``observation.environment_state`` (n): privileged object / task state (every non-robot, non-image
  observation of the task, flattened in a fixed order; names are listed in ``meta/info.json``).
* ``observation.images.<camera>``: RGB video per camera. Images are re-rendered from the recorded
  simulator states, so any camera and resolution can be exported from a states-only recording.
* ``action`` (8): the command in the chosen representation, ``joint_pos`` (7 joint targets + gripper
  closure) or ``ee_pose`` (xyz + wxyz quaternion + gripper closure).
* ``action.joint_pos`` / ``action.ee_pose``: both representations, whichever ``action`` is.
* ``observation.images.target_mask``: binary mask (0/255, stored losslessly as PNG, 3 identical
  channels) of the episode's target entity seen from ``mask_camera`` (default the top camera): the
  nail to hit, the connector to use, the peg to insert, the button to press. Amodal: the arm never
  hides it. Only for single-target tasks; multi-step tasks (all gears, kitting lists, button
  sequences) and reach have no mask. ``FrankaEnv.render_target_mask`` gives the same image live.
* ``cond.<key>``: the episode's conditioning variables (e.g. ``cond.target`` = nail index).
* DAgger recordings (``franka_sim.dagger``) add ``dagger.actor`` (0 learner, 1 human, 2 scripted
  expert) and ``dagger.intervention`` (a human was in control) per frame, so training can weight or
  mask the loss without cutting episodes apart; plain demo files in the same dataset count as
  ``actor = 2``. When every file has scripted expert labels, ``expert.joint_pos`` and
  ``expert.ee_pose`` (8 each, gripper included) hold them, and ``labels="expert"`` makes them the
  ``action``.
* ``next.reward``, ``next.success``, ``next.done``.
* ``task``: the natural-language instruction, e.g. "pick up the hammer and hit the leftmost nail".
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import mujoco
import numpy as np

import franka_sim
from franka_sim.masks import render_mask, target_geoms
from franka_sim.recording import dataset_info, iter_episodes

MASK_KEY = "observation.images.target_mask"

ROBOT_KEYS = {"joint_pos", "joint_vel", "gripper_width", "ee_pos", "ee_quat"}
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def _env_state_keys(obs: dict) -> list[str]:
    return sorted(k for k, v in obs.items() if k not in ROBOT_KEYS and not k.startswith("rgb_"))


def _names(key: str, dim: int) -> list[str]:
    return [key] if dim == 1 else [f"{key}_{j}" for j in range(dim)]


def _file_kind(path: Path) -> tuple[bool, bool]:
    """(has DAgger actor/intervention arrays, has scripted expert labels) of a recording."""
    import h5py

    with h5py.File(path, "r") as f:
        demos = list(f["data"].keys())
        if not demos:
            return False, False
        g = f["data"][demos[0]]
        return "actor" in g, "expert" in g and "joint_pos" in g["expert"]


def build_features(sample_ep: dict, cameras: Sequence[str], image_size: tuple[int, int], use_videos: bool) -> dict:
    obs = sample_ep["obs"]
    env_keys = _env_state_keys(obs)
    env_names = [n for k in env_keys for n in _names(k, int(np.prod(obs[k].shape[1:])))]
    grip = ["gripper"]
    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + ["gripper_width"]},
        "observation.ee_pose": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"]},
        "observation.environment_state": {"dtype": "float32", "shape": (len(env_names),), "names": env_names},
        "action": {"dtype": "float32", "shape": (8,), "names": None},
        "action.joint_pos": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES + grip},
        "action.ee_pose": {"dtype": "float32", "shape": (8,), "names": ["x", "y", "z", "qw", "qx", "qy", "qz"] + grip},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": None},
        "next.success": {"dtype": "bool", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for k, v in sample_ep["cond"].items():
        shape = tuple(np.asarray(v).shape) or (1,)
        dtype = "int64" if np.issubdtype(np.asarray(v).dtype, np.integer) else "float32"
        features[f"cond.{k}"] = {"dtype": dtype, "shape": shape, "names": None}
    h, w = image_size
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if use_videos else "image", "shape": (h, w, 3), "names": ["height", "width", "channels"],
        }
    return features, env_keys


class _Renderer:
    """Re-renders recorded simulator states for one task configuration."""

    def __init__(self, info: dict, image_size: tuple[int, int]):
        task_kwargs = info.get("task_kwargs") or {}
        if isinstance(task_kwargs, str):
            task_kwargs = json.loads(task_kwargs)
        self.env = franka_sim.make(info["task"], task_kwargs=task_kwargs, image_size=image_size)
        self.task = self.env.task
        self.model = self.env.model
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, *image_size)

    def render(self, states: dict, t: int, cameras: Sequence[str],
               mask: tuple[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """RGB per camera; plus ``MASK_KEY`` when ``mask = (camera, target geom ids)``."""
        d = self.data
        d.qpos[:], d.qvel[:] = states["qpos"][t], states["qvel"][t]
        d.mocap_pos[:], d.mocap_quat[:] = states["mocap_pos"][t], states["mocap_quat"][t]
        mujoco.mj_forward(self.model, d)
        out = {}
        for cam in cameras:
            self.renderer.update_scene(d, camera=cam)
            out[cam] = self.renderer.render().copy()
        if mask is not None:
            m = render_mask(self.renderer, d, mask[0], mask[1])
            out[MASK_KEY] = np.repeat(m[..., None].astype(np.uint8) * 255, 3, axis=2)
        return out

    def close(self):
        self.renderer.close()
        self.env.close()


def convert(hdf5_paths: Sequence[str | Path], repo_id: str, root: str | Path, *,
            cameras: Sequence[str] = ("front", "wrist"), image_size: tuple[int, int] = (256, 256),
            action: str = "joint_pos", labels: str = "executed", use_videos: bool = True, only_success: bool = True,
            mask_camera: str | None = "top", encoder_preset: int | None = 10,
            progress: Callable[[str], None] | None = print):
    """Write every (successful) episode of ``hdf5_paths`` into a new LeRobot dataset at ``root``.

    All files must come from the same task configuration (same observation layout). Returns the
    finalized ``LeRobotDataset``; upload it with ``dataset.push_to_hub()``.

    ``labels``: ``"executed"`` (``action`` = what was executed: the expert's commands in demos, the
    operator's on intervention steps, the learner's own elsewhere) or ``"expert"`` (``action`` = the
    scripted expert's label on every frame; needs DAgger files recorded with expert labels).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if action not in ("joint_pos", "ee_pose"):
        raise ValueError("action must be 'joint_pos' or 'ee_pose'")
    if labels not in ("executed", "expert"):
        raise ValueError("labels must be 'executed' or 'expert'")
    paths = [Path(p) for p in hdf5_paths]
    kinds = [_file_kind(p) for p in paths]
    with_dagger = any(k[0] for k in kinds)
    with_expert = all(k[1] for k in kinds)
    if labels == "expert" and not with_expert:
        raise ValueError("labels='expert' needs scripted expert labels in every file (DAgger recordings)")
    infos = [dataset_info(p) for p in paths]
    if len({(i["task"], json.dumps(i.get("task_kwargs", {}), sort_keys=True)) for i in infos}) != 1:
        raise ValueError("all files must come from the same task configuration")
    info = infos[0]
    fps = int(round(info["control_freq"]))

    first = next(iter_episodes(paths[0], only_success=only_success))
    features, env_keys = build_features(first, cameras, image_size, use_videos)
    renderer = _Renderer(info, image_size) if cameras or mask_camera else None
    with_mask = mask_camera is not None and renderer.task.mask_bodies(first["cond"]) is not None
    if with_mask:
        features[MASK_KEY] = {"dtype": "image", "shape": (*image_size, 3), "names": ["height", "width", "channels"]}
    features["action"]["names"] = features[f"action.{action}"]["names"]
    if with_dagger:
        features["dagger.actor"] = {"dtype": "int64", "shape": (1,), "names": None}
        features["dagger.intervention"] = {"dtype": "bool", "shape": (1,), "names": None}
    if with_expert:
        for rep_ in ("joint_pos", "ee_pose"):
            features[f"expert.{rep_}"] = {"dtype": "float32", "shape": (8,), "names": features[f"action.{rep_}"]["names"]}
    root = Path(root)
    if root.exists():
        if renderer is not None:
            renderer.close()
        raise FileExistsError(f"{root} already exists; choose a new --root or delete it")
    kwargs = {}
    if use_videos and encoder_preset is not None:
        from lerobot.configs.video import RGBEncoderConfig

        # LeRobot's default AV1 encoding is slow; a faster SVT-AV1 preset keeps the same format.
        kwargs["rgb_encoder"] = RGBEncoderConfig(preset=encoder_preset)
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=root, robot_type="franka_panda",
                               use_videos=use_videos, image_writer_threads=4, **kwargs)
    n_eps = 0
    try:
        for path in paths:
            for ep in iter_episodes(path, only_success=only_success):
                T = int(ep["attrs"]["num_samples"])
                obs, cmd = ep["obs"], ep["commands"]
                instruction = str(ep["attrs"]["instruction"])
                env_state = np.concatenate([obs[k].reshape(T, -1) for k in env_keys], axis=1).astype(np.float32)
                state = np.concatenate([obs["joint_pos"], obs["gripper_width"]], axis=1).astype(np.float32)
                ee = np.concatenate([obs["ee_pos"], obs["ee_quat"]], axis=1).astype(np.float32)
                grip = cmd["gripper"].reshape(T, 1).astype(np.float32)
                act = {"joint_pos": np.concatenate([cmd["joint_pos"], grip], 1).astype(np.float32),
                       "ee_pose": np.concatenate([cmd["ee_pose"], grip], 1).astype(np.float32)}
                if "actor" in ep:
                    actor = ep["actor"].astype(np.int64).reshape(T, 1)
                    intervention = ep["intervention"].astype(bool).reshape(T, 1)
                else:  # a plain demo file: recorded by the scripted expert
                    actor = np.full((T, 1), 2, dtype=np.int64)
                    intervention = np.zeros((T, 1), dtype=bool)
                expert = None
                if with_expert:
                    eg = ep["expert"]["gripper"].reshape(T, 1).astype(np.float32)
                    expert = {r: np.concatenate([ep["expert"][r], eg], 1).astype(np.float32)
                              for r in ("joint_pos", "ee_pose")}
                label = expert if labels == "expert" else act
                cond = {k: np.atleast_1d(np.asarray(v)) for k, v in ep["cond"].items()}
                mask = (mask_camera, target_geoms(renderer.model, renderer.task.mask_bodies(ep["cond"]))) \
                    if with_mask else None
                for t in range(T):
                    frame = {
                        "observation.state": state[t],
                        "observation.ee_pose": ee[t],
                        "observation.environment_state": env_state[t],
                        "action": label[action][t],
                        "action.joint_pos": act["joint_pos"][t],
                        "action.ee_pose": act["ee_pose"][t],
                        "next.reward": np.array([ep["rewards"][t]], dtype=np.float32),
                        "next.success": np.array([bool(ep["success"][t])]),
                        "next.done": np.array([bool(ep["dones"][t])]),
                        "task": instruction,
                    }
                    if with_dagger:
                        frame["dagger.actor"] = actor[t]
                        frame["dagger.intervention"] = intervention[t]
                    if expert is not None:
                        frame["expert.joint_pos"] = expert["joint_pos"][t]
                        frame["expert.ee_pose"] = expert["ee_pose"][t]
                    for k, v in cond.items():
                        frame[f"cond.{k}"] = v.astype(np.int64 if features[f"cond.{k}"]["dtype"] == "int64" else np.float32)
                    if renderer is not None:
                        for key, img in renderer.render(ep["states"], t, cameras, mask).items():
                            frame[key if key == MASK_KEY else f"observation.images.{key}"] = img
                    ds.add_frame(frame)
                ds.save_episode(parallel_encoding=False)  # a process pool per episode costs seconds
                n_eps += 1
                if progress:
                    progress(f"  episode {n_eps}: {T} frames  '{instruction}'")
    finally:
        if renderer is not None:
            renderer.close()
    ds.finalize()
    return ds
