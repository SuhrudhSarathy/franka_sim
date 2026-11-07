"""Render recorded episodes to mp4 by replaying their saved simulator states.

    uv run --extra video python scripts/replay.py data/hammer_nails.hdf5 --episodes 0 1 2 --camera front
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

import franka_sim
from franka_sim.recording import dataset_info, load_episode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--episodes", nargs="*", type=int, default=[0])
    p.add_argument("--camera", nargs="*", default=["front"], help="one or more cameras, tiled side by side")
    p.add_argument("--size", nargs=2, type=int, default=[384, 512], metavar=("H", "W"))
    p.add_argument("--out-dir", default="videos")
    args = p.parse_args()

    info = dataset_info(args.path)
    task_kwargs = info.get("task_kwargs", {})
    if isinstance(task_kwargs, str):
        task_kwargs = json.loads(task_kwargs)
    env = franka_sim.make(info["task"], task_kwargs=task_kwargs, image_size=tuple(args.size))
    model = env.model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, *args.size)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.episodes:
        ep = load_episode(args.path, idx)
        st = ep["states"]
        frames = []
        for t in range(len(st["qpos"])):
            data.qpos[:], data.qvel[:] = st["qpos"][t], st["qvel"][t]
            data.mocap_pos[:], data.mocap_quat[:] = st["mocap_pos"][t], st["mocap_quat"][t]
            mujoco.mj_forward(model, data)
            tiles = []
            for cam in args.camera:
                renderer.update_scene(data, camera=cam)
                tiles.append(renderer.render())
            frames.append(np.concatenate(tiles, axis=1))
        path = out_dir / f"{Path(args.path).stem}_ep{idx}.mp4"
        imageio.mimsave(path, frames, fps=int(round(info["control_freq"])))
        print(f"{path}: {len(frames)} frames, success={ep['attrs']['success']}, '{ep['attrs']['instruction']}'")
    env.close()


if __name__ == "__main__":
    main()
