"""Run the scripted expert on every environment and save one mp4 per episode.

    uv run --extra video python scripts/make_videos.py                      # all envs, 5 episodes each
    uv run --extra video python scripts/make_videos.py --envs hammer_nails --episodes 3 --camera front wrist

Each video tiles the chosen cameras side by side and overlays the episode's instruction and status.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import franka_sim
from franka_sim.policies import make_policy

# name -> (task, task_kwargs)
ENVS = {
    "reach": ("reach", {}),
    "hammer_nails": ("hammer_nails", {}),
    "peg_insertion": ("peg_insertion", {}),
    "peg_insertion_all": ("peg_insertion", {"mode": "all"}),
    "buttons": ("buttons", {}),
    "buttons_sequence": ("buttons", {"sequence_length": 3}),
    "gear_assembly": ("gear_assembly", {}),
    "kitting": ("kitting", {}),
    "kitting_all": ("kitting", {"mode": "all"}),
    "pcb_insertion": ("pcb_insertion", {}),
}


def overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.load_default(size=18)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    h = 24 * len(lines) + 8
    draw.rectangle([0, 0, img.width, h], fill=(0, 0, 0, 140))
    for k, line in enumerate(lines):
        draw.text((10, 6 + 24 * k), line, fill=(255, 255, 255, 255), font=font)
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="*", default=list(ENVS), choices=list(ENVS))
    p.add_argument("--episodes", type=int, default=5, help="episodes (= videos) per environment")
    p.add_argument("--camera", nargs="*", default=["front", "side", "wrist"])
    p.add_argument("--size", nargs=2, type=int, default=[384, 512], metavar=("H", "W"))
    p.add_argument("--post-success-steps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="videos")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.envs:
        task, kwargs = ENVS[name]
        n = args.episodes
        # One parallel env per episode; each records until success (+ a short settle) or timeout.
        env = franka_sim.make(task, num_envs=n, action_mode="ee_pose", seed=args.seed, task_kwargs=kwargs,
                              image_size=tuple(args.size))
        obs, _ = env.reset()
        policy = make_policy(env, noise=0.002, seed=args.seed)
        policy.reset(range(n), obs)
        instructions = env.instructions()
        frames = [[] for _ in range(n)]
        success_at = np.full(n, -1)
        active = np.ones(n, dtype=bool)
        fps = int(round(1 / env.control_dt))

        def capture(i, status):
            tiles = [env.render(cam, env_id=i) for cam in args.camera]
            lines = [f"{name}  |  episode {i}  |  t={env.elapsed_steps[i] * env.control_dt:4.1f}s  |  {status}",
                     instructions[i]]
            frames[i].append(overlay(np.concatenate(tiles, axis=1), lines))

        for i in range(n):
            capture(i, "running")
        for t in range(env.max_episode_steps):
            obs, reward, term, trunc, info = env.step(policy.act(obs))
            for i in np.flatnonzero(active):
                if info["success"][i] and success_at[i] < 0:
                    success_at[i] = t
                capture(i, "SUCCESS" if success_at[i] >= 0 else "running")
                if (success_at[i] >= 0 and t - success_at[i] >= args.post_success_steps) or trunc[i]:
                    active[i] = False
            if not active.any():
                break

        for i in range(n):
            path = out_dir / f"{name}_ep{i}.mp4"
            imageio.mimsave(path, frames[i], fps=fps, macro_block_size=8)
            status = f"success after {(success_at[i] + 1) * env.control_dt:.1f}s" if success_at[i] >= 0 else "FAILED"
            print(f"{path}  ({len(frames[i])} frames, {status})  '{instructions[i]}'")
        env.close()


if __name__ == "__main__":
    main()
