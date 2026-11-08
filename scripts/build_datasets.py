"""Record scripted demos for every task and convert each to a LeRobot dataset.

    uv run --extra lerobot python scripts/build_datasets.py                       # everything
    uv run --extra lerobot python scripts/build_datasets.py --tasks hammer_nails --episodes 100
    uv run --extra lerobot python scripts/build_datasets.py --namespace <hf-user> --jobs 4

Outputs ``data/<name>.hdf5`` (states + low-dim obs) and ``lerobot_data/<name>/`` (LeRobot v3,
repo id ``<namespace>/franka_sim_<name>``). Upload with scripts/to_lerobot.py --push.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# name -> (task, task_kwargs, balanced condition: number of targets to cycle through, or None)
DATASETS = {
    "reach": ("reach", {}, None),
    "hammer_nails": ("hammer_nails", {}, 4),
    "buttons_sequence": ("buttons", {"sequence_length": 2}, None),
    "peg_insertion": ("peg_insertion", {}, 3),
    "gear_assembly": ("gear_assembly", {}, None),
    "kitting": ("kitting", {}, None),
    "pcb_insertion": ("pcb_insertion", {}, 3),
}


def build(name: str, episodes: int, namespace: str, cameras: list[str], image_size: tuple[int, int],
          num_envs: int, seed: int, overwrite: bool) -> dict:
    import shutil

    import franka_sim
    from franka_sim.lerobot_export import convert
    from franka_sim.policies import make_policy
    from franka_sim.recording import DemoWriter, collect_demos

    task, task_kwargs, balanced = DATASETS[name]
    h5 = Path("data") / f"{name}.hdf5"
    root = Path("lerobot_data") / name
    if overwrite:
        h5.unlink(missing_ok=True)
        shutil.rmtree(root, ignore_errors=True)
    t0 = time.time()
    env = franka_sim.make(task, num_envs=num_envs, action_mode="ee_pose", seed=seed, task_kwargs=task_kwargs)
    policy = make_policy(env, noise=0.002, seed=seed)
    conditions = (lambda k: {"target": k % balanced}) if balanced else None
    with DemoWriter(h5, env, task_kwargs=task_kwargs, seed=seed) as w:
        stats = collect_demos(env, policy, episodes, w, conditions=conditions, seed=seed)
    env.close()
    t1 = time.time()
    ds = convert([h5], f"{namespace}/franka_sim_{name}", root, cameras=cameras, image_size=image_size,
                 progress=None)
    return {"name": name, "episodes": ds.num_episodes, "frames": ds.num_frames,
            "scripted_success": f"{stats.succeeded}/{stats.attempted}",
            "record_s": round(t1 - t0), "convert_s": round(time.time() - t1), "root": str(root)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--namespace", default="franka_sim", help="Hugging Face user/org for the repo ids")
    p.add_argument("--cameras", nargs="*", default=["front", "wrist"])
    p.add_argument("--image-size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--jobs", type=int, default=3, help="datasets built in parallel")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    kw = dict(episodes=args.episodes, namespace=args.namespace, cameras=args.cameras,
              image_size=tuple(args.image_size), num_envs=args.num_envs, seed=args.seed, overwrite=args.overwrite)
    with ProcessPoolExecutor(args.jobs) as pool:
        futures = {name: pool.submit(build, name, **kw) for name in args.tasks}
        results = []
        for name, fut in futures.items():
            res = fut.result()
            results.append(res)
            print(json.dumps(res), flush=True)
    summary_path = Path("lerobot_data") / "summary.json"
    summary_path.parent.mkdir(exist_ok=True)
    summary = {r["name"]: r for r in json.loads(summary_path.read_text())} if summary_path.exists() else {}
    summary.update({r["name"]: r for r in results})  # keep entries of datasets not rebuilt now
    summary_path.write_text(json.dumps([summary[k] for k in DATASETS if k in summary], indent=2))


if __name__ == "__main__":
    main()
