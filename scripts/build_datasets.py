"""Record scripted demos for every task and convert each to a LeRobot dataset.

    uv run --extra lerobot python scripts/build_datasets.py                       # the whole suite
    uv run --extra lerobot python scripts/build_datasets.py --tasks hammer_nails --episodes 100
    uv run --extra lerobot python scripts/build_datasets.py --namespace <hf-user> --jobs 4
    uv run --extra lerobot python scripts/build_datasets.py --hdf5-dir /mnt/big/demos --lerobot-dir /mnt/big/lerobot

Outputs ``<hdf5-dir>/<name>.hdf5`` (states + low-dim obs, default ``data/``) and
``<lerobot-dir>/<name>/`` (default ``lerobot_data/``; LeRobot v3,
repo id ``<namespace>/franka_sim_<name>``). Upload with scripts/to_lerobot.py --push.
"""

from __future__ import annotations

import argparse
import json
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from franka_sim.benchmark import EVAL_SEED, SUITE

# name -> (task, task_kwargs, balanced condition: number of targets to cycle through, or None).
# One dataset per benchmark task; single tasks cycle their target so every target is equally common.
DATASETS = {name: (name, {}, n or None) for name, (_kind, n) in SUITE.items()}


def build(name: str, episodes: int, namespace: str, cameras: list[str], image_size: tuple[int, int],
          num_envs: int, seed: int, overwrite: bool, action: str = "joint_pos", hdf5_dir: str = "data",
          lerobot_dir: str = "lerobot_data", slots=None) -> dict:
    import shutil

    task, task_kwargs, balanced = DATASETS[name]
    h5 = Path(hdf5_dir) / f"{name}.hdf5"
    h5.parent.mkdir(parents=True, exist_ok=True)
    root = Path(lerobot_dir) / name
    if overwrite:
        h5.unlink(missing_ok=True)
        shutil.rmtree(root, ignore_errors=True)
    slot = slots.get() if slots is not None else 0  # a terminal line for this job's progress bars
    try:
        return _build(name, task, task_kwargs, balanced, h5, root, episodes, namespace, cameras, image_size,
                      num_envs, seed, slot, action)
    finally:
        if slots is not None:
            slots.put(slot)


def _build(name, task, task_kwargs, balanced, h5, root, episodes, namespace, cameras, image_size, num_envs,
           seed, slot, action="joint_pos") -> dict:
    import franka_sim
    from franka_sim.lerobot_export import convert
    from franka_sim.policies import make_policy
    from franka_sim.recording import DemoWriter, collect_demos

    t0 = time.time()
    env = franka_sim.make(task, num_envs=num_envs, action_mode="ee_pose", seed=seed, task_kwargs=task_kwargs)
    policy = make_policy(env, noise=0.002, seed=seed)
    conditions = (lambda k: {"target": k % balanced}) if balanced else None
    with DemoWriter(h5, env, task_kwargs=task_kwargs, seed=seed) as w:
        stats = collect_demos(env, policy, episodes, w, conditions=conditions, seed=seed,
                              progress={"desc": f"{name}: record", "position": slot, "leave": False})
    env.close()
    t1 = time.time()
    ds = convert([h5], f"{namespace}/franka_sim_{name}", root, cameras=cameras, image_size=image_size,
                 action=action, progress=None, bar={"desc": f"{name}: convert", "position": slot, "leave": False})
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
    p.add_argument("--action", default="joint_pos", choices=["joint_pos", "ee_pose", "delta_ee_pose"],
                   help="representation stored as `action` (all three are always in action.*)")
    p.add_argument("--hdf5-dir", default="data", help="where the recorded HDF5 demos go")
    p.add_argument("--lerobot-dir", default="lerobot_data", help="where the LeRobot datasets go")
    args = p.parse_args()
    if args.seed + 10 * args.episodes >= EVAL_SEED:
        raise SystemExit(f"--seed must stay well below the evaluation seeds ({EVAL_SEED})")

    kw = dict(episodes=args.episodes, namespace=args.namespace, cameras=args.cameras,
              image_size=tuple(args.image_size), num_envs=args.num_envs, action=args.action, seed=args.seed, overwrite=args.overwrite,
              hdf5_dir=args.hdf5_dir, lerobot_dir=args.lerobot_dir)
    from tqdm import tqdm

    jobs = max(1, min(args.jobs, len(args.tasks)))
    with multiprocessing.Manager() as manager, ProcessPoolExecutor(jobs) as pool:
        slots = manager.Queue()  # one terminal line per worker, for its progress bars
        for k in range(jobs):
            slots.put(k)
        futures = {pool.submit(build, name, slots=slots, **kw): name for name in args.tasks}
        results = []
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            tqdm.write(json.dumps(res))
    summary_path = Path(args.lerobot_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {r["name"]: r for r in json.loads(summary_path.read_text())} if summary_path.exists() else {}
    summary.update({r["name"]: r for r in results})  # keep entries of datasets not rebuilt now
    summary_path.write_text(json.dumps([summary[k] for k in DATASETS if k in summary], indent=2))


if __name__ == "__main__":
    main()
