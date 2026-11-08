"""Record scripted demonstrations to HDF5.

Examples:
    # 40 hammer demos, balanced over the 4 nails, with front + wrist cameras
    uv run python scripts/record_demos.py --task hammer_nails --episodes 40 --condition balanced \
        --cameras front wrist --out data/hammer.hdf5

    # Only episodes that hit the third nail from the left (index 2)
    uv run python scripts/record_demos.py --task hammer_nails --episodes 10 --condition target=2

    # Two-button sequences on 8 parallel envs
    uv run python scripts/record_demos.py --task buttons --task-kwargs sequence_length=2 --num-envs 8
"""

from __future__ import annotations

import argparse
import ast
import time

import franka_sim
from franka_sim.policies import make_policy
from franka_sim.recording import DemoWriter, collect_demos


def parse_kv(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="hammer_nails")
    p.add_argument("--task-kwargs", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--backend", default="cpu", choices=["cpu", "warp"])
    p.add_argument("--cameras", nargs="*", default=["front", "wrist"])
    p.add_argument("--image-size", nargs=2, type=int, default=[128, 128], metavar=("H", "W"))
    p.add_argument("--condition", default=None,
                   help="'balanced' to cycle the target over all choices, or KEY=VALUE (e.g. target=2, "
                        "sequence=[3,1]) to fix it; default: random per episode")
    p.add_argument("--noise", type=float, default=0.002, help="action noise std-dev (m) for demo diversity")
    p.add_argument("--keep-failures", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output .hdf5 (default: data/<task>.hdf5)")
    args = p.parse_args()

    task_kwargs = parse_kv(args.task_kwargs)
    env = franka_sim.make(args.task, num_envs=args.num_envs, backend=args.backend, action_mode="ee_pose",
                          cameras=args.cameras, image_size=tuple(args.image_size), seed=args.seed,
                          task_kwargs=task_kwargs)
    policy = make_policy(env, noise=args.noise, seed=args.seed)

    conditions = None
    if args.condition == "balanced":
        n_choices = next(getattr(env.task, a) for a in ("num_nails", "num_pegs", "num_buttons")
                         if hasattr(env.task, a))
        if args.task == "buttons" and getattr(env.task, "sequence_length", 1) > 1:
            raise SystemExit("--condition balanced supports single targets; pass --condition sequence=[...]")
        conditions = lambda k: {"target": k % n_choices}  # noqa: E731
    elif args.condition:
        fixed = parse_kv([args.condition])
        conditions = lambda k: fixed  # noqa: E731

    out = args.out or f"data/{args.task}.hdf5"
    t0 = time.time()

    def report(k, ep, ok):
        cond = {k: v.tolist() for k, v in ep.cond.items()}
        print(f"episode {k:4d}  {'success' if ok else 'FAILED '}  steps={len(ep):4d}  cond={cond}  "
              f"'{ep.instruction}'")

    with DemoWriter(out, env, task_kwargs=task_kwargs, backend=args.backend, seed=args.seed) as writer:
        stats = collect_demos(env, policy, args.episodes, writer, conditions=conditions,
                              keep_failures=args.keep_failures, on_episode=report, seed=args.seed)
    env.close()
    print(f"\nsaved {stats.saved} episodes ({sum(stats.lengths)} steps) to {out} in {time.time() - t0:.1f}s; "
          f"scripted success rate {stats.success_rate:.0%} ({stats.succeeded}/{stats.attempted})")


if __name__ == "__main__":
    main()
