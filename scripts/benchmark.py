"""Measure env steps per second for each backend.

    uv run --extra warp python scripts/benchmark.py --task reach --num-envs 1 64 1024 --backends cpu warp
"""

import argparse
import time

import numpy as np

import franka_sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="reach")
    p.add_argument("--backends", nargs="+", default=["cpu"])
    p.add_argument("--num-envs", nargs="+", type=int, default=[1, 16])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--action-mode", default="joint_delta",
                   help="joint_delta skips IK, which runs on the host and dominates at large batch sizes")
    args = p.parse_args()

    for backend in args.backends:
        for n in args.num_envs:
            env = franka_sim.make(args.task, num_envs=n, backend=backend, action_mode=args.action_mode)
            env.reset(seed=0)
            action = np.zeros((n, env.action_dim))
            env.step(action)  # compile / warm up
            t = time.perf_counter()
            for _ in range(args.steps):
                env.step(action)
            dt = time.perf_counter() - t
            sps = n * args.steps / dt
            phys = sps * env.nsubsteps
            print(f"{backend:5s} envs={n:6d}  {sps:10.0f} env-steps/s  {phys:12.0f} physics-steps/s")
            env.close()


if __name__ == "__main__":
    main()
