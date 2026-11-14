"""Evaluate a policy on the franka_sim benchmark suite (see src/franka_sim/benchmark.py).

    # reference: the scripted expert with privileged state (the upper bound of the suite)
    uv run python scripts/evaluate.py --policy scripted

    # your policy: a factory ``make(env, **kwargs) -> policy`` with reset(env_ids, obs) / act(obs)
    uv run python scripts/evaluate.py --policy my_pkg.policies:make --policy-kwargs '{"ckpt": "run1"}' \
        --tasks hammer_nails hammer_nails_long --episodes 50 --cameras front wrist --mask-camera top

    # a policy served elsewhere (scripts/serve_policy.py, e.g. on a GPU box)
    uv run --extra server python scripts/evaluate.py --policy remote:ws://gpu-box:9000

Prints a table and writes every metric to --out (JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from franka_sim.benchmark import EVAL_SEED, SUITE, evaluate, scripted_factory


def load_factory(spec: str, kwargs: dict):
    if spec == "scripted":
        return scripted_factory(**kwargs)
    from franka_sim.dagger.remote import load_factory as _load

    try:
        return _load(spec, kwargs)  # "module.path:factory" or "remote:ws://host:port"
    except ValueError as e:
        raise SystemExit(str(e)) from None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="scripted")
    p.add_argument("--policy-kwargs", default="{}", help="JSON dict passed to the policy factory")
    p.add_argument("--tasks", nargs="*", default=list(SUITE))
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--seed", type=int, default=EVAL_SEED)
    p.add_argument("--obs-mode", choices=["vision", "privileged"], default=None,
                   help="default: privileged for --policy scripted, vision otherwise")
    p.add_argument("--cameras", nargs="*", default=None, help="default: none for scripted, front wrist otherwise")
    p.add_argument("--image-size", nargs=2, type=int, default=[256, 256])
    p.add_argument("--mask-camera", default=None, help="add obs['target_mask'] rendered from this camera")
    p.add_argument("--action-mode", default=None, help="default: ee_pose for scripted, joint_pos otherwise")
    p.add_argument("--backend", default="cpu")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--out", default="results/eval.json")
    args = p.parse_args()

    scripted = args.policy == "scripted"
    factory = load_factory(args.policy, json.loads(args.policy_kwargs))
    obs_mode = args.obs_mode or ("privileged" if scripted else "vision")
    cameras = args.cameras if args.cameras is not None else ([] if scripted else ["front", "wrist"])
    action_mode = args.action_mode or ("ee_pose" if scripted else "joint_pos")

    results = []
    print(f"{'task':<22}{'kind':<8}{'success':>9}{'wrong':>8}{'progress':>10}{'steps':>8}  per-target success")
    for task in args.tasks:
        r = evaluate(task, factory, episodes=args.episodes, num_envs=args.num_envs, seed=args.seed,
                     obs_mode=obs_mode, cameras=cameras, image_size=tuple(args.image_size),
                     mask_camera=args.mask_camera, action_mode=action_mode, backend=args.backend,
                     progress=print if args.verbose else None)
        results.append(r.to_dict())
        steps = f"{r.mean_success_steps:.0f}" if r.mean_success_steps is not None else "-"
        per = " ".join(f"{k}:{v:.2f}" for k, v in r.per_target_success.items())
        print(f"{task:<22}{r.kind:<8}{r.success_rate:>9.2f}{r.wrong_target_rate:>8.2f}{r.mean_progress:>10.2f}"
              f"{steps:>8}  {per}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"policy": args.policy, "policy_kwargs": json.loads(args.policy_kwargs), "seed": args.seed,
            "obs_mode": obs_mode, "cameras": cameras, "mask_camera": args.mask_camera, "action_mode": action_mode}
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
