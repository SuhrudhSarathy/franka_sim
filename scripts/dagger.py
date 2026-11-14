"""DAgger data collection: human-gated (keyboard teleop) or automatic (scripted expert labels).

Each run is one DAgger round and writes one HDF5 file (the usual demo format plus ``actor``,
``intervention`` and ``expert/*`` per step; see ``franka_sim.dagger``). Train on the union of all
rounds, then run the next round with the new policy.

Human-gated (HG-DAgger): the policy drives, you take over with the keyboard when it goes wrong::

    # macOS: --viewer needs mjpython; the teleop console works either way
    uv run --extra teleop --extra server mjpython scripts/dagger.py teleop --task peg_insertion \\
        --policy my_pkg.policies:make --action-mode joint_pos --cameras front wrist --round 1 --viewer

    # a policy running on a GPU box (scripts/serve_policy.py there)
    uv run --extra teleop --extra server python scripts/dagger.py teleop --task peg_insertion \\
        --policy remote:ws://gpu-box:9000 --action-mode joint_pos --cameras front wrist --round 1

    # no policy: plain teleop demonstrations
    uv run --extra teleop python scripts/dagger.py teleop --task kitting --episodes 10

    # try the loop with a deliberately biased "learner"
    uv run --extra teleop mjpython scripts/dagger.py teleop --task peg_insertion --viewer \\
        --policy franka_sim.dagger.toy:biased_expert --obs-mode privileged

Automatic (classic DAgger with the scripted expert labeling every visited state)::

    uv run python scripts/dagger.py auto --task peg_insertion --round 0 --episodes 100        # beta = 1
    uv run python scripts/dagger.py auto --task peg_insertion --round 1 --episodes 100 \\
        --policy my_pkg.policies:make --beta 0.5 --action-mode joint_pos --cameras front wrist

Episode ``k`` of round ``r`` uses seed ``--seed`` (default ``20000 + 1000 * r``) + ``k``, always
below the evaluation seeds. Single-target tasks cycle through the targets.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import franka_sim
from franka_sim.benchmark import EVAL_SEED, SUITE
from franka_sim.recording import DemoWriter


def conditions_for(task: str):
    kind, n = SUITE.get(task, ("single", 0))
    return (lambda k: {"target": k % n}) if kind == "single" and n else None


def build(args, num_envs: int):
    from franka_sim.dagger.remote import load_factory

    if args.seed is None:
        args.seed = 20_000 + 1000 * args.round
    if args.seed + 10 * args.episodes >= EVAL_SEED:
        raise SystemExit(f"--seed must stay well below the evaluation seeds ({EVAL_SEED})")
    env = franka_sim.make(args.task, num_envs=num_envs, action_mode=args.action_mode, cameras=args.cameras,
                          image_size=tuple(args.image_size), seed=args.seed)
    learner = None
    if args.policy:
        learner = load_factory(args.policy, json.loads(args.policy_kwargs))(env)
    out = Path(args.out or f"data/dagger/{args.task}/round_{args.round}.hdf5")
    if out.exists() and not args.append:
        raise SystemExit(f"{out} exists; pass --append to add episodes to it, or choose --out")
    meta = {"mode": args.command, "round": args.round, "policy": args.policy or None,
            "policy_kwargs": json.loads(args.policy_kwargs), "obs_mode": args.obs_mode,
            "mask_camera": args.mask_camera, "seed": args.seed}
    if args.command == "auto":
        meta.update(beta=args.beta if args.policy else 1.0, mixing=args.mixing)
    writer = DemoWriter(out, env, dagger=json.dumps(meta))
    return env, learner, writer, out


def run_teleop(args) -> None:
    from franka_sim.dagger.session import TeleopSession
    from franka_sim.teleop import KeyboardTeleop

    env, learner, writer, out = build(args, num_envs=1)
    device = KeyboardTeleop(num_frames=len(args.console_cameras), frame_size=tuple(args.console_size),
                            title=f"franka-sim teleop: {args.task}")

    def report(k, ep, ok, saved):
        n_int = int(sum(ep.extras.get("intervention", [])))
        print(f"  episode {k}: {'success' if ok else 'fail'}, {len(ep)} steps, {n_int} intervention steps, "
              f"{'saved' if saved else 'discarded'}", flush=True)

    session = TeleopSession(env, device, learner, writer, obs_mode=args.obs_mode, mask_camera=args.mask_camera,
                            console_cameras=args.console_cameras, console_size=tuple(args.console_size),
                            speed=args.speed, auto_takeover=not args.no_auto_takeover,
                            scripted_labels=not args.no_scripted_labels, realtime_factor=args.realtime_factor,
                            conditions=conditions_for(args.task), seed=args.seed, viewer=args.viewer,
                            on_episode=report)
    print(f"{'HG-DAgger' if learner else 'teleop'} on {args.task}: focus the teleop window to drive "
          f"(Tab takes over / hands back, Esc quits). Writing {out}", flush=True)
    try:
        summary = session.run(args.episodes)
    finally:
        device.close()
        writer.close()
        env.close()
    print(f"done: {summary}")


def run_auto(args) -> None:
    from franka_sim.dagger import collect_dagger

    env, learner, writer, out = build(args, num_envs=args.num_envs)
    t0 = time.time()

    def report(k, ep, ok):
        print(f"  episode {k}: {'success' if ok else 'fail'} after {len(ep)} steps", flush=True)

    try:
        stats = collect_dagger(env, learner, args.episodes, writer, beta=args.beta, mixing=args.mixing,
                               obs_mode=args.obs_mode, mask_camera=args.mask_camera,
                               conditions=conditions_for(args.task), seed=args.seed,
                               on_episode=report if args.verbose else None)
    finally:
        writer.close()
        env.close()
    share = sum(stats.expert_fraction) / max(len(stats.expert_fraction), 1)
    print(f"round {args.round}: {stats.saved} episodes -> {out}; mixture success {stats.success_rate:.2f}, "
          f"expert executed {share:.0%} of steps, {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", required=True, choices=sorted(franka_sim.TASKS))
    common.add_argument("--policy", default=None,
                        help="learner: module.path:factory or remote:ws://host:port (none: expert/teleop only)")
    common.add_argument("--policy-kwargs", default="{}")
    common.add_argument("--action-mode", default="joint_pos", help="the learner's action mode (and of `actions`)")
    common.add_argument("--obs-mode", default="vision", choices=["vision", "privileged"])
    common.add_argument("--cameras", nargs="*", default=[], help="cameras in the learner's observations")
    common.add_argument("--image-size", nargs=2, type=int, default=[256, 256])
    common.add_argument("--mask-camera", default=None)
    common.add_argument("--episodes", type=int, default=20)
    common.add_argument("--round", type=int, default=0)
    common.add_argument("--seed", type=int, default=None)
    common.add_argument("--out", default=None, help="default: data/dagger/<task>/round_<round>.hdf5")
    common.add_argument("--append", action="store_true", help="add to an existing --out file")

    t = sub.add_parser("teleop", parents=[common], help="human-gated DAgger / teleop demos")
    t.add_argument("--console-cameras", nargs="*", default=["back", "wrist"])
    t.add_argument("--console-size", nargs=2, type=int, default=[300, 400])
    t.add_argument("--speed", type=float, default=0.15, help="teleop speed at full key press (m/s)")
    t.add_argument("--realtime-factor", type=float, default=1.0, help="<1 slows the clock (slow policies)")
    t.add_argument("--no-auto-takeover", action="store_true", help="only Tab takes control")
    t.add_argument("--no-scripted-labels", action="store_true", help="do not record scripted expert labels")
    t.add_argument("--viewer", action="store_true", help="also open the MuJoCo viewer (macOS: mjpython)")

    a = sub.add_parser("auto", parents=[common], help="DAgger with scripted expert labels")
    a.add_argument("--beta", type=float, default=0.0, help="probability that the expert acts")
    a.add_argument("--mixing", choices=["step", "episode"], default="step")
    a.add_argument("--num-envs", type=int, default=8)
    a.add_argument("--verbose", action="store_true")

    args = p.parse_args()
    run_teleop(args) if args.command == "teleop" else run_auto(args)


if __name__ == "__main__":
    main()
