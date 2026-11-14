"""Serve a policy over the network so the simulator (evaluate.py, dagger.py) can use it remotely.

    # on the GPU machine
    uv run --extra server python scripts/serve_policy.py --policy my_pkg.policies:make \\
        --policy-kwargs '{"ckpt": "runs/act_round2"}' --host 0.0.0.0 --port 9000

    # on the simulation machine
    uv run --extra server python scripts/evaluate.py --policy remote:ws://gpu-box:9000
    uv run --extra server --extra teleop python scripts/dagger.py teleop --task hammer_nails \\
        --policy remote:ws://gpu-box:9000

``my_pkg.policies:make(env_spec, **kwargs)`` gets an ``EnvSpec`` (task, action_mode, action_dim,
num_envs, cameras, image_size, control_dt) in place of the env, and returns an object with
``reset(env_ids, obs)``, ``act(obs) -> (num_envs, action_dim)`` and optionally
``interrupt(env_ids, obs)`` (called when a human hands control back mid-episode).
"""

import argparse
import json

from franka_sim.dagger.remote import serve_policy


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="module.path:factory")
    p.add_argument("--policy-kwargs", default="{}", help="JSON dict passed to the factory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    args = p.parse_args()
    print(f"serving {args.policy} on ws://{args.host}:{args.port}", flush=True)
    serve_policy(args.policy, args.host, args.port, json.loads(args.policy_kwargs))


if __name__ == "__main__":
    main()
