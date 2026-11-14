"""Run a simulated Franka as a proxy robot: MuJoCo's native viewer + a model API.

    # macOS: the viewer needs mjpython
    uv run --extra server mjpython scripts/serve.py --task hammer_nails
    # Linux / Windows
    uv run --extra server python scripts/serve.py --task hammer_nails

    # without a window (servers), or paused at start (lockstep: advances only on step requests)
    uv run --extra server python scripts/serve.py --task peg_insertion --headless --paused

Models connect with ``franka_sim.client.RobotClient("ws://127.0.0.1:8000/ws")``. In the viewer,
double-click a body to select it and Ctrl + right-drag to push it. Closing the window stops the server.
"""

import argparse

import franka_sim  # noqa: F401  (sets MUJOCO_GL on headless machines before mujoco is imported)
from franka_sim.server import serve
from franka_sim.tasks import TASKS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="hammer_nails", choices=sorted(TASKS))
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept connections from other machines")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--control-freq", type=float, default=20.0)
    p.add_argument("--image-size", nargs=2, type=int, default=[224, 224], help="default observation size (H W)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paused", action="store_true", help="start paused (lockstep)")
    p.add_argument("--headless", action="store_true", help="no viewer window")
    args = p.parse_args()
    print(f"franka-sim robot ({args.task})  |  model API: ws://{args.host}:{args.port}/ws  |  "
          f"status: http://{args.host}:{args.port}/api/state  |  Ctrl+C or close the viewer to stop", flush=True)
    serve(args.task, args.host, args.port, control_freq=args.control_freq, image_size=tuple(args.image_size),
          seed=args.seed, realtime=not args.paused, viewer=not args.headless)


if __name__ == "__main__":
    main()
