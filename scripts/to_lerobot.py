"""Convert recorded HDF5 demos into a LeRobot dataset, ready for ``push_to_hub``.

    uv run --extra lerobot python scripts/to_lerobot.py data/hammer_nails.hdf5 \
        --repo-id <hf-user>/franka_sim_hammer_nails --root lerobot_data/franka_sim_hammer_nails \
        --cameras front wrist --image-size 256 256

    # upload (needs `huggingface-cli login`):
    uv run --extra lerobot python scripts/to_lerobot.py --push --repo-id <hf-user>/franka_sim_hammer_nails \
        --root lerobot_data/franka_sim_hammer_nails
"""

from __future__ import annotations

import argparse
import time

from franka_sim.lerobot_export import convert


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hdf5", nargs="*", help="recorded demo files (same task configuration)")
    p.add_argument("--repo-id", required=True, help="Hugging Face dataset id, e.g. user/franka_sim_hammer")
    p.add_argument("--root", required=True, help="local output directory of the LeRobot dataset")
    p.add_argument("--cameras", nargs="*", default=["front", "wrist"])
    p.add_argument("--image-size", nargs=2, type=int, default=[256, 256], metavar=("H", "W"))
    p.add_argument("--action", default="joint_pos", choices=["joint_pos", "ee_pose"])
    p.add_argument("--images", action="store_true", help="store PNG frames instead of mp4 videos")
    p.add_argument("--include-failures", action="store_true")
    p.add_argument("--push", action="store_true", help="upload an existing dataset at --root to the Hub")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if args.push:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(args.repo_id, root=args.root)
        ds.push_to_hub(private=args.private)
        print(f"pushed {ds.num_episodes} episodes to https://huggingface.co/datasets/{args.repo_id}")
        return

    t0 = time.time()
    ds = convert(args.hdf5, args.repo_id, args.root, cameras=args.cameras, image_size=tuple(args.image_size),
                 action=args.action, use_videos=not args.images, only_success=not args.include_failures)
    print(f"\nwrote {ds.num_episodes} episodes / {ds.num_frames} frames to {args.root} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
