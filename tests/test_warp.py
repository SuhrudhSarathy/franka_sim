import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

import franka_sim  # noqa: E402


def _policy(obs, t):
    a = np.zeros((len(obs["ee_pos"]), 7))
    a[:, :3] = np.clip(obs["goal_pos"] - obs["ee_pos"], -0.03, 0.03)
    a[:, -1] = float(t > 10)
    return a


def test_warp_matches_cpu():
    trajs = {}
    for backend in ("cpu", "warp"):
        env = franka_sim.make("reach", num_envs=2, backend=backend, seed=0)
        obs, _ = env.reset(seed=0)
        traj = []
        for t in range(20):
            obs, *_ = env.step(_policy(obs, t))
            traj.append(obs["ee_pos"])
        trajs[backend] = np.array(traj)
        env.close()
    np.testing.assert_allclose(trajs["cpu"], trajs["warp"], atol=1e-3)


def test_warp_partial_reset_and_render():
    env = franka_sim.make("reach", num_envs=3, backend="warp", seed=0, cameras=("wrist",), image_size=(32, 32))
    obs, _ = env.reset()
    for t in range(3):
        obs, *_ = env.step(_policy(obs, t))
    keep = obs["joint_pos"][2].copy()
    obs, _ = env.reset(env_ids=[0, 1])
    np.testing.assert_allclose(obs["joint_pos"][2], keep, atol=1e-6)
    assert np.abs(env.sim.read("qvel")[:2]).max() == 0.0
    assert obs["rgb_wrist"].shape == (3, 32, 32, 3)
    env.close()
