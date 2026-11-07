import numpy as np
import pytest

import franka_sim
from franka_sim.gym_env import FrankaGymEnv


def _reach_policy(obs):
    a = np.zeros((len(obs["ee_pos"]), 7))
    a[:, :3] = np.clip(obs["goal_pos"] - obs["ee_pos"], -0.03, 0.03)
    return a


def test_obs_shapes_batched():
    env = franka_sim.make("reach", num_envs=3, seed=0)
    obs, info = env.reset()
    assert obs["joint_pos"].shape == (3, 7)
    assert obs["ee_quat"].shape == (3, 4)
    assert obs["gripper_width"].shape == (3, 1)
    assert info["cond_goal_pos"].shape == (3, 3)
    env.close()


def test_reach_ee_delta_succeeds():
    env = franka_sim.make("reach", num_envs=4, seed=0)
    obs, _ = env.reset()
    success = np.zeros(4, dtype=bool)
    for _ in range(80):
        obs, reward, term, trunc, info = env.step(_reach_policy(obs))
        success |= info["success"]
    assert success.all()
    env.close()


@pytest.mark.parametrize("mode", ["ee_pose", "joint_pos", "joint_delta"])
def test_action_modes_hold_pose(mode):
    env = franka_sim.make("reach", action_mode=mode, seed=0, init_noise=0.0)
    obs, _ = env.reset()
    start = obs["ee_pos"].copy()
    if mode == "ee_pose":
        a = np.concatenate([obs["ee_pos"], obs["ee_quat"], [[0.0]]], -1)
    elif mode == "joint_pos":
        a = np.concatenate([obs["joint_pos"], [[0.0]]], -1)
    else:
        a = np.zeros((1, 8))
    for _ in range(20):
        obs, *_ = env.step(a)
    assert np.linalg.norm(obs["ee_pos"] - start) < 5e-3
    env.close()


def test_partial_reset_only_touches_selected_envs():
    env = franka_sim.make("reach", num_envs=2, seed=0)
    obs, _ = env.reset()
    for _ in range(5):
        obs, *_ = env.step(_reach_policy(obs))
    before = obs["joint_pos"][1].copy()
    obs, _ = env.reset(env_ids=[0])
    assert env.elapsed_steps.tolist() == [0, 5]
    np.testing.assert_allclose(obs["joint_pos"][1], before)
    env.close()


def test_gripper_closes():
    env = franka_sim.make("reach", seed=0)
    obs, _ = env.reset()
    assert obs["gripper_width"][0, 0] > 0.07
    a = np.zeros((1, 7))
    a[:, -1] = 1.0
    for _ in range(10):
        obs, *_ = env.step(a)
    assert obs["gripper_width"][0, 0] < 0.01
    env.close()


def test_camera_observations():
    env = franka_sim.make("reach", num_envs=2, cameras=("front", "wrist"), image_size=(64, 80))
    obs, _ = env.reset()
    assert obs["rgb_front"].shape == (2, 64, 80, 3)
    assert obs["rgb_wrist"].dtype == np.uint8
    env.close()


def test_gym_wrapper():
    env = FrankaGymEnv("reach", render_mode="rgb_array", image_size=(48, 64))
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert isinstance(info["instruction"], str)
    obs, r, term, trunc, info = env.step(env.action_space.sample().clip(-0.01, 1.0))
    assert env.render().shape == (48, 64, 3)
    env.close()
