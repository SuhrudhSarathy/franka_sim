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


def test_reach_delta_ee_pose_succeeds():
    env = franka_sim.make("reach", num_envs=4, seed=0)
    obs, _ = env.reset()
    success = np.zeros(4, dtype=bool)
    for _ in range(80):
        obs, reward, term, trunc, info = env.step(_reach_policy(obs))
        success |= info["success"]
    assert success.all()
    env.close()


@pytest.mark.parametrize("mode", ["ee_pose", "joint_pos", "joint_delta", "joint_vel", "delta_ee_pose"])
def test_action_modes_hold_pose(mode):
    env = franka_sim.make("reach", action_mode=mode, seed=0, init_noise=0.0)
    obs, _ = env.reset()
    start = obs["ee_pos"].copy()
    if mode == "ee_pose":
        a = np.concatenate([obs["ee_pos"], obs["ee_quat"], [[0.0]]], -1)
    elif mode == "joint_pos":
        a = np.concatenate([obs["joint_pos"], [[0.0]]], -1)
    else:
        a = np.zeros((1, env.action_dim))
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


def test_headless_linux_defaults_to_egl():
    from franka_sim import _default_gl_backend

    assert _default_gl_backend({}, "linux") == "egl"
    assert _default_gl_backend({"DISPLAY": ":0"}, "linux") is None
    assert _default_gl_backend({"MUJOCO_GL": "osmesa"}, "linux") is None  # the user's choice wins
    assert _default_gl_backend({}, "darwin") is None


def test_ee_delta_is_an_alias():
    env = franka_sim.make("reach", action_mode="ee_delta")
    assert env.action_mode == "delta_ee_pose" and env.action_dim == 7
    env.close()


def test_delta_ee_pose_tracks_a_straight_line():
    """5 mm/step along y: the hand moves at the commanded speed and stays on the line."""
    env = franka_sim.make("reach", action_mode="delta_ee_pose", seed=0, init_noise=0.0, randomize_start=False)
    obs, _ = env.reset()
    start = obs["ee_pos"][0].copy()
    a = np.zeros((1, 7))
    a[0, 1] = 0.005
    path = []
    for _ in range(30):
        obs, *_ = env.step(a)
        path.append(obs["ee_pos"][0] - start)
    path = np.array(path)
    assert abs(path[-1, 1] - 0.15) < 0.01  # commanded 15 cm, small lag only
    assert np.abs(path[:, [0, 2]]).max() < 2e-3  # stays on the line
    np.testing.assert_allclose(np.diff(path[10:, 1]), 0.005, atol=5e-4)  # constant speed, no ringing
    env.close()


def test_joint_vel_moves_at_commanded_rate():
    env = franka_sim.make("reach", action_mode="joint_vel", seed=0, init_noise=0.0, randomize_start=False)
    obs, _ = env.reset()
    q0 = obs["joint_pos"][0].copy()
    a = np.zeros((1, 8))
    a[0, 0] = 0.5  # rad/s on joint 1
    for _ in range(20):  # 1 s
        obs, *_ = env.step(a)
    assert abs(obs["joint_pos"][0, 0] - q0[0] - 0.5) < 0.02
    assert np.abs(obs["joint_pos"][0, 1:] - q0[1:]).max() < 0.01
    env.close()


def test_commands_respect_joint_velocity_limits():
    from franka_sim import robot

    env = franka_sim.make("reach", action_mode="joint_pos", seed=0)
    obs, _ = env.reset()
    a = np.concatenate([obs["joint_pos"] + 1.0, [[0.0]]], -1)  # far beyond one step
    prev = obs["joint_pos"][0].copy()
    env.step(a)
    step = np.abs(env.last_command["joint_pos"][0] - prev)
    assert np.all(step <= robot.JOINT_VEL_LIMITS * env.control_dt + 1e-9)
    env.close()


def test_wrist_servo_does_not_ring():
    """A joint step on the wrist settles without overshoot (Menagerie's gains overshoot ~30%)."""
    env = franka_sim.make("reach", action_mode="joint_pos", seed=0, init_noise=0.0, randomize_start=False,
                          velocity_feedforward=0.0)
    obs, _ = env.reset()
    target = obs["joint_pos"][0].copy()
    target[5] += 0.1
    traj = []
    for _ in range(15):
        obs, *_ = env.step(np.r_[target, 0.0][None])
        traj.append(obs["joint_pos"][0, 5] - target[5])
    assert max(traj) < 0.1 * 0.05  # < 5% overshoot
    assert abs(traj[-1]) < 1e-3
    env.close()


def test_all_cameras_render():
    from franka_sim.scene import CAMERAS

    assert len(CAMERAS) >= 7
    env = franka_sim.make("hammer_nails", cameras=CAMERAS, image_size=(32, 32))
    obs, _ = env.reset(seed=0)
    for cam in CAMERAS:
        img = obs[f"rgb_{cam}"]
        assert img.shape == (1, 32, 32, 3) and img.std() > 1  # not blank
    with pytest.raises(ValueError, match="unknown cameras"):
        franka_sim.make("reach", cameras=("nope",))
    env.close()
