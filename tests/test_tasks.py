import numpy as np
import pytest

import franka_sim
from franka_sim import robot
from franka_sim.tasks import objects
from franka_sim.tasks import TASKS

BACKENDS = ["cpu"]
try:
    import mujoco_warp  # noqa: F401

    BACKENDS.append("warp")
except ImportError:
    pass


@pytest.mark.parametrize("task", sorted(TASKS))
def test_task_builds_and_steps(task):
    env = franka_sim.make(task, num_envs=2, seed=0)
    obs, info = env.reset()
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(np.zeros((2, env.action_dim)))
    assert reward.shape == (2,)
    assert all(v.shape[0] == 2 for v in obs.values())
    assert len(env.instructions()) == 2
    env.close()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("task", ["hammer_nails", "peg_insertion", "buttons"])
def test_objects_at_rest(task, backend):
    """Objects settle without drifting, flying off or triggering success on their own."""
    env = franka_sim.make(task, num_envs=2, backend=backend, seed=1)
    obs, _ = env.reset()
    for _ in range(30):
        obs, reward, term, trunc, info = env.step(np.zeros((2, env.action_dim)))
    assert not info["success"].any()
    if task == "hammer_nails":
        assert np.abs(obs["nail_depth"]).max() < 0.05
        assert np.abs(obs["hammer_pos"][:, 2] - objects.HAMMER_HANDLE_Z).max() < 0.005
    if task == "peg_insertion":
        assert np.abs(obs["peg_pos"].reshape(2, -1, 3)[..., 2] - objects.PEG_HALF[2]).max() < 0.005
    env.close()


def test_conditioning_options():
    env = franka_sim.make("hammer_nails", num_envs=3, seed=0)
    obs, info = env.reset(options={"target": 2})
    assert (info["cond_target"] == 2).all()
    np.testing.assert_allclose(obs["target_onehot"], np.tile([0, 0, 1, 0], (3, 1)))
    assert env.instructions()[0] == "pick up the hammer and hit the third nail from the left"
    obs, info = env.reset(env_ids=[1], options={"target": 0})
    assert info["cond_target"].tolist() == [2, 0, 2]

    env = franka_sim.make("buttons", task_kwargs={"sequence_length": 2}, seed=0)
    _, info = env.reset(options={"sequence": [3, 1]})
    assert info["cond_sequence"].tolist() == [[3, 1]]
    assert env.instructions()[0] == "press the yellow, then the green buttons in order"


@pytest.mark.parametrize("backend", BACKENDS)
def test_press_button_with_gripper(backend):
    env = franka_sim.make("buttons", backend=backend, action_mode="ee_pose", seed=0)
    obs, _ = env.reset(options={"target": 1})
    quat = env.home_ee_quat
    target = obs["next_button_pos"][0]
    pressed = False
    for t in range(60):
        # Fingertips are FINGERTIP_OFFSET below the EE site; push them 1.5 cm below the cap top.
        z = target[2] + (0.08 if t < 25 else robot.FINGERTIP_OFFSET - 0.015)
        action = np.concatenate([[target[0], target[1], z], quat, [1.0]])[None]
        obs, reward, term, trunc, info = env.step(action)
        pressed |= bool(info["success"][0])
    assert pressed
    assert not info["failed"][0]
    env.close()


def test_nail_layout_randomized_and_ordered():
    env = franka_sim.make("hammer_nails", num_envs=3, seed=0)
    obs, _ = env.reset()
    nails = obs["nail_pos"].reshape(3, -1, 3)
    # Different layouts across envs, not just a shifted copy of one row.
    rel = nails[..., :2] - nails[:, :1, :2]
    assert np.abs(rel[0] - rel[1]).max() > 0.01
    # Index order = left to right from the robot (decreasing y), nails well separated.
    assert (np.diff(nails[..., 1], axis=1) < 0).all()
    d = np.linalg.norm(nails[:, :, None, :2] - nails[:, None, :, :2], axis=-1) + np.eye(4) * 9
    assert d.min() >= 0.075 - 1e-9
    obs2, _ = env.reset()
    assert np.abs(obs2["nail_pos"] - obs["nail_pos"]).max() > 0.01
    env.close()
