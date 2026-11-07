import numpy as np
import pytest

import franka_sim
from franka_sim.policies import make_policy
from franka_sim.recording import DemoWriter, collect_demos, dataset_info, iter_episodes, load_episode


@pytest.mark.parametrize("task,kwargs", [
    ("reach", {}),
    ("hammer_nails", {}),
    ("peg_insertion", {}),
    ("peg_insertion", {"mode": "all"}),
    ("buttons", {"sequence_length": 3}),
    ("gear_assembly", {}),
    ("kitting", {}),
    ("pcb_insertion", {}),
])
def test_scripted_policy_succeeds(task, kwargs):
    env = franka_sim.make(task, num_envs=3, action_mode="ee_pose", seed=7, task_kwargs=kwargs)
    stats = collect_demos(env, make_policy(env, noise=0.002, seed=7), num_episodes=3)
    assert stats.success_rate == 1.0, stats
    env.close()


def test_hammer_hits_selected_nail_only():
    env = franka_sim.make("hammer_nails", num_envs=4, action_mode="ee_pose", seed=3)
    obs, _ = env.reset(options={"target": 3})
    policy = make_policy(env, seed=3)
    policy.reset(range(4), obs)
    for _ in range(env.max_episode_steps):
        obs, reward, term, trunc, info = env.step(policy.act(obs))
    depth = obs["nail_depth"]
    assert (depth[:, 3] > 0.75).all()
    assert (depth[:, :3] < 0.05).all()


def test_record_and_load(tmp_path):
    path = tmp_path / "demos.hdf5"
    env = franka_sim.make("hammer_nails", num_envs=2, action_mode="ee_pose", seed=0, cameras=("wrist",),
                          image_size=(32, 32))
    policy = make_policy(env, seed=0)
    with DemoWriter(path, env) as w:
        stats = collect_demos(env, policy, 4, w, conditions=lambda k: {"target": k % 4})
    env.close()
    assert stats.saved == 4

    info = dataset_info(path)
    assert info["task"] == "hammer_nails" and info["episodes"] == 4 and info["cameras"] == ["wrist"]
    targets = []
    for ep in iter_episodes(path):
        T = ep["attrs"]["num_samples"]
        assert ep["actions"].shape == (T, 8)
        assert ep["obs"]["rgb_wrist"].shape == (T, 32, 32, 3)
        assert ep["commands"]["joint_pos"].shape == (T, 7)
        assert ep["states"]["qpos"].shape[0] == T
        assert ep["success"][-1] and ep["dones"][-1]
        assert ep["attrs"]["options"]["target"] == int(ep["cond"]["target"])
        assert ep["obs"]["target_onehot"][0].argmax() == int(ep["cond"]["target"])
        targets.append(int(ep["cond"]["target"]))
    assert sorted(targets) == [0, 1, 2, 3]


def test_replay_states_reproduce_observations(tmp_path):
    """Saved states are enough to reconstruct the recorded end-effector trajectory."""
    import mujoco

    path = tmp_path / "reach.hdf5"
    env = franka_sim.make("reach", action_mode="ee_pose", seed=0)
    with DemoWriter(path, env) as w:
        collect_demos(env, make_policy(env), 1, w)
    ep = load_episode(path, 0)
    data = mujoco.MjData(env.model)
    site = env.model.site("gripper").id
    for t in (0, len(ep["actions"]) // 2, len(ep["actions"]) - 1):
        data.qpos[:] = ep["states"]["qpos"][t]
        mujoco.mj_kinematics(env.model, data)
        np.testing.assert_allclose(data.site_xpos[site], ep["obs"]["ee_pos"][t], atol=1e-5)
    env.close()


def test_bc_dataset_windows(tmp_path):
    from franka_sim.dataset import BCDataset

    path = tmp_path / "buttons.hdf5"
    env = franka_sim.make("buttons", num_envs=2, action_mode="ee_pose", seed=0)
    with DemoWriter(path, env) as w:
        collect_demos(env, make_policy(env), 2, w)
    env.close()
    ds = BCDataset(path, obs_keys=["ee_pos", "sequence_onehot"], obs_horizon=2, action_horizon=8,
                   action_key="commands/joint_pos")
    s = ds[0]
    assert s["obs/ee_pos"].shape == (2, 3)
    np.testing.assert_allclose(s["obs/ee_pos"][0], s["obs/ee_pos"][1])  # padded at episode start
    assert s["actions"].shape == (8, 8)
    assert s["cond/sequence"].shape == (1,)
    last = ds[len(ds) - 1]
    assert last["action_pad_mask"].sum() == 1
    assert ds.stats()["actions"]["mean"].shape == (8,)
    assert ds.instruction(0).startswith("press the")
