import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.lerobot_dataset")

import franka_sim  # noqa: E402
from franka_sim.lerobot_export import convert  # noqa: E402
from franka_sim.policies import make_policy  # noqa: E402
from franka_sim.recording import DemoWriter, collect_demos  # noqa: E402


def test_convert_to_lerobot(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    h5 = tmp_path / "pcb.hdf5"
    env = franka_sim.make("pcb_insertion", num_envs=2, action_mode="ee_pose", seed=0)
    with DemoWriter(h5, env) as w:
        collect_demos(env, make_policy(env), 2, w, conditions=lambda k: {"target": k % 3})
    env.close()

    root = tmp_path / "lerobot"
    ds = convert([h5], "local/franka_sim_test", root, cameras=("wrist",), image_size=(64, 64), progress=None)
    assert ds.num_episodes == 2

    ds = LeRobotDataset("local/franka_sim_test", root=root)
    item = ds[5]
    assert item["observation.state"].shape == (8,)
    assert item["action"].shape == (8,)
    assert item["observation.images.wrist"].shape == (3, 64, 64)
    assert item["task"].startswith("insert the card into the")
    np.testing.assert_allclose(item["action"].numpy(), item["action.joint_pos"].numpy())
    assert item["action.delta_ee_pose"].shape == (7,)
    targets = {int(ds[i]["cond.target"]) for i in (0, ds.num_frames - 1)}
    assert targets == {0, 1}
    mask = item["observation.images.target_mask"]
    assert mask.shape == (3, 64, 64)
    assert set(mask.unique().tolist()) <= {0.0, 1.0} and mask.sum() > 0


def test_convert_dagger_rounds(tmp_path):
    """DAgger files export with actor / intervention / expert labels; plain demos mix in as actor 2."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from franka_sim.dagger import collect_dagger
    from franka_sim.dagger.toy import biased_expert

    env = franka_sim.make("buttons", num_envs=2, action_mode="joint_pos", max_episode_steps=40)
    with DemoWriter(tmp_path / "r0.hdf5", env) as w:
        collect_dagger(env, None, 2, w, seed=0)
    with DemoWriter(tmp_path / "r1.hdf5", env) as w:
        collect_dagger(env, biased_expert(env), 2, w, beta=0.5, obs_mode="privileged", seed=10)
    env.close()
    files = [tmp_path / "r0.hdf5", tmp_path / "r1.hdf5"]
    ds = convert(files, "local/dagger", tmp_path / "lr", cameras=(), mask_camera=None, labels="expert",
                 only_success=False, progress=None)
    ds = LeRobotDataset("local/dagger", root=tmp_path / "lr")
    item = ds[3]
    for key in ("dagger.actor", "dagger.intervention", "expert.joint_pos", "expert.ee_pose", "expert.delta_ee_pose"):
        assert key in item
    np.testing.assert_allclose(item["action"].numpy(), item["expert.joint_pos"].numpy())
    actors = {int(ds[i]["dagger.actor"]) for i in range(ds.num_frames)}
    assert actors == {0, 2}

    ds = convert(files, "local/dagger_delta", tmp_path / "lr_delta", cameras=(), mask_camera=None,
                 action="delta_ee_pose", only_success=False, progress=None)
    item = LeRobotDataset("local/dagger_delta", root=tmp_path / "lr_delta")[3]
    assert item["action"].shape == (7,)
    np.testing.assert_allclose(item["action"].numpy(), item["action.delta_ee_pose"].numpy())

    with pytest.raises(ValueError, match="expert labels"):  # plain demos have no expert labels
        h5 = tmp_path / "plain.hdf5"
        env = franka_sim.make("buttons", num_envs=1, action_mode="ee_pose", seed=0)
        with DemoWriter(h5, env) as w:
            collect_demos(env, make_policy(env), 1, w)
        env.close()
        convert([h5, files[0]], "local/x", tmp_path / "lr2", cameras=(), mask_camera=None, labels="expert",
                progress=None)
