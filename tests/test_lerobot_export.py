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
    targets = {int(ds[i]["cond.target"]) for i in (0, ds.num_frames - 1)}
    assert targets == {0, 1}
    mask = item["observation.images.target_mask"]
    assert mask.shape == (3, 64, 64)
    assert set(mask.unique().tolist()) <= {0.0, 1.0} and mask.sum() > 0
