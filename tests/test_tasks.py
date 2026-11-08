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
@pytest.mark.parametrize("task", ["hammer_nails", "peg_insertion", "buttons", "gear_assembly", "kitting",
                                  "pcb_insertion"])
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
        assert np.abs(obs["obj_pos"].reshape(2, -1, 3)[..., 2] - objects.PEG_HALF[2]).max() < 0.005
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
    assert not info["wrong_target"][0]
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


def test_insertion_conditioning():
    env = franka_sim.make("kitting", num_envs=2, seed=0, task_kwargs={"mode": "subset"})
    obs, info = env.reset(options={"targets": [3, 1]})
    assert info["cond_targets"].tolist() == [[3, 1], [3, 1]]
    np.testing.assert_allclose(obs["targets_mask"][0], [0, 1, 0, 1])
    assert obs["target_onehot"][0].argmax() == 3
    assert env.instructions()[0] == "kit the cube, then the cylinder"

    env = franka_sim.make("pcb_insertion", num_envs=2, seed=0)
    obs, info = env.reset(options={"target": 2})
    assert info["cond_target"].tolist() == [2, 2]
    np.testing.assert_allclose(obs["goal_pos"], obs["slot_pos"][:, 6:9])
    assert env.instructions()[0] == "insert the card into the rightmost connector"


def test_everything_is_randomized():
    """Robot start pose, objects and fixtures all change between resets."""
    for task in ("hammer_nails", "peg_insertion", "gear_assembly", "kitting", "pcb_insertion", "buttons"):
        env = franka_sim.make(task, seed=0)
        first, _ = env.reset()
        second, _ = env.reset()
        assert np.linalg.norm(first["ee_pos"] - second["ee_pos"]) > 0.01, task
        for key, v in first.items():
            if key.endswith(("_pos", "_quat")) and not key.startswith(("ee_", "target_")) and key in second:
                if np.abs(v).sum() > 0:
                    assert np.abs(v - second[key]).max() > 1e-3, (task, key)
        env.close()


def test_start_pose_fixed_when_disabled():
    env = franka_sim.make("reach", seed=0, randomize_start=False, init_noise=0.0)
    a, _ = env.reset()
    b, _ = env.reset()
    np.testing.assert_allclose(a["joint_pos"], b["joint_pos"])


@pytest.mark.parametrize("task,kwargs,body", [
    ("hammer_nails", {}, "nail{}"),
    ("pcb_insertion", {}, "motherboard_conn{}"),
    ("peg_insertion", {}, "peg{}"),
    ("buttons", {}, "button{}"),
    ("gear_assembly", {}, "gear{}"),
    ("kitting", {}, "part{}"),
])
def test_target_mask(task, kwargs, body):
    env = franka_sim.make(task, num_envs=3, seed=1, task_kwargs=kwargs, image_size=(128, 128))
    env.reset()
    cond = env.task.conditioning()
    for i in range(3):
        k = int(np.ravel(next(iter(cond.values()))[i])[0])
        assert env.task.mask_bodies({key: v[i] for key, v in cond.items()}) == [body.format(k)]
        mask = env.render_target_mask("top", env_id=i)
        assert mask.shape == (128, 128) and set(np.unique(mask)) == {0, 255}
    env.close()


@pytest.mark.parametrize("task,kwargs", [
    ("gear_assembly_long", {}), ("kitting", {"mode": "subset"}), ("buttons", {"sequence_length": 2}), ("reach", {}),
])
def test_no_mask_for_multi_step_tasks(task, kwargs):
    env = franka_sim.make(task, task_kwargs=kwargs)
    env.reset()
    assert env.render_target_mask() is None
    env.close()


SINGLE = ["hammer_nails", "buttons", "peg_insertion", "gear_assembly", "kitting", "pcb_insertion"]
LONG = ["hammer_nails_long", "buttons_long", "peg_insertion_long", "gear_assembly_long", "kitting_long"]


@pytest.mark.parametrize("task", SINGLE)
def test_single_variants(task):
    env = franka_sim.make(task, num_envs=4, seed=0)
    _, info = env.reset()
    assert env.task.num_steps == 1 and "cond_target" in info
    assert env.max_episode_steps == franka_sim.tasks.HORIZONS[task]
    _, _, _, _, info = env.step(np.zeros((4, env.action_dim)))
    assert info["progress"].shape == (4,) and not info["wrong_target"].any()
    env.close()


@pytest.mark.parametrize("task", LONG)
def test_long_variants(task):
    env = franka_sim.make(task, num_envs=8, seed=0)
    _, info = env.reset()
    order = info["cond_order"]
    assert env.task.num_steps == order.shape[1] > 1
    # The order is a random permutation per episode, but the instruction does not depend on it.
    assert all(sorted(o) == list(range(order.shape[1])) for o in order.tolist())
    assert len({tuple(o) for o in order.tolist()}) > 1
    assert len(set(env.instructions())) == 1
    assert env.render_target_mask() is None
    env.close()


def test_buttons_long_any_order():
    from franka_sim.tasks import objects as obj

    env = franka_sim.make("buttons_long", seed=0)
    env.reset(options={"order": [0, 1, 2, 3]})
    qpos = env.sim.read("qpos")
    for k, b in enumerate([3, 1, 0, 2]):  # press in a different order than the sampled one
        qpos[0, env.task.button_qadr[b]] = -obj.BUTTON_TRAVEL
        env.sim.write("qpos", qpos)
        ev = env.task.evaluate(env.sim)
        assert ev.info["progress"][0] == k + 1 and not ev.info["wrong_target"][0]
    assert ev.success[0]


def test_wrong_target_reported():
    from franka_sim.tasks import objects as obj

    env = franka_sim.make("hammer_nails", seed=0)
    env.reset(options={"target": 0})
    qpos = env.sim.read("qpos")
    qpos[0, env.task.nail_qadr[2]] = -obj.NAIL_TRAVEL  # drive a different nail
    env.sim.write("qpos", qpos)
    ev = env.task.evaluate(env.sim)
    assert ev.info["wrong_target"][0] and not ev.success[0]
