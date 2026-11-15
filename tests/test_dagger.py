import socket
import threading
import time

import numpy as np
import pytest

import franka_sim
from franka_sim.controller import ACTION_DIMS
from franka_sim.dagger import ACTOR_EXPERT, ACTOR_HUMAN, ACTOR_LEARNER, collect_dagger
from franka_sim.dagger.session import TeleopSession
from franka_sim.dagger.toy import BiasedExpert, biased_expert
from franka_sim.recording import DemoWriter, dataset_info, load_episode
from franka_sim.teleop import ScriptedDevice, TeleopInput
from franka_sim.teleop.keyboard import keys_to_twist


@pytest.mark.parametrize("mode", list(ACTION_DIMS))
def test_convert_action_round_trips(mode):
    """An ee_pose command restated in any mode reproduces the same joint command when executed."""
    env = franka_sim.make("reach", action_mode="ee_pose", seed=0)
    obs, _ = env.reset(seed=0)
    target = np.c_[obs["ee_pos"] + [[0.03, -0.02, 0.01]], obs["ee_quat"], [[1.0]]]
    converted = env.convert_action(target, "ee_pose", mode)
    q_meas = env.sim.read("qpos")[:, env.arm_qadr]
    via_mode = env.controller.preview(mode, converted[:, :-1], q_meas)["joint_pos"]
    direct = env.controller.preview("ee_pose", target[:, :-1], q_meas)["joint_pos"]
    np.testing.assert_allclose(via_mode, direct, atol=1e-9)
    env.step(target)
    np.testing.assert_allclose(env.executed_action(mode), converted, atol=1e-12)
    env.close()


def test_expert_only_round_executes_its_labels(tmp_path):
    env = franka_sim.make("peg_insertion", num_envs=3, action_mode="joint_pos")
    with DemoWriter(tmp_path / "r0.hdf5", env) as w:
        stats = collect_dagger(env, None, 3, w, seed=0)
    assert stats.success_rate == 1.0
    ep = load_episode(tmp_path / "r0.hdf5", 0)
    assert (ep["actor"] == ACTOR_EXPERT).all() and not ep["intervention"].any()
    np.testing.assert_allclose(ep["actions"], ep["expert"]["actions"], atol=1e-6)
    np.testing.assert_allclose(ep["commands"]["joint_pos"], ep["expert"]["joint_pos"], atol=1e-5)
    env.close()


def test_learner_states_get_corrective_labels(tmp_path):
    """beta=0: the biased learner drives (and fails); the expert labels pull against its bias."""
    env = franka_sim.make("peg_insertion", num_envs=2, action_mode="joint_pos", max_episode_steps=80)
    with DemoWriter(tmp_path / "r1.hdf5", env, dagger='{"round": 1}') as w:
        stats = collect_dagger(env, biased_expert(env, bias=(0.0, 0.02, 0.0)), 2, w, beta=0.0,
                               obs_mode="privileged", seed=0)
    assert stats.saved == 2 and stats.succeeded == 0
    assert dataset_info(tmp_path / "r1.hdf5")["dagger"] == {"round": 1}
    ep = load_episode(tmp_path / "r1.hdf5", 0)
    assert (ep["actor"] == ACTOR_LEARNER).all()
    late = slice(20, None)  # once the learner's offset has accumulated
    dy = ep["expert"]["ee_pose"][late, 1] - ep["commands"]["ee_pose"][late, 1]
    assert (dy < 0).mean() > 0.9 and dy.mean() < -0.002  # the label consistently points back toward -y
    env.close()


def test_keys_to_twist():
    np.testing.assert_array_equal(keys_to_twist({"w", "a", "q"}), [1, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(keys_to_twist({"w", "s"}), np.zeros(6))  # opposite keys cancel
    np.testing.assert_array_equal(keys_to_twist({"f", "unrelated"}), [0, 0, -1, 0, 0, 0])


def test_teleop_session_takeover_handback_and_labels(tmp_path):
    env = franka_sim.make("peg_insertion", action_mode="joint_pos")
    learner = BiasedExpert(env)
    interrupts = []
    orig = learner.interrupt
    learner.interrupt = lambda ids, obs: (interrupts.append(1), orig(ids, obs))
    fwd = np.array([1.0, 0, 0, 0, 0, 0])
    inputs = ([TeleopInput()] * 5 + [TeleopInput(twist=fwd)] * 10 + [TeleopInput(events=["toggle"])]
              + [TeleopInput()] * 10 + [TeleopInput(events=["save"])] + [TeleopInput()] * 3
              + [TeleopInput(events=["discard"])] + [TeleopInput(events=["quit"])])
    device = ScriptedDevice(inputs)
    with DemoWriter(tmp_path / "hg.hdf5", env) as w:
        session = TeleopSession(env, device, learner, w, obs_mode="privileged", realtime_factor=0,
                                console_size=(48, 64), frame="world")
        summary = session.run(5)
    assert summary == {"saved": 1, "successes": 0, "discarded": 1, "episodes": 3}
    assert len(interrupts) == 1 and device.shown > 0
    ep = load_episode(tmp_path / "hg.hdf5", 0)
    actor = ep["actor"]
    assert actor.tolist() == [ACTOR_LEARNER] * 5 + [ACTOR_HUMAN] * 10 + [ACTOR_LEARNER] * 11
    assert ep["intervention"].tolist() == (actor == ACTOR_HUMAN).tolist()
    assert "expert" in ep  # scripted shadow labels
    x = ep["commands"]["ee_pose"][:, 0]
    assert abs((x[14] - x[4]) - 10 * 0.15 * env.control_dt) < 0.01  # the operator's +x motion
    # human steps are stored in the learner's action space (joint_pos)
    np.testing.assert_allclose(ep["actions"][5:15, :7], ep["commands"]["joint_pos"][5:15], atol=1e-6)
    env.close()


def test_pure_teleop_without_learner(tmp_path):
    env = franka_sim.make("reach", action_mode="ee_pose")
    up = np.array([0, 0, 1.0, 0, 0, 0])
    device = ScriptedDevice([TeleopInput(twist=up)] * 8 + [TeleopInput(events=["grip"]), TeleopInput(events=["save"])])
    with DemoWriter(tmp_path / "t.hdf5", env) as w:
        summary = TeleopSession(env, device, None, w, realtime_factor=0, console_size=(48, 64),
                                frame="world").run(1)
    assert summary["saved"] == 1
    ep = load_episode(tmp_path / "t.hdf5", 0)
    assert (ep["actor"] == ACTOR_HUMAN).all() and ep["intervention"].all()
    assert ep["commands"]["ee_pose"][-1, 2] - ep["commands"]["ee_pose"][0, 2] > 0.04
    assert ep["actions"].shape[1] == 8 and ep["commands"]["gripper"][-1] == 1.0
    env.close()


class _HoldPolicy:
    """Remote test policy: holds the measured joints (joint_pos), counts calls."""

    def __init__(self, spec):
        assert spec.action_mode == "joint_pos" and spec.action_dim == 8
        self.action_mode = "joint_pos"
        self.calls = 0

    def reset(self, env_ids, obs):
        assert isinstance(obs["instruction"][0], str)

    def act(self, obs):
        self.calls += 1
        q = np.asarray(obs["joint_pos"])
        return np.concatenate([q, np.zeros((len(q), 1))], axis=1)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_remote_policy_round_trip():
    pytest.importorskip("fastapi")
    pytest.importorskip("websockets")
    import uvicorn

    from franka_sim.dagger.remote import create_policy_app, load_factory

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_policy_app(_HoldPolicy), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        env = franka_sim.make("buttons", num_envs=2, action_mode="joint_pos", max_episode_steps=15)
        for _ in range(100):
            try:
                policy = load_factory(f"remote:ws://127.0.0.1:{port}")(env)
                break
            except OSError:
                time.sleep(0.05)
        stats = collect_dagger(env, policy, 2, None, beta=0.0, seed=0)
        assert stats.attempted == 2 and stats.succeeded == 0  # holding still never presses a button
        policy.close()
        env.close()
    finally:
        server.should_exit = True
        thread.join(timeout=5)



def test_teleop_frames_world_heading_and_spin():
    import mujoco

    from franka_sim.teleop.base import twist_to_world

    env = franka_sim.make("peg_insertion")
    sid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
    for seed in range(3):  # different start yaws
        env.reset(seed=seed)
        site = env.sim.mjdata(0).site_xmat[sid].reshape(3, 3)
        quat = env.last_command["ee_pose"][0, 3:7]
        e = np.eye(6)
        for k in range(3):  # world: fixed axes, whatever the gripper's yaw
            np.testing.assert_allclose(twist_to_world(e[k], "world", quat)[:3], np.eye(3)[k], atol=1e-12)
        fwd = twist_to_world(e[0], "heading", quat)[:3]
        heading = site[:2, 0] / np.linalg.norm(site[:2, 0])
        np.testing.assert_allclose(fwd, [*heading, 0.0], atol=1e-3)  # heading: the gripper's x, on the floor
        np.testing.assert_allclose(twist_to_world(e[2], "heading", quat)[:3], [0, 0, 1], atol=1e-12)
        for frame in ("world", "heading"):  # Q/E spins about the approach axis (a yaw while pointing down)
            spin = twist_to_world(e[5], frame, quat)[3:]
            np.testing.assert_allclose(spin, -site[:, 2], atol=0.05)
            assert spin[2] > 0.99
    env.close()


def test_frame_event_switches_the_teleop_frame(tmp_path):
    env = franka_sim.make("reach", action_mode="ee_pose")
    fwd = TeleopInput(twist=np.array([1.0, 0, 0, 0, 0, 0]))
    device = ScriptedDevice([fwd] * 3 + [TeleopInput(events=["frame"])] + [TeleopInput(events=["save"])])
    session = TeleopSession(env, device, None, None, realtime_factor=0, console_size=(48, 64))
    assert session.frame == "world"
    session.run(1)
    assert session.frame == "heading"
    env.close()
