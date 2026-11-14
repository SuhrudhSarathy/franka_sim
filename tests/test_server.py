import socket
import threading
import time

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("websockets")

from franka_sim.server import RobotServer, SimRobot  # noqa: E402


def test_delta_commands_accumulate_and_apply_once():
    robot = SimRobot("reach", seed=0, env_kwargs={"randomize_start": False, "init_noise": 0.0})
    p0 = robot.observation()["ee_pos"].copy()
    for _ in range(4):  # four deltas between two ticks add up
        robot.command([0, 0.01, 0, 0, 0, 0, 0], "delta_ee_pose")
    for _ in range(20):  # ... and are applied once: the arm then holds
        robot.tick()
    moved = robot.observation()["ee_pos"] - p0
    assert abs(moved[1] - 0.04) < 3e-3 and np.abs(moved[[0, 2]]).max() < 3e-3
    robot.close()


def test_absolute_targets_hold_and_joint_vel_times_out():
    robot = SimRobot("reach", seed=0, watchdog=0.05)
    q0 = robot.observation()["joint_pos"].copy()
    target = q0 + np.array([0.2, 0, 0, 0, 0, 0, 0])
    robot.command(np.r_[target, 0.0], "joint_pos")
    for _ in range(20):
        robot.tick()
    assert np.abs(robot.observation()["joint_pos"] - target).max() < 0.01

    robot.command([0.5, 0, 0, 0, 0, 0, 0, 0], "joint_vel")
    time.sleep(0.1)  # longer than the watchdog: the velocity command has expired
    q1 = robot.observation()["joint_pos"].copy()
    for _ in range(10):
        robot.tick()
    assert abs(robot.observation()["joint_pos"][0] - q1[0]) < 0.01
    with pytest.raises(ValueError, match="8 values"):
        robot.command([0, 0, 0], "joint_pos")
    robot.close()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_server_lockstep_and_realtime_over_websocket():
    from franka_sim.client import RobotClient, RobotError

    port = _free_port()
    server = RobotServer(SimRobot("buttons", seed=0), port=port, realtime=False, viewer=False)
    results, errors = {}, []

    def client():
        try:
            for _ in range(100):  # wait for the server to listen
                try:
                    robot = RobotClient(f"ws://127.0.0.1:{port}/ws", cameras=("wrist",), image_size=(32, 48))
                    break
                except OSError:
                    time.sleep(0.05)
            with robot:
                results["info"] = robot.info()
                obs, info = robot.reset(seed=5, options={"target": 1})
                results["reset_obs"] = obs
                results["cond"] = info["cond_target"]
                q = obs["joint_pos"]
                obs, reward, term, trunc, info = robot.step(np.r_[q, 0.0], mode="joint_pos")
                results["step_obs"] = obs
                results["privileged"] = robot.observe(privileged=True)
                robot.run()
                with pytest.raises(RobotError, match="pause"):
                    robot.step(np.r_[q, 0.0], mode="joint_pos")
                p0 = robot.observe()["ee_pos"]
                for _ in range(10):
                    robot.act([0, 0, 0.005, 0, 0, 0, 0], mode="delta_ee_pose")
                    time.sleep(0.05)
                time.sleep(0.3)
                results["rise"] = robot.observe()["ee_pos"][2] - p0[2]
                results["state"] = robot.state()
        except Exception as e:  # surfaced below
            errors.append(e)
        finally:
            server.shutdown()

    t = threading.Thread(target=client)
    t.start()
    server.serve_forever()  # rendering must happen on this (main) thread
    t.join(timeout=10)
    assert not errors, errors

    assert results["info"]["action_modes"]["delta_ee_pose"] == 7
    assert results["cond"] == 1
    obs = results["step_obs"]
    assert obs["rgb_wrist"].shape == (32, 48, 3) and obs["rgb_wrist"].dtype == np.uint8
    assert "button" in obs["instruction"]
    from franka_sim.benchmark import ROBOT_KEYS

    assert set(obs) == {*ROBOT_KEYS, "rgb_wrist", "instruction"}  # vision protocol by default
    assert set(results["privileged"]) > set(ROBOT_KEYS) | {"instruction"}  # plus task state
    assert 0.03 < results["rise"] < 0.06  # 10 x 5 mm streamed in real time
    assert results["state"]["running"] and results["state"]["task"] == "buttons"
