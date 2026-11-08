import numpy as np

import franka_sim  # noqa: F401
from franka_sim.benchmark import ROBOT_KEYS, SUITE, episode_options, evaluate, scripted_factory
from franka_sim.policies import make_policy
from franka_sim.tasks import TASKS


def test_suite_is_registered():
    assert set(SUITE) <= set(TASKS)
    assert episode_options("hammer_nails", 5) == {"target": 1}
    assert episode_options("hammer_nails_long", 5) == {}


class _Idle:
    def __init__(self, env):
        self.env = env
        self.seen = None

    def reset(self, env_ids, obs):
        self.seen = obs

    def act(self, obs):
        self.seen = obs
        return np.zeros((self.env.num_envs, self.env.action_dim))


def test_idle_policy_scores_zero_and_sees_only_vision_keys():
    holder = {}

    def factory(env):
        holder["p"] = _Idle(env)
        return holder["p"]

    r = evaluate("buttons", factory, episodes=3, num_envs=3, cameras=("wrist",), image_size=(64, 64),
                 mask_camera="top", action_mode="joint_delta", max_episode_steps=5)
    assert r.success_rate == 0 and r.mean_progress == 0 and r.wrong_target_rate == 0
    obs = holder["p"].seen
    assert set(obs) == set(ROBOT_KEYS) | {"rgb_wrist", "instruction", "target_mask"}
    assert obs["target_mask"].shape == (3, 64, 64) and obs["target_mask"].max() == 255
    assert all(s.startswith("press the") for s in obs["instruction"])


def test_scripted_expert_and_misled_expert():
    ok = evaluate("hammer_nails", scripted_factory(), episodes=4, num_envs=4, obs_mode="privileged",
                  cameras=(), action_mode="ee_pose")
    assert ok.success_rate == 1 and ok.per_target_success == {0: 1, 1: 1, 2: 1, 3: 1}

    class Misled:  # the expert aiming at the nail right of the requested one
        def __init__(self, env):
            self.p = make_policy(env)

        def _shift(self, obs):
            return {**obs, "target_onehot": np.roll(obs["target_onehot"], 1, axis=1)}

        def reset(self, env_ids, obs):
            self.p.reset(env_ids, self._shift(obs))

        def act(self, obs):
            return self.p.act(self._shift(obs))

    bad = evaluate("hammer_nails", Misled, episodes=4, num_envs=4, obs_mode="privileged", cameras=(),
                   action_mode="ee_pose")
    assert bad.success_rate == 0 and bad.wrong_target_rate == 1


def test_evaluation_episodes_are_fixed():
    # Episode k is reset with seed + k: same layout whichever env slot runs it.
    a = franka_sim.make("peg_insertion", num_envs=2)
    b = franka_sim.make("peg_insertion", num_envs=3)
    oa, _ = a.reset(seed=7, env_ids=[1], options=episode_options("peg_insertion", 2))
    ob, _ = b.reset(seed=7, env_ids=[0], options=episode_options("peg_insertion", 2))
    np.testing.assert_allclose(oa["obj_pos"][1], ob["obj_pos"][0])
    np.testing.assert_allclose(oa["joint_pos"][1], ob["joint_pos"][0])
