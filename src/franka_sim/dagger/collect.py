"""DAgger data collection with the scripted expert as the labeler (no human needed).

Each step, the learner and the expert both look at the current state. With probability ``beta``
the expert's action is executed (classic DAgger's mixture policy ``beta * expert + (1 - beta) *
learner``), otherwise the learner's; either way the expert's action is recorded as the label. Round
0 is usually ``beta=1`` (plain demonstrations); later rounds lower ``beta`` toward 0, so the data
covers the states the *learner* visits.

The expert is the task's scripted policy. It is re-anchored to the robot's actual commanded pose
every step (``ScriptedPolicy.track``), so its label is a correction from where the learner put the
robot, not a continuation of its own plan.

Recorded per step, on top of the usual demo format (``recording.py``):

* ``actions``: what was executed, in the env's (= the learner's) action mode;
* ``actor``: who executed it: 0 learner, 1 human, 2 scripted expert (``ACTORS``);
* ``intervention``: True where a human was in control (always False here);
* ``expert/actions``: the expert's label in the learner's action mode;
* ``expert/joint_pos``, ``expert/ee_pose``, ``expert/gripper``: the same label in the absolute
  representations (what ``commands/*`` holds for the executed action).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from franka_sim.benchmark import policy_observation
from franka_sim.recording import STATE_FIELDS, EpisodeBuffer

ACTOR_LEARNER, ACTOR_HUMAN, ACTOR_EXPERT = 0, 1, 2
ACTORS = {ACTOR_LEARNER: "learner", ACTOR_HUMAN: "human", ACTOR_EXPERT: "expert"}


def expert_labels(env, expert_action: np.ndarray) -> dict[str, np.ndarray]:
    """The expert's ``ee_pose`` actions ``(N, 8)`` as labels in every stored representation."""
    a = np.asarray(expert_action, dtype=np.float64)
    quat = a[:, 3:7] / np.linalg.norm(a[:, 3:7], axis=1, keepdims=True)
    grip = np.clip(a[:, 7], 0.0, 1.0)
    # One IK preview, restated in every representation (same as ``env.convert_action``).
    q_meas = env.sim.read("qpos")[:, env.arm_qadr]
    last = env.controller.preview("ee_pose", a[:, :7], q_meas)["last"]
    return {
        "expert/actions": np.concatenate([env.controller.express(env.action_mode, last), grip[:, None]], axis=1),
        "expert/joint_pos": env.controller.express("joint_pos", last),
        "expert/ee_pose": np.concatenate([a[:, :3], quat], axis=1),
        "expert/delta_ee_pose": env.controller.express("delta_ee_pose", last),
        "expert/gripper": grip,
    }


def make_expert(env):
    """The task's scripted expert as a noise-free labeler for an env in any action mode."""
    from franka_sim.policies import make_policy

    return make_policy(env, noise=0.0, any_env_mode=True)


def record_step(ep: EpisodeBuffer, env, i: int, prev_obs: dict, states: dict, executed: np.ndarray,
                reward: float, success: bool, actor: int, intervention: bool, labels: dict | None) -> None:
    """Append one transition of env ``i`` (``executed`` is the env-mode action actually applied)."""
    for k, v in prev_obs.items():
        ep.obs.setdefault(k, []).append(v[i])
    ep.actions.append(executed[i])
    for k in ep.commands:
        ep.commands[k].append(np.array(env.last_command[k][i]))
    for f in STATE_FIELDS:
        ep.states[f].append(states[f][i])
    ep.rewards.append(float(reward))
    ep.success.append(bool(success))
    ep.extras.setdefault("actor", []).append(np.int8(actor))
    ep.extras.setdefault("intervention", []).append(bool(intervention))
    for k, v in (labels or {}).items():
        ep.extras.setdefault(k, []).append(v[i])


@dataclass
class DaggerStats:
    attempted: int = 0
    saved: int = 0
    succeeded: int = 0
    lengths: list = field(default_factory=list)
    expert_fraction: list = field(default_factory=list)  # share of executed steps by the expert

    @property
    def success_rate(self) -> float:
        return self.succeeded / max(self.attempted, 1)


def collect_dagger(env, learner, num_episodes: int, writer=None, *, beta: float = 0.0, mixing: str = "step",
                   obs_mode: str = "vision", mask_camera: str | None = None,
                   conditions: Callable[[int], dict] | list[dict] | None = None, keep_failures: bool = True,
                   post_success_steps: int = 10, seed: int | None = None,
                   on_episode: Callable[[int, EpisodeBuffer, bool], None] | None = None) -> DaggerStats:
    """One DAgger round: roll out the beta-mixture in every env of ``env``, label with the expert.

    Args:
        env: a ``FrankaEnv`` in the learner's action mode (any batch size).
        learner: benchmark-style policy (``reset(env_ids, obs)`` / ``act(obs)``) in ``env.action_mode``
            seeing ``policy_observation(env, raw, obs_mode, mask_camera)``; ``None`` = expert only.
        beta: probability that the expert acts (per step, or per episode with ``mixing="episode"``).
        keep_failures: save episodes that failed too; their labels are the point of DAgger.
        conditions: reset options per episode (as in ``collect_demos``).
        seed: base seed; episode ``k`` is reset with ``seed + k`` (keep below ``EVAL_SEED``).
    """
    if learner is None:
        beta = 1.0
    if mixing not in ("step", "episode"):
        raise ValueError("mixing must be 'step' or 'episode'")
    n = env.num_envs
    rng = np.random.default_rng(seed)
    expert = make_expert(env)
    stats = DaggerStats()
    buffers: list[EpisodeBuffer | None] = [None] * n
    expert_episode = np.zeros(n, dtype=bool)
    ep_wrong = np.zeros(n, dtype=bool)
    started = 0
    base = 0 if seed is None else seed

    def options_for(k: int) -> dict:
        if conditions is None:
            return {}
        return dict(conditions(k) if callable(conditions) else conditions[k % len(conditions)])

    def start(i: int):
        nonlocal started
        k, opts = started, options_for(started)
        started += 1
        raw, info = env.reset(seed=base + k, env_ids=[i], options=opts or None)
        buffers[i] = EpisodeBuffer(
            cond={key[5:]: np.asarray(v[i]).copy() for key, v in info.items() if key.startswith("cond_")},
            instruction=env.task.instruction(i), options=opts)
        expert_episode[i] = rng.random() < beta
        ep_wrong[i] = False
        return raw

    raw = None
    for i in range(n):
        raw = start(i)
    expert.reset(np.arange(n), raw)
    if learner is not None:
        learner.reset(np.arange(n), policy_observation(env, raw, obs_mode, mask_camera))

    while stats.saved < num_episodes and stats.attempted < 4 * num_episodes:
        states = {f: env.sim.read(f) for f in STATE_FIELDS}
        expert.track(env.last_command["ee_pose"])
        labels = expert_labels(env, expert.act(raw))
        if learner is not None:
            learner_action = np.asarray(learner.act(policy_observation(env, raw, obs_mode, mask_camera)), float)
            use_expert = expert_episode if mixing == "episode" else rng.random(n) < beta
            action = np.where(use_expert[:, None], labels["expert/actions"], learner_action)
        else:
            use_expert = np.ones(n, dtype=bool)
            action = labels["expert/actions"]
        prev_raw = raw
        raw, reward, terminated, truncated, info = env.step(action)
        executed = env.executed_action()
        ep_wrong |= info["wrong_target"]
        reset_ids = []
        for i in range(n):
            ep = buffers[i]
            if ep is None:
                continue
            actor = ACTOR_EXPERT if use_expert[i] else ACTOR_LEARNER
            record_step(ep, env, i, prev_raw, states, executed, reward[i], info["success"][i], actor, False, labels)
            if info["success"][i] and ep.steps_since_success < 0:
                ep.steps_since_success = 0
            elif ep.steps_since_success >= 0:
                ep.steps_since_success += 1
            succeeded = ep.steps_since_success >= post_success_steps
            if not (succeeded or truncated[i] or ep_wrong[i]):
                continue
            ok = ep.steps_since_success >= 0 and not ep_wrong[i]
            stats.attempted += 1
            stats.succeeded += int(ok)
            stats.lengths.append(len(ep))
            stats.expert_fraction.append(float(np.mean(np.asarray(ep.extras["actor"]) == ACTOR_EXPERT)))
            if (ok or keep_failures) and stats.saved < num_episodes:
                if writer is not None:
                    writer.write(ep, ok, env_id=i)
                stats.saved += 1
            if on_episode is not None:
                on_episode(stats.attempted, ep, ok)
            buffers[i] = None
            if stats.saved + sum(b is not None for b in buffers) < num_episodes:
                raw = start(i)
                reset_ids.append(i)
        if reset_ids:
            ids = np.array(reset_ids)
            expert.reset(ids, raw)
            if learner is not None:
                learner.reset(ids, policy_observation(env, raw, obs_mode, mask_camera))
        if all(b is None for b in buffers):
            break
    return stats
