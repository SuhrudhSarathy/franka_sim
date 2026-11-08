"""The franka_sim benchmark: task suite, seeds, observation protocol and the evaluation loop.

Two kinds of tasks (see ``SUITE``):

* **single**: the instruction names one target among several (nail, button, peg, gear, part,
  connector). Tests language adherence: ``wrong_target`` counts episodes where the policy acted on
  another target (which also fails the episode).
* **long**: every target of the scene must be done, in any order. Tests long-horizon completion:
  ``progress`` counts sub-goals done.

Episode ``k`` of an evaluation is reset with seed ``seed + k`` (single tasks also get target
``k % num_targets``), so every policy is scored on exactly the same layouts and conditions however
fast it finishes episodes. Demonstrations use seeds below ``EVAL_SEED``; evaluation uses
``EVAL_SEED`` and above.

Policy interface (the scripted experts implement it)::

    policy = factory(env)            # env: FrankaEnv built by ``evaluate``
    policy.reset(env_ids, obs)       # called after env_ids were reset
    actions = policy.act(obs)        # (num_envs, env.action_dim)

``obs`` holds the env observations filtered by ``obs_mode`` plus ``instruction`` (an object array
of strings) and, with ``mask_camera``, ``target_mask`` ((N, H, W) uint8, zeros when the task has
no single target).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

import franka_sim

EVAL_SEED = 100_000

# name -> (kind, number of distinct targets the condition chooses from; 0 for long tasks)
SUITE: dict[str, tuple[str, int]] = {
    "hammer_nails": ("single", 4),
    "hammer_nails_long": ("long", 0),
    "buttons": ("single", 4),
    "buttons_long": ("long", 0),
    "peg_insertion": ("single", 3),
    "peg_insertion_long": ("long", 0),
    "gear_assembly": ("single", 3),
    "gear_assembly_long": ("long", 0),
    "kitting": ("single", 4),
    "kitting_long": ("long", 0),
    "pcb_insertion": ("single", 3),
}

# Observations a deployable policy may use ("vision" mode). Everything else the env returns is
# privileged simulator state (object poses, target positions...) for oracle baselines only.
ROBOT_KEYS = ("joint_pos", "joint_vel", "gripper_width", "ee_pos", "ee_quat")


def episode_options(task: str, k: int) -> dict:
    """Reset options of evaluation episode ``k``: balanced targets for single tasks."""
    kind, n = SUITE.get(task, ("single", 0))
    return {"target": k % n} if kind == "single" and n else {}


@dataclass
class EvalResult:
    task: str
    kind: str
    episodes: int
    success_rate: float
    wrong_target_rate: float
    mean_progress: float  # fraction of sub-goals done at the end of the episode
    mean_success_steps: float | None  # steps to success, over successful episodes
    per_target_success: dict[int, float] = field(default_factory=dict)  # single tasks
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _filter(obs: dict, obs_mode: str) -> dict:
    if obs_mode == "privileged":
        return dict(obs)
    if obs_mode == "vision":
        return {k: v for k, v in obs.items() if k in ROBOT_KEYS or k.startswith("rgb_")}
    raise ValueError("obs_mode must be 'vision' or 'privileged'")


def evaluate(task: str, policy_factory: Callable, *, episodes: int = 50, num_envs: int = 10,
             seed: int = EVAL_SEED, obs_mode: str = "vision", cameras=("front", "wrist"),
             image_size: tuple[int, int] = (256, 256), mask_camera: str | None = None,
             action_mode: str = "joint_pos", backend: str = "cpu", max_episode_steps: int | None = None,
             progress: Callable[[str], None] | None = None) -> EvalResult:
    """Roll out a policy on ``episodes`` fixed evaluation episodes of ``task`` and score it."""
    t0 = time.time()
    kind = SUITE.get(task, ("single", 0))[0]
    num_envs = min(num_envs, episodes)
    env = franka_sim.make(task, num_envs=num_envs, backend=backend, action_mode=action_mode,
                          cameras=cameras, image_size=image_size, max_episode_steps=max_episode_steps)
    num_steps = env.task.num_steps
    policy = policy_factory(env)

    def augment(raw: dict) -> dict:
        obs = _filter(raw, obs_mode)
        obs["instruction"] = np.array(env.instructions(), dtype=object)
        if mask_camera is not None:
            masks = [env.render_target_mask(mask_camera, env_id=i) for i in range(num_envs)]
            obs["target_mask"] = np.stack([m if m is not None else np.zeros(image_size, np.uint8) for m in masks])
        return obs

    slot_episode = np.full(num_envs, -1)
    next_episode = 0
    obs = None

    def start(i: int):
        nonlocal next_episode, obs
        k = next_episode
        next_episode += 1
        slot_episode[i] = k
        raw, _ = env.reset(seed=seed + k, env_ids=[i], options=episode_options(task, k))
        return raw

    raw = None
    for i in range(num_envs):
        raw = start(i)
    obs = augment(raw)
    policy.reset(np.arange(num_envs), obs)

    success = np.zeros(episodes, dtype=bool)
    wrong = np.zeros(episodes, dtype=bool)
    prog = np.zeros(episodes)
    steps = np.zeros(episodes, dtype=np.int64)
    targets = np.full(episodes, -1)
    ep_wrong = np.zeros(num_envs, dtype=bool)
    finished = 0
    while finished < episodes:
        raw, _, terminated, truncated, info = env.step(policy.act(obs))
        ep_wrong |= info["wrong_target"]
        done_ids = []
        for i in np.flatnonzero(slot_episode >= 0):
            if not (terminated[i] or truncated[i] or ep_wrong[i]):
                continue
            k = slot_episode[i]
            success[k] = bool(info["success"][i]) and not ep_wrong[i]
            wrong[k] = ep_wrong[i]
            prog[k] = float(info["progress"][i]) / num_steps
            steps[k] = info["elapsed_steps"][i]
            if "cond_target" in info:
                targets[k] = int(np.ravel(info["cond_target"][i])[0])
            finished += 1
            ep_wrong[i] = False
            if next_episode < episodes:
                raw = start(i)
                done_ids.append(i)
            else:
                slot_episode[i] = -1
            if progress:
                progress(f"  {task} episode {k}: {'success' if success[k] else 'fail'} after {steps[k]} steps")
        obs = augment(raw)
        if done_ids:
            policy.reset(np.array(done_ids), obs)
    env.close()

    per_target = {int(t): float(success[targets == t].mean()) for t in np.unique(targets[targets >= 0])}
    return EvalResult(task=task, kind=kind, episodes=episodes, success_rate=float(success.mean()),
                      wrong_target_rate=float(wrong.mean()), mean_progress=float(prog.mean()),
                      mean_success_steps=float(steps[success].mean()) if success.any() else None,
                      per_target_success=per_target, seconds=round(time.time() - t0, 1))


def scripted_factory(noise: float = 0.0, seed: int = 0) -> Callable:
    """The scripted expert as a benchmark policy (needs ``obs_mode="privileged"``, ``ee_pose``)."""
    from franka_sim.policies import make_policy

    return lambda env: make_policy(env, noise=noise, seed=seed)
