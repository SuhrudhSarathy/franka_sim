"""Record demonstrations to HDF5 and read them back.

File layout (robomimic-style)::

    /data                         attrs: task, action_mode, control_freq, env_kwargs (json),
                                         num_episodes, total_steps
    /data/demo_<k>                attrs: num_samples, success, instruction, options (json), env_id
        obs/<key>                 (T, ...) observation before each action (images uint8, gzip)
        actions                   (T, A)   action sent to the env (in its action_mode)
        commands/joint_pos        (T, 7)   the same command as IK joint targets
        commands/ee_pose          (T, 7)   ... as an end-effector pose (xyz + wxyz)
        commands/delta_ee_pose    (T, 6)   ... as a delta_ee_pose action (dxyz + world-frame rotation
                                           vector about the gripper tip; replays the command exactly)
        commands/gripper          (T,)     ... gripper closure in [0, 1]
        rewards, success, dones   (T,)
        <extra>                   (T, ...) optional per-step arrays, e.g. DAgger's ``actor``,
                                           ``intervention`` and ``expert/<representation>`` labels
        states/<field>            (T, ...) qpos, qvel, ctrl, mocap_pos, mocap_quat (exact replay)
        cond/<key>                conditioning variables of the episode (e.g. target nail index)
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

STATE_FIELDS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")


@dataclass
class EpisodeBuffer:
    obs: dict[str, list] = field(default_factory=dict)
    actions: list = field(default_factory=list)
    commands: dict[str, list] = field(default_factory=lambda: {"joint_pos": [], "ee_pose": [], "delta_ee_pose": [], "gripper": []})
    states: dict[str, list] = field(default_factory=lambda: {f: [] for f in STATE_FIELDS})
    rewards: list = field(default_factory=list)
    success: list = field(default_factory=list)
    cond: dict[str, np.ndarray] = field(default_factory=dict)
    instruction: str = ""
    options: dict = field(default_factory=dict)
    steps_since_success: int = -1
    extras: dict[str, list] = field(default_factory=dict)  # per-step arrays, path -> list

    def __len__(self) -> int:
        return len(self.actions)


class DemoWriter:
    """Append episodes to an HDF5 file (created, or appended to if it exists)."""

    def __init__(self, path: str | Path, env=None, compress_images: bool = True, **attrs):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "a")
        self.data = self.file.require_group("data")
        self.compress_images = compress_images
        if env is not None:
            meta = {
                "task": env.task.name,
                "action_mode": env.action_mode,
                "control_freq": 1.0 / env.control_dt,
                "cameras": list(env.cameras),
                "image_size": list(env.image_size),
                "num_envs": env.num_envs,
            }
            for k, v in meta.items():
                self.data.attrs[k] = json.dumps(v) if isinstance(v, list) else v
        for k, v in attrs.items():
            self.data.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
        self.num_episodes = int(self.data.attrs.get("num_episodes", 0))
        self.total_steps = int(self.data.attrs.get("total_steps", 0))

    def write(self, ep: EpisodeBuffer, success: bool, env_id: int = 0) -> str:
        name = f"demo_{self.num_episodes}"
        g = self.data.create_group(name)
        T = len(ep)
        g.attrs.update(num_samples=T, success=bool(success), instruction=ep.instruction,
                       options=json.dumps(_jsonable(ep.options)), env_id=env_id)
        obs = g.create_group("obs")
        for k, v in ep.obs.items():
            arr = np.stack(v)
            if arr.dtype == np.uint8 and self.compress_images:
                obs.create_dataset(k, data=arr, compression="gzip", compression_opts=4,
                                   chunks=(1,) + arr.shape[1:])
            else:
                obs.create_dataset(k, data=arr.astype(np.float32) if arr.dtype == np.float64 else arr)
        g.create_dataset("actions", data=np.stack(ep.actions).astype(np.float32))
        cmd = g.create_group("commands")
        for k, v in ep.commands.items():
            cmd.create_dataset(k, data=np.stack(v).astype(np.float32))
        st = g.create_group("states")
        for k, v in ep.states.items():
            st.create_dataset(k, data=np.stack(v))
        for k, v in ep.extras.items():
            arr = np.stack(v)
            g.create_dataset(k, data=arr.astype(np.float32) if arr.dtype == np.float64 else arr)
        g.create_dataset("rewards", data=np.asarray(ep.rewards, dtype=np.float32))
        g.create_dataset("success", data=np.asarray(ep.success, dtype=bool))
        dones = np.zeros(T, dtype=bool)
        dones[-1] = True
        g.create_dataset("dones", data=dones)
        cond = g.create_group("cond")
        for k, v in ep.cond.items():
            cond.create_dataset(k, data=np.asarray(v))
        self.num_episodes += 1
        self.total_steps += T
        self.data.attrs["num_episodes"] = self.num_episodes
        self.data.attrs["total_steps"] = self.total_steps
        self.file.flush()
        return name

    def close(self) -> None:
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


@dataclass
class CollectStats:
    attempted: int = 0
    saved: int = 0
    succeeded: int = 0
    lengths: list = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.succeeded / max(self.attempted, 1)


def collect_demos(env, policy, num_episodes: int, writer: DemoWriter | None = None, *,
                  conditions: Callable[[int], dict] | list[dict] | None = None, keep_failures: bool = False,
                  post_success_steps: int = 10, max_attempts: int | None = None,
                  on_episode: Callable[[int, EpisodeBuffer, bool], None] | None = None,
                  seed: int | None = None, progress: bool | dict = False) -> CollectStats:
    """Roll out ``policy`` in every env of ``env`` until ``num_episodes`` episodes are saved.

    Args:
        conditions: reset options per episode, as a list or a function of the episode number,
            e.g. ``lambda k: {"target": k % 4}`` for a dataset balanced over nails. ``None`` lets
            the task sample its own condition.
        keep_failures: also save episodes that did not succeed (``attrs["success"]`` tells them apart).
        post_success_steps: keep recording this many steps after success (the "settle" phase).
        max_attempts: stop after this many attempted episodes (default ``4 * num_episodes``).
        on_episode: callback ``(episode_number, buffer, success)`` after each finished episode.
        progress: show a tqdm bar of saved episodes (a dict: extra ``tqdm`` arguments, e.g. ``desc``,
            ``position``).
    """
    n = env.num_envs
    stats = CollectStats()
    max_attempts = max_attempts or 4 * num_episodes
    started = 0

    def options_for(k: int) -> dict:
        if conditions is None:
            return {}
        return dict(conditions(k) if callable(conditions) else conditions[k % len(conditions)])

    buffers: list[EpisodeBuffer | None] = [None] * n
    obs, info = env.reset(seed=seed)

    def start(i: int, obs, info):
        nonlocal started
        opts = options_for(started)
        started += 1
        if opts:
            obs, info = env.reset(env_ids=[i], options=opts)
        policy.reset([i], obs)
        buffers[i] = EpisodeBuffer(
            cond={k[5:]: np.asarray(v[i]).copy() for k, v in info.items() if k.startswith("cond_")},
            instruction=env.task.instruction(i), options=opts)
        return obs, info

    for i in range(n):
        obs, info = start(i, obs, info)

    bar = _progress_bar(progress, total=num_episodes, unit="ep", desc="recording")
    while stats.saved < num_episodes and stats.attempted < max_attempts:
        states = {f: env.sim.read(f) for f in STATE_FIELDS}
        action = policy.act(obs)
        prev_obs = obs
        obs, reward, terminated, truncated, step_info = env.step(action)
        step_success = step_info["success"]
        for i in range(n):
            ep = buffers[i]
            if ep is None:
                continue
            for k, v in prev_obs.items():
                ep.obs.setdefault(k, []).append(v[i])
            ep.actions.append(action[i])
            for k in ep.commands:
                ep.commands[k].append(np.array(env.last_command[k][i]))
            for f in STATE_FIELDS:
                ep.states[f].append(states[f][i])
            ep.rewards.append(float(reward[i]))
            ep.success.append(bool(step_success[i]))
            if step_success[i] and ep.steps_since_success < 0:
                ep.steps_since_success = 0
            elif ep.steps_since_success >= 0:
                ep.steps_since_success += 1

            succeeded = ep.steps_since_success >= post_success_steps
            stuck = policy.finished(i) and ep.steps_since_success < 0 and len(ep) > 20
            if not (succeeded or truncated[i] or stuck):
                continue
            ok = ep.steps_since_success >= 0
            stats.attempted += 1
            stats.succeeded += int(ok)
            stats.lengths.append(len(ep))
            if (ok or keep_failures) and stats.saved < num_episodes:
                if writer is not None:
                    writer.write(ep, ok, env_id=i)
                stats.saved += 1
                if bar is not None:
                    bar.update(1)
            if bar is not None:
                bar.set_postfix(success=f"{stats.succeeded}/{stats.attempted}", refresh=False)
            if on_episode is not None:
                on_episode(stats.attempted, ep, ok)
            buffers[i] = None
            if stats.saved + sum(b is not None for b in buffers) < num_episodes:
                obs, reset_info = env.reset(env_ids=[i])
                obs, _ = start(i, obs, reset_info)
    if bar is not None:
        bar.close()
    return stats


def _progress_bar(progress: bool | dict, **defaults):
    """A tqdm bar (or ``None`` when ``progress`` is falsy). Logs to a file update once a minute."""
    if not progress:
        return None
    import sys

    from tqdm import tqdm

    kw = {"dynamic_ncols": True, "smoothing": 0.05, **defaults, **(progress if isinstance(progress, dict) else {})}
    if not sys.stderr.isatty():
        kw.setdefault("mininterval", 60.0)
    return tqdm(**kw)


# ------------------------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------------------------
def _read_group(g: h5py.Group) -> dict:
    return {k: (_read_group(v) if isinstance(v, h5py.Group) else v[()]) for k, v in g.items()}


def load_episode(path: str | Path, index: int | str) -> dict:
    """Load one episode into a nested dict of numpy arrays (plus its attrs under ``"attrs"``)."""
    with h5py.File(path, "r") as f:
        g = f["data"][index if isinstance(index, str) else f"demo_{index}"]
        ep = _read_group(g)
        ep["attrs"] = dict(g.attrs)
        ep["attrs"]["options"] = json.loads(ep["attrs"].get("options", "{}"))
    return ep


def iter_episodes(path: str | Path, only_success: bool = False) -> Iterator[dict]:
    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
        succ = {n: bool(f["data"][n].attrs["success"]) for n in names}
    for n in names:
        if succ[n] or not only_success:
            yield load_episode(path, n)


def dataset_info(path: str | Path) -> dict:
    with h5py.File(path, "r") as f:
        info = {k: (json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v) for k, v in f["data"].attrs.items()}
        info["episodes"] = len(f["data"])
    return info
