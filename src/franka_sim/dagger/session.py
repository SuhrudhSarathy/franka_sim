"""Human-gated DAgger (HG-DAgger) and teleop demo collection, in real time.

The learner drives the robot; the operator watches (MuJoCo viewer and/or the teleop console) and
takes over whenever it goes wrong, steers it back on track, and hands control back. Every step is
recorded with who was in control; the operator's steps (``intervention``) are the corrective labels.
Without a learner the operator is always in control, which collects plain teleop demonstrations.

Control switches are bumpless: the operator's commands are ``delta_ee_pose`` steps from the
robot's *commanded* pose, and the gripper starts where the learner left it. When control returns to
the learner, ``learner.interrupt(env_ids, obs)`` is called if it exists (``reset`` otherwise) so a
policy that executes action chunks drops the actions it queued before the takeover.

Whatever the operator does is recorded in the learner's action space (``env.executed_action``), so
interventions are labels the learner can be trained on directly; ``commands/*`` additionally hold
the absolute joint and end-effector representations. With ``scripted_labels``, the scripted expert
shadows the whole episode and its labels are stored as ``expert/*`` too (as in ``collect``).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from franka_sim._signals import interruptible
from franka_sim.benchmark import policy_observation
from franka_sim.dagger.collect import (
    ACTOR_HUMAN,
    ACTOR_LEARNER,
    expert_labels,
    make_expert,
    record_step,
)
from franka_sim.recording import STATE_FIELDS, EpisodeBuffer
from franka_sim.teleop.base import TELEOP_FRAMES, TeleopDevice, twist_to_world

LEARNER, HUMAN = "LEARNER", "HUMAN"


class TeleopSession:
    """Real-time loop: learner and/or operator drive one env; episodes go to ``writer``.

    Args:
        env: a single-env ``FrankaEnv`` in the learner's action mode (without a learner: the mode
            the recorded ``actions`` should use).
        device: a ``TeleopDevice`` (e.g. ``KeyboardTeleop``).
        learner: benchmark-style policy, or ``None`` for pure teleop.
        writer: a ``DemoWriter`` (``None``: nothing is saved).
        obs_mode / mask_camera: the learner's observation protocol (see ``policy_observation``).
        console_cameras: cameras streamed to the device's console.
        console_size: ``(height, width)`` of those frames.
        speed: operator translation speed at full deflection (m/s); ``rot_speed`` in rad/s.
        frame: initial operator frame of the device's twist, ``"world"`` or ``"heading"`` (see
            ``teleop.base``); the ``frame`` event (key M) switches between them.
        auto_takeover: any motion input while the learner drives takes control immediately.
        scripted_labels: also record the scripted expert's labels (if the task has one).
        realtime_factor: wall-clock pace (0.5 = half speed, for slow learners; 0 = as fast as possible).
        post_success_steps: keep recording this long after success before ending the episode.
        keep_failures: save episodes that end without success (they hold interventions too).
        conditions: reset options per episode, e.g. ``lambda k: {"target": k % 4}``.
        seed: episode ``k`` is reset with ``seed + k`` (keep below ``EVAL_SEED``).
        viewer: also open MuJoCo's native viewer on the env (CPU backend; macOS: ``mjpython``).
        on_episode: callback ``(index, buffer, success, saved)``.
    """

    def __init__(self, env, device: TeleopDevice, learner=None, writer=None, *, obs_mode: str = "vision",
                 mask_camera: str | None = None, console_cameras=("back", "wrist"),
                 console_size: tuple[int, int] = (300, 400), speed: float = 0.15, rot_speed: float = 1.0,
                 frame: str = "world", auto_takeover: bool = True, scripted_labels: bool = True, realtime_factor: float = 1.0,
                 post_success_steps: int = 10, keep_failures: bool = True,
                 conditions: Callable[[int], dict] | list[dict] | None = None, seed: int = 0,
                 viewer: bool = False, on_episode: Callable | None = None):
        if env.num_envs != 1:
            raise ValueError("TeleopSession drives a single env (num_envs=1)")
        self.env, self.device, self.learner, self.writer = env, device, learner, writer
        self.obs_mode, self.mask_camera = obs_mode, mask_camera
        self.console_cameras, self.console_size = tuple(console_cameras), tuple(console_size)
        self.speed, self.rot_speed, self.speed_scale = speed, rot_speed, 1.0
        if frame not in TELEOP_FRAMES:
            raise ValueError(f"unknown teleop frame {frame!r}; expected one of {TELEOP_FRAMES}")
        self.frame = frame
        self.auto_takeover = auto_takeover
        self.realtime_factor = realtime_factor
        self.post_success_steps = post_success_steps
        self.keep_failures = keep_failures
        self.conditions = conditions
        self.seed = seed
        self.on_episode = on_episode
        self.expert = None
        if scripted_labels:
            try:
                self.expert = make_expert(env)
            except ValueError:  # no scripted expert for this task
                self.expert = None
        self._viewer = None
        if viewer:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(env.model, env.sim.datas[0])
        self.saved = self.discarded = self.episodes_started = 0
        self.successes = 0
        self.learner_ms = 0.0

    # ------------------------------------------------------------------------------------------
    def _options(self, k: int) -> dict:
        if self.conditions is None:
            return {}
        c = self.conditions
        return dict(c(k) if callable(c) else c[k % len(c)])

    def _start_episode(self):
        k = self.episodes_started
        self.episodes_started += 1
        opts = self._options(k)
        raw, info = self.env.reset(seed=self.seed + k, options=opts or None)
        self.ep = EpisodeBuffer(cond={key[5:]: np.asarray(v[0]).copy() for key, v in info.items()
                                      if key.startswith("cond_")},
                                instruction=self.env.instructions()[0], options=opts)
        self.control = LEARNER if self.learner is not None else HUMAN
        self.grip = 0.0
        self.interventions = 0
        self.ep_wrong = False
        self.raw = raw
        if self.expert is not None:
            self.expert.reset([0], raw)
        if self.learner is not None:
            self.learner.reset(np.array([0]), self._policy_obs())

    def _policy_obs(self) -> dict:
        return policy_observation(self.env, self.raw, self.obs_mode, self.mask_camera)

    def _set_control(self, who: str) -> None:
        if who == self.control or (who == LEARNER and self.learner is None):
            return
        if who == HUMAN:
            self.grip = float(self.env.last_command["gripper"][0])  # continue from the learner's gripper
            self.interventions += 1
        else:
            fn = getattr(self.learner, "interrupt", None) or self.learner.reset
            fn(np.array([0]), self._policy_obs())
        self.control = who

    def _end_episode(self, save: bool) -> None:
        ep = self.ep
        ok = ep.steps_since_success >= 0 and not self.ep_wrong
        saved = save and len(ep) > 0 and (ok or self.keep_failures)
        if saved and self.writer is not None:
            self.writer.write(ep, ok)
        if saved:
            self.saved += 1
            self.successes += int(ok)
        else:
            self.discarded += 1
        if self.on_episode is not None:
            self.on_episode(self.episodes_started - 1, ep, ok, saved)

    # ------------------------------------------------------------------------------------------
    def _human_action(self, inp) -> np.ndarray:
        scale = self.speed_scale * (0.25 if inp.fine else 1.0)
        dt = self.env.control_dt
        # the frame follows the commanded pose, which the delta is applied to (not the lagging measured one)
        twist = twist_to_world(inp.twist, self.frame, self.env.last_command["ee_pose"][0, 3:7])
        a = np.zeros(7)
        a[:3] = twist[:3] * self.speed * scale * dt
        a[3:6] = twist[3:] * self.rot_speed * scale * dt
        a[6] = self.grip
        return a

    def _show(self, paused: bool) -> None:
        env = self.env
        h, w = self.console_size
        frames = [env.render(cam, height=h, width=w) for cam in self.console_cameras]
        ep, n = self.ep, env.task.num_steps
        info = self._last_info
        mode = "PAUSED" if paused else self.control
        status = (f"episode {self.episodes_started}  step {len(ep)}/{env.max_episode_steps}  "
                  f"progress {int(info.get('progress', 0))}/{n}  "
                  f"{'SUCCESS' if ep.steps_since_success >= 0 else ''}{'WRONG TARGET' if self.ep_wrong else ''}")
        detail = (f"saved {self.saved} ({self.successes} ok)  discarded {self.discarded}  "
                  f"interventions {self.interventions}  speed x{self.speed_scale:.2f}  frame {self.frame}  "
                  f"gripper {'closed' if self.grip > 0.5 else 'open'}"
                  + (f"  learner {self.learner_ms:.0f} ms" if self.learner is not None else ""))
        self.device.show(frames, {"mode": mode, "status": status, "instruction": ep.instruction,
                                  "detail": detail, "cameras": list(self.console_cameras),
                                  "frame": self.frame})

    def _sync_viewer(self) -> bool:
        if self._viewer is None:
            return True
        if not self._viewer.is_running():
            return False
        self._viewer.sync()
        return True

    def run(self, num_episodes: int) -> dict:
        """Collect until ``num_episodes`` episodes are saved or the operator quits (Esc, closing a
        window, or Ctrl+C; the episode in progress is dropped)."""
        with interruptible():
            try:
                return self._run(num_episodes)
            except KeyboardInterrupt:
                return self.summary()

    def _run(self, num_episodes: int) -> dict:
        env = self.env
        dt_wall = env.control_dt / self.realtime_factor if self.realtime_factor > 0 else 0.0
        paused = False
        self._last_info = {}
        self._start_episode()
        try:
            while self.saved < num_episodes:
                t0 = time.perf_counter()
                inp = self.device.poll()
                end, save = False, True
                for ev in inp.events:
                    if ev == "quit":
                        return self.summary()
                    if ev == "pause":
                        paused = not paused
                    elif ev == "toggle":
                        self._set_control(LEARNER if self.control == HUMAN else HUMAN)
                    elif ev == "grip":
                        self._set_control(HUMAN)
                        self.grip = 0.0 if self.grip > 0.5 else 1.0
                    elif ev == "frame":
                        self.frame = TELEOP_FRAMES[(TELEOP_FRAMES.index(self.frame) + 1) % len(TELEOP_FRAMES)]
                    elif ev == "faster":
                        self.speed_scale = min(self.speed_scale * 1.25, 4.0)
                    elif ev == "slower":
                        self.speed_scale = max(self.speed_scale / 1.25, 0.1)
                    elif ev in ("save", "discard"):
                        end, save = True, ev == "save"
                if not self._sync_viewer():
                    return self.summary()
                if end:
                    self._end_episode(save)
                    if self.saved >= num_episodes:
                        break
                    self._start_episode()
                    continue
                if paused:
                    self._show(paused=True)
                    time.sleep(0.03)
                    continue
                if self.control == LEARNER and inp.moving and self.auto_takeover:
                    self._set_control(HUMAN)
                self._step(inp)
                self._show(paused=False)
                ep = self.ep
                done = (ep.steps_since_success >= self.post_success_steps or self.ep_wrong
                        or len(ep) >= env.max_episode_steps)
                if done:
                    self._end_episode(save=True)
                    if self.saved >= num_episodes:
                        break
                    self._start_episode()
                if dt_wall:
                    time.sleep(max(0.0, dt_wall - (time.perf_counter() - t0)))
        finally:
            if self._viewer is not None:
                self._viewer.close()
        return self.summary()

    def _step(self, inp) -> None:
        env, ep = self.env, self.ep
        states = {f: env.sim.read(f) for f in STATE_FIELDS}
        labels = None
        if self.expert is not None:
            self.expert.track(env.last_command["ee_pose"])
            labels = expert_labels(env, self.expert.act(self.raw))
        if self.control == HUMAN:
            action, mode, actor = self._human_action(inp)[None], "delta_ee_pose", ACTOR_HUMAN
        else:
            t = time.perf_counter()
            action = np.asarray(self.learner.act(self._policy_obs()), dtype=float).reshape(1, -1)
            self.learner_ms = 0.8 * self.learner_ms + 0.2 * 1e3 * (time.perf_counter() - t)
            mode, actor = env.action_mode, ACTOR_LEARNER
        prev = self.raw
        self.raw, reward, _, _, info = env.step(action, action_mode=mode)
        self._last_info = {k: np.asarray(v)[0] for k, v in info.items()}
        self.ep_wrong |= bool(info["wrong_target"][0])
        record_step(ep, env, 0, prev, states, env.executed_action(), reward[0], info["success"][0], actor,
                    actor == ACTOR_HUMAN, labels)
        if info["success"][0] and ep.steps_since_success < 0:
            ep.steps_since_success = 0
        elif ep.steps_since_success >= 0:
            ep.steps_since_success += 1

    def summary(self) -> dict:
        return {"saved": self.saved, "successes": self.successes, "discarded": self.discarded,
                "episodes": self.episodes_started}
