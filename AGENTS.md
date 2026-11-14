# AGENTS.md: working on franka-sim

This is a guide for coding agents and new contributors: what the project is, how it is put
together, the rules the code relies on, and what to do next. `README.md` is the user-facing
documentation. Read both before changing anything.

## What this is

franka-sim is a MuJoCo Franka Panda simulator for **conditional behavior cloning** research.
- **Backends:** a batched env API over two physics backends, MuJoCo on CPU and MuJoCo Warp on GPU.
- **Tasks:** multi-object manipulation tasks whose episodes are conditioned on a target, in a
  language instruction and in `cond_*` metadata.
- **Experts and data:** scripted experts, an HDF5 demo recorder, and a LeRobot v3 dataset exporter.
- **Benchmark:** a suite of tasks with a fixed evaluation protocol.

The benchmark measures three things:
- **Language adherence.** In *single* variants the instruction picks one target, and acting on
  another target fails the episode.
- **Long horizon.** *Long* variants (`*_long`) require every target in the scene, in any order.
- **Task completion.** Measured as success and progress.

## Setup and everyday commands

```sh
uv sync --extra lerobot --extra video --extra server   # all extras except warp; Python 3.12, uv-managed
uv run --extra lerobot --extra server pytest -q        # full suite, ~100 s on CPU (85 tests + 1 warp); run before every commit
uv run --extra server mjpython scripts/serve.py --task hammer_nails   # proxy robot + MuJoCo viewer (macOS: mjpython)
uv run python scripts/evaluate.py --policy scripted --episodes 50   # expert must stay at 1.00 on every task
uv run --extra video python scripts/make_videos.py --envs hammer_nails_long --episodes 2   # eyeball a change
```

- pytest lives in the `dev` dependency group (installed by `uv sync` by default), not in an extra.
- `uv sync`/`uv add`/`uv remove` drop every extra you don't name. Pass the extras you need to
  each `uv run`.
- The `warp` extra installs `mujoco-warp`. Without a GPU it falls back to CPU kernels, which are
  correct but slow. `tests/test_warp.py` covers it.
- Commit after each milestone, with a descriptive message. The remote is `origin` on GitHub
  (`git@github.com:SuhrudhSarathy/franka_sim.git`, over SSH; HTTPS has no credentials on the
  author's machine). Push only when the user asks.

## Architecture (src/franka_sim)

| module | role |
|---|---|
| `env.py` | `FrankaEnv`: batched `reset(seed, env_ids, options)` / `step(action, action_mode=None)`, observations, set-point interpolation + velocity feedforward, `render`, `render_target_mask`, `instructions()` |
| `controller.py` | action modes (`joint_pos`, `joint_vel`, `joint_delta`, `ee_pose`, `delta_ee_pose`; `ee_delta` alias) -> commanded joints: `MinkIK` (mink QP on an arm-only model), velocity limits (optional acceleration limits), anti-windup |
| `server/` | proxy robot: `SimRobot` (one world, real-time `tick`, commands held like a driver, lockstep `step`), `RobotServer` (main-thread loop + native viewer + uvicorn thread), `web.py` (WebSocket `/ws`, `/api/state`), `protocol.py` (msgpack + numpy) |
| `client.py` | `RobotClient`: `observe`/`act`/`stop` (real time), `pause`/`step`/`reset`/`run` (lockstep) |
| `scene.py` | builds the `MjSpec`: Panda (vendored Menagerie `mjx_panda.xml`), gravity compensation, stiff gripper servo, floor, lights, critically damped arm servos, floor, lights, cameras (`scene.CAMERAS`: `front`/`side`/`left`/`top`/`back`/`angle`/`wrist`); calls `task.build(spec)` |
| `backends/` | `cpu.py` (one `MjData` per env, thread pool) and `warp.py` (one batched Warp `Data`, CUDA graph capture). Same `Backend` interface: `read`/`write`/`reset`/`step`/`forward`/`mjdata` |
| `ik.py` | damped-least-squares IK (used only to sample start poses on reset) and quaternion helpers |
| `tasks/` | one module per family; `objects.py` builds all assets from primitives; `insertion.py` is the shared base for pegs, gears, kitting and PCB; `__init__.py` is the registry (`TASKS`, `HORIZONS`, `make_task`) |
| `policies/` | scripted experts: waypoint state machines emitting `ee_pose` actions. `make_policy(env)` picks the expert by task class |
| `masks.py` | exact, amodal binary target masks via segmentation rendering (~1 ms per 256² frame) |
| `recording.py` | `DemoWriter`/`collect_demos`: batched collection into robomimic-style HDF5, with full simulator states |
| `dataset.py` | `BCDataset`: observation history plus action chunks from HDF5, for PyTorch |
| `lerobot_export.py` | HDF5 to LeRobot v3. Images are re-rendered from the stored states, so any camera or resolution can be exported later |
| `benchmark.py` | `SUITE`, `EVAL_SEED`, observation protocol, `evaluate()` |

Scripts (`scripts/`):
- `evaluate.py`: runs the benchmark.
- `record_demos.py`: writes HDF5 demos.
- `build_datasets.py`: records every suite task and converts it to LeRobot.
- `to_lerobot.py`: converts HDF5 to LeRobot; with `--push`, uploads.
- `replay.py`: renders saved episodes.
- `make_videos.py`: one mp4 per expert episode.
- `benchmark.py`: physics *speed*, not the task benchmark.
- `serve.py`: the proxy robot (MuJoCo viewer + model API).

## Rules the code relies on (do not break)

**Arrays and coordinates**
- Every array is batched as `(num_envs, ...)`. Tasks work on whole batches. `reset` gets `env_ids`
  and must touch only those rows.
- Quaternions are **wxyz** (MuJoCo convention). The robot base is at the origin, facing +x. The
  robot's left is **+y**.
- In the `front` camera, the robot's left appears on the image *right*.

**Targets and conditioning**
- Target index 0 is the target furthest to the robot's left (nails are sorted by y after sampling;
  connectors and buttons are laid out that way). Instructions say "leftmost", "second from the
  left" and so on, from the robot's point of view.
- Conditioning keys (`task.conditioning()`, exposed as `info["cond_<key>"]`, saved as `cond/<key>`
  and `cond.<key>`):
  - `target`: single variants, a scalar index.
  - `order`: long variants, a permutation. It is the order the expert follows. Success never
    depends on it.
  - `targets`: ordered subset mode (insertion `mode="subset"`).
  - `sequence`: ordered button sequences.
  - `goal_pos`: reach.
  - Reset options use the same keys, e.g. `env.reset(options={"target": 2})`.
- Every task's `step` info has `success`, `progress` (sub-goals done, 0..`task.num_steps`) and
  `wrong_target`. A wrong target must also make `success` false; the benchmark ends the episode on
  it.
- Long-variant instructions are fixed strings (`long_instruction`, "... all the ..."). Single
  instructions name the target. Instructions must never leak the expert's sampled order in long
  variants.
- `task.mask_bodies(cond)` returns the body names of the single target, or `None` for multi-step
  tasks. Masks exist only for single variants.

**Registry and horizons**
- `make_task(name)` sets `task.name = name`. Recordings store that name, and the exporter rebuilds
  the task from it plus the stored `task_kwargs`.
- Variants are `functools.partial`s in `TASKS`. Experts are chosen by **class**
  (`isinstance`), not by name.
- `HORIZONS[name]` is the episode limit for the registered configuration only. It is 2× the longest
  of 64 expert demos, rounded up to 50. Passing custom `task_kwargs` falls back to the task's own
  default. If you change a task or an expert, re-measure: sweep with `max_episode_steps=2000`, take
  the max steps to success, then update `HORIZONS` and the README table.

**Scene and physics**
- Fixtures (board, tray, rack, receptacle, panel, plate) are **mocap bodies**, so every world can
  place them independently on both backends. Free objects get poses through `qpos` in
  `task.reset`.
- Collision bitmasks keep contact counts low for Warp. Gear bores and shafts use bit 4
  (`contype`/`conaffinity` 4); each gear's grip disc uses bit 1. On new assets, check
  `nconmax` overflow warnings on the Warp backend.
- Both backends run `mj_forward` / `mjw.forward` after the substeps, so site positions are current
  when `evaluate`/`observe` read them. `WarpBackend.read` returns copies: Warp's `.numpy()` on CPU
  is a live view.

**Control**
- The arm servos are critically damped (`robot.ARM_KV`). Menagerie's wrist servos ring at a
  damping ratio of about 0.3.
- Set-points ramp across the control period (`interp_substeps`). They lead the command by
  `(kv + damping) / kp * v` (`velocity_feedforward`), which removes servo lag.
- Without the feedforward, each joint lags by its own time constant. The hand then drifts off its
  path by 6-8 mm, which broke gear grasps.
- Acceleration limiting is off by default. It made the insertion experts' closed-loop yaw control
  oscillate (peg 0.04, PCB 0.08 success).
- `MinkIK` solves to convergence with *no* velocity limit. `ArmController` applies the limits.
  A velocity-limited IK target is only one step away, so braking cannot anticipate.
- Changing gains, interpolation or the controller changes the expert's timing. Re-run
  `evaluate.py` and re-measure `HORIZONS`.

**Experts and the benchmark**
- Experts emit `ee_pose` actions and read **privileged** observations. Precise phases (final
  insertion) use `Target(noise=0)`: demo noise (2 mm) is larger than PCB and peg clearances.
- Benchmark episode `k` is reset with seed `EVAL_SEED + k` (100 000 + k) on its own env row, so
  layouts do not depend on policy timing. Demo collection must use seeds below `EVAL_SEED`.
- The `vision` observation mode exposes only `benchmark.ROBOT_KEYS`, `rgb_*`, `instruction` and
  `target_mask`. Anything in `observation.environment_state` can reveal the target, such as
  `target_*_pos` or `target_onehot`. Never feed it to a policy that is scored as non-privileged.

## Adding a task

1. **Assets.** Add them in `tasks/objects.py`, using primitives only. Use mocap for fixtures and a
   free joint for objects. Set collision bits deliberately.
2. **The task.** Subclass `Task`, or `InsertionTask` for pick-and-place-into-fixture, which gives you
   modes, observations, the expert and the metrics for free. Implement:
   - `build`, `bind`, `reset` (rejection-sample layouts so objects don't start where the open
     gripper works at the fixture);
   - `observe`, `evaluate` (put `progress` and `wrong_target` in `Evaluation.info`);
   - `instruction`, `conditioning`, `num_steps`, `mask_bodies`.
3. **Registry.** Register the single and `_long` variants in `TASKS`. Add a `SUITE` entry in
   `benchmark.py`, with the number of targets for single variants.
4. **Expert.** Add one in `policies/scripted.py`, plus a `make_policy` branch if it's a new class.
   It must reach 1.00 on `evaluate.py --policy scripted`.
5. **Horizon.** Measure it and add it to `HORIZONS`.
6. **Tests.** Extend `tests/test_tasks.py`. The parametrized single and long tests and the mask
   tests pick new names up from lists at the bottom of the file. If the expert needs special
   handling, add a recording test in `tests/test_policies_recording.py`.
7. **Docs.** Update the README task table and benchmark table, and the status section below.

## Gotchas already paid for

- **Masks:** compare MuJoCo enums as `int(mujoco.mjtObj.X)`. Comparing a numpy array with the
  pybind enum object is slow, about 30 ms per image.
- **LeRobot (0.7):**
  - `image` features must be 3-channel; 1-channel PNGs crash the stats.
  - Call `save_episode(parallel_encoding=False)`; otherwise it spawns a process pool per episode.
  - Every frame needs a `task` string.
  - Masks are stored as `image` (lossless PNG), not `video`. AV1 would blur the edges.
- **Conversion cost:** about 0.1–0.15 s per frame. Rendering two cameras takes about 18 ms of that;
  the rest is LeRobot's write path (a PNG per frame, then encoding, then stats). Recording takes
  seconds. On an NVIDIA box, use `MUJOCO_GL=egl` and parallelize across datasets with `--jobs`.
- **Headless Linux:** MuJoCo's default GLFW backend needs a display. `franka_sim/__init__.py`
  sets `MUJOCO_GL=egl` when there is none. This must happen before `mujoco` is first imported, so
  in scripts import `franka_sim` before `mujoco`. Without it, `mujoco.Renderer` fails with
  "an OpenGL platform library has not been loaded".
- **numpy:** lerobot needs numpy < 2.3, so the pin is `numpy>=1.26`.
- **Grasping:** the gripper needs the stiff servo in `robot.py` (kp 3000, 70 N). The Menagerie
  default squeezes about 4 N and the hammer slips. Gears are grasped on a hidden smooth disc,
  because polygon facets squeeze them out of the fingers.
- **Rendering threads:** MuJoCo offscreen rendering hangs when it is called from a non-main thread
  on macOS. The server therefore keeps physics and rendering on the main thread and runs uvicorn
  on a side thread, passing requests in through a job queue.
  - `mujoco.viewer.launch_passive` needs `mjpython` on macOS. Under `mjpython` the script still
    runs as Python's main thread, and offscreen rendering works.
  - Closing a passive viewer finishes asynchronously, so reopening it right away (on a task
    switch) raises "another MuJoCo viewer is already open". `_open_viewer` retries.
- **Recording:** `collect_demos` keeps the reset info and the step info separate. Mixing them once
  overwrote `cond_*`.

## Project status (2026-11-14)

Done:
- **Tasks:** single and long variants for hammer, buttons, pegs, gears and kitting, plus PCB
  (single only) and reach (smoke test).
- **Metrics and horizons:** `progress` and `wrong_target` on every task; measured horizons.
- **Evaluation:** `scripts/evaluate.py` with fixed seeds. The scripted expert scores 1.00 on all
  11 suite tasks on held-out seeds.
- **Masks and export:** target masks (top camera) in the env and in the LeRobot export.
- **Control (2026-11-14):**
  - mink controller with 5 action modes, critically damped servos, interpolation and feedforward;
    the arm no longer oscillates.
  - Expert still at 1.00 everywhere. Horizons re-measured.
- **Cameras:** 7 in total (`front`, `side`, `left`, `top`, `back`, `angle`, `wrist`).
- **Proxy robot:** `scripts/serve.py` with the native MuJoCo viewer and a WebSocket model API. The
  user chose the native viewer over a web UI. There is no keyboard teleop in it (MuJoCo's default
  viewer only), but a teleop client could drive the API.

On hold (the user asked to wait; do not start without asking):
- **Recording and converting the benchmark datasets.** When asked, run
  `uv run --extra lerobot python scripts/build_datasets.py --overwrite --jobs 3` (add
  `--namespace <hf-user>` once known, `--episodes N` once the user picks a count, and
  `--hdf5-dir` / `--lerobot-dir` to write somewhere other than `data/` and `lerobot_data/`). The whole
  suite is about 280k frames at 100 episodes per task, roughly 2–2.5 h on the M-series Mac.
- **A combined multi-task dataset** (`franka_sim_suite`).
- **Uploading to the Hugging Face Hub.** `to_lerobot.py --push` needs `hf auth login` and the
  user's go-ahead. Pushing is public by default; use `--private` if asked.

Out of date on disk (gitignored):
- `lerobot_data/` and `data/` hold datasets built *before* the variants existed: `reach`,
  `buttons_sequence`, `gear_assembly` (all gears), `kitting` (2-part lists), plus `hammer_nails`,
  `peg_insertion` and `pcb_insertion`, which already have masks. Their cond keys predate the
  current scheme (`cond.targets` and `cond.sequence` instead of `cond.target` and `cond.order`).
  Rebuild them rather than mixing them with new data.

Ideas not started:
- **Batched GPU rendering for evaluation.** The `mujoco_warp.render` / `create_render_context`
  ray tracer could render image observations for hundreds of worlds at once. It is untested on
  these scenes; check that its images match `mujoco.Renderer`, which produced the training data,
  before using it.
- **GPU testing.** The Warp backend has never run on an NVIDIA GPU in this project; only its CPU
  fallback is tested.
- **Third benchmark tier:** ordered sequences (instruction-dictated order). The user decided
  against it for now.
