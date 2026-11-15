# franka-sim

A Franka Panda simulator built on MuJoCo. It runs on two physics backends:

* `cpu`: MuJoCo's C engine, with one `MjData` per env, stepped on a thread pool.
* `warp`: [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), with every env batched on a GPU (CUDA), or on CPU when no GPU is present.

Both backends use the same Python API, which is built for behavior cloning. You get batched, gym-like
`reset`/`step` calls, several action representations, camera observations and per-episode conditioning.

## Install

```sh
uv sync                     # CPU backend
uv sync --extra warp        # + MuJoCo Warp backend
uv sync --extra video       # + mp4 writing for the scripts
uv sync --extra server      # + proxy-robot server and client (see "Proxy robot")
uv sync --extra teleop      # + keyboard teleop console (see "Teleop and DAgger")
```

**Headless servers.** MuJoCo renders through OpenGL. On Linux with no display (`DISPLAY` and
`WAYLAND_DISPLAY` both unset), `import franka_sim` sets `MUJOCO_GL=egl`. That gives GPU rendering
with the NVIDIA driver, or Mesa's software EGL otherwise. Two cases need a manual setting:
- A machine with no EGL library at all: install OSMesa and set `MUJOCO_GL=osmesa`.
- A script that imports `mujoco` before `franka_sim`: set `MUJOCO_GL=egl` yourself.

## Quick start

```python
import numpy as np
import franka_sim

env = franka_sim.make("reach", num_envs=8, backend="cpu", action_mode="delta_ee_pose",
                      cameras=("front", "wrist"), image_size=(128, 128))
obs, info = env.reset(seed=0)          # every array is batched: obs["ee_pos"].shape == (8, 3)
for _ in range(100):
    action = np.zeros((8, env.action_dim))
    action[:, :3] = np.clip(obs["goal_pos"] - obs["ee_pos"], -0.03, 0.03)
    obs, reward, terminated, truncated, info = env.step(action)
env.reset(env_ids=np.flatnonzero(terminated | truncated))   # partial resets
```

To get a standard single-env Gymnasium environment, use `franka_sim.gym_env.FrankaGymEnv("reach", ...)`.

### Action modes

The last action dimension is always the gripper closure in `[0, 1]`, where 0 is open and 1 is closed.

| `action_mode` | dims | meaning |
|---|---|---|
| `joint_pos` | 8 | 7 absolute joint targets (rad), gripper |
| `joint_vel` | 8 | 7 joint velocities (rad/s), gripper |
| `joint_delta` | 8 | 7 joint offsets (rad) from the measured joint positions, gripper |
| `ee_pose` | 8 | absolute `x y z` (m), quaternion `w x y z` (world frame), gripper |
| `delta_ee_pose` | 7 | `dx dy dz` (m) and `rx ry rz` (rotation vector, rad), both world frame, from the commanded pose, gripper |

`ee_delta` is still accepted as an alias of `delta_ee_pose`.

How actions become motion (`franka_sim/controller.py`):

* **Cartesian modes** are solved by differential IK with [mink](https://github.com/kevinzakka/mink):
  a QP with the end-effector task, a light posture term for the redundant joint, and the joint limits
  as hard constraints.
* **Every mode** moves a commanded joint configuration within the Panda's joint velocity limits (and,
  with `controller_kwargs={"accel_scale": 0.5}`, its acceleration limits too).
* **Incremental modes** (`joint_vel`, `delta_ee_pose`) build on the previous *command*, not the
  measured state, so the arm does not drift from tracking lag. If contact holds the arm back by
  more than 0.2 rad, the reference is pulled back toward the measured state, so it never winds up.
* **Servo tracking.** The arm's position servos are critically damped and gravity compensated.
  Within each control period their set-point ramps from the previous command to the new one (at
  100 Hz, `interp_substeps=5`). A velocity feedforward (`velocity_feedforward=1.0`) cancels their
  tracking lag: a 0.1 m/s Cartesian move is followed with no lag and a constant speed, and with
  sub-millimetre deviation from the line. Command smoothly. A target that jumps 10 cm in a single
  step is overshot by about 7 mm (about 1 mm with `velocity_feedforward=0.5`).

After every step, `env.last_command` holds the command in each representation (`joint_pos`,
`ee_pose`, `gripper`), so a dataset can store all of them.

### Observations

`joint_pos`, `joint_vel`, `gripper_width`, `ee_pos`, `ee_quat` (wxyz), any task-specific keys, and
`rgb_<camera>` for each requested camera.

| camera | view |
|---|---|
| `front` | facing the robot |
| `side` | from the robot's right |
| `left` | from the robot's left |
| `top` | bird's eye |
| `back` | over the robot's right shoulder, as a teleoperator standing behind it would see |
| `angle` | three-quarter view from the front left, close on the workspace |
| `wrist` | on the hand, looking along the gripper |

`franka_sim.scene.CAMERAS` lists them. Pass any subset as `cameras=(...)`.

## Backends

```python
env = franka_sim.make("reach", num_envs=4096, backend="warp",
                      backend_kwargs={"device": "cuda:0", "nconmax": 128, "njmax": 384})
```

* With `backend="warp"`, all envs are `nworld` worlds in a single MuJoCo Warp `Data`. On CUDA, each
  control step's physics substeps are captured once in a CUDA graph and then replayed. Without a
  GPU, Warp runs its kernels on the CPU. That is correct but slow, and is only useful for testing.
* `nconmax` and `njmax` are per-world buffer sizes. The backend warns when the contact buffer
  overflows.
* Rendering and IK run on the host. To render or solve IK for a world, its state is copied into an
  `MjData`, so camera observations and Cartesian action modes cost host time per env (about
  0.1 ms per mink QP iteration). For large `num_envs`, use joint-space actions and no cameras.
* Physics agrees with the CPU backend to float32 precision (see `tests/test_warp.py`).

Benchmark: `uv run --extra warp python scripts/benchmark.py --backends cpu warp --num-envs 1 64 1024`.

## Proxy robot

`scripts/serve.py` runs one simulated Franka the way a real robot runs behind its driver:
- **Clock.** The simulation advances in real time on its own clock (20 Hz control by default).
- **Viewer.** MuJoCo's native viewer shows it live. Double-click a body to select it, then
  Ctrl + right-drag to push it. Closing the window stops the server.
- **Model API.** Models connect over a WebSocket.

```sh
uv run --extra server mjpython scripts/serve.py --task hammer_nails   # macOS: the viewer needs mjpython
uv run --extra server python scripts/serve.py --task hammer_nails     # Linux / Windows
uv run --extra server python scripts/serve.py --task kitting --headless --paused   # no window, lockstep
```

```python
from franka_sim.client import RobotClient

with RobotClient("ws://127.0.0.1:8000/ws", cameras=("front", "wrist"), image_size=(224, 224)) as robot:
    obs = robot.observe()   # joint_pos, joint_vel, gripper_width, ee_pos, ee_quat, rgb_front, rgb_wrist, instruction
    robot.act([0, 0.005, 0, 0, 0, 0, 0.0], mode="delta_ee_pose")   # real time: streamed commands

    robot.pause()           # lockstep: each step advances exactly one control period
    obs, info = robot.reset(seed=100_000, options={"target": 2}, task="hammer_nails")
    obs, reward, terminated, truncated, info = robot.step(action, mode="joint_pos")
    robot.run()
```

In real time, commands are held as a robot driver holds them:
- Absolute targets (`joint_pos`, `ee_pose`) stay in force until the next command.
- `joint_vel` stops after a 0.25 s watchdog.
- Deltas (`delta_ee_pose`, `joint_delta`) sent between two ticks are summed and applied once.
- With no command, the arm holds still.

Any action mode can be used on any command.

Observations follow the benchmark's `vision` protocol. Pass `privileged=True` for the task's
simulator state (oracle use only). `GET /api/state` returns the robot and episode status as JSON.

Everything runs on one thread (MuJoCo's OpenGL needs it), and the network layer queues requests
between control ticks. The scripted expert solves tasks through the socket in lockstep, at about
500 steps/s headless and 260 with the viewer.

## Teleop and DAgger

`scripts/dagger.py` collects DAgger data. Each run is one round and writes one HDF5 file: the usual
demo format plus, per step:
- `actor`: 0 learner, 1 human, 2 scripted expert;
- `intervention`: a human was in control;
- `expert/*`: the scripted expert's label for the state, in the learner's action mode and as
  absolute joint and end-effector targets.

Train on the union of all rounds, then run the next round with the new policy.

**Human-gated (HG-DAgger).** The policy drives in real time. When it goes wrong, you take over with
the keyboard, steer, and hand back.

```sh
# macOS: --viewer (MuJoCo's window) needs mjpython. The teleop console works with plain python too.
uv run --extra teleop --extra server mjpython scripts/dagger.py teleop --task peg_insertion --round 1 \
    --policy my_pkg.policies:make --action-mode joint_pos --cameras front wrist --viewer

# try the loop now, with a deliberately biased "learner" that misses the hole
uv run --extra teleop mjpython scripts/dagger.py teleop --task peg_insertion --viewer \
    --policy franka_sim.dagger.toy:biased_expert --obs-mode privileged

# no policy: plain teleop demonstrations
uv run --extra teleop python scripts/dagger.py teleop --task kitting --episodes 10
```

The teleop console is a small window showing the `back` and `wrist` cameras, who is in control, the
instruction and the progress. Click it to drive:

| keys | |
|---|---|
| `W`/`S`, `A`/`D`, `R`/`F` | move ±x (away from / toward the base), ±y (robot's left / right), ±z |
| `Q`/`E`, `I`/`K`, `J`/`L` | yaw, pitch, roll |
| `Shift` (held) | precision mode (1/4 speed) |
| `[` / `]` | slower / faster (default 0.15 m/s) |
| `Space` | toggle the gripper |
| `Tab` or `Return` | take over / hand back to the policy |
| `P` | pause |
| `N` | end the episode and save it |
| `Backspace` | discard the episode |
| `Esc` or close a window | quit |

How a session behaves:
- **Taking over.** Any movement key takes control from the policy (`--no-auto-takeover`: only
  `Tab`). The switch is bumpless: you move from the robot's *commanded* pose, and the gripper
  starts where the policy left it.
- **Handing back.** The policy's `interrupt(env_ids, obs)` is called if it has one (`reset`
  otherwise), so a policy that executes action chunks drops the actions it queued before you took
  over.
- **Labels.** Your corrections are stored in the policy's action space (`actions`), and as
  absolute targets (`commands/*`).
- **Episode end.** An episode ends on success (plus 10 settling steps), on a wrong target, or at
  the horizon. Failed episodes are kept, because their interventions are the point.
- **Slow policies.** `--realtime-factor 0.5` runs the clock at half speed.

**Automatic DAgger.** The scripted expert labels every state the learner visits. No human is
needed, which suits sweeps and ablations:

```sh
uv run python scripts/dagger.py auto --task peg_insertion --round 0 --episodes 100        # expert demos (beta = 1)
uv run python scripts/dagger.py auto --task peg_insertion --round 1 --episodes 100 \
    --policy my_pkg.policies:make --beta 0.5 --action-mode joint_pos --cameras front wrist
```

- **Mixing.** Each step, the expert acts with probability `beta` (`--mixing episode`: a coin per
  episode). Its label is recorded either way.
- **Re-anchoring.** The expert is re-anchored to the robot's commanded pose each step, so its label
  is a correction from where the learner left the robot.

**Policies** use the benchmark interface (`factory(env)` returning an object with
`reset(env_ids, obs)`, `act(obs)` and optionally `interrupt`), so the same policy runs in
`evaluate.py` and in DAgger.
- **Local:** `module.path:factory`.
- **Remote, e.g. on a GPU box:**
  `uv run --extra server python scripts/serve_policy.py --policy my_pkg.policies:make --host 0.0.0.0`
  there. Then pass `--policy remote:ws://gpu-box:9000` here (`evaluate.py` accepts it too). The
  remote factory gets an `EnvSpec` (task, action mode and dimension, cameras, image size, control
  period) instead of the env.

**Export** every round into one LeRobot dataset, so failures (which hold interventions) are kept:

```sh
uv run --extra lerobot python scripts/to_lerobot.py data/dagger/peg_insertion/round_*.hdf5 \
    --repo-id <hf-user>/peg_dagger --root lerobot_data/peg_dagger --include-failures [--labels expert]
```

The export adds `dagger.actor` and `dagger.intervention` per frame. HG-DAgger usually trains on
human-labelled frames, so mask the loss with them rather than cutting episodes apart. With expert
labels in every file, `expert.joint_pos` and `expert.ee_pose` are added too, and `--labels expert`
makes them the `action` (classic DAgger).

## Tasks

Every task family comes in two variants. A **single** variant asks for one target picked out by
the condition and named in the instruction, which tests language adherence. A **long** variant asks
for every target in the scene, in any order, which tests long-horizon completion.

| family | objects | single variant (`cond_target`) | long variant (`cond_order`) |
|---|---|---|---|
| `hammer_nails` | hammer on a stand; board with 4 identical nails at random positions, on friction slide joints | drive the target nail ≥ 75 % (0 = furthest to the robot's left), no other nail > 25 % | `hammer_nails_long`: drive all 4 nails |
| `buttons` | panel with 4 colored spring-loaded buttons | press the target button | `buttons_long`: press all 4 |
| `peg_insertion` | 3 colored square pegs, receptacle with 3 matching holes (5 mm clearance) | seat the target peg | `peg_insertion_long`: seat all 3 |
| `gear_assembly` | 3 spur gears (small / medium / large) lying flat; plate with 3 shafts (3 mm bore clearance) spaced so the gears mesh | put the target gear on its shaft | `gear_assembly_long`: all 3 gears |
| `kitting` | 4 machined parts (block, cylinder, bar, cube); kit tray with a fitted compartment per part | put the target part in its compartment | `kitting_long`: all 4 parts |
| `pcb_insertion` | expansion card standing in a rack; motherboard with 3 card-edge connectors (1.4 mm clearance per side) | seat the card in the target connector (0 = leftmost) | none |
| `reach` | mocap target sphere | gripper within 2 cm of `goal_pos` (smoke test, not in the benchmark) | none |

* **Wrong targets fail.** In a single variant, acting on any target other than the requested one
  sets `info["wrong_target"]` and fails the episode. That means driving another nail past 25 %,
  pressing another button, inserting another object, or seating the card in another connector.
* **Order is free in long variants.** Each episode still samples a random order (`cond_order`), and
  the scripted expert follows it, so recorded demos vary in order. Success ignores the order, and
  the instruction does not mention it ("pick up the hammer and hit all the nails").
* **Progress.** `info["progress"]` counts the sub-goals done so far, from 0 to
  `env.task.num_steps`.
* **Horizons.** `franka_sim.tasks.HORIZONS` sets each episode limit to 2× the longest of 64
  scripted demos.
* **Other modes.** Ordered sequences are still available as task options: `buttons` with
  `sequence_length=2`, and `kitting` with `mode="subset"`. They are not part of the benchmark.

**Randomization.** Every reset samples:
- the robot start pose (end-effector offset of ±8/15/8 cm and ±0.5 rad yaw, solved with IK, plus
  joint noise);
- the position and yaw of every fixture: nail board, hammer stand, receptacle, gear plate, kit tray,
  card rack, motherboard and button panel;
- the position and yaw of every free object: the hammer (on either side of the board), pegs, gears,
  parts and the card;
- the per-episode condition.

Layouts are rejection-sampled, so objects never start where the gripper would hit them while it
works at the fixture. Pass `randomize_start=False` to the env, or `task_kwargs={"randomize": False}`,
for fixed layouts.

`peg_insertion`, `gear_assembly`, `kitting` and `pcb_insertion` share `InsertionTask`, which supports
`mode="single" | "subset" | "all"` (the `_long` variants are `mode="all"`). They also share an observation layout: `obj_pos`, `obj_quat`,
`goal_pos`, `fixture_quat`, `obj_inserted`, `targets_mask`, `target_onehot`, `target_obj_pos`,
`target_goal_pos`. So one policy, or one multi-task model, works across all of them.

Each task has a language instruction per env (`env.instructions()`). Examples:
- "pick up the hammer and hit the second nail from the left"
- "press the red button"
- "put the small gear on its shaft"
- "kit the cube"
- "insert the card into the rightmost connector"
- "put all the parts in the kit tray"

Pass task options as `franka_sim.make("kitting", task_kwargs={"mode": "subset"})`. Custom kwargs keep
the task's own default horizon, not the measured one.

All assets are made from primitive geoms, and collision bitmasks keep contact counts low for
MuJoCo Warp. The fixtures are mocap bodies, so they can be placed independently in every world on
both backends. To add your own task, subclass `franka_sim.Task` (or `InsertionTask` for
pick-and-insert) and call `franka_sim.register_task`. `AGENTS.md` has a checklist.

## Scripted demos and recording

Every task has a scripted expert (`franka_sim.policies.make_policy(env)`). These are waypoint
state machines that emit `ee_pose` actions and close the loop on object poses. For example, the
hammer policy aims the hammer *face*, not the gripper, at the target nail, and keeps striking until
the nail is driven in. All insertion tasks use one `InsertionPolicy`, which:
- opens the gripper only as wide as the part needs;
- steers the part's leading edge;
- matches the fixture's yaw, modulo the part's symmetry;
- re-grasps if a part slips.

On the 50 held-out evaluation episodes of every benchmark task, the scripted experts succeed 100 % of
the time (`scripts/evaluate.py --policy scripted`). The hammer expert keeps hold of the hammer
through `hammer_nails_long`. Each policy keeps its own state per env, so it works with batched envs whose
episodes start and end at different times.

```sh
# 40 hammer demos, balanced over the 4 nails (conditioning = target nail), front + wrist cameras
uv run python scripts/record_demos.py --task hammer_nails --episodes 40 --condition balanced \
    --cameras front wrist --num-envs 8 --out data/hammer.hdf5

# only episodes that hit the third nail from the left
uv run python scripts/record_demos.py --task hammer_nails --episodes 10 --condition target=2

# long-horizon variants, an ordered button sequence, the Warp backend
uv run python scripts/record_demos.py --task hammer_nails_long --episodes 20
uv run python scripts/record_demos.py --task buttons --task-kwargs sequence_length=2
uv run --extra warp python scripts/record_demos.py --task hammer_nails --backend warp --num-envs 64 --cameras

# run the scripted expert on reach and every benchmark task and save one mp4 per episode (5 each) to videos/
uv run --extra video python scripts/make_videos.py

# render saved episodes from their simulator states (any camera, any resolution)
uv run --extra video python scripts/replay.py data/hammer.hdf5 --episodes 0 1 --camera front wrist
```

From Python:

```python
from franka_sim.policies import make_policy
from franka_sim.recording import DemoWriter, collect_demos

env = franka_sim.make("hammer_nails", num_envs=8, action_mode="ee_pose", cameras=("front", "wrist"))
with DemoWriter("data/hammer.hdf5", env) as writer:
    stats = collect_demos(env, make_policy(env, noise=0.002), 100, writer,
                          conditions=lambda k: {"target": k % 4})
```

The HDF5 layout is robomimic-style, with one `data/demo_<k>` group per episode. Each group holds
the observations (images are gzipped uint8), `actions`, the same command as `commands/joint_pos`,
`commands/ee_pose` and `commands/gripper`, plus `rewards`, `success` and `dones`. It also stores the
full simulator `states` (for exact replay), and `cond/<key>` with the episode's conditioning
(e.g. `cond/target`). The language instruction and reset options are stored as attributes. See
`franka_sim/recording.py` for details.

### Training data

```python
from franka_sim.dataset import BCDataset

ds = BCDataset("data/hammer.hdf5", obs_keys=["joint_pos", "gripper_width", "rgb_wrist", "target_onehot"],
               obs_horizon=2, action_horizon=16, action_key="commands/joint_pos")
sample = ds[0]   # {"obs/rgb_wrist": (2,128,128,3) uint8, "actions": (16, 8), "cond/target": (), ...}
# works directly with torch.utils.data.DataLoader; ds.stats() gives normalization statistics
```

## LeRobot datasets

```sh
uv sync --extra lerobot

# record scripted demos for every benchmark task and convert each into a LeRobot (v3) dataset
uv run --extra lerobot python scripts/build_datasets.py --namespace <hf-user> --episodes 50
# outputs default to data/ (HDF5) and lerobot_data/ (LeRobot); put them elsewhere with
#   --hdf5-dir /path/to/demos --lerobot-dir /path/to/lerobot

# or convert your own recordings
uv run --extra lerobot python scripts/to_lerobot.py data/hammer_nails.hdf5 \
    --repo-id <hf-user>/franka_sim_hammer_nails --root lerobot_data/hammer_nails --cameras front wrist

# upload (after `hf auth login`)
uv run --extra lerobot python scripts/to_lerobot.py --push --repo-id <hf-user>/franka_sim_hammer_nails \
    --root lerobot_data/hammer_nails
```

| key | shape | content |
|---|---|---|
| `observation.state` | 8 | 7 joint positions + gripper width |
| `observation.ee_pose` | 7 | end-effector xyz + quaternion (wxyz) |
| `observation.environment_state` | task-specific | privileged object state; names in `meta/info.json` |
| `observation.images.{front,wrist}` | 256×256×3 video | re-rendered from the recorded simulator states |
| `observation.images.target_mask` | 256×256×3 PNG | binary mask (0/255) of the target entity from the top camera; single-target tasks only |
| `action` | 8 | joint position targets + gripper closure (`--action ee_pose` for xyz+quat+gripper); with `--labels expert`, the scripted expert's labels |
| `action.joint_pos`, `action.ee_pose` | 8 | both representations of what was executed |
| `dagger.actor` | 1 | DAgger files only: who acted, 0 learner / 1 human / 2 scripted expert (plain demos in the same dataset count as 2) |
| `dagger.intervention` | 1 | DAgger files only: a human was in control (mask the loss with it for HG-DAgger) |
| `expert.joint_pos`, `expert.ee_pose` | 8 | the scripted expert's labels, when every file has them (`dagger.py auto`, or `teleop` without `--no-scripted-labels`) |
| `cond.*` | | episode conditioning: `cond.target` (single variants: nail / button / peg / gear / part / connector index) or `cond.order` (long variants: the order the expert followed) |
| `next.reward`, `next.success`, `next.done` | 1 | |
| `task` | string | language instruction, e.g. "pick up the hammer and hit the leftmost nail" |

Demos are recorded as low-dimensional data plus simulator states, so recording is fast. Cameras
are rendered at conversion time, which means any camera set and resolution can be exported from the
same recording without simulating again.

### Target masks

Single variants also get `observation.images.target_mask`. It is a binary image of the entity the
episode's condition selects, seen from the `top` camera: the nail, button, peg, gear, part or
connector. It comes from MuJoCo's segmentation rendering, so it is exact. It is also amodal: the arm
and the hammer never hide the target. Masks are stored losslessly as PNG with three identical
channels, and cost about 1 ms per frame. Long variants, ordered sequences and reach have no mask. Use `--mask-camera <camera>` to mask from another camera, `none` to drop it.

At rollout time the same image comes from the env:

```python
mask = env.render_target_mask("top", env_id=0)  # (H, W) uint8 0/255, or None
```

## Benchmark

`franka_sim.benchmark` defines the suite: the 11 single and long tasks above. It also fixes the
evaluation protocol, and `scripts/evaluate.py` runs it:

```sh
# the scripted expert with privileged state: 100 % on every task (upper bound, sanity check)
uv run python scripts/evaluate.py --policy scripted

# your policy: a factory make(env, **kwargs) returning an object with reset(env_ids, obs) / act(obs)
uv run python scripts/evaluate.py --policy my_pkg.policies:make --policy-kwargs '{"ckpt": "run1"}' \
    --episodes 50 --cameras front wrist --mask-camera top
```

* **Fixed evaluation episodes.** Episode `k` is reset with seed `EVAL_SEED + k` (100 000 + k).
  Single variants also get target `k % num_targets`. Every policy faces the same layouts and
  conditions, and the targets are balanced. Demonstrations must use seeds below `EVAL_SEED`, and
  `build_datasets.py` checks this.
* **What the policy sees.** In the default `vision` mode, a policy gets robot proprioception
  (`joint_pos`, `joint_vel`, `gripper_width`, `ee_pos`, `ee_quat`), `rgb_<camera>`, `instruction`,
  and with `--mask-camera` also `target_mask`. `privileged` mode adds the simulator's object state,
  which is only for oracle baselines. Actions default to `joint_pos`; pass `--action-mode` to change
  it.
* **Metrics per task.**
  - `success_rate`.
  - `wrong_target_rate`: single variants, language adherence. A wrong target also ends the episode
    as a failure.
  - `mean_progress`: the fraction of sub-goals done, the main long-horizon signal.
  - `mean_success_steps`.
  - `per_target_success`.

  Everything is written to `results/eval.json`.

| task | kind | scripted expert success | expert steps (mean) | horizon |
|---|---|---|---|---|
| hammer_nails | single | 1.00 | 79 | 200 |
| hammer_nails_long | long | 1.00 | 132 | 350 |
| buttons | single | 1.00 | 33 | 100 |
| buttons_long | long | 1.00 | 105 | 300 |
| peg_insertion | single | 1.00 | 94 | 250 |
| peg_insertion_long | long | 1.00 | 304 | 700 |
| gear_assembly | single | 1.00 | 106 | 250 |
| gear_assembly_long | long | 1.00 | 396 | 1000 |
| kitting | single | 1.00 | 120 | 400 |
| kitting_long | long | 1.00 | 561 | 1500 |
| pcb_insertion | single | 1.00 | 109 | 250 |

## Layout

```
src/franka_sim/
  env.py            FrankaEnv: batched reset/step, action modes, observations, rendering
  gym_env.py        single-env gymnasium wrapper
  scene.py          MjSpec scene: Panda (MuJoCo Menagerie, MJX variant) + floor, lights, cameras
  controller.py     action modes -> joint commands: mink differential IK, velocity limits, anti-windup
  server/           proxy robot: SimRobot (held commands, watchdog), RobotServer (real-time loop, viewer, WebSocket API)
  client.py         RobotClient for the proxy robot
  teleop/           teleop devices: keyboard console (pygame, child process), scripted device for tests
  dagger/           DAgger: collect.py (scripted-expert rounds), session.py (HG-DAgger / teleop),
                    remote.py (policy server + RemotePolicy), toy.py (a biased learner to try it)
  ik.py             damped-least-squares IK (start-pose sampling) and quaternion helpers
  benchmark.py      benchmark suite, seeds, observation protocol, evaluation loop
  masks.py          binary target masks from segmentation rendering
  backends/         cpu.py (MuJoCo), warp.py (MuJoCo Warp)
  tasks/            reach, hammer_nails, buttons, insertion.py (pegs, gears, kitting, pcb); objects.py = assets
  policies/         scripted experts
  recording.py      HDF5 demo writer / reader, batched collection loop
  dataset.py        BCDataset (observation history + action chunks)
  lerobot_export.py HDF5 demos -> LeRobot dataset
scripts/            serve.py (proxy robot), dagger.py (DAgger rounds, teleop), serve_policy.py, evaluate.py (benchmark evaluation), record_demos.py, replay.py, make_videos.py,
                    to_lerobot.py, build_datasets.py, benchmark.py (physics speed, not the task benchmark)
```

The Panda model is `mjx_panda.xml` from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0, see `src/franka_sim/assets/panda/LICENSE`). The scene builder adds gravity
compensation to the arm, critically damps its joint servos (Menagerie's wrist servos ring at a
damping ratio of about 0.3) and stiffens the gripper.
