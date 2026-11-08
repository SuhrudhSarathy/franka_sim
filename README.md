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
```

## Quick start

```python
import numpy as np
import franka_sim

env = franka_sim.make("reach", num_envs=8, backend="cpu", action_mode="ee_delta",
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
| `ee_delta` | 7 | `dx dy dz` (m), `rx ry rz` (world-frame rotation vector), gripper |
| `ee_pose` | 8 | absolute `x y z`, quaternion `w x y z`, gripper |
| `joint_pos` | 8 | 7 absolute joint targets, gripper |
| `joint_delta` | 8 | 7 joint offsets from the current joint positions, gripper |

End-effector commands are converted to joint targets by damped-least-squares IK. The arm's joint
position actuators then track those targets, with gravity compensation. After every step,
`env.last_command` holds the command in each representation (`joint_pos`, `ee_pose`, `gripper`), so
a dataset can store all of them.

### Observations

`joint_pos`, `joint_vel`, `gripper_width`, `ee_pos`, `ee_quat` (wxyz), any task-specific keys, and
`rgb_<camera>` for each requested camera. The cameras are `front`, `side`, `top` and `wrist`.

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
  `MjData`, so camera observations and `ee_*` action modes cost host time per env. For large
  `num_envs`, use `joint_pos`/`joint_delta` actions and no cameras.
* Physics agrees with the CPU backend to float32 precision (see `tests/test_warp.py`).

Benchmark: `uv run --extra warp python scripts/benchmark.py --backends cpu warp --num-envs 1 64 1024`.

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
| `action` | 8 | joint position targets + gripper closure (`--action ee_pose` for xyz+quat+gripper) |
| `action.joint_pos`, `action.ee_pose` | 8 | both action representations |
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
| hammer_nails | single | 1.00 | 81 | 200 |
| hammer_nails_long | long | 1.00 | 142 | 350 |
| buttons | single | 1.00 | 35 | 100 |
| buttons_long | long | 1.00 | 111 | 300 |
| peg_insertion | single | 1.00 | 98 | 250 |
| peg_insertion_long | long | 1.00 | 320 | 750 |
| gear_assembly | single | 1.00 | 114 | 300 |
| gear_assembly_long | long | 1.00 | 429 | 1000 |
| kitting | single | 1.00 | 113 | 400 |
| kitting_long | long | 1.00 | 568 | 1250 |
| pcb_insertion | single | 1.00 | 114 | 250 |

## Layout

```
src/franka_sim/
  env.py            FrankaEnv: batched reset/step, action modes, observations, rendering
  gym_env.py        single-env gymnasium wrapper
  scene.py          MjSpec scene: Panda (MuJoCo Menagerie, MJX variant) + floor, lights, cameras
  ik.py             damped-least-squares IK
  benchmark.py      benchmark suite, seeds, observation protocol, evaluation loop
  masks.py          binary target masks from segmentation rendering
  backends/         cpu.py (MuJoCo), warp.py (MuJoCo Warp)
  tasks/            reach, hammer_nails, buttons, insertion.py (pegs, gears, kitting, pcb); objects.py = assets
  policies/         scripted experts
  recording.py      HDF5 demo writer / reader, batched collection loop
  dataset.py        BCDataset (observation history + action chunks)
  lerobot_export.py HDF5 demos -> LeRobot dataset
scripts/            evaluate.py (benchmark evaluation), record_demos.py, replay.py, make_videos.py,
                    to_lerobot.py, build_datasets.py, benchmark.py (physics speed, not the task benchmark)
```

The Panda model is `mjx_panda.xml` from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0, see `src/franka_sim/assets/panda/LICENSE`). The scene builder adds gravity
compensation to the arm.
