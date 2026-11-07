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
                      backend_kwargs={"device": "cuda:0", "nconmax": 64, "njmax": 256})
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
