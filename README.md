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
