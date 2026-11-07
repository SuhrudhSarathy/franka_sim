"""MuJoCo (C, CPU) backend: one ``MjData`` per env, optionally stepped on a thread pool."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from franka_sim.backends.base import Backend, as_env_ids


class CpuBackend(Backend):
    name = "cpu"

    def __init__(self, model: mujoco.MjModel, num_envs: int = 1, num_threads: int | None = None):
        super().__init__(model, num_envs)
        self.datas = [mujoco.MjData(model) for _ in range(num_envs)]
        # mj_step releases the GIL, so threads give real parallelism for larger batches.
        threads = num_threads if num_threads is not None else min(num_envs, 8)
        self._pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.forward()

    def read(self, field: str) -> np.ndarray:
        return np.stack([np.array(getattr(d, field), copy=True) for d in self.datas])

    def write(self, field: str, values: np.ndarray, env_ids=None) -> None:
        ids = as_env_ids(env_ids, self.num_envs)
        values = np.asarray(values)
        if values.shape[:1] != (len(ids),):
            values = np.broadcast_to(values, (len(ids),) + values.shape)
        for i, v in zip(ids, values):
            dst = getattr(self.datas[i], field)
            if isinstance(dst, np.ndarray):
                dst[...] = v.reshape(dst.shape)
            else:  # scalar fields such as ``time``
                setattr(self.datas[i], field, float(v))

    def reset(self, env_ids=None) -> None:
        for i in as_env_ids(env_ids, self.num_envs):
            mujoco.mj_resetData(self.model, self.datas[i])

    def _map(self, fn) -> None:
        if self._pool is None:
            for d in self.datas:
                fn(d)
        else:
            list(self._pool.map(fn, self.datas))

    def forward(self) -> None:
        self._map(lambda d: mujoco.mj_forward(self.model, d))

    def step(self, ctrl: np.ndarray, nsubsteps: int = 1) -> None:
        ctrl = np.broadcast_to(np.asarray(ctrl, dtype=np.float64), (self.num_envs, self.model.nu))
        for d, c in zip(self.datas, ctrl):
            d.ctrl[:] = c

        def run(d):
            for _ in range(nsubsteps):
                mujoco.mj_step(self.model, d)

        self._map(run)

    def sync_mjdata(self, env_id: int, data: mujoco.MjData) -> mujoco.MjData:
        src = self.datas[env_id]
        data.time = src.time
        data.qpos[:] = src.qpos
        data.qvel[:] = src.qvel
        data.act[:] = src.act
        data.ctrl[:] = src.ctrl
        data.mocap_pos[:] = src.mocap_pos
        data.mocap_quat[:] = src.mocap_quat
        mujoco.mj_forward(self.model, data)
        return data

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
